"""FastAPI application — REST + WebSocket for the web UI."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, Coroutine, Literal

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .config import AppConfig, load_config, save_config
from .export import _XLSX_MEDIA_TYPE, build_force_xlsx, saved_run_to_xlsx
from .flow_analysis import flow_analysis_to_dict
from .flow_gen import build_flow_ramp
from .flow_runner import (
    FlowRun,
    flow_params_from_config,
    list_flow_runs,
    replay_flow,
    run_flow_test,
)
from .gcode_gen import build_sweep
from .gcode_preamble import is_indx, normalise_printer_model, validated_tool_index
from .gcode_parse import ParsedGcode, layer_polylines, parse_gcode
from .livemap_map import MappedRun, layer_detail, mapped_summary
from .livemap_runner import (
    LiveMapRun,
    list_livemap_runs,
    replay_livemap,
    run_live_map,
)
from .netutil import local_ip_toward
from .optimiser import (
    OPTIMISABLE_METRICS,
    discover_annotated_runs,
    optimise_weights,
    result_to_dict,
    write_weights_json,
)
from .probe_analysis import probe_analysis_to_dict
from .probe_gen import build_probe_test
from .probe_runner import (
    ProbeRun,
    list_probe_runs,
    probe_params_from_config,
    replay_probe,
    run_probe_test,
)
from .prusalink import PrusaLinkClient
from .replay import annotate_run, list_runs, read_annotation, replay
from .runner import TuningRun, _analysis_to_dict, params_from_config, run_tuning
from .udp_metrics import MetricStream

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class AppState:
    cfg: AppConfig
    stream: MetricStream
    current_run: TuningRun | None = None
    run_task: asyncio.Task | None = None
    # Max Flow test run (separate from the PA tuning run; only one of the
    # two may run at a time since they share the single loadcell stream).
    flow_run: FlowRun | None = None
    flow_run_task: asyncio.Task | None = None
    # Live Map run + the uploaded-but-not-yet-run gcode ("pending") + the
    # saved run currently opened for review (cached so per-layer paging
    # doesn't re-map every request). All three loadcell consumers
    # (PA / flow / livemap) are mutually exclusive -- one stream.
    livemap_run: LiveMapRun | None = None
    livemap_run_task: asyncio.Task | None = None
    livemap_pending: dict[str, Any] | None = None  # {name, gcode, parsed}
    livemap_review: dict[str, Any] | None = None    # {filename, mapped}
    # Touch-probe lateral characterisation run. Shares the single loadcell
    # stream with the other three modes -- mutually exclusive.
    probe_run: ProbeRun | None = None
    probe_run_task: asyncio.Task | None = None
    ws_clients: set[WebSocket]
    # Background optimiser jobs keyed by job_id. Each entry is a dict
    # the polling endpoint returns directly; updated in-place by the
    # progress callback (single-key writes are atomic enough for our
    # producer/consumer pattern across the thread boundary).
    optimise_jobs: dict[str, dict[str, Any]]

    def __init__(self):
        self.cfg = load_config()
        self.stream = MetricStream(port=self.cfg.udp_port)
        self.ws_clients = set()
        self.optimise_jobs = {}


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await state.stream.start()
    log.info("PrusaPATuner v%s started — UDP on port %d", __version__, state.cfg.udp_port)
    try:
        yield
    finally:
        state.stream.stop()


app = FastAPI(title="PrusaPATuner", version=__version__, lifespan=lifespan)

# Serve the bundled JS/CSS/assets under /static. The index.html the root route
# returns references /static/styles.css and /static/app.js, so this mount is
# what makes the page actually render with styling and behaviour.
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------- models ----------

class ConfigModel(BaseModel):
    printer_host: str = ""
    printer_api_key: str = ""
    printer_user: str = "maker"
    printer_password: str = ""
    printer_model: Literal["COREONE", "COREONEINDX"] = "COREONE"
    tool_index: int = Field(0, ge=0, le=7, strict=True)
    udp_port: int = 8514
    nozzle_temp: float = 215.0
    preheat_temp: float = 225.0
    nozzle_diameter: float = 0.4
    filament_diameter: float = 1.75
    slow_flow_mm3_s: float = Field(1.92, gt=0)
    fast_flow_mm3_s: float = Field(19.24, gt=0)
    slow_volume_mm3: float = Field(1.92, gt=0)
    fast_volume_mm3: float = Field(4.81, gt=0)
    cycles_per_K: int = Field(14, ge=1, le=64)
    accel_mm_s2: float = Field(5000.0, gt=0)
    k_min: float = Field(0.0, ge=0)
    k_max: float = Field(0.10, ge=0)
    k_step: float = Field(0.002, gt=0)
    purge_x: float = 30.0
    purge_y: float = 30.0
    purge_z: float = 50.0
    coupled_dx_mm: float = Field(0.05, ge=0)
    coupled_dy_mm: float = Field(0.0, ge=0)
    coupled_dz_mm: float = Field(0.0, ge=0)
    first_slow_leg_factor: float = Field(10.0, ge=1)
    filament_label: str = "PLA"

    # --- Max Flow test ---
    flow_min_mm3_s: float = Field(5.0, gt=0)
    flow_max_mm3_s: float = Field(30.0, gt=0)
    flow_step_mm3_s: float = Field(1.0, gt=0)
    flow_dwell_s: float = Field(3.0, gt=0)
    flow_settle_frac: float = Field(0.5, ge=0, lt=1)
    flow_warmup_s: float = Field(3.0, ge=0)
    flow_tare_dwell_s: float = Field(1.5, ge=0)

    # --- Touch probe (lateral characterisation) ---
    probe_axis: str = "X"
    probe_dir: str = "+"
    probe_start_x: float = 125.0
    probe_start_y: float = 110.0
    probe_z: float = Field(5.0, ge=0)
    probe_creep_mm: float = Field(1.0, gt=0, le=20)
    probe_slow_feed_mm_min: float = Field(30.0, gt=0)
    probe_travel_feed_mm_min: float = Field(3000.0, gt=0)
    probe_n_touches: int = Field(5, ge=1, le=50)
    probe_backoff_mm: float = Field(1.0, ge=0)
    probe_settle_ms: int = Field(300, ge=0, le=10000)
    probe_temp: float = Field(0.0, ge=0)

    @classmethod
    def from_appconfig(cls, c: AppConfig) -> "ConfigModel":
        values = {f: getattr(c, f) for f in cls.model_fields if hasattr(c, f)}
        values["printer_model"] = normalise_printer_model(c.printer_model)
        return cls(**values)

    def apply(self, c: AppConfig) -> AppConfig:
        for f in self.model_fields:
            if hasattr(c, f):
                setattr(c, f, getattr(self, f))
        return c


# ---------- routes ----------

@app.get("/")
async def root():
    index = STATIC_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return PlainTextResponse(
        "PrusaPATuner is running but static/index.html is missing — "
        "see the README for setup.",
        status_code=200,
    )


@app.get("/api/version")
async def get_version():
    return {"version": __version__}


@app.get("/api/config", response_model=ConfigModel)
async def get_config():
    return ConfigModel.from_appconfig(state.cfg)


@app.post("/api/config", response_model=ConfigModel)
async def post_config(model: ConfigModel):
    model.apply(state.cfg)
    save_config(state.cfg)
    return ConfigModel.from_appconfig(state.cfg)


@app.get("/api/status")
async def get_status():
    udp_stats = state.stream.stats
    run_dict = state.current_run.to_dict() if state.current_run else None
    return {
        "udp": udp_stats,
        "run": run_dict,
        "running": state.run_task is not None and not state.run_task.done(),
    }


@app.get("/api/preview")
async def get_preview():
    """Show the generated G-code without uploading anything."""
    try:
        udp_host = local_ip_toward(state.cfg.printer_host or "8.8.8.8")
    except Exception:
        udp_host = "192.168.1.10"
    plan = build_sweep(params_from_config(state.cfg, udp_host=udp_host))
    return PlainTextResponse(plan.gcode, media_type="text/x.gcode")


# All four run modes (PA / flow / livemap / probe) share the single
# loadcell stream, so at most one may run at a time. `_RUN_SLOTS` maps
# each mode to its AppState task attribute plus the two 409 message
# variants: the "already in progress" text used when the mode blocks
# itself, and the base text used when it blocks another mode.
_RUN_SLOTS: dict[str, tuple[str, str, str]] = {
    "pa": (
        "run_task",
        "A run is already in progress",
        "A PA tuning run is in progress",
    ),
    "flow": (
        "flow_run_task",
        "A flow run is already in progress",
        "A Max Flow run is in progress",
    ),
    "livemap": (
        "livemap_run_task",
        "A Live Map run is already in progress",
        "A Live Map run is in progress",
    ),
    "probe": (
        "probe_run_task",
        "A Touch Probe run is already in progress",
        "A Touch Probe run is in progress",
    ),
}


def _busy_reason(*, starting: str) -> str | None:
    """Return a human-readable 409 reason if any run task is active, else None.

    `starting` is the mode about to start (a `_RUN_SLOTS` key); it selects
    the endpoint-specific phrasing. Every endpoint checks its own slot
    first (the Live Map endpoint keeps its historical fixed order) and
    appends "; wait for it to finish" when blocked by another mode (the
    Live Map endpoint historically omits that suffix).
    """
    order = list(_RUN_SLOTS)
    if starting != "livemap":
        order.remove(starting)
        order.insert(0, starting)
    for key in order:
        attr, self_msg, other_msg = _RUN_SLOTS[key]
        task: asyncio.Task | None = getattr(state, attr)
        if task is None or task.done():
            continue
        if key == starting:
            return self_msg
        if starting == "livemap":
            return other_msg
        return other_msg + "; wait for it to finish"
    return None


def _require_printer_config() -> None:
    """400 unless the printer host + one credential are configured."""
    if not state.cfg.printer_host:
        raise HTTPException(400, "printer_host must be configured")
    if not (state.cfg.printer_api_key or state.cfg.printer_password):
        raise HTTPException(
            400, "either printer_api_key or printer_password must be configured"
        )
    try:
        model = normalise_printer_model(state.cfg.printer_model)
        validated_tool_index(model, state.cfg.tool_index)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _start_run(
    run_attr: str, task_attr: str, coro: Coroutine[Any, Any, Any]
) -> dict[str, str]:
    """Clear the mode's finished-run slot, launch `coro` as its task.

    `run_attr` / `task_attr` are AppState attribute names; the completed
    run object is stored back into `run_attr` when `coro` finishes.
    """
    setattr(state, run_attr, None)

    async def _go():
        setattr(state, run_attr, await coro)

    setattr(state, task_attr, asyncio.create_task(_go()))
    return {"status": "started"}


@app.post("/api/run")
async def post_run():
    busy = _busy_reason(starting="pa")
    if busy:
        raise HTTPException(409, busy)
    _require_printer_config()
    return _start_run(
        "current_run",
        "run_task",
        run_tuning(
            state.cfg, state.stream, on_update=partial(_broadcast_update, "run")
        ),
    )


@app.post("/api/cancel")
async def post_cancel():
    if state.run_task is not None and not state.run_task.done():
        state.run_task.cancel()
    return {"status": "ok"}


@app.get("/api/runs")
async def get_runs():
    """List `runs/run_*.npz` dumps available for replay.

    The frontend renders these in a dropdown; selecting one fires
    `POST /api/runs/<filename>/analyse` and renders the result in the
    same UI used for live sweeps.
    """
    runs = list_runs("runs")
    return {
        "runs": [
            {
                "filename": r.filename,
                "path": r.path,
                "mtime_unix": r.mtime_unix,
                "n_force": r.n_force,
                "n_pos": r.n_pos,
                "n_K": r.n_K,
                "cycles_per_K": r.cycles_per_K,
                "slow_half_s": r.slow_half_s,
                "fast_half_s": r.fast_half_s,
                "duration_s": r.duration_s,
                "filament_label": r.filament_label,
                "nozzle_temp": r.nozzle_temp,
                "printer_model": r.printer_model,
                "tool_index": r.tool_index,
                "user_k_opt": r.user_k_opt,
                "user_k_opt_notes": r.user_k_opt_notes,
            }
            for r in runs
        ]
    }


def _resolve_run_path(filename: str, prefix: str) -> Path:
    """Validate `filename` against path-traversal and return the resolved path.

    `prefix` is the per-mode filename prefix ("run_", "flow_", "livemap_",
    "probe_"). Raises HTTPException with the appropriate status on bad
    input or missing file. Used by every endpoint that takes a
    `{filename}` path parameter so the whitelist stays consistent.
    """
    if not filename.startswith(prefix) or not filename.endswith(".npz"):
        raise HTTPException(400, f"filename must match {prefix}*.npz")
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "filename must not contain path separators")
    path = Path("runs") / filename
    if not path.exists():
        raise HTTPException(404, f"run {filename} not found")
    return path


@app.post("/api/runs/{filename}/analyse")
async def post_run_analyse(filename: str):
    """Re-run `analyse_sweep` on a saved npz.

    Returns the same shape as the `analysis` field on `/api/status`,
    so the frontend can swap directly into its render path.
    """
    path = _resolve_run_path(filename, "run_")
    try:
        plan, analysis = replay(path)
    except Exception as exc:
        raise HTTPException(500, f"replay failed: {type(exc).__name__}: {exc}")
    user_k_opt, user_k_opt_notes = read_annotation(path)
    return {
        "filename": filename,
        "k_values": [seg.k for seg in plan.segments],
        "printer_model": normalise_printer_model(plan.params.printer_model),
        "tool_index": (
            plan.params.tool_index if is_indx(plan.params.printer_model) else None
        ),
        "analysis": _analysis_to_dict(analysis),
        "user_k_opt": user_k_opt,
        "user_k_opt_notes": user_k_opt_notes,
    }


def _xlsx_response(data: bytes, filename: str) -> Response:
    """Wrap xlsx bytes in a download response with the right headers."""
    return Response(
        content=data,
        media_type=_XLSX_MEDIA_TYPE,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _saved_run_xlsx(path: Path) -> Response:
    """Shared body of the four saved-run xlsx export endpoints."""
    try:
        data, out_name = saved_run_to_xlsx(path)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"export failed: {type(exc).__name__}: {exc}")
    return _xlsx_response(data, out_name)


def _current_run_xlsx(
    run: Any, prefix: str, missing_msg: str, fallback_name: str
) -> Response:
    """Shared body of the current-run xlsx export endpoints.

    The in-memory run carries no wall-clock anchor (only the npz dump
    does), so we capture one now: `force_t` are monotonic seconds and the
    monotonic/system clocks advance together, making
    `(time.monotonic(), time.time())` a valid conversion anchor.
    """
    if run is None or not run.force_t:
        raise HTTPException(404, missing_msg)
    try:
        data = build_force_xlsx(
            run.force_t,
            run.force_y,
            mono_anchor_mono=time.monotonic(),
            mono_anchor_unix=time.time(),
        )
    except Exception as exc:
        raise HTTPException(500, f"export failed: {type(exc).__name__}: {exc}")
    ts = int(run.state.started_at) if run.state.started_at else 0
    out_name = f"{prefix}{ts}.xlsx" if ts else fallback_name
    return _xlsx_response(data, out_name)


@app.get("/api/runs/{filename}/export.xlsx")
async def export_run_xlsx(filename: str):
    """Download a saved run's loadcell trace (timestamp + force) as xlsx.

    Two columns the user actually inspects -- wall-clock timestamp and
    nozzle force (grams) -- plus a zero-based `t_rel_s` for charting.
    See `export.saved_run_to_xlsx`.
    """
    return _saved_run_xlsx(_resolve_run_path(filename, "run_"))


@app.get("/api/current_run/export.xlsx")
async def export_current_run_xlsx():
    """Download the just-finished live run's loadcell trace as xlsx."""
    return _current_run_xlsx(
        state.current_run,
        "run_",
        "no completed run in memory -- finish a tuning run, or export a "
        "saved run from the replay dropdown",
        "current_run.xlsx",
    )


# ---------- Max Flow test ----------

@app.get("/api/flow/preview")
async def get_flow_preview():
    """Show the generated max-flow G-code without uploading anything."""
    try:
        udp_host = local_ip_toward(state.cfg.printer_host or "8.8.8.8")
    except Exception:
        udp_host = "192.168.1.10"
    plan = build_flow_ramp(flow_params_from_config(state.cfg, udp_host=udp_host))
    return PlainTextResponse(plan.gcode, media_type="text/x.gcode")


@app.post("/api/flow/run")
async def post_flow_run():
    busy = _busy_reason(starting="flow")
    if busy:
        raise HTTPException(409, busy)
    _require_printer_config()
    if state.cfg.flow_max_mm3_s <= state.cfg.flow_min_mm3_s:
        raise HTTPException(400, "flow_max_mm3_s must be greater than flow_min_mm3_s")
    return _start_run(
        "flow_run",
        "flow_run_task",
        run_flow_test(
            state.cfg, state.stream, on_update=partial(_broadcast_update, "flow_run")
        ),
    )


@app.post("/api/flow/cancel")
async def post_flow_cancel():
    if state.flow_run_task is not None and not state.flow_run_task.done():
        state.flow_run_task.cancel()
    return {"status": "ok"}


@app.get("/api/flow/status")
async def get_flow_status():
    return {
        "run": state.flow_run.to_dict() if state.flow_run else None,
        "running": state.flow_run_task is not None and not state.flow_run_task.done(),
    }


@app.get("/api/flow/runs")
async def get_flow_runs():
    """List `runs/flow_*.npz` dumps available for replay."""
    runs = list_flow_runs("runs")
    return {
        "runs": [
            {
                "filename": r.filename,
                "mtime_unix": r.mtime_unix,
                "n_force": r.n_force,
                "n_levels": r.n_levels,
                "min_flow": r.min_flow,
                "max_flow": r.max_flow,
                "flow_step": r.flow_step,
                "duration_s": r.duration_s,
                "filament_label": r.filament_label,
                "nozzle_temp": r.nozzle_temp,
                "printer_model": r.printer_model,
                "tool_index": r.tool_index,
            }
            for r in runs
        ]
    }


@app.post("/api/flow/runs/{filename}/analyse")
async def post_flow_run_analyse(filename: str):
    """Re-run `analyse_flow` on a saved flow npz."""
    path = _resolve_run_path(filename, "flow_")
    try:
        plan, analysis = replay_flow(path)
    except Exception as exc:
        raise HTTPException(500, f"replay failed: {type(exc).__name__}: {exc}")
    return {
        "filename": filename,
        "printer_model": normalise_printer_model(plan.params.printer_model),
        "tool_index": (
            plan.params.tool_index if is_indx(plan.params.printer_model) else None
        ),
        "analysis": flow_analysis_to_dict(analysis),
    }


@app.get("/api/flow/runs/{filename}/export.xlsx")
async def export_flow_run_xlsx(filename: str):
    """Download a saved flow run's loadcell trace (timestamp + force) as xlsx."""
    return _saved_run_xlsx(_resolve_run_path(filename, "flow_"))


@app.get("/api/flow/current_run/export.xlsx")
async def export_flow_current_run_xlsx():
    """Download the just-finished live flow run's loadcell trace as xlsx."""
    return _current_run_xlsx(
        state.flow_run,
        "flow_",
        "no completed flow run in memory -- finish a max-flow test, or "
        "export a saved one from the replay dropdown",
        "flow_run.xlsx",
    )


# ---------- Live Map ----------
# Upload a sliced ASCII .gcode, print it, and map the live nozzle-force
# stream onto a 2D per-layer preview. The three loadcell consumers
# (PA / flow / livemap) are mutually exclusive -- one physical stream.

def _livemap_pending_summary(parsed: ParsedGcode, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "n_layers": len(parsed.layers),
        "n_moves": len(parsed.moves),
        "n_extruding": parsed.n_extruding,
        "features": list(parsed.features),
        "bbox": list(parsed.bbox),
        "est_print_s": parsed.total_time_s,
        "layer_zs": [lyr.z for lyr in parsed.layers],
    }


@app.post("/api/livemap/upload")
async def post_livemap_upload(file: UploadFile = File(...)):
    """Accept a sliced ASCII .gcode, parse it, and cache it as the pending run.

    Rejects binary .bgcode (we can't parse feature/layer/speed out of it) with
    a message pointing at the PrusaSlicer ASCII-export toggle.
    """
    raw = await file.read()
    if raw[:4] == b"GCDE":
        raise HTTPException(
            400,
            "binary .bgcode is not supported -- re-export as plain ASCII .gcode "
            "(untick 'Export as binary G-code' in PrusaSlicer)",
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            400, "file is not valid UTF-8 text -- upload a plain ASCII .gcode"
        )
    parsed = await asyncio.to_thread(parse_gcode, text)
    if not parsed.moves:
        raise HTTPException(400, "no G0/G1 moves found -- is this a sliced .gcode?")
    name = file.filename or "upload.gcode"
    state.livemap_pending = {"name": name, "gcode": text, "parsed": parsed}
    return _livemap_pending_summary(parsed, name)


@app.get("/api/livemap/pending")
async def get_livemap_pending():
    if not state.livemap_pending:
        raise HTTPException(404, "no gcode uploaded yet")
    p = state.livemap_pending
    return _livemap_pending_summary(p["parsed"], p["name"])


@app.get("/api/livemap/pending/layer/{idx}")
async def get_livemap_pending_layer(idx: int):
    """Geometry backdrop (feature-grouped polylines) for one uploaded layer."""
    if not state.livemap_pending:
        raise HTTPException(404, "no gcode uploaded yet")
    parsed: ParsedGcode = state.livemap_pending["parsed"]
    if idx < 0 or idx >= len(parsed.layers):
        raise HTTPException(404, f"layer {idx} out of range")
    lyr = parsed.layers[idx]
    return {"index": idx, "z": lyr.z, "polylines": layer_polylines(parsed, idx)}


@app.post("/api/livemap/run")
async def post_livemap_run():
    busy = _busy_reason(starting="livemap")
    if busy:
        raise HTTPException(409, busy)
    if not state.livemap_pending:
        raise HTTPException(400, "upload a .gcode first")
    _require_printer_config()

    pending = state.livemap_pending
    return _start_run(
        "livemap_run",
        "livemap_run_task",
        run_live_map(
            state.cfg, state.stream, pending["gcode"], pending["parsed"],
            pending["name"], on_update=partial(_broadcast_update, "livemap_run"),
        ),
    )


@app.post("/api/livemap/attach")
async def post_livemap_attach():
    """Capture the pending G-code against the printer's current active job.

    This deliberately performs no upload or printer-control write.  It is the
    recovery path when PrusaLink starts a large upload but its HTTP response
    times out, and it is also useful when the tuner is opened just after a
    telemetry-enabled print has begun.
    """
    busy = _busy_reason(starting="livemap")
    if busy:
        raise HTTPException(409, busy)
    if not state.livemap_pending:
        raise HTTPException(400, "upload the matching .gcode first")
    _require_printer_config()

    pending = state.livemap_pending
    return _start_run(
        "livemap_run",
        "livemap_run_task",
        run_live_map(
            state.cfg,
            state.stream,
            pending["gcode"],
            pending["parsed"],
            pending["name"],
            on_update=partial(_broadcast_update, "livemap_run"),
            attach_existing=True,
        ),
    )


@app.post("/api/livemap/cancel")
async def post_livemap_cancel():
    if state.livemap_run_task is not None and not state.livemap_run_task.done():
        state.livemap_run_task.cancel()
    return {"status": "ok"}


@app.get("/api/livemap/status")
async def get_livemap_status():
    return {
        "run": state.livemap_run.to_dict() if state.livemap_run else None,
        "running": state.livemap_run_task is not None and not state.livemap_run_task.done(),
        "has_pending": bool(state.livemap_pending),
    }


@app.get("/api/livemap/runs")
async def get_livemap_runs():
    """List runs/livemap_*.npz dumps available for review."""
    runs = list_livemap_runs("runs")
    return {
        "runs": [
            {
                "filename": r.filename,
                "mtime_unix": r.mtime_unix,
                "n_force": r.n_force,
                "n_layers": r.n_layers,
                "n_moves": r.n_moves,
                "duration_s": r.duration_s,
                "est_print_s": r.est_print_s,
                "gcode_name": r.gcode_name,
            }
            for r in runs
        ]
    }


async def _load_livemap_review(filename: str) -> MappedRun:
    """Return the cached MappedRun for `filename`, mapping it if not cached.

    Mapping (parse + interpolate + cross-check) is CPU-bound, so it runs in a
    thread to keep the event loop responsive on big prints.
    """
    review = state.livemap_review
    if review and review.get("filename") == filename:
        return review["mapped"]
    path = _resolve_run_path(filename, "livemap_")
    mapped = await asyncio.to_thread(replay_livemap, path)
    state.livemap_review = {"filename": filename, "mapped": mapped}
    return mapped


@app.post("/api/livemap/runs/{filename}/analyse")
async def post_livemap_analyse(filename: str):
    """Map a saved run (tare + layering + feature stats + cross-check)."""
    _resolve_run_path(filename, "livemap_")  # validate before threading
    try:
        mapped = await _load_livemap_review(filename)
    except Exception as exc:
        raise HTTPException(500, f"replay failed: {type(exc).__name__}: {exc}")
    return mapped_summary(mapped, filename)


@app.get("/api/livemap/runs/{filename}/layer/{idx}")
async def get_livemap_run_layer(filename: str, idx: int):
    """Geometry + mapped force points for one layer of a saved run."""
    _resolve_run_path(filename, "livemap_")
    try:
        mapped = await _load_livemap_review(filename)
    except Exception as exc:
        raise HTTPException(500, f"replay failed: {type(exc).__name__}: {exc}")
    if idx < 0 or idx >= len(mapped.parsed.layers):
        raise HTTPException(404, f"layer {idx} out of range")
    return layer_detail(mapped, idx)


@app.get("/api/livemap/runs/{filename}/export.xlsx")
async def export_livemap_run_xlsx(filename: str):
    """Download a saved Live Map run's loadcell trace (timestamp + force)."""
    return _saved_run_xlsx(_resolve_run_path(filename, "livemap_"))


# ---------- Touch Probe (lateral characterisation) ----------
# Push the nozzle tip sideways into a rigid target and capture loadcell +
# position to see whether lateral contact registers cleanly. Shares the
# single loadcell stream with the other three modes (mutually exclusive).

@app.get("/api/probe/preview")
async def get_probe_preview():
    """Show the generated touch-probe G-code without uploading anything."""
    try:
        udp_host = local_ip_toward(state.cfg.printer_host or "8.8.8.8")
    except Exception:
        udp_host = "192.168.1.10"
    plan = build_probe_test(probe_params_from_config(state.cfg, udp_host=udp_host))
    return PlainTextResponse(plan.gcode, media_type="text/x.gcode")


@app.post("/api/probe/run")
async def post_probe_run():
    busy = _busy_reason(starting="probe")
    if busy:
        raise HTTPException(409, busy)
    _require_printer_config()
    return _start_run(
        "probe_run",
        "probe_run_task",
        run_probe_test(
            state.cfg, state.stream, on_update=partial(_broadcast_update, "probe_run")
        ),
    )


@app.post("/api/probe/cancel")
async def post_probe_cancel():
    if state.probe_run_task is not None and not state.probe_run_task.done():
        state.probe_run_task.cancel()
    return {"status": "ok"}


@app.get("/api/probe/status")
async def get_probe_status():
    return {
        "run": state.probe_run.to_dict() if state.probe_run else None,
        "running": state.probe_run_task is not None and not state.probe_run_task.done(),
    }


@app.get("/api/probe/runs")
async def get_probe_runs():
    """List `runs/probe_*.npz` dumps available for replay."""
    runs = list_probe_runs("runs")
    return {
        "runs": [
            {
                "filename": r.filename,
                "mtime_unix": r.mtime_unix,
                "n_force": r.n_force,
                "axis": r.axis,
                "dir": r.dir,
                "n_touches": r.n_touches,
                "creep_mm": r.creep_mm,
                "duration_s": r.duration_s,
                "printer_model": r.printer_model,
                "tool_index": r.tool_index,
            }
            for r in runs
        ]
    }


@app.post("/api/probe/runs/{filename}/analyse")
async def post_probe_run_analyse(filename: str):
    """Re-run `analyse_probe` on a saved probe npz."""
    path = _resolve_run_path(filename, "probe_")
    try:
        plan, analysis = replay_probe(path)
    except Exception as exc:
        raise HTTPException(500, f"replay failed: {type(exc).__name__}: {exc}")
    return {
        "filename": filename,
        "printer_model": normalise_printer_model(plan.params.printer_model),
        "tool_index": (
            plan.params.tool_index if is_indx(plan.params.printer_model) else None
        ),
        "analysis": probe_analysis_to_dict(analysis),
    }


@app.get("/api/livemap/printer-z")
async def get_livemap_printer_z():
    """Read the printer's logical Z for authoritative live-layer tracking."""
    _require_printer_config()
    try:
        async with PrusaLinkClient(
            state.cfg.printer_host,
            state.cfg.printer_api_key,
            password=state.cfg.printer_password,
            user=state.cfg.printer_user,
            timeout=4.0,
        ) as printer:
            status = await printer.status()
    except Exception as exc:
        log.warning("livemap printer-z status failed: %s", type(exc).__name__)
        raise HTTPException(503, "printer status is temporarily unavailable") from exc
    printer_state = status.get("printer") if isinstance(status, dict) else None
    axis_z = printer_state.get("axis_z") if isinstance(printer_state, dict) else None
    if not isinstance(axis_z, (int, float)):
        raise HTTPException(503, "printer status did not include axis_z")
    return {
        "axis_z": float(axis_z),
        "state": printer_state.get("state"),
        "online": True,
        "stale_sec": 0,
        "run_started_at": (
            state.livemap_run.state.started_at if state.livemap_run else None
        ),
    }


@app.get("/api/probe/runs/{filename}/export.xlsx")
async def export_probe_run_xlsx(filename: str):
    """Download a saved probe run's loadcell trace (timestamp + force) as xlsx."""
    return _saved_run_xlsx(_resolve_run_path(filename, "probe_"))


@app.get("/api/probe/current_run/export.xlsx")
async def export_probe_current_run_xlsx():
    """Download the just-finished live probe run's loadcell trace as xlsx."""
    return _current_run_xlsx(
        state.probe_run,
        "probe_",
        "no completed probe run in memory -- finish a touch-probe test, or "
        "export a saved one from the replay dropdown",
        "probe_run.xlsx",
    )


class AnnotateModel(BaseModel):
    """Request body for `POST /api/runs/<filename>/annotate`.

    `k` is the user's measured ground-truth K from a test print, or
    `null` to clear the annotation. `notes` is free-text context
    (filament, test method, date — anything the user wants to remember
    when they look at this run again later).
    """
    k: float | None = None
    notes: str = ""


@app.post("/api/runs/{filename}/annotate")
async def post_run_annotate(filename: str, body: AnnotateModel):
    """Attach (or clear) the user-supplied ground-truth K on a saved run.

    Re-writes the npz atomically. The new `user_k_opt` / `user_k_opt_notes`
    flow back through `/api/runs` and `/api/runs/<f>/analyse` so the UI
    can pre-fill the input on re-open.
    """
    path = _resolve_run_path(filename, "run_")
    try:
        annotate_run(path, body.k, body.notes)
    except Exception as exc:
        raise HTTPException(500, f"annotate failed: {type(exc).__name__}: {exc}")
    k, notes = read_annotation(path)
    return {"filename": filename, "user_k_opt": k, "user_k_opt_notes": notes}


class OptimiseStartModel(BaseModel):
    """Request body for `POST /api/optimise_weights/start`.

    `excluded_metrics` is a list of metric names (subset of the 8
    optimisable metrics) to lock at weight 0 -- the UI checkboxes feed
    this. `alpha` trades bias against bootstrap stability (default 1.0
    weights them equally). `n_boot`, `popsize`, `maxiter`, `seed` are
    inner-loop knobs exposed for power-user tuning.
    """
    excluded_metrics: list[str] = Field(default_factory=list)
    alpha: float = 1.0
    n_boot: int = Field(200, ge=20, le=2000)
    popsize: int = Field(15, ge=4, le=64)
    maxiter: int = Field(50, ge=5, le=500)
    seed: int = 0xC0FFEE


def _optimise_worker(job_id: str, body: OptimiseStartModel) -> None:
    """Runs in a thread (DE is CPU-bound, blocks the event loop otherwise).

    Writes progress + final result into `state.optimise_jobs[job_id]`.
    """
    job = state.optimise_jobs[job_id]

    def progress(frac: float, msg: str) -> None:
        job["progress"] = float(frac)
        job["message"] = msg

    try:
        paths = discover_annotated_runs("runs")
        if not paths:
            raise ValueError(
                "no annotated runs found -- annotate at least one saved npz "
                "with its ground-truth K via the replay panel first"
            )
        result = optimise_weights(
            paths,
            excluded_metrics=body.excluded_metrics,
            alpha=body.alpha,
            n_boot=body.n_boot,
            popsize=body.popsize,
            maxiter=body.maxiter,
            seed=body.seed,
            progress_cb=progress,
        )
        job["result"] = result_to_dict(result)
        job["status"] = "done"
    except Exception as exc:
        log.exception("optimiser job %s failed", job_id)
        job["status"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        job["progress"] = 1.0


@app.get("/api/optimise_weights/info")
async def get_optimise_info():
    """Return the metric names the optimiser can tune + count of annotated runs.

    The GUI uses this to render the checkbox list and the "N annotated
    runs available" hint above the Run button.
    """
    return {
        "optimisable_metrics": list(OPTIMISABLE_METRICS),
        "annotated_run_count": len(discover_annotated_runs("runs")),
    }


@app.post("/api/optimise_weights/start")
async def post_optimise_start(body: OptimiseStartModel):
    """Kick off a background optimiser job. Returns a `job_id` the UI
    polls via `GET /api/optimise_weights/<job_id>`.
    """
    import uuid

    job_id = uuid.uuid4().hex[:12]
    state.optimise_jobs[job_id] = {
        "status": "running",
        "progress": 0.0,
        "message": "queued",
        "result": None,
        "error": None,
    }
    asyncio.create_task(asyncio.to_thread(_optimise_worker, job_id, body))
    return {"job_id": job_id}


@app.get("/api/optimise_weights/{job_id}")
async def get_optimise_job(job_id: str):
    """Poll the status of a running optimiser job."""
    job = state.optimise_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    return {"job_id": job_id, **job}


@app.post("/api/optimise_weights/{job_id}/apply")
async def post_optimise_apply(job_id: str):
    """Persist the optimised weights from a done job to `runs/weights_opt.json`.

    The server preloads that file at startup; existing UI weight
    sliders pre-fill from it on the next analysis render.
    """
    job = state.optimise_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"job {job_id} not found")
    if job.get("status") != "done":
        raise HTTPException(409, f"job is in state {job.get('status')!r}, not done")
    result_dict = job.get("result")
    if not result_dict:
        raise HTTPException(500, "job marked done but has no result")
    # Reconstruct a minimal OptimiseResult-shaped object for write_weights_json.
    # write_weights_json only reads a handful of attributes; pass a small
    # SimpleNamespace so we don't have to round-trip through OptimiseResult.
    from types import SimpleNamespace
    payload = SimpleNamespace(
        weights_display=result_dict["weights_display"],
        alpha=result_dict["alpha"],
        n_runs=result_dict["n_runs"],
        rms_error=result_dict["rms_error"],
        median_abs_error=result_dict["median_abs_error"],
        excluded_metrics=result_dict["excluded_metrics"],
        timestamp_unix=result_dict["timestamp_unix"],
    )
    try:
        write_weights_json(payload)
    except Exception as exc:
        raise HTTPException(500, f"failed to write weights file: {exc}")
    return {"job_id": job_id, "written": "runs/weights_opt.json"}


@app.get("/api/metrics_seen")
async def get_metrics_seen():
    """Diagnostic: which metric names have we observed and how many samples."""
    stats = state.stream.stats
    return {
        "stats": stats,
        "names": {
            name: len(state.stream.snapshot(name))
            for name in sorted(state.stream._rings.keys())  # noqa: SLF001 — diagnostic only
        },
    }


@app.get("/api/diagnostics")
async def get_diagnostics(window_s: float = 5.0):
    """Live diagnostics: packet stats + per-metric sample rate.

    Use this to verify each metric is actually streaming and at what rate.
    If `loadcell_value` reads zero or far below ~100 Hz the M331 enable
    almost certainly silently failed (most likely cause: the firmware on
    your printer doesn't expose that metric name -- the M331 handler
    writes "Metric not found" to serial, which PrusaLink discards). The
    Buddy throttle is COMPILE-TIME per `METRIC_DEF` -- gcode can't change
    it, only enable/disable.
    """
    rates = state.stream.metric_rates(window_s=window_s)
    return {
        "stats": state.stream.stats,
        "rates_hz": rates,
        "window_s": window_s,
    }


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws.accept()
    state.ws_clients.add(ws)
    # bootstrap with current run state
    if state.current_run is not None:
        try:
            await ws.send_json({"type": "run", "data": state.current_run.to_dict()})
        except Exception:
            pass
    if state.flow_run is not None:
        try:
            await ws.send_json({"type": "flow_run", "data": state.flow_run.to_dict()})
        except Exception:
            pass
    if state.livemap_run is not None:
        try:
            await ws.send_json({"type": "livemap_run", "data": state.livemap_run.to_dict()})
        except Exception:
            pass
    if state.probe_run is not None:
        try:
            await ws.send_json({"type": "probe_run", "data": state.probe_run.to_dict()})
        except Exception:
            pass
    # Live loadcell fan-out: single subscriber to loadcell_value (the only
    # loadcell metric this firmware emits). We BATCH samples instead of
    # sending one WS frame per sample -- at ~180 Hz, per-sample frames
    # overflowed the subscriber queue. Batching at ~50 ms windows
    # (~9 samples per frame) drops WebSocket overhead and lets the queue
    # drain. Sample timestamps are already spread within each UDP packet
    # by MetricStream._on_packet, so the receiver sees a continuous time
    # series rather than vertical clusters at packet-arrival times.
    BATCH_S = 0.05  # 20 Hz flush rate

    async def forward(metric: str, msg_type: str) -> None:
        import time as _time
        batch_t: list[float] = []
        batch_v: list[float] = []
        last_flush = _time.monotonic()
        try:
            async for sample in state.stream.subscribe(metric):
                v = _first_numeric(sample.fields)
                if v is None:
                    continue
                batch_t.append(sample.recv_monotonic)
                batch_v.append(v)
                now = _time.monotonic()
                if now - last_flush >= BATCH_S or len(batch_t) >= 64:
                    try:
                        await ws.send_json(
                            {
                                "type": msg_type,
                                "metric": metric,
                                "t": batch_t,
                                "v": batch_v,
                            }
                        )
                    except WebSocketDisconnect:
                        return
                    except Exception:
                        return
                    batch_t = []
                    batch_v = []
                    last_flush = now
        except Exception:
            return

    # Two parallel forwarders: the force trace and the X-position trace.
    # Each rides its own batched WebSocket message type so the client can
    # plot them on independent y-axes without metric-name disambiguation.
    tasks = [
        asyncio.create_task(forward("loadcell_value", "force_batch")),
        asyncio.create_task(forward("pos_x", "pos_batch")),
        # pos_y / pos_z power the Live Map 2D placement + layer detection.
        # They only stream when M331-enabled (the Live Map preamble does so);
        # PA / flow modes simply ignore these message types.
        asyncio.create_task(forward("pos_y", "pos_y_batch")),
        asyncio.create_task(forward("pos_z", "pos_z_batch")),
    ]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        for t in tasks:
            t.cancel()
        state.ws_clients.discard(ws)


def _first_numeric(fields: dict) -> float | None:
    import math
    for v in fields.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                continue
            return x
    return None


async def _broadcast_update(
    msg_type: str, run: TuningRun | FlowRun | LiveMapRun | ProbeRun
) -> None:
    """Fan a run snapshot out to all WebSocket clients as `msg_type`.

    The runners' `on_update` callbacks take just the run object, so the
    run-start endpoints bind `msg_type` ("run" / "flow_run" /
    "livemap_run" / "probe_run") with `functools.partial`.
    """
    # Keep the in-memory snapshot current while a task is running, not only
    # after it finishes.  This makes REST status and a newly reconnected
    # WebSocket able to bootstrap an attached Live Map with its Z/layer seed.
    run_attr = {
        "run": "current_run",
        "flow_run": "flow_run",
        "livemap_run": "livemap_run",
        "probe_run": "probe_run",
    }.get(msg_type)
    if run_attr is not None:
        setattr(state, run_attr, run)

    payload = {"type": msg_type, "data": run.to_dict()}
    dead: list[WebSocket] = []
    for ws in list(state.ws_clients):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.ws_clients.discard(ws)
