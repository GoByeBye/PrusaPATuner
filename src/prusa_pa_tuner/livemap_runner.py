"""End-to-end orchestration for one Live Map run.

Mirrors flow_runner / runner: inject the metric preamble into the user's
uploaded gcode, upload + auto-print, capture loadcell_value + pos_x/y/z while
polling PrusaLink to completion, then dump the raw streams + the embedded
gcode to runs/livemap_<ts>.npz for self-contained per-layer review.

Live placement happens client-side off the WebSocket (force interpolated onto
streamed position). The offline mapping / cross-check lives in livemap_map and
runs on the saved npz when a run is opened for review.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import httpx

from .config import AppConfig
from .gcode_parse import ParsedGcode, parse_gcode
from .livemap_gen import build_livemap_gcode
from .livemap_map import MappedRun, map_run
from .netutil import local_ip_toward
from .prusalink import JobStatus, PrusaLinkClient
from .run_lifecycle import (
    TRANSIENT_HTTPX_ERRORS,
    anchor_fields,
    cancel_collectors,
    collect_metric,
    npz_scalar,
    poll_job_until_done,
    runs_dir,
    schedule_on_update,
)
from .sampling import extract_force, extract_numeric, sort_by_time
from .udp_metrics import MetricStream

log = logging.getLogger(__name__)

_ACTIVE_JOB_STATES = {"PRINTING", "PAUSED", "BUSY"}
_TERMINAL_JOB_STATES = {"FINISHED", "STOPPED", "CANCELLED", "IDLE"}


def _job_filenames(status: JobStatus) -> list[str]:
    """Return usable current-job names, preferring PrusaLink display_name."""
    file_info = status.raw.get("file")
    if isinstance(file_info, str):
        values = [file_info]
    elif isinstance(file_info, dict):
        values = [
            file_info.get("display_name"),
            file_info.get("name"),
            file_info.get("path"),
        ]
    else:
        values = []
    return [Path(str(value)).name for value in values if value]


def _job_matches_filename(status: JobStatus, expected_filename: str) -> bool:
    expected = Path(expected_filename).name.casefold()
    return any(name.casefold() == expected for name in _job_filenames(status))


async def _wait_for_uploaded_job(
    pl: PrusaLinkClient,
    expected_filename: str,
    *,
    grace_s: float,
    poll_interval_s: float,
    upload_error: httpx.TransportError | None = None,
) -> JobStatus:
    """Reconcile one upload PUT with the exact job that appears afterward.

    PrusaLink may return 204 for several seconds while it finalises a file, and
    the upload response itself may time out after the firmware accepted it.
    Polling is safe and read-only; retrying the PUT is not.
    """
    deadline = time.monotonic() + max(0.0, grace_s)
    last_detail = "no job visible"
    while True:
        try:
            status = await pl.get_job_status()
        except TRANSIENT_HTTPX_ERRORS as exc:
            status = None
            last_detail = f"status check failed: {type(exc).__name__}"

        if status is not None:
            state = status.state.upper()
            names = _job_filenames(status)
            matches = _job_matches_filename(status, expected_filename)
            if matches and (state in _ACTIVE_JOB_STATES or state == "FINISHED"):
                return status
            if state in _ACTIVE_JOB_STATES and names and not matches:
                raise RuntimeError(
                    "printer started a different job while confirming upload: "
                    f"expected {expected_filename!r}, got {names[0]!r}; upload "
                    "was not retried"
                ) from upload_error
            if matches and state in _TERMINAL_JOB_STATES:
                raise RuntimeError(
                    f"uploaded job entered {state} before collection started"
                ) from upload_error
            last_detail = f"state={state}, files={names or ['unknown']}"

        if time.monotonic() >= deadline:
            raise RuntimeError(
                "upload outcome is unknown after waiting for the exact printer "
                f"job {expected_filename!r} ({last_detail}); upload was not retried"
            ) from upload_error
        await asyncio.sleep(max(0.0, poll_interval_s))


@dataclass(slots=True)
class LiveMapRunState:
    state: str = "idle"  # idle | preparing | running | saving | done | error
    message: str = ""
    progress_pct: float = 0.0
    started_at: float = 0.0
    error: str | None = None
    saved_filename: str | None = None  # set once the npz is on disk


@dataclass(slots=True)
class LiveMapRun:
    cfg: AppConfig
    name: str               # original upload filename (for display)
    user_gcode: str         # the uploaded ASCII gcode, untouched
    parsed: ParsedGcode
    attached_existing: bool = False
    attach_axis_z: float | None = None
    attach_layer_index: int | None = None
    state: LiveMapRunState = field(default_factory=LiveMapRunState)
    force_t: list[float] = field(default_factory=list)
    force_y: list[float] = field(default_factory=list)
    pos_x_t: list[float] = field(default_factory=list)
    pos_x: list[float] = field(default_factory=list)
    pos_y_t: list[float] = field(default_factory=list)
    pos_y: list[float] = field(default_factory=list)
    pos_z_t: list[float] = field(default_factory=list)
    pos_z: list[float] = field(default_factory=list)
    on_update: Callable[["LiveMapRun"], None] | None = None

    def emit(self) -> None:
        schedule_on_update(self.on_update, self, log, "livemap on_update callback failed")

    def to_dict(self) -> dict[str, Any]:
        s = self.state
        return {
            "state": s.state,
            "message": s.message,
            "progress_pct": s.progress_pct,
            "started_at": s.started_at,
            "error": s.error,
            "saved_filename": s.saved_filename,
            "name": self.name,
            "n_force_samples": len(self.force_t),
            "n_layers": len(self.parsed.layers),
            "n_moves": len(self.parsed.moves),
            "bbox": list(self.parsed.bbox),
            "est_print_s": self.parsed.total_time_s,
            "attached_existing": self.attached_existing,
            "attach_axis_z": self.attach_axis_z,
            "attach_layer_index": self.attach_layer_index,
        }


def _nearest_layer_index(parsed: ParsedGcode, axis_z: float) -> int | None:
    """Return a nearby parsed layer, rejecting setup travel and large Z hops."""
    if not math.isfinite(axis_z) or not parsed.layers:
        return None
    best = min(
        range(len(parsed.layers)), key=lambda i: abs(parsed.layers[i].z - axis_z)
    )
    zs = [layer.z for layer in parsed.layers]
    pitch = (
        float(np.median(np.diff(np.asarray(zs, dtype=float))))
        if len(zs) >= 2 else 0.2
    )
    tolerance = max(0.6 * pitch, 0.3)
    return best if abs(zs[best] - axis_z) <= tolerance else None


async def run_live_map(
    cfg: AppConfig,
    stream: MetricStream,
    user_gcode: str,
    parsed: ParsedGcode,
    name: str,
    *,
    on_update: Callable[["LiveMapRun"], None] | None = None,
    loadcell_metric: str = "loadcell_value",
    poll_interval_s: float = 1.0,
    job_timeout_s: float | None = None,
    attach_existing: bool = False,
    upload_start_grace_s: float = 60.0,
    upload_status_poll_s: float = 0.75,
) -> LiveMapRun:
    udp_host = local_ip_toward(cfg.printer_host, port=80)
    upload_gcode = None
    if not attach_existing:
        upload_gcode = build_livemap_gcode(
            user_gcode, udp_host=udp_host, udp_port=cfg.udp_port,
            loadcell_metric=loadcell_metric,
        )

    if job_timeout_s is None:
        est = parsed.total_time_s
        job_timeout_s = max(900.0, 2.0 * est + 600.0)

    run = LiveMapRun(
        cfg=cfg,
        name=name,
        user_gcode=user_gcode,
        parsed=parsed,
        attached_existing=attach_existing,
        on_update=on_update,
    )
    run.state.started_at = time.time()
    run.state.state = "preparing"
    if attach_existing:
        run.state.message = "Attaching to the print already in progress…"
    else:
        run.state.message = (
            f"Uploading {name} ({len(parsed.layers)} layers, "
            f"~{parsed.total_time_s / 60:.0f} min est)…"
        )
    # Attach mode waits until the read-only printer status query supplies its
    # current Z/layer seed.  Its first WebSocket snapshot is then immediately
    # useful to a client joining partway through the print.
    if not attach_existing:
        run.emit()

    stop_collect = asyncio.Event()

    collectors = [
        asyncio.create_task(collect_metric(
            stream, loadcell_metric, stop_collect, extract_force,
            run.force_t, run.force_y,
        )),
        asyncio.create_task(collect_metric(
            stream, "pos_x", stop_collect, extract_numeric,
            run.pos_x_t, run.pos_x,
        )),
        asyncio.create_task(collect_metric(
            stream, "pos_y", stop_collect, extract_numeric,
            run.pos_y_t, run.pos_y,
        )),
        asyncio.create_task(collect_metric(
            stream, "pos_z", stop_collect, extract_numeric,
            run.pos_z_t, run.pos_z,
        )),
    ]

    try:
        async with PrusaLinkClient(
            cfg.printer_host, cfg.printer_api_key,
            password=cfg.printer_password, user=cfg.printer_user or "maker",
        ) as pl:
            if attach_existing:
                status = await pl.get_job_status()
                if status is None or status.state.upper() not in {
                    "PRINTING", "PAUSED", "BUSY",
                }:
                    current = "no active job" if status is None else status.state
                    raise RuntimeError(
                        "cannot attach Live Map: printer is not actively printing "
                        f"({current})"
                    )
                run.state.progress_pct = status.progress_pct
                try:
                    printer_status = await pl.status()
                    raw_axis_z = printer_status.get("printer", {}).get("axis_z")
                    axis_z = float(raw_axis_z)
                    layer_index = _nearest_layer_index(parsed, axis_z)
                    if layer_index is not None:
                        run.attach_axis_z = axis_z
                        run.attach_layer_index = layer_index
                except Exception as exc:
                    # Z seeding improves the live UI but is not required for
                    # the server-side capture or its saved offline mapping.
                    log.warning("could not seed attached Live Map layer from Z: %s", exc)
            else:
                filename = (
                    f"livemap_{int(time.time())}_{uuid.uuid4().hex[:10]}.gcode"
                )
                upload_error: httpx.TransportError | None = None
                try:
                    await pl.upload_and_print(filename, upload_gcode or b"")
                except httpx.TransportError as exc:
                    # The firmware may have accepted Print-After-Upload before
                    # the response failed. Never repeat this non-idempotent PUT;
                    # reconcile it by exact display filename instead.
                    upload_error = exc
                    log.warning(
                        "livemap upload response failed (%s); waiting for exact "
                        "job %s without retrying upload",
                        type(exc).__name__, filename,
                    )
                status = await _wait_for_uploaded_job(
                    pl,
                    filename,
                    grace_s=upload_start_grace_s,
                    poll_interval_s=upload_status_poll_s,
                    upload_error=upload_error,
                )
                run.state.progress_pct = status.progress_pct
            run.state.state = "running"
            if attach_existing and run.attach_layer_index is not None:
                run.state.message = (
                    f"Attached at layer {run.attach_layer_index + 1} "
                    f"(Z {run.attach_axis_z:.2f} mm); mapping nozzle force live…"
                )
            elif attach_existing:
                run.state.message = "Attached to current print; mapping nozzle force live…"
            else:
                run.state.message = "Printing; mapping nozzle force live…"
            run.emit()

            warned_no_pos = False

            def warn_if_no_pos(elapsed_s: float) -> None:
                # Fail-fast: if force is streaming but position isn't, the map
                # will be empty (e.g. the wrong metric -- 'ipos' instead of
                # 'pos' -- was enabled). Warn within ~20 s rather than after the
                # whole print.
                nonlocal warned_no_pos
                if (not warned_no_pos and elapsed_s > 20.0
                        and len(run.force_t) > 50 and len(run.pos_x) == 0):
                    warned_no_pos = True
                    run.state.message = (
                        "⚠ No position telemetry (pos_x = 0 samples) — the load "
                        "map will be EMPTY. Enable the pos_x/pos_y/pos_z metrics "
                        "on the printer (note: 'pos', not 'ipos'), then re-run."
                    )
                    run.emit()
                    log.warning(
                        "livemap: %d force samples but 0 pos_x after %ds -- "
                        "position metrics not streaming; map will be empty",
                        len(run.force_t), int(elapsed_s),
                    )

            await poll_job_until_done(
                pl, run,
                job_timeout_s=job_timeout_s,
                poll_interval_s=poll_interval_s,
                verb="Printing",
                log_timeout_done=lambda final_state: log.warning(
                    "livemap job_timeout reached but final status %r -- "
                    "treating as done", final_state,
                ),
                on_tick=warn_if_no_pos,
            )

        run.state.state = "saving"
        run.state.message = "Saving captured data…"
        run.emit()
        saved = _dump_livemap_run(run)
        run.state.saved_filename = saved
        run.state.state = "done"
        run.state.message = "Done." if saved else "Done (save failed; see log)."
        run.state.progress_pct = 100.0
        run.emit()

    except Exception as exc:
        run.state.state = "error"
        run.state.error = f"{type(exc).__name__}: {exc}"
        run.state.message = run.state.error
        log.exception("livemap run failed")
        run.emit()
    finally:
        stop_collect.set()
        await cancel_collectors(collectors, log)
        log.info(
            "livemap captured: %d force, %d pos_x, %d pos_y, %d pos_z samples",
            len(run.force_t), len(run.pos_x), len(run.pos_y), len(run.pos_z),
        )

    return run


def _dump_livemap_run(run: LiveMapRun) -> str | None:
    """Save raw streams + the embedded gcode to runs/livemap_<ts>.npz.

    Returns the filename on success (None on failure). Compressed because the
    embedded gcode can be large; np.load reads it back transparently.
    """
    try:
        fname = f"livemap_{int(run.state.started_at)}.npz"
        dump_path = runs_dir() / fname

        force_t = np.asarray(run.force_t, dtype=float)
        force_y = np.asarray(run.force_y, dtype=float)
        force_t, force_y = sort_by_time(force_t, force_y)
        px_t = np.asarray(run.pos_x_t, dtype=float)
        px = np.asarray(run.pos_x, dtype=float)
        px_t, px = sort_by_time(px_t, px)
        py_t = np.asarray(run.pos_y_t, dtype=float)
        py = np.asarray(run.pos_y, dtype=float)
        py_t, py = sort_by_time(py_t, py)
        pz_t = np.asarray(run.pos_z_t, dtype=float)
        pz = np.asarray(run.pos_z, dtype=float)
        pz_t, pz = sort_by_time(pz_t, pz)

        gcode_bytes = np.frombuffer(run.user_gcode.encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(
            dump_path,
            force_t=force_t, force_y=force_y,
            pos_x_t=px_t, pos_x=px,
            pos_y_t=py_t, pos_y=py,
            pos_z_t=pz_t, pos_z=pz,
            attached_existing=np.array([run.attached_existing], dtype=bool),
            attach_axis_z=np.array([
                run.attach_axis_z if run.attach_axis_z is not None else np.nan
            ], dtype=float),
            attach_layer_index=np.array([
                run.attach_layer_index if run.attach_layer_index is not None else -1
            ], dtype=int),
            **anchor_fields(),
            started_at_unix=np.array([float(run.state.started_at)]),
            gcode_bytes=gcode_bytes,
            gcode_name=np.array([run.name], dtype="U256"),
            n_layers=np.array([len(run.parsed.layers)]),
            n_moves=np.array([len(run.parsed.moves)]),
            est_print_s=np.array([float(run.parsed.total_time_s)]),
            bbox=np.asarray(run.parsed.bbox, dtype=float),
        )
        log.info("dumped livemap run to %s", dump_path)
        return fname
    except Exception:
        log.exception("livemap raw data dump failed")
        return None


# ---------- replay / listing ----------

@dataclass(slots=True)
class LiveMapRunInfo:
    filename: str
    path: str
    mtime_unix: float
    n_force: int
    n_layers: int
    n_moves: int
    duration_s: float
    est_print_s: float
    gcode_name: str = ""


def list_livemap_runs(runs_dir: str | os.PathLike = "runs") -> list[LiveMapRunInfo]:
    """Enumerate runs/livemap_*.npz, newest first. Never raises.

    Reads only the cheap scalar metadata -- it does NOT re-parse the embedded
    gcode, so the dropdown stays fast even with large saved prints.
    """
    p = Path(runs_dir)
    if not p.exists():
        return []
    out: list[LiveMapRunInfo] = []
    for f in p.glob("livemap_*.npz"):
        try:
            with np.load(f, allow_pickle=True) as d:
                ft = d["force_t"] if "force_t" in d else np.array([])
                _s = partial(npz_scalar, d)
                name = ""
                if "gcode_name" in d and len(d["gcode_name"]):
                    try:
                        name = str(d["gcode_name"][0])
                    except Exception:
                        name = ""
                out.append(
                    LiveMapRunInfo(
                        filename=f.name,
                        path=str(f.resolve()),
                        mtime_unix=f.stat().st_mtime,
                        n_force=int(len(ft)),
                        n_layers=int(_s("n_layers", 0)),
                        n_moves=int(_s("n_moves", 0)),
                        duration_s=float(ft[-1] - ft[0]) if len(ft) >= 2 else 0.0,
                        est_print_s=_s("est_print_s", 0.0),
                        gcode_name=name,
                    )
                )
        except Exception:
            continue
    out.sort(key=lambda r: r.mtime_unix, reverse=True)
    return out


def replay_livemap(path: str | os.PathLike) -> MappedRun:
    """Re-parse the embedded gcode + re-map the saved streams into a MappedRun."""
    with np.load(path, allow_pickle=True) as d:
        def _arr(key: str) -> np.ndarray | None:
            return np.asarray(d[key], dtype=float) if key in d and len(d[key]) else None

        force_t = _arr("force_t")
        force_y = _arr("force_y")
        px_t, px = _arr("pos_x_t"), _arr("pos_x")
        py_t, py = _arr("pos_y_t"), _arr("pos_y")
        pz_t, pz = _arr("pos_z_t"), _arr("pos_z")
        attach_axis_z = npz_scalar(d, "attach_axis_z", float("nan"))
        raw_start_layer = npz_scalar(d, "attach_layer_index", -1)
        gbytes = d["gcode_bytes"] if "gcode_bytes" in d else np.array([], dtype=np.uint8)
        gcode = bytes(np.asarray(gbytes, dtype=np.uint8)).decode("utf-8", errors="replace")

    if force_t is None or force_y is None:
        raise ValueError("run contains no loadcell samples")
    parsed = parse_gcode(gcode)
    start_layer_index = (
        int(raw_start_layer)
        if math.isfinite(raw_start_layer) and raw_start_layer >= 0
        else 0
    )
    mapped = map_run(
        force_t, force_y, px_t, px, py_t, py, pz_t, pz, parsed,
        start_layer_index=start_layer_index,
    )
    # Keep the attach seed visible in review diagnostics. Old/full-run files
    # have neither key and retain the layer-zero defaults.
    mapped.cross["attach_axis_z"] = (
        float(attach_axis_z) if math.isfinite(attach_axis_z) else None
    )
    return mapped
