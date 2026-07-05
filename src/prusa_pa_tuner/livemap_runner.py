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
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import AppConfig
from .gcode_parse import ParsedGcode, parse_gcode
from .livemap_gen import build_livemap_gcode
from .livemap_map import MappedRun, map_run
from .netutil import local_ip_toward
from .prusalink import PrusaLinkClient
from .runner import _extract_force, _extract_numeric, _sort_by_time
from .udp_metrics import MetricStream

log = logging.getLogger(__name__)


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
        if self.on_update is None:
            return
        try:
            result = self.on_update(self)
            if asyncio.iscoroutine(result):
                try:
                    asyncio.get_running_loop().create_task(result)
                except RuntimeError:
                    result.close()
        except Exception:
            log.exception("livemap on_update callback failed")

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
        }


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
) -> LiveMapRun:
    udp_host = local_ip_toward(cfg.printer_host, port=80)
    upload_gcode = build_livemap_gcode(
        user_gcode, udp_host=udp_host, udp_port=cfg.udp_port,
        loadcell_metric=loadcell_metric,
    )

    if job_timeout_s is None:
        est = parsed.total_time_s
        job_timeout_s = max(900.0, 2.0 * est + 600.0)

    run = LiveMapRun(cfg=cfg, name=name, user_gcode=user_gcode, parsed=parsed,
                     on_update=on_update)
    run.state.started_at = time.time()
    run.state.state = "preparing"
    run.state.message = (
        f"Uploading {name} ({len(parsed.layers)} layers, "
        f"~{parsed.total_time_s / 60:.0f} min est)…"
    )
    run.emit()

    stop_collect = asyncio.Event()

    async def collect_force() -> None:
        async for sample in stream.subscribe(loadcell_metric):
            if stop_collect.is_set():
                return
            v = _extract_force(sample)
            if v is None:
                continue
            run.force_t.append(sample.recv_monotonic)
            run.force_y.append(v)

    async def collect_pos(metric: str, t_list: list[float], y_list: list[float]) -> None:
        async for sample in stream.subscribe(metric):
            if stop_collect.is_set():
                return
            v = _extract_numeric(sample)
            if v is None:
                continue
            t_list.append(sample.recv_monotonic)
            y_list.append(v)

    collectors = [
        asyncio.create_task(collect_force()),
        asyncio.create_task(collect_pos("pos_x", run.pos_x_t, run.pos_x)),
        asyncio.create_task(collect_pos("pos_y", run.pos_y_t, run.pos_y)),
        asyncio.create_task(collect_pos("pos_z", run.pos_z_t, run.pos_z)),
    ]

    try:
        async with PrusaLinkClient(
            cfg.printer_host, cfg.printer_api_key,
            password=cfg.printer_password, user=cfg.printer_user or "maker",
        ) as pl:
            filename = f"livemap_{int(time.time())}.gcode"
            await pl.upload_and_print(filename, upload_gcode)
            run.state.state = "running"
            run.state.message = "Printing; mapping nozzle force live…"
            run.emit()

            import httpx
            transient = (
                httpx.ReadError, httpx.ReadTimeout, httpx.ConnectError,
                httpx.ConnectTimeout, httpx.RemoteProtocolError,
            )
            t_start = time.monotonic()
            last_progress = -1.0
            last_status_state: str | None = None
            consecutive_failures = 0
            warned_no_pos = False
            while True:
                # Fail-fast: if force is streaming but position isn't, the map
                # will be empty (e.g. the wrong metric -- 'ipos' instead of
                # 'pos' -- was enabled). Warn within ~20 s rather than after the
                # whole print.
                if (not warned_no_pos and time.monotonic() - t_start > 20.0
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
                        len(run.force_t), int(time.monotonic() - t_start),
                    )
                if time.monotonic() - t_start > job_timeout_s:
                    try:
                        final_status = await pl.get_job_status()
                    except transient:
                        final_status = None
                    final_state = (
                        final_status.state.upper() if final_status is not None
                        else (last_status_state or "")
                    )
                    if final_state in ("FINISHED", "STOPPED", "CANCELLED", "IDLE", ""):
                        log.warning(
                            "livemap job_timeout reached but final status %r -- "
                            "treating as done", final_state,
                        )
                        break
                    raise TimeoutError(
                        f"job exceeded timeout ({job_timeout_s:.0f}s); "
                        f"last known state: {final_state}"
                    )
                try:
                    status = await pl.get_job_status()
                    consecutive_failures = 0
                except transient as exc:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        raise RuntimeError(
                            f"PrusaLink unresponsive after {consecutive_failures} "
                            f"failures: {type(exc).__name__}: {exc}"
                        )
                    run.state.message = (
                        f"Printing… (PrusaLink busy, retrying "
                        f"{consecutive_failures}/10)"
                    )
                    run.emit()
                    await asyncio.sleep(poll_interval_s)
                    continue
                if status is None:
                    break
                last_status_state = status.state.upper()
                if last_status_state in ("FINISHED", "STOPPED", "CANCELLED", "IDLE"):
                    break
                if last_status_state in ("ERROR",):
                    raise RuntimeError(f"printer reported ERROR: {status.raw}")
                if status.progress_pct != last_progress:
                    run.state.progress_pct = status.progress_pct
                    run.state.message = f"Printing… {status.progress_pct:.0f}%"
                    run.emit()
                    last_progress = status.progress_pct
                await asyncio.sleep(poll_interval_s)

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
        for task in collectors:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
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
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)
        fname = f"livemap_{int(run.state.started_at)}.npz"
        dump_path = runs_dir / fname

        force_t = np.asarray(run.force_t, dtype=float)
        force_y = np.asarray(run.force_y, dtype=float)
        force_t, force_y = _sort_by_time(force_t, force_y)
        px_t = np.asarray(run.pos_x_t, dtype=float)
        px = np.asarray(run.pos_x, dtype=float)
        px_t, px = _sort_by_time(px_t, px)
        py_t = np.asarray(run.pos_y_t, dtype=float)
        py = np.asarray(run.pos_y, dtype=float)
        py_t, py = _sort_by_time(py_t, py)
        pz_t = np.asarray(run.pos_z_t, dtype=float)
        pz = np.asarray(run.pos_z, dtype=float)
        pz_t, pz = _sort_by_time(pz_t, pz)

        gcode_bytes = np.frombuffer(run.user_gcode.encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(
            dump_path,
            force_t=force_t, force_y=force_y,
            pos_x_t=px_t, pos_x=px,
            pos_y_t=py_t, pos_y=py,
            pos_z_t=pz_t, pos_z=pz,
            mono_anchor_mono=np.array([float(time.monotonic())]),
            mono_anchor_unix=np.array([float(time.time())]),
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

                def _s(key: str, default: float) -> float:
                    return float(d[key][0]) if key in d and len(d[key]) else default

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
        gbytes = d["gcode_bytes"] if "gcode_bytes" in d else np.array([], dtype=np.uint8)
        gcode = bytes(np.asarray(gbytes, dtype=np.uint8)).decode("utf-8", errors="replace")

    if force_t is None or force_y is None:
        raise ValueError("run contains no loadcell samples")
    parsed = parse_gcode(gcode)
    return map_run(force_t, force_y, px_t, px, py_t, py, pz_t, pz, parsed)
