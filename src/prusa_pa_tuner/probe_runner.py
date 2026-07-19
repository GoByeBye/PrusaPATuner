"""End-to-end orchestration for one lateral touch-probe characterisation run.

Mirrors flow_runner: build the probe gcode, upload + auto-print via PrusaLink,
capture loadcell_value + pos_x/pos_y/pos_z while polling to completion, run
probe_analysis.analyse_probe on the probe-axis position stream, and dump the
raw arrays to runs/probe_<ts>.npz for offline replay.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .config import AppConfig
from .gcode_preamble import is_indx, normalise_printer_model
from .probe_analysis import ProbeAnalysis, analyse_probe, probe_analysis_to_dict
from .probe_gen import ProbeParams, ProbePlan, build_probe_test
from .netutil import local_ip_toward
from .prusalink import PrusaLinkClient
from .run_lifecycle import (
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


@dataclass(slots=True)
class ProbeRunState:
    state: str = "idle"  # idle | preparing | running | analyzing | done | error
    message: str = ""
    progress_pct: float = 0.0
    started_at: float = 0.0
    analysis: ProbeAnalysis | None = None
    error: str | None = None


@dataclass(slots=True)
class ProbeRun:
    cfg: AppConfig
    plan: ProbePlan
    state: ProbeRunState = field(default_factory=ProbeRunState)
    force_t: list[float] = field(default_factory=list)
    force_y: list[float] = field(default_factory=list)
    pos_x_t: list[float] = field(default_factory=list)
    pos_x: list[float] = field(default_factory=list)
    pos_y_t: list[float] = field(default_factory=list)
    pos_y: list[float] = field(default_factory=list)
    pos_z_t: list[float] = field(default_factory=list)
    pos_z: list[float] = field(default_factory=list)
    on_update: Callable[["ProbeRun"], None] | None = None

    def emit(self) -> None:
        schedule_on_update(self.on_update, self, log, "probe on_update callback failed")

    def to_dict(self) -> dict[str, Any]:
        s = self.state
        p = self.plan.params
        model = normalise_printer_model(p.printer_model)
        return {
            "state": s.state,
            "message": s.message,
            "progress_pct": s.progress_pct,
            "started_at": s.started_at,
            "error": s.error,
            "n_force_samples": len(self.force_t),
            "axis": self.plan.axis,
            "dir": "+" if self.plan.dir_sign > 0 else "-",
            "n_touches": p.n_touches,
            "printer_model": model,
            "tool_index": p.tool_index if is_indx(model) else None,
            "analysis": probe_analysis_to_dict(s.analysis) if s.analysis else None,
        }


def probe_params_from_config(cfg: AppConfig, udp_host: str) -> ProbeParams:
    printer_model = normalise_printer_model(cfg.printer_model)
    return ProbeParams(
        nozzle_diameter=cfg.nozzle_diameter,
        printer_model=printer_model,
        tool_index=cfg.tool_index,
        probe_axis=cfg.probe_axis,
        probe_dir=cfg.probe_dir,
        start_x=cfg.probe_start_x,
        start_y=cfg.probe_start_y,
        probe_z=cfg.probe_z,
        creep_mm=cfg.probe_creep_mm,
        slow_feed_mm_min=cfg.probe_slow_feed_mm_min,
        travel_feed_mm_min=cfg.probe_travel_feed_mm_min,
        n_touches=cfg.probe_n_touches,
        backoff_mm=cfg.probe_backoff_mm,
        settle_ms=cfg.probe_settle_ms,
        probe_temp=cfg.probe_temp,
        udp_host=udp_host,
        udp_port=cfg.udp_port,
        label=(
            f"Touch-probe lateral characterisation -- T{cfg.tool_index}"
            if is_indx(printer_model)
            else "Touch-probe lateral characterisation"
        ),
    )


async def run_probe_test(
    cfg: AppConfig,
    stream: MetricStream,
    *,
    on_update: Callable[["ProbeRun"], None] | None = None,
    loadcell_metric: str = "loadcell_value",
    poll_interval_s: float = 1.0,
    job_timeout_s: float | None = None,
) -> ProbeRun:
    udp_host = local_ip_toward(cfg.printer_host, port=80)
    params = probe_params_from_config(cfg, udp_host=udp_host)
    plan = build_probe_test(params)

    if job_timeout_s is None:
        if plan.touches:
            sweep_dur = plan.touches[-1].start_offset_s + plan.touches[-1].duration_s
        else:
            sweep_dur = 0.0
        job_timeout_s = max(600.0, 3.0 * sweep_dur + 300.0)

    run = ProbeRun(cfg=cfg, plan=plan, on_update=on_update)
    run.state.started_at = time.time()
    run.state.state = "preparing"
    run.state.message = (
        f"Generated touch-probe test ({plan.axis}"
        f"{'+' if plan.dir_sign > 0 else '-'}, {params.n_touches} touches, "
        f"{params.creep_mm:g} mm creep), uploading…"
    )
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
            tool_part = (
                f"_t{params.tool_index}" if is_indx(params.printer_model) else ""
            )
            filename = f"probe_test{tool_part}_{int(time.time())}.gcode"
            await pl.upload_and_print(filename, plan.gcode)
            run.state.state = "running"
            run.state.message = "Job started; capturing loadcell + position…"
            run.emit()

            await poll_job_until_done(
                pl, run,
                job_timeout_s=job_timeout_s,
                poll_interval_s=poll_interval_s,
                verb="Probing",
                log_timeout_done=lambda final_state: log.warning(
                    "probe job_timeout reached but final status %r -- "
                    "treating as done", final_state,
                ),
            )

        run.state.state = "analyzing"
        run.state.message = "Crunching force-vs-position…"
        run.emit()

        force_t = np.asarray(run.force_t, dtype=float)
        force_y = np.asarray(run.force_y, dtype=float)
        force_t, force_y = sort_by_time(force_t, force_y)
        px_t = np.asarray(run.pos_x_t, dtype=float) if run.pos_x_t else None
        px = np.asarray(run.pos_x, dtype=float) if run.pos_x else None
        py_t = np.asarray(run.pos_y_t, dtype=float) if run.pos_y_t else None
        py = np.asarray(run.pos_y, dtype=float) if run.pos_y else None
        if px_t is not None and px is not None:
            px_t, px = sort_by_time(px_t, px)
        if py_t is not None and py is not None:
            py_t, py = sort_by_time(py_t, py)

        if plan.axis == "X":
            probe_pos_t, probe_pos = px_t, px
        else:
            probe_pos_t, probe_pos = py_t, py

        analysis = analyse_probe(
            force_t=force_t, force_y=force_y,
            pos_t=probe_pos_t, pos=probe_pos, plan=plan,
        )
        run.state.analysis = analysis

        _dump_probe_run(run, force_t, force_y, px_t, px, py_t, py)

        run.state.state = "done"
        run.state.message = f"Done — {analysis.verdict}."
        run.state.progress_pct = 100.0
        run.emit()

    except Exception as exc:
        run.state.state = "error"
        run.state.error = f"{type(exc).__name__}: {exc}"
        run.state.message = run.state.error
        log.exception("probe run failed")
        run.emit()
    finally:
        stop_collect.set()
        await cancel_collectors(collectors, log)
        log.info(
            "probe captured: %d loadcell, %d pos_x, %d pos_y, %d pos_z samples",
            len(run.force_t), len(run.pos_x), len(run.pos_y), len(run.pos_z),
        )

    return run


def _dump_probe_run(
    run: ProbeRun,
    force_t: np.ndarray,
    force_y: np.ndarray,
    px_t: np.ndarray | None,
    px: np.ndarray | None,
    py_t: np.ndarray | None,
    py: np.ndarray | None,
) -> None:
    """Save the raw arrays + plan scalars to runs/probe_<ts>.npz."""
    try:
        p = run.plan.params
        tool_part = f"_t{p.tool_index}" if is_indx(p.printer_model) else ""
        dump_path = runs_dir() / f"probe{tool_part}_{int(run.state.started_at)}.npz"
        np.savez(
            dump_path,
            force_t=force_t,
            force_y=force_y,
            pos_x_t=px_t if px_t is not None else np.array([]),
            pos_x=px if px is not None else np.array([]),
            pos_y_t=py_t if py_t is not None else np.array([]),
            pos_y=py if py is not None else np.array([]),
            pos_z_t=np.asarray(run.pos_z_t, dtype=float),
            pos_z=np.asarray(run.pos_z, dtype=float),
            **anchor_fields(),
            started_at_unix=np.array([float(run.state.started_at)]),
            probe_axis=np.array([run.plan.axis], dtype="U2"),
            dir_sign=np.array([run.plan.dir_sign]),
            start_x=np.array([p.start_x]),
            start_y=np.array([p.start_y]),
            probe_z=np.array([p.probe_z]),
            creep_mm=np.array([p.creep_mm]),
            slow_feed_mm_min=np.array([p.slow_feed_mm_min]),
            travel_feed_mm_min=np.array([p.travel_feed_mm_min]),
            n_touches=np.array([p.n_touches]),
            backoff_mm=np.array([p.backoff_mm]),
            settle_ms=np.array([p.settle_ms]),
            probe_temp=np.array([p.probe_temp]),
            printer_model=np.array(
                [normalise_printer_model(p.printer_model)], dtype="U32"
            ),
            tool_index=np.array([p.tool_index if is_indx(p.printer_model) else -1]),
        )
        if run.state.analysis is not None:
            run.state.analysis.notes.append(f"raw data dumped to {dump_path.resolve()}")
        log.info("dumped raw probe run to %s", dump_path)
    except Exception:
        log.exception("probe raw data dump failed (analysis still succeeded)")


# ---------- replay / listing ----------

@dataclass(slots=True)
class ProbeRunInfo:
    filename: str
    path: str
    mtime_unix: float
    n_force: int
    axis: str
    dir: str
    n_touches: int
    creep_mm: float
    duration_s: float
    printer_model: str = "COREONE"
    tool_index: int | None = None


def list_probe_runs(runs_dir: str | os.PathLike = "runs") -> list[ProbeRunInfo]:
    """Enumerate runs/probe_*.npz, newest first. Never raises."""
    p = Path(runs_dir)
    if not p.exists():
        return []
    out: list[ProbeRunInfo] = []
    for f in p.glob("probe_*.npz"):
        try:
            with np.load(f, allow_pickle=True) as d:
                ft = d["force_t"] if "force_t" in d else np.array([])
                _s = partial(npz_scalar, d)
                axis = "X"
                if "probe_axis" in d and len(d["probe_axis"]):
                    try:
                        axis = str(d["probe_axis"][0])
                    except Exception:
                        axis = "X"
                sign = _s("dir_sign", 1.0)
                model = (
                    str(d["printer_model"][0])
                    if "printer_model" in d
                    else "COREONE"
                )
                raw_tool = int(_s("tool_index", -1))
                out.append(
                    ProbeRunInfo(
                        filename=f.name,
                        path=str(f.resolve()),
                        mtime_unix=f.stat().st_mtime,
                        n_force=int(len(ft)),
                        axis=axis,
                        dir="+" if sign > 0 else "-",
                        n_touches=int(_s("n_touches", 0)),
                        creep_mm=_s("creep_mm", 0.0),
                        duration_s=float(ft[-1] - ft[0]) if len(ft) >= 2 else 0.0,
                        printer_model=model,
                        tool_index=(
                            raw_tool
                            if is_indx(model) and 0 <= raw_tool <= 7
                            else None
                        ),
                    )
                )
        except Exception:
            continue
    out.sort(key=lambda r: r.mtime_unix, reverse=True)
    return out


def load_probe_run(path: str | os.PathLike) -> tuple[ProbePlan, dict[str, Any]]:
    """Rebuild a ProbePlan + analyse_probe kwargs from a saved probe npz."""
    d = np.load(path, allow_pickle=True)
    _s = partial(npz_scalar, d)

    axis = "X"
    if "probe_axis" in d and len(d["probe_axis"]):
        try:
            axis = str(d["probe_axis"][0])
        except Exception:
            axis = "X"
    sign = _s("dir_sign", 1.0)
    printer_model = (
        str(d["printer_model"][0]) if "printer_model" in d else "COREONE"
    )
    raw_tool = int(_s("tool_index", 0))
    params = ProbeParams(
        printer_model=printer_model,
        tool_index=raw_tool if is_indx(printer_model) else 0,
        probe_axis=axis,
        probe_dir="+" if sign > 0 else "-",
        start_x=_s("start_x", 125.0),
        start_y=_s("start_y", 110.0),
        probe_z=_s("probe_z", 5.0),
        creep_mm=_s("creep_mm", 1.0),
        slow_feed_mm_min=_s("slow_feed_mm_min", 30.0),
        travel_feed_mm_min=_s("travel_feed_mm_min", 3000.0),
        n_touches=int(_s("n_touches", 5)),
        backoff_mm=_s("backoff_mm", 1.0),
        settle_ms=int(_s("settle_ms", 300)),
        probe_temp=_s("probe_temp", 0.0),
    )
    plan = build_probe_test(params)

    force_t = np.asarray(d["force_t"], dtype=float)
    force_y = np.asarray(d["force_y"], dtype=float)
    force_t, force_y = sort_by_time(force_t, force_y)

    def _arr(key: str) -> np.ndarray | None:
        return np.asarray(d[key], dtype=float) if key in d and len(d[key]) else None

    px_t, px = _arr("pos_x_t"), _arr("pos_x")
    py_t, py = _arr("pos_y_t"), _arr("pos_y")
    if px_t is not None and px is not None:
        px_t, px = sort_by_time(px_t, px)
    if py_t is not None and py is not None:
        py_t, py = sort_by_time(py_t, py)
    if axis == "X":
        pos_t, pos = px_t, px
    else:
        pos_t, pos = py_t, py

    kwargs: dict[str, Any] = {
        "force_t": force_t,
        "force_y": force_y,
        "pos_t": pos_t,
        "pos": pos,
        "plan": plan,
    }
    return plan, kwargs


def replay_probe(path: str | os.PathLike) -> tuple[ProbePlan, ProbeAnalysis]:
    plan, kwargs = load_probe_run(path)
    analysis = analyse_probe(**kwargs)
    analysis.notes.insert(0, f"REPLAY: {Path(path).name}")
    return plan, analysis
