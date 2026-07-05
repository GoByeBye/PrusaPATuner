"""End-to-end orchestration for one Max Flow test run.

Mirrors runner.run_tuning but for the stepped free-air flow ramp:
upload the flow gcode, poll PrusaLink to completion while collecting the
loadcell + pos_z streams, run flow_analysis.analyse_flow, and dump the raw
arrays to runs/flow_<ts>.npz for offline replay.
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
from .flow_analysis import FlowAnalysis, analyse_flow, flow_analysis_to_dict
from .flow_gen import FlowPlan, FlowRampParams, build_flow_ramp
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
class FlowRunState:
    state: str = "idle"  # idle | preparing | running | analyzing | done | error
    message: str = ""
    progress_pct: float = 0.0
    started_at: float = 0.0
    sweep_t0: float | None = None
    analysis: FlowAnalysis | None = None
    error: str | None = None


@dataclass(slots=True)
class FlowRun:
    cfg: AppConfig
    plan: FlowPlan
    state: FlowRunState = field(default_factory=FlowRunState)
    force_t: list[float] = field(default_factory=list)
    force_y: list[float] = field(default_factory=list)
    pos_z_t: list[float] = field(default_factory=list)
    pos_z: list[float] = field(default_factory=list)
    on_update: Callable[["FlowRun"], None] | None = None

    def emit(self) -> None:
        schedule_on_update(self.on_update, self, log, "flow on_update callback failed")

    def to_dict(self) -> dict[str, Any]:
        s = self.state
        return {
            "state": s.state,
            "message": s.message,
            "progress_pct": s.progress_pct,
            "started_at": s.started_at,
            "error": s.error,
            "n_force_samples": len(self.force_t),
            "n_levels": len(self.plan.segments),
            "flow_levels": [seg.flow_mm3_s for seg in self.plan.segments],
            "analysis": flow_analysis_to_dict(s.analysis) if s.analysis else None,
        }


def flow_params_from_config(cfg: AppConfig, udp_host: str) -> FlowRampParams:
    return FlowRampParams(
        nozzle_temp=cfg.nozzle_temp,
        preheat_temp=cfg.preheat_temp,
        nozzle_diameter=cfg.nozzle_diameter,
        filament_diameter=cfg.filament_diameter,
        filament_label=cfg.filament_label,
        min_flow_mm3_s=cfg.flow_min_mm3_s,
        max_flow_mm3_s=cfg.flow_max_mm3_s,
        flow_step_mm3_s=cfg.flow_step_mm3_s,
        dwell_s=cfg.flow_dwell_s,
        settle_frac=cfg.flow_settle_frac,
        warmup_s=cfg.flow_warmup_s,
        tare_dwell_s=cfg.flow_tare_dwell_s,
        accel_mm_s2=cfg.accel_mm_s2,
        purge_x=cfg.purge_x,
        purge_y=cfg.purge_y,
        purge_z=cfg.purge_z,
        udp_host=udp_host,
        udp_port=cfg.udp_port,
        label=f"Max flow test -- {cfg.filament_label}",
    )


async def run_flow_test(
    cfg: AppConfig,
    stream: MetricStream,
    *,
    on_update: Callable[["FlowRun"], None] | None = None,
    loadcell_metric: str = "loadcell_value",
    poll_interval_s: float = 1.0,
    job_timeout_s: float | None = None,
) -> FlowRun:
    udp_host = local_ip_toward(cfg.printer_host, port=80)
    params = flow_params_from_config(cfg, udp_host=udp_host)
    plan = build_flow_ramp(params)

    if job_timeout_s is None:
        if plan.segments:
            sweep_dur = plan.segments[-1].start_offset_s + plan.segments[-1].duration_s
        else:
            sweep_dur = 0.0
        job_timeout_s = max(600.0, 2.0 * sweep_dur + 300.0)

    run = FlowRun(cfg=cfg, plan=plan, on_update=on_update)
    run.state.started_at = time.time()
    run.state.state = "preparing"
    run.state.message = (
        f"Generated flow ramp ({len(plan.segments)} levels "
        f"{params.min_flow_mm3_s:g}→{params.max_flow_mm3_s:g} mm³/s), uploading…"
    )
    run.emit()

    stop_collect = asyncio.Event()

    def latch_sweep_t0(sample, v) -> None:
        if run.state.sweep_t0 is None and run.state.state == "running":
            run.state.sweep_t0 = sample.recv_monotonic

    collector_force = asyncio.create_task(collect_metric(
        stream, loadcell_metric, stop_collect, extract_force,
        run.force_t, run.force_y, on_sample=latch_sweep_t0,
    ))
    collector_pos_z = asyncio.create_task(collect_metric(
        stream, "pos_z", stop_collect, extract_numeric,
        run.pos_z_t, run.pos_z,
    ))

    try:
        async with PrusaLinkClient(
            cfg.printer_host,
            cfg.printer_api_key,
            password=cfg.printer_password,
            user=cfg.printer_user or "maker",
        ) as pl:
            filename = f"flow_test_{int(time.time())}.gcode"
            await pl.upload_and_print(filename, plan.gcode)
            run.state.state = "running"
            run.state.message = "Job started; capturing loadcell stream…"
            run.emit()

            await poll_job_until_done(
                pl, run,
                job_timeout_s=job_timeout_s,
                poll_interval_s=poll_interval_s,
                verb="Printing",
                log_timeout_done=lambda final_state: log.warning(
                    "flow job_timeout reached but final status %r -- "
                    "treating as done", final_state,
                ),
            )

        run.state.state = "analyzing"
        run.state.message = "Crunching flow-vs-force…"
        run.emit()

        if run.state.sweep_t0 is None and run.force_t:
            run.state.sweep_t0 = run.force_t[0]
        if run.state.sweep_t0 is None:
            raise RuntimeError(
                "No loadcell samples received — verify the metric streams during print"
            )

        force_t = np.asarray(run.force_t, dtype=float)
        force_y = np.asarray(run.force_y, dtype=float)
        pos_z_t = np.asarray(run.pos_z_t, dtype=float) if run.pos_z_t else None
        pos_z = np.asarray(run.pos_z, dtype=float) if run.pos_z else None
        force_t, force_y = sort_by_time(force_t, force_y)
        if pos_z_t is not None and pos_z is not None:
            pos_z_t, pos_z = sort_by_time(pos_z_t, pos_z)

        analysis = analyse_flow(
            sweep_t0=run.state.sweep_t0,
            force_t=force_t,
            force_y=force_y,
            plan=plan,
            pos_z_t=pos_z_t,
            pos_z=pos_z,
            z_marker_lift_mm=plan.params.z_marker_lift_mm,
        )
        run.state.sweep_t0 = analysis.sweep_t0
        run.state.analysis = analysis

        _dump_flow_run(run, force_t, force_y, pos_z_t, pos_z)

        run.state.state = "done"
        run.state.message = "Done."
        run.state.progress_pct = 100.0
        run.emit()

    except Exception as exc:
        run.state.state = "error"
        run.state.error = f"{type(exc).__name__}: {exc}"
        run.state.message = run.state.error
        log.exception("flow run failed")
        run.emit()
    finally:
        stop_collect.set()
        await cancel_collectors((collector_force, collector_pos_z), log)
        log.info(
            "flow captured: %d loadcell, %d pos_z samples",
            len(run.force_t), len(run.pos_z),
        )

    return run


def _dump_flow_run(
    run: FlowRun,
    force_t: np.ndarray,
    force_y: np.ndarray,
    pos_z_t: np.ndarray | None,
    pos_z: np.ndarray | None,
) -> None:
    """Save the raw arrays + plan scalars to runs/flow_<ts>.npz."""
    try:
        dump_path = runs_dir() / f"flow_{int(run.state.started_at)}.npz"
        p = run.plan.params
        np.savez(
            dump_path,
            force_t=force_t,
            force_y=force_y,
            pos_z_t=pos_z_t if pos_z_t is not None else np.array([]),
            pos_z=pos_z if pos_z is not None else np.array([]),
            sweep_t0=np.array([float(run.state.sweep_t0)]),
            **anchor_fields(),
            started_at_unix=np.array([float(run.state.started_at)]),
            flow_levels=np.array([s.flow_mm3_s for s in run.plan.segments], dtype=float),
            min_flow_mm3_s=np.array([p.min_flow_mm3_s]),
            max_flow_mm3_s=np.array([p.max_flow_mm3_s]),
            flow_step_mm3_s=np.array([p.flow_step_mm3_s]),
            dwell_s=np.array([p.dwell_s]),
            settle_frac=np.array([p.settle_frac]),
            warmup_s=np.array([p.warmup_s]),
            tare_dwell_s=np.array([p.tare_dwell_s]),
            accel_mm_s2=np.array([p.accel_mm_s2]),
            purge_x=np.array([p.purge_x]),
            purge_y=np.array([p.purge_y]),
            purge_z=np.array([p.purge_z]),
            z_marker_lift_mm=np.array([p.z_marker_lift_mm]),
            filament_diameter=np.array([p.filament_diameter]),
            nozzle_temp=np.array([p.nozzle_temp]),
            filament_label=np.array([p.filament_label], dtype="U128"),
        )
        if run.state.analysis is not None:
            run.state.analysis.notes.append(
                f"raw data dumped to {dump_path.resolve()}"
            )
        log.info("dumped raw flow run to %s", dump_path)
    except Exception:
        log.exception("flow raw data dump failed (analysis still succeeded)")


# ---------- replay / listing ----------

@dataclass(slots=True)
class FlowRunInfo:
    filename: str
    path: str
    mtime_unix: float
    n_force: int
    n_levels: int
    min_flow: float
    max_flow: float
    flow_step: float
    duration_s: float
    filament_label: str = ""
    nozzle_temp: float = 0.0


def list_flow_runs(runs_dir: str | os.PathLike = "runs") -> list[FlowRunInfo]:
    """Enumerate runs/flow_*.npz, newest first. Never raises."""
    p = Path(runs_dir)
    if not p.exists():
        return []
    out: list[FlowRunInfo] = []
    for f in p.glob("flow_*.npz"):
        try:
            with np.load(f, allow_pickle=True) as d:
                ft = d["force_t"] if "force_t" in d else np.array([])
                levels = d["flow_levels"] if "flow_levels" in d else np.array([])
                _s = partial(npz_scalar, d)
                label = ""
                if "filament_label" in d and len(d["filament_label"]):
                    try:
                        label = str(d["filament_label"][0])
                    except Exception:
                        label = ""
                out.append(
                    FlowRunInfo(
                        filename=f.name,
                        path=str(f.resolve()),
                        mtime_unix=f.stat().st_mtime,
                        n_force=int(len(ft)),
                        n_levels=int(len(levels)),
                        min_flow=_s("min_flow_mm3_s", 0.0),
                        max_flow=_s("max_flow_mm3_s", 0.0),
                        flow_step=_s("flow_step_mm3_s", 0.0),
                        duration_s=float(ft[-1] - ft[0]) if len(ft) >= 2 else 0.0,
                        filament_label=label,
                        nozzle_temp=_s("nozzle_temp", 0.0),
                    )
                )
        except Exception:
            continue
    out.sort(key=lambda r: r.mtime_unix, reverse=True)
    return out


def load_flow_run(path: str | os.PathLike) -> tuple[FlowPlan, dict[str, Any]]:
    """Rebuild a FlowPlan + analyse_flow kwargs from a saved flow npz."""
    d = np.load(path, allow_pickle=True)
    _s = partial(npz_scalar, d)

    params = FlowRampParams(
        nozzle_temp=_s("nozzle_temp", 215.0),
        filament_diameter=_s("filament_diameter", 1.75),
        min_flow_mm3_s=_s("min_flow_mm3_s", 5.0),
        max_flow_mm3_s=_s("max_flow_mm3_s", 30.0),
        flow_step_mm3_s=_s("flow_step_mm3_s", 1.0),
        dwell_s=_s("dwell_s", 3.0),
        settle_frac=_s("settle_frac", 0.5),
        warmup_s=_s("warmup_s", 3.0),
        tare_dwell_s=_s("tare_dwell_s", 1.5),
        accel_mm_s2=_s("accel_mm_s2", 5000.0),
        purge_x=_s("purge_x", 30.0),
        purge_y=_s("purge_y", 30.0),
        purge_z=_s("purge_z", 50.0),
        z_marker_lift_mm=_s("z_marker_lift_mm", 2.0),
    )
    plan = build_flow_ramp(params)

    force_t = np.asarray(d["force_t"], dtype=float)
    force_y = np.asarray(d["force_y"], dtype=float)
    force_t, force_y = sort_by_time(force_t, force_y)
    pos_z_t = np.asarray(d["pos_z_t"], dtype=float) if "pos_z_t" in d and len(d["pos_z_t"]) else None
    pos_z = np.asarray(d["pos_z"], dtype=float) if "pos_z" in d and len(d["pos_z"]) else None
    if pos_z_t is not None and pos_z is not None:
        pos_z_t, pos_z = sort_by_time(pos_z_t, pos_z)

    kwargs: dict[str, Any] = {
        "sweep_t0": _s("sweep_t0", float(force_t[0]) if len(force_t) else 0.0),
        "force_t": force_t,
        "force_y": force_y,
        "plan": plan,
        "pos_z_t": pos_z_t,
        "pos_z": pos_z,
        "z_marker_lift_mm": params.z_marker_lift_mm,
    }
    return plan, kwargs


def replay_flow(path: str | os.PathLike) -> tuple[FlowPlan, FlowAnalysis]:
    plan, kwargs = load_flow_run(path)
    analysis = analyse_flow(**kwargs)
    analysis.notes.insert(0, f"REPLAY: {Path(path).name}")
    return plan, analysis
