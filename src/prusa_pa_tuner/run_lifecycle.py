"""Shared run-lifecycle machinery for the four job runners.

Extracted from runner / flow_runner / probe_runner / livemap_runner, which
were largely copy-paste of each other: the transient-HTTP-error set, the
async on_update dispatch, the PrusaLink poll-to-completion loop, the metric
collector plumbing, and the NPZ dump/read helpers.

Behaviour is unchanged by design -- every user-visible state message, log
line, and NPZ field layout stays byte-identical. The small deliberate
per-module differences (timeout formulas, message verbs, extra log lines)
are injected by each runner via parameters/callbacks; the numeric timeout
formulas themselves stay in the runners.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import httpx
import numpy as np

if TYPE_CHECKING:
    from .prusalink import PrusaLinkClient
    from .udp_metrics import MetricSample, MetricStream

# The Core One occasionally drops HTTP connections during long heatups /
# homing -- httpx surfaces this as ReadError / ReadTimeout / ConnectError.
# None of those mean the job actually failed; just back off and retry.
TRANSIENT_HTTPX_ERRORS = (
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.RemoteProtocolError,
)


def schedule_on_update(
    callback: Callable[[Any], Any] | None,
    run: Any,
    log: logging.Logger,
    err_msg: str = "on_update callback failed",
) -> None:
    """Invoke a run's on_update callback, scheduling coroutine results.

    If the callback is async (e.g. the FastAPI WebSocket broadcaster),
    schedule it as a task so the coroutine actually runs. Without this
    the call site just creates an unawaited coroutine and Python emits
    "coroutine X was never awaited" while no client ever sees updates.
    """
    if callback is None:
        return
    try:
        result = callback(run)
        if asyncio.iscoroutine(result):
            try:
                asyncio.get_running_loop().create_task(result)
            except RuntimeError:
                # no running loop (e.g. called from a sync test) -- drop it
                result.close()
    except Exception:  # broadcast must never fail the run
        log.exception(err_msg)


async def poll_job_until_done(
    pl: "PrusaLinkClient",
    run: Any,
    *,
    job_timeout_s: float,
    poll_interval_s: float,
    log_timeout_done: Callable[[str], None],
    verb: str = "Printing",
    unresponsive_noun: str = "failures",
    transient_retry_log: Callable[[str, int], None] | None = None,
    on_tick: Callable[[float], None] | None = None,
    max_consecutive_failures: int = 10,
) -> None:
    """Poll PrusaLink job status until the print finishes (or fails).

    `run` is any of the four run dataclasses (duck-typed: needs
    `.state.message`, `.state.progress_pct` and `.emit()`).

    Deliberate per-runner differences are parameterized so each runner's
    user-visible messages / log lines stay byte-identical:
      * `verb` -- "Printing" vs "Probing" in the retry + progress messages.
      * `log_timeout_done(final_state)` -- the module-specific warning
        logged when the timeout fires but a final status check shows the
        print actually completed.
      * `unresponsive_noun` -- "failures" vs "consecutive failures" in the
        gave-up RuntimeError.
      * `transient_retry_log(exc_name, n)` -- runner.py additionally logs
        each transient retry; the others don't.
      * `on_tick(elapsed_s)` -- optional per-iteration hook run before the
        timeout check (livemap's early no-position-telemetry warning).
    """
    t_start = time.monotonic()
    last_progress = -1.0
    last_status_state: str | None = None
    consecutive_failures = 0
    while True:
        if on_tick is not None:
            on_tick(time.monotonic() - t_start)
        if time.monotonic() - t_start > job_timeout_s:
            # Before giving up, do one final blocking status check. The
            # user's machine sometimes finishes the gcode while we're
            # between polls; if a final poll confirms FINISHED/STOPPED/
            # IDLE, the print is actually done and we should analyse,
            # not error.
            try:
                final_status = await pl.get_job_status()
            except TRANSIENT_HTTPX_ERRORS:
                final_status = None
            final_state = (
                final_status.state.upper() if final_status is not None
                else (last_status_state or "")
            )
            if final_state in ("FINISHED", "STOPPED", "CANCELLED", "IDLE", ""):
                log_timeout_done(final_state)
                break
            raise TimeoutError(
                f"job exceeded timeout ({job_timeout_s:.0f}s); "
                f"last known state: {final_state}"
            )
        try:
            status = await pl.get_job_status()
            consecutive_failures = 0
        except TRANSIENT_HTTPX_ERRORS as exc:
            consecutive_failures += 1
            # Tolerate a handful of transient drops; only fail if they
            # pile up (suggesting the printer is actually gone).
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    f"PrusaLink unresponsive after {consecutive_failures} "
                    f"{unresponsive_noun}: {type(exc).__name__}: {exc}"
                )
            if transient_retry_log is not None:
                transient_retry_log(type(exc).__name__, consecutive_failures)
            run.state.message = (
                f"{verb}… (PrusaLink busy, retrying "
                f"{consecutive_failures}/10)"
            )
            run.emit()
            await asyncio.sleep(poll_interval_s)
            continue
        if status is None:
            # job already gone — assume finished
            break
        last_status_state = status.state.upper()
        if last_status_state in ("FINISHED", "STOPPED", "CANCELLED", "IDLE"):
            break
        if last_status_state in ("ERROR",):
            raise RuntimeError(f"printer reported ERROR: {status.raw}")
        if status.progress_pct != last_progress:
            run.state.progress_pct = status.progress_pct
            run.state.message = f"{verb}… {status.progress_pct:.0f}%"
            run.emit()
            last_progress = status.progress_pct
        await asyncio.sleep(poll_interval_s)


async def collect_metric(
    stream: "MetricStream",
    metric: str,
    stop_event: asyncio.Event,
    extract_fn: Callable[["MetricSample"], float | None],
    t_list: list[float],
    y_list: list[float],
    on_sample: Callable[["MetricSample", float], None] | None = None,
) -> None:
    """Append (recv_monotonic, extracted value) pairs for one metric stream.

    Replaces the copy-pasted collect_loadcell / collect_force / collect_pos
    inner coroutines. `on_sample` is the optional per-sample hook the PA and
    flow runners use to latch sweep_t0 on the first sample seen while the
    run state is "running".
    """
    async for sample in stream.subscribe(metric):
        if stop_event.is_set():
            return
        v = extract_fn(sample)
        if v is None:
            continue
        t_list.append(sample.recv_monotonic)
        y_list.append(v)
        if on_sample is not None:
            on_sample(sample, v)


async def cancel_collectors(
    tasks: Sequence[asyncio.Task], log: logging.Logger,
) -> None:
    """Cancel + await the collector tasks (the shared finally-block teardown)."""
    for task in tasks:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


def runs_dir() -> Path:
    """Return the runs/ output directory, creating it if needed."""
    p = Path("runs")
    p.mkdir(exist_ok=True)
    return p


def anchor_fields() -> dict[str, np.ndarray]:
    """Shared NPZ time-domain anchor entries for monotonic → wall-clock
    conversion.

    Pairs the current monotonic instant with the corresponding wall-clock
    unix timestamp. With both, every monotonic sample can be converted to
    a UTC datetime in post-processing:
      wall_unix(sample) = mono_anchor_unix + (sample.mono - mono_anchor_mono)
    """
    return {
        "mono_anchor_mono": np.array([float(time.monotonic())]),
        "mono_anchor_unix": np.array([float(time.time())]),
    }


def npz_scalar(d: Mapping[str, Any], key: str, default: float) -> float:
    """Read a 1-element float array from a loaded NPZ, with a default.

    Replaces the `_s(key, default)` closure that was redefined inside every
    `list_*_runs` / `load_*_run` function.
    """
    return float(d[key][0]) if key in d and len(d[key]) else default
