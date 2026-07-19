"""Replay a recorded sweep from `runs/run_<ts>.npz`.

Used by:
  * the `replay_run.py` CLI (fast iteration: tweak `analysis.py`,
    `python replay_run.py runs/run_<ts>.npz`, see new metrics);
  * the web UI's "replay" dropdown (`GET /api/runs`,
    `POST /api/runs/<filename>/analyse`), which routes through the
    same code so the rendered view matches a live sweep exactly.

Since 2026-07 the npz dump stores the FULL SweepParams as JSON
(`sweep_params_json`), so replay rebuilds the exact plan the run used.
Older dumps only carried per-cycle timing knobs (slow/fast halves,
cycle count, K values); for those we reconstruct a minimal SweepParams
and, when `coupled_dx_mm` is missing, derive an effective amplitude
from the actual pos_x swing inside the burst region so the pos_x
transition detector picks up real motion even when the run used a
non-default coupling.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .analysis import SweepAnalysis, analyse_sweep
from .gcode_gen import SweepParams, SweepPlan, build_sweep
from .gcode_preamble import is_indx


@dataclass(slots=True)
class RunInfo:
    """One row in the `/api/runs` listing."""
    filename: str       # basename, e.g. "run_1778953709.npz"
    path: str           # absolute filesystem path
    mtime_unix: float   # file modification time (unix seconds)
    n_force: int
    n_pos: int
    n_K: int
    cycles_per_K: int
    slow_half_s: float
    fast_half_s: float
    duration_s: float   # end - start of force_t
    # Filament label + nozzle temp the run was started with. Newer NPZ
    # dumps record both so the UI can display them under the dropdown
    # ("PLA @ 215°C"). Older dumps don't have these fields; we report
    # empty / 0 and the UI hides the line.
    filament_label: str = ""
    nozzle_temp: float = 0.0
    printer_model: str = "COREONE"
    tool_index: int | None = None
    # User-supplied ground-truth K (the value the user measured from a
    # test print) and free-text notes. Set via annotate_run() after the
    # dump is written. None when the run has never been annotated.
    user_k_opt: float | None = None
    user_k_opt_notes: str = ""


def list_runs(runs_dir: str | os.PathLike = "runs") -> list[RunInfo]:
    """Enumerate npz files in `runs_dir`, sorted newest first.

    Returns empty list if the directory doesn't exist. Files that fail
    to load are silently skipped — listing must never break the UI.
    """
    p = Path(runs_dir)
    if not p.exists():
        return []
    out: list[RunInfo] = []
    for f in p.glob("run_*.npz"):
        try:
            # allow_pickle=True so we can still read older NPZs that
            # stored filament_label as an object-dtype array. New NPZs
            # use a fixed-width Unicode dtype that doesn't need pickle,
            # but the flag is harmless when there's nothing to unpickle.
            # `with` releases the file handle promptly; without it, an
            # atomic rewrite by annotate_run can hit WinError 32 on
            # Windows because numpy's NpzFile still holds the handle
            # until GC.
            with np.load(f, allow_pickle=True) as d:
                ft = d["force_t"] if "force_t" in d else np.array([])
                pt = d["pos_t"] if "pos_t" in d else np.array([])
                cycles = (
                    int(d["cycles_per_K"][0])
                    if "cycles_per_K" in d and len(d["cycles_per_K"])
                    else 0
                )
                slow_h = (
                    float(d["slow_half_s"][0])
                    if "slow_half_s" in d and len(d["slow_half_s"])
                    else 0.0
                )
                fast_h = (
                    float(d["fast_half_s"][0])
                    if "fast_half_s" in d and len(d["fast_half_s"])
                    else 0.0
                )
                n_k = int(len(d["k_values"])) if "k_values" in d else 0
                duration = float(ft[-1] - ft[0]) if len(ft) >= 2 else 0.0
                filament_label = ""
                if "filament_label" in d and len(d["filament_label"]):
                    try:
                        filament_label = str(d["filament_label"][0])
                    except Exception:
                        filament_label = ""
                nozzle_temp = 0.0
                if "nozzle_temp" in d and len(d["nozzle_temp"]):
                    try:
                        nozzle_temp = float(d["nozzle_temp"][0])
                    except Exception:
                        nozzle_temp = 0.0
                printer_model = (
                    str(d["printer_model"][0])
                    if "printer_model" in d and len(d["printer_model"])
                    else "COREONE"
                )
                raw_tool = (
                    int(d["tool_index"][0])
                    if "tool_index" in d and len(d["tool_index"])
                    else -1
                )
                user_k_opt: float | None = None
                if "user_k_opt" in d and len(d["user_k_opt"]):
                    try:
                        v = float(d["user_k_opt"][0])
                        user_k_opt = v if np.isfinite(v) else None
                    except Exception:
                        user_k_opt = None
                user_k_opt_notes = ""
                if "user_k_opt_notes" in d and len(d["user_k_opt_notes"]):
                    try:
                        user_k_opt_notes = str(d["user_k_opt_notes"][0])
                    except Exception:
                        user_k_opt_notes = ""
            out.append(
                RunInfo(
                    filename=f.name,
                    path=str(f.resolve()),
                    mtime_unix=f.stat().st_mtime,
                    n_force=int(len(ft)),
                    n_pos=int(len(pt)),
                    n_K=n_k,
                    cycles_per_K=cycles,
                    slow_half_s=slow_h,
                    fast_half_s=fast_h,
                    duration_s=duration,
                    filament_label=filament_label,
                    nozzle_temp=nozzle_temp,
                    printer_model=printer_model,
                    tool_index=(
                        raw_tool
                        if is_indx(printer_model) and 0 <= raw_tool <= 7
                        else None
                    ),
                    user_k_opt=user_k_opt,
                    user_k_opt_notes=user_k_opt_notes,
                )
            )
        except Exception:
            continue
    out.sort(key=lambda r: r.mtime_unix, reverse=True)
    return out


def _params_from_json(d: Any) -> SweepParams | None:
    """Rebuild SweepParams from the `sweep_params_json` NPZ field.

    Returns None when the field is absent (legacy dump) or unparseable.
    Unknown keys in the JSON (from a future SweepParams revision) are
    dropped; missing keys fall back to the dataclass defaults."""
    if "sweep_params_json" not in d or not len(d["sweep_params_json"]):
        return None
    try:
        import dataclasses
        import json

        raw = json.loads(str(d["sweep_params_json"][0]))
        if not isinstance(raw, dict):
            return None
        known = {f.name for f in dataclasses.fields(SweepParams)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "K_values" in kwargs:
            kwargs["K_values"] = tuple(float(k) for k in kwargs["K_values"])
        return SweepParams(**kwargs)
    except Exception:
        return None


def _derive_coupled_dx_mm(pos_x: np.ndarray) -> float:
    """Best-effort estimate of `coupled_dx_mm` from the recorded pos_x.

    The npz dump doesn't store SweepParams.coupled_dx_mm, but the
    pos_x transition detector needs an amplitude hint to set its
    deadband. The previous heuristic (p10..p90 of the full pos_x
    distribution) was wrong on any run that included homing /
    parking moves: it returned ~215 mm when the actual cycle
    amplitude was 1 mm, because pos_x ranges from ~0 (home) to
    ~252 (park). With that bogus amplitude the deadband (=
    0.3 × amplitude = ~64 mm) is much larger than the real cycle
    motion, so EVERY real transition is missed.

    Strategy that survives parking motions:
      1. Compute the median of pos_x. The toolhead spends most of its
         time at `purge_x` during the burst, so the median sits near
         purge_x.
      2. Filter to samples within ±5 mm of the median. This excludes
         homing (X→0), parking (X→240+), and any leveling moves -- the
         remaining samples are the burst oscillation around purge_x.
      3. Compute the amplitude from the filtered subset: p95 − p5.
      4. If the filtered subset has too few samples (< 50) we have no
         tight cluster around the median (probably this is a single-
         shot move not a cycle); fall back to the configured default
         of 0.05 mm so the detector at least uses a sane deadband.

    Clamps to a 0.02 mm minimum (the encoder/throttle quantum below
    which detection is unreliable regardless of the configured amount).
    """
    if len(pos_x) < 10:
        return 0.05
    median_x = float(np.median(pos_x))
    near_median = np.abs(pos_x - median_x) <= 5.0
    sub = pos_x[near_median]
    if len(sub) < 50:
        return 0.05
    p5 = float(np.percentile(sub, 5))
    p95 = float(np.percentile(sub, 95))
    spread = max(p95 - p5, 0.02)
    return spread


def load_run(path: str | os.PathLike) -> tuple[SweepPlan, dict[str, Any]]:
    """Load an npz dump and reconstruct a SweepPlan + the arrays
    `analyse_sweep` needs. Returns `(plan, kwargs)` where `kwargs`
    can be splatted directly into `analyse_sweep(**kwargs)`.

    `plan.gcode` will be the re-emitted gcode; the analyser doesn't
    consume it but the dataclass needs it populated.
    """
    d = np.load(path, allow_pickle=True)
    pos_x = d["pos_x"] if "pos_x" in d and len(d["pos_x"]) else np.array([])

    # PREFERRED: the full SweepParams JSON that the runner dumps since
    # 2026-07. Replay then rebuilds the plan from the exact parameters
    # the run used -- no defaults, no heuristics.
    params = _params_from_json(d)
    if params is None:
        # Legacy NPZs: reconstruct from the individual scalar fields,
        # with fallback defaults for knobs older dumps didn't store.
        # When `coupled_dx_mm` is missing we INFER it from the data so
        # the pos_x transition detector gets a sane deadband.
        def _get(key: str, default: float) -> float:
            return float(d[key][0]) if key in d and len(d[key]) else default

        coupled_dx_mm_saved = _get("coupled_dx_mm", -1.0)
        printer_model = (
            str(d["printer_model"][0])
            if "printer_model" in d and len(d["printer_model"])
            else "COREONE"
        )
        raw_tool = int(_get("tool_index", 0.0))
        params = SweepParams(
            printer_model=printer_model,
            tool_index=raw_tool if is_indx(printer_model) else 0,
            K_values=tuple(float(k) for k in d["k_values"]),
            cycles_per_K=int(d["cycles_per_K"][0]),
            slow_half_s=float(d["slow_half_s"][0]),
            fast_half_s=float(d["fast_half_s"][0]),
            slow_feed_mm_s=float(d["slow_feed_mm_s"][0]),
            fast_feed_mm_s=float(d["fast_feed_mm_s"][0]),
            coupled_dx_mm=(
                coupled_dx_mm_saved
                if coupled_dx_mm_saved > 0
                else _derive_coupled_dx_mm(pos_x)
            ),
            coupled_dy_mm=_get("coupled_dy_mm", 0.0),
            coupled_dz_mm=_get("coupled_dz_mm", 0.0),
            first_slow_leg_factor=_get("first_slow_leg_factor", 10.0),
            purge_x=_get("purge_x", 30.0),
            purge_y=_get("purge_y", 30.0),
            purge_z=_get("purge_z", 50.0),
            z_marker_lift_mm=_get("z_marker_lift_mm", 2.0),
        )
    z_marker_lift_mm = params.z_marker_lift_mm
    plan = build_sweep(params)

    pos_t = d["pos_t"] if "pos_t" in d and len(d["pos_t"]) else None
    pos_x_arr = pos_x if len(pos_x) else None
    pos_z_t = (
        d["pos_z_t"] if "pos_z_t" in d and len(d["pos_z_t"]) else None
    )
    pos_z = d["pos_z"] if "pos_z" in d and len(d["pos_z"]) else None

    # Defensive sort: NPZs dumped before the udp_metrics monotonic-clip
    # fix carry out-of-order timestamps (overlapping firmware-time spans
    # in consecutive packets). Sort each stream so window slicing and
    # plot rendering see a strictly-monotonic timeline. Without this,
    # plotly draws backwards diagonals on the rising/falling edges
    # (observed on run_1779015193.npz K=0.05 seg 1).
    force_t = np.asarray(d["force_t"], dtype=float)
    force_y = np.asarray(d["force_y"], dtype=float)
    force_t, force_y = _sort_by_time(force_t, force_y)
    if pos_t is not None and pos_x_arr is not None:
        pos_t = np.asarray(pos_t, dtype=float)
        pos_x_arr = np.asarray(pos_x_arr, dtype=float)
        pos_t, pos_x_arr = _sort_by_time(pos_t, pos_x_arr)
    if pos_z_t is not None and pos_z is not None:
        pos_z_t = np.asarray(pos_z_t, dtype=float)
        pos_z = np.asarray(pos_z, dtype=float)
        pos_z_t, pos_z = _sort_by_time(pos_z_t, pos_z)

    kwargs: dict[str, Any] = {
        "sweep_t0": float(d["sweep_t0"][0]),
        "force_t": force_t,
        "force_y": force_y,
        "plan": plan,
        "pos_t": pos_t,
        "pos_x": pos_x_arr,
        "pos_z_t": pos_z_t,
        "pos_z": pos_z,
        "z_marker_lift_mm": z_marker_lift_mm,
    }
    return plan, kwargs


def _sort_by_time(
    t: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stable-sort (t, y) by ascending t. Skips the sort when already
    monotonic so the common case has zero copies."""
    if len(t) <= 1:
        return t, y
    if bool(np.all(np.diff(t) >= 0)):
        return t, y
    order = np.argsort(t, kind="stable")
    return t[order], y[order]


def replay(path: str | os.PathLike) -> tuple[SweepPlan, SweepAnalysis]:
    """One-shot: load the npz, run analyse_sweep, return (plan, analysis)."""
    plan, kwargs = load_run(path)
    analysis = analyse_sweep(**kwargs)
    analysis.notes.insert(
        0,
        f"REPLAY: {Path(path).name} "
        f"(coupled_dx_mm derived: {plan.params.coupled_dx_mm:.3f})",
    )
    return plan, analysis


def read_annotation(
    path: str | os.PathLike,
) -> tuple[float | None, str]:
    """Return `(user_k_opt, user_k_opt_notes)` for a saved run.

    Either field may be absent (legacy dumps, or runs that have never
    been annotated). `user_k_opt` is None when missing, non-finite, or
    unparseable; notes default to empty string.
    """
    with np.load(path, allow_pickle=True) as d:
        k: float | None = None
        if "user_k_opt" in d and len(d["user_k_opt"]):
            try:
                v = float(d["user_k_opt"][0])
                k = v if np.isfinite(v) else None
            except Exception:
                k = None
        notes = ""
        if "user_k_opt_notes" in d and len(d["user_k_opt_notes"]):
            try:
                notes = str(d["user_k_opt_notes"][0])
            except Exception:
                notes = ""
    return k, notes


def annotate_run(
    path: str | os.PathLike,
    user_k_opt: float | None,
    notes: str = "",
) -> None:
    """Set the user-supplied ground-truth K and free-text notes on a saved run.

    Re-writes the npz file: load every array, overwrite (or add) the
    `user_k_opt` and `user_k_opt_notes` fields, atomic-rename a tmp file
    over the original so a crash mid-write can't truncate the data.

    Passing `user_k_opt=None` clears the annotation (stores NaN, which
    `read_annotation` then reports as None).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    with np.load(p, allow_pickle=True) as d:
        # Copy every original array into a plain dict so we can write
        # them back. `np.load`'s NpzFile is lazy; eager-load each entry
        # so we don't keep the file open across the rename.
        payload = {name: np.asarray(d[name]) for name in d.files}

    k_value = float("nan") if user_k_opt is None else float(user_k_opt)
    payload["user_k_opt"] = np.array([k_value], dtype=float)
    # Fixed-width Unicode dtype mirrors the `filament_label` convention
    # so `np.load` without `allow_pickle` can read the file back.
    notes_str = notes or ""
    payload["user_k_opt_notes"] = np.array([notes_str], dtype="U512")

    # np.savez auto-appends ".npz" if missing, so the tmp name must
    # already end in ".npz" or we'd produce "<stem>.tmp.npz.npz".
    tmp = p.parent / (p.stem + ".tmp.npz")
    try:
        np.savez(tmp, **payload)
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
