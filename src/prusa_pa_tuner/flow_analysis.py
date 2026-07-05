"""Analyse a stepped free-air flow ramp to find the maximum acceptable flow.

Physics (see the research synthesis in the project notes):
  * In the well-behaved regime the steady-state back-pressure follows a
    sub-linear power law  F(Q) ≈ a·Qᵇ + c  with b < 1 (shear-thinning
    melt; the +c absorbs the entrance / static-load offset).
  * When the hotend can't melt fast enough, force breaks UPWARD off that
    curve (a partially-molten / solid core raises required pressure).
  * Per-level force VARIANCE rises sharply at the onset of instability,
    typically BEFORE the mean breaks and before the extruder skips. Skip
    itself shows up as the mean force COLLAPSING downward.

So we extract three independent markers and report the most conservative:
  1. deviation_flow      -- first level whose mean force exceeds the
                            low-flow power-law fit by > n_sigma·residualσ.
  2. variance_onset_flow -- first level whose force std exceeds
                            var_factor × the quiet-region baseline std.
  3. collapse_flow       -- first level after the force peak where the
                            mean drops by > collapse_frac (skip / grind).

`recommended_max_flow` is the earliest of (1)/(2) derated by `derate_frac`
(mirroring the 10-20% margins community max-flow methods apply). This is a
heuristic v1 -- everything per-level is exposed so the thresholds can be
re-tuned once we see real hardware data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .analysis import _detect_z_marker_anchor
from .flow_gen import FlowPlan


@dataclass(slots=True)
class FlowLevelResult:
    flow_mm3_s: float
    feed_mm_s: float
    n_samples: int          # samples in the measured (post-settle) window
    force_mean: float       # tared steady-state force over the measured window
    force_std: float
    force_median: float
    t_window_start: float   # dwell start (incl. settle), relative to sweep_t0
    t_settle: float         # measurement start (settle discarded), relative
    t_end: float            # dwell end, relative
    # Raw samples across the FULL dwell window (settle + measure), tared,
    # for the per-step viewer. Time is relative to sweep_t0.
    t: list[float] = field(default_factory=list)
    force: list[float] = field(default_factory=list)


@dataclass(slots=True)
class FlowFit:
    """Power-law baseline fit F = a·Qᵇ + c over the low-flow region."""
    a: float
    b: float
    c: float
    n_points: int
    fit_max_flow: float   # highest flow included in the baseline fit
    residual_std: float
    r_squared: float

    def eval(self, q: np.ndarray | float) -> np.ndarray | float:
        return self.a * np.power(q, self.b) + self.c


@dataclass(slots=True)
class FlowAnalysis:
    sweep_t0: float                # detected sweep START (host-clock seconds)
    sweep_end: float = 0.0         # detected sweep END
    step_width: float = 0.0        # per-step duration used for slicing (s)
    detect_method: str = ""        # how the sweep was located
    tare: float = 0.0  # static-load offset subtracted from all forces (raw units)
    levels: list[FlowLevelResult] = field(default_factory=list)
    fit: FlowFit | None = None
    deviation_flow: float | None = None
    variance_onset_flow: float | None = None
    collapse_flow: float | None = None
    recommended_max_flow: float | None = None
    derate_frac: float = 0.15
    n_sigma: float = 4.0
    var_factor: float = 3.0
    sample_rate_hz: float = 0.0
    notes: list[str] = field(default_factory=list)


def _sort_by_time(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(t) <= 1 or bool(np.all(np.diff(t) >= 0)):
        return t, y
    order = np.argsort(t, kind="stable")
    return t[order], y[order]


def _rolling_mean(y: np.ndarray, win: int) -> np.ndarray:
    """Centered moving average, edge-padded to preserve length."""
    if win <= 1:
        return y.astype(float, copy=True)
    c = np.cumsum(np.insert(y.astype(float), 0, 0.0))
    out = (c[win:] - c[:-win]) / win
    pad = win // 2
    head = np.full(pad, out[0])
    tail = np.full(len(y) - len(out) - pad, out[-1])
    return np.concatenate([head, out, tail])


def detect_flow_sweep(
    force_t: np.ndarray, force_y: np.ndarray, n_steps: int, dwell_s: float,
) -> tuple[float, float, float] | None:
    """Locate the extrusion sweep [start, end] from the force signal alone.

    The planned-timing + Z-marker approach proved unreliable on real
    hardware: pos_z was often unavailable, and the long, variable
    heat-up/home/park delay before the first step means "first metric
    seen" is a useless anchor. The force signal itself carries the
    structure unambiguously -- the sweep is N back-to-back equal-duration
    extrusion steps, one long contiguous region where the loadcell is
    ACTIVE (force elevated above idle and/or erratic at breakdown),
    bracketed by idle before (heat/home/tare) and a clean idle tail after
    (cooldown). We:

      1. estimate an idle baseline + noise from the quiet capture head,
      2. build smoothed activity = |force - baseline|,
      3. threshold + merge short (<1 s) gaps -> contiguous active runs,
      4. pick the highest-ENERGY run of plausible duration -- the sweep
         (robust against brief homing probes and post-sweep spikes),
      5. anchor its END (reliable: the highest-flow steps are far from the
         idle tail and drop into it sharply) and back-compute
         start = end - N*dwell_s. The START is NOT trusted directly --
         the lowest-flow steps can fall under the activity threshold.

    Returns (start_t, end_t, step_width) or None. step_width = dwell_s:
    each constant-velocity E move runs for ~dwell_s with no inter-step gap
    (the tiny accel ramp is negligible vs a multi-second dwell).
    """
    n = len(force_t)
    if n < 50 or n_steps < 1 or dwell_s <= 0:
        return None
    dt = float(np.median(np.diff(force_t)))
    if dt <= 0:
        return None
    fs = 1.0 / dt
    expected = n_steps * dwell_s

    # Idle baseline from the first 3 s (capture starts pre-extrusion);
    # median is robust to brief homing/probing spikes.
    head = force_t < force_t[0] + 3.0
    if int(head.sum()) > 5:
        base = float(np.median(force_y[head]))
        res = force_y[head] - base
    else:
        base = float(np.median(force_y))
        res = force_y - base
    noise = 1.4826 * float(np.median(np.abs(res - np.median(res))))
    if not np.isfinite(noise) or noise <= 0:
        noise = float(np.std(res)) or 1.0

    win = max(1, int(round(0.25 * fs)))
    sm = _rolling_mean(np.abs(force_y - base), win)
    amp = float(np.percentile(sm, 99))
    thr = max(8.0 * noise, 0.05 * amp)
    active = sm > thr

    # contiguous active runs, bridging only short (<1 s) gaps so post-sweep
    # spikes (separated by idle) don't merge into the sweep.
    gap_max = int(round(1.0 * fs))
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not active[i]:
            i += 1
            continue
        j = i
        while j < n:
            if active[j]:
                j += 1
            else:
                k = j
                while k < n and not active[k]:
                    k += 1
                if k < n and (k - j) <= gap_max:
                    j = k
                else:
                    break
        runs.append((i, j - 1))
        i = j
    if not runs:
        return None

    def energy(r: tuple[int, int]) -> float:
        return float(np.sum(sm[r[0]:r[1] + 1]))

    # Prefer runs of plausible duration; the sweep dominates total energy.
    candidates = [
        r for r in runs
        if (force_t[r[1]] - force_t[r[0]]) >= 0.4 * expected
    ] or runs
    best = max(candidates, key=energy)
    end_t = float(force_t[best[1]])
    start_t = max(end_t - expected, float(force_t[0]))
    # W is physically exact (constant-velocity E moves, no inter-step gap).
    # The end-anchor fixes WHERE the sweep is; the sub-step PHASE is refined
    # separately in analyse_flow (it needs settle_frac) via _align_grid_phase.
    return start_t, end_t, dwell_s


def _align_grid_phase(
    force_t: np.ndarray,
    force_y: np.ndarray,
    start: float,
    n_steps: int,
    width: float,
    settle_frac: float,
    fs: float,
) -> float:
    """Shift the uniform step grid so each step's RISE falls in the
    discarded settle region, not the measured window.

    The end-anchor can be off by a fraction of a step (post-extrusion force
    decay reads as "active"), which pushes each measured window into the
    NEXT step's rising edge -- so the steady-state value gets contaminated
    by the transient. We search a phase shift in [-W/2, W/2] and pick the
    one that MINIMIZES positive-gradient (rise) energy inside the measured
    windows over the clean lower-flow region. Ties are broken toward the
    smallest shift (robust against a noisy far-side dip). Returns the
    adjusted start; width is unchanged (it's exact).
    """
    if settle_frac <= 0.0 or settle_frac >= 1.0 or n_steps < 2:
        return start
    win = max(1, int(round(0.08 * fs)))           # light smoothing: sharp rises
    sf = _rolling_mean(force_y, win)
    g = np.clip(np.gradient(sf), 0.0, None)
    n_clean = max(3, int(0.6 * n_steps))
    settle = settle_frac * width
    deltas = np.linspace(-width / 2.0, width / 2.0, 121)
    scores = np.empty_like(deltas)
    for di, dl in enumerate(deltas):
        e = 0.0
        base_i = start + dl
        for i in range(n_clean):
            ms = base_i + i * width + settle
            me = base_i + (i + 1) * width
            lo = int(np.searchsorted(force_t, ms, "left"))
            hi = int(np.searchsorted(force_t, me, "left"))
            if hi > lo:
                e += float(g[lo:hi].sum())
        scores[di] = e
    smin = float(scores.min())
    near = np.where(scores <= smin * 1.05 + 1e-9)[0]
    best = int(near[int(np.argmin(np.abs(deltas[near])))])
    return start + float(deltas[best])


def _fit_power_law(
    q: np.ndarray, f: np.ndarray,
) -> tuple[float, float, float, float, float] | None:
    """Fit F = a·Qᵇ + c. Returns (a, b, c, residual_std, r_squared) or None.

    Uses scipy if available; falls back to a log-log linear fit (which
    implicitly assumes c≈0) when scipy or the non-linear solve is
    unavailable. Bounds keep b in a physically sane sub-/mildly-super-
    linear band and a ≥ 0.
    """
    if len(q) < 3:
        return None
    try:
        from scipy.optimize import curve_fit

        def model(qq, a, b, c):
            return a * np.power(qq, b) + c

        f_span = float(np.ptp(f)) or 1.0
        p0 = [f_span / max(float(np.ptp(q)) ** 0.5, 1e-6), 0.5, float(np.min(f))]
        popt, _ = curve_fit(
            model, q, f, p0=p0,
            bounds=([0.0, 0.05, -np.inf], [np.inf, 2.0, np.inf]),
            maxfev=20000,
        )
        a, b, c = (float(x) for x in popt)
        resid = f - model(q, a, b, c)
    except Exception:
        # log-log fallback: log F = log a + b log Q  (c forced 0)
        if np.any(f <= 0) or np.any(q <= 0):
            return None
        b, log_a = np.polyfit(np.log(q), np.log(f), 1)
        a, c = float(np.exp(log_a)), 0.0
        b = float(b)
        resid = f - (a * np.power(q, b) + c)

    residual_std = float(np.std(resid, ddof=1)) if len(resid) > 1 else 0.0
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((f - np.mean(f)) ** 2)) or 1.0
    r_squared = 1.0 - ss_res / ss_tot
    return a, b, c, residual_std, r_squared


def analyse_flow(
    *,
    sweep_t0: float,
    force_t: np.ndarray,
    force_y: np.ndarray,
    plan: FlowPlan,
    pos_z_t: np.ndarray | None = None,
    pos_z: np.ndarray | None = None,
    z_marker_lift_mm: float = 2.0,
    derate_frac: float = 0.15,
    n_sigma: float = 4.0,
    var_factor: float = 3.0,
    collapse_frac: float = 0.15,
    rel_tol: float = 0.10,
) -> FlowAnalysis:
    p = plan.params
    force_t = np.asarray(force_t, dtype=float)
    force_y = np.asarray(force_y, dtype=float)
    force_t, force_y = _sort_by_time(force_t, force_y)

    notes: list[str] = []
    n_steps = len(plan.segments)

    sample_rate_hz = 0.0
    if len(force_t) > 1:
        dt = float(np.median(np.diff(force_t)))
        sample_rate_hz = 1.0 / dt if dt > 0 else 0.0

    # --- locate the sweep DIRECTLY in the force signal (primary) ---
    # This is the reliable anchor: it finds where extrusion actually
    # happened, independent of planned timing and the (often absent) pos_z
    # Z-marker. See detect_flow_sweep for the method.
    detected = detect_flow_sweep(force_t, force_y, n_steps, p.dwell_s)
    if detected is not None:
        sweep_start, sweep_end, step_width = detected
        # Refine the sub-step phase so step rises land in the discarded
        # settle region, not the measured window.
        sweep_start = _align_grid_phase(
            force_t, force_y, sweep_start, n_steps, step_width,
            p.settle_frac, sample_rate_hz or 100.0,
        )
        sweep_end = sweep_start + n_steps * step_width
        detect_method = "force-activity (end-anchored + phase-aligned)"
        notes.append(
            f"sweep located by force activity: "
            f"[{sweep_start:.2f}, {sweep_end:.2f}] s, "
            f"step width {step_width:.3f}s, {n_steps} steps"
        )
    else:
        # Fallback: planned timing relative to the provided sweep_t0 seed.
        first_off = plan.segments[0].start_offset_s if plan.segments else 0.0
        sweep_start = sweep_t0 + first_off
        step_width = p.dwell_s
        sweep_end = sweep_start + n_steps * step_width
        detect_method = "planned-timing fallback"
        notes.append(
            "could not locate sweep from force activity; using planned timing "
            "(check that the loadcell streamed during extrusion)"
        )

    # --- Z-marker cross-check when pos_z is present ---
    # pos_z anchors the marker; planned offset to step0 = segments[0]
    # start_offset (pre-roll + tare hold). We don't override the
    # force-based detection with it -- just log agreement so a future
    # discrepancy is visible. If detection failed, fall back to it.
    if pos_z is not None and pos_z_t is not None and len(pos_z) > 10:
        anchor = _detect_z_marker_anchor(
            np.asarray(pos_z_t, dtype=float),
            np.asarray(pos_z, dtype=float),
            expected_lift_mm=z_marker_lift_mm,
        )
        if anchor is not None and plan.segments:
            marker_start = float(anchor) + plan.segments[0].start_offset_s
            notes.append(
                f"Z-marker cross-check: predicts step 0 at {marker_start:.2f}s "
                f"(force-detected {sweep_start:.2f}s, Δ{sweep_start - marker_start:+.2f}s)"
            )
            if detected is None:
                sweep_start = marker_start
                sweep_end = sweep_start + n_steps * step_width
                detect_method = "Z-marker (planned offset)"

    # --- tare from the no-flow window just before the first step ---
    # The 1 s before step 0 is the tare hold + pre-roll (extruder idle), so
    # it reads the static head load; subtract its median so plotted force
    # starts at ~0 and reads as back-pressure above baseline.
    tare = 0.0
    tmask = (force_t >= sweep_start - 1.2) & (force_t < sweep_start - 0.2)
    tvals = force_y[tmask]
    if tvals.size >= 8:
        tare = float(np.median(tvals))
        notes.append(
            f"tared to no-flow baseline ({tare:.4g} raw) over {tvals.size} samples"
        )
    else:
        notes.append("tare window too short -- showing raw force")

    settle_s = step_width * p.settle_frac
    levels: list[FlowLevelResult] = []
    # Cap raw points sent per level so a long dwell doesn't bloat the payload.
    MAX_PTS = 800
    for i, seg in enumerate(plan.segments):
        w0 = sweep_start + i * step_width              # full dwell start
        ts = w0 + settle_s                             # measurement start
        t1 = w0 + step_width                           # dwell end
        # measured (post-settle) window -> steady-state stats
        m_mask = (force_t >= ts) & (force_t < t1)
        m_vals = force_y[m_mask] - tare
        n = int(m_vals.size)
        if n >= 3:
            mean = float(np.mean(m_vals))
            std = float(np.std(m_vals, ddof=1))
            median = float(np.median(m_vals))
        elif n > 0:
            mean = float(np.mean(m_vals))
            std = float("nan")
            median = float(np.median(m_vals))
        else:
            mean = std = median = float("nan")
        # full window (settle + measure) -> raw trace for the viewer
        f_mask = (force_t >= w0) & (force_t < t1)
        ft_win = force_t[f_mask]
        fy_win = force_y[f_mask] - tare
        if ft_win.size > MAX_PTS:
            stride = int(np.ceil(ft_win.size / MAX_PTS))
            ft_win = ft_win[::stride]
            fy_win = fy_win[::stride]
        levels.append(
            FlowLevelResult(
                flow_mm3_s=seg.flow_mm3_s,
                feed_mm_s=seg.feed_mm_s,
                n_samples=n,
                force_mean=mean,
                force_std=std,
                force_median=median,
                t_window_start=w0 - sweep_start,
                t_settle=ts - sweep_start,
                t_end=t1 - sweep_start,
                t=[float(x) for x in (ft_win - sweep_start)],
                force=[float(x) for x in fy_win],
            )
        )

    analysis = FlowAnalysis(
        sweep_t0=sweep_start,
        sweep_end=sweep_end,
        step_width=step_width,
        detect_method=detect_method,
        tare=tare,
        levels=levels,
        derate_frac=derate_frac,
        n_sigma=n_sigma,
        var_factor=var_factor,
        sample_rate_hz=sample_rate_hz,
        notes=notes,
    )

    # Valid levels with a finite mean, sorted by flow.
    valid = [lv for lv in levels if math.isfinite(lv.force_mean)]
    valid.sort(key=lambda lv: lv.flow_mm3_s)
    if len(valid) < 4:
        notes.append(
            f"only {len(valid)} levels had enough samples -- no detection "
            "(check that the loadcell streamed during extrusion)"
        )
        analysis.recommended_max_flow = None
        return analysis

    q = np.array([lv.flow_mm3_s for lv in valid], dtype=float)
    fmean = np.array([lv.force_mean for lv in valid], dtype=float)
    fstd = np.array(
        [lv.force_std if math.isfinite(lv.force_std) else 0.0 for lv in valid],
        dtype=float,
    )

    # --- baseline power-law fit over the low-flow region ---
    # Fit the lowest ~45% of levels (>= 5 points): the regime that should
    # be well-behaved. A larger region reduces exponent bias in the
    # extrapolation. Refining the region adaptively (grow until residuals
    # break) is a future step once we have real hardware curves.
    n_fit = max(5, int(round(0.45 * len(valid))))
    n_fit = min(n_fit, len(valid))
    fit_res = _fit_power_law(q[:n_fit], fmean[:n_fit])
    if fit_res is not None:
        a, b, c, residual_std, r2 = fit_res
        analysis.fit = FlowFit(
            a=a, b=b, c=c, n_points=n_fit,
            fit_max_flow=float(q[n_fit - 1]),
            residual_std=residual_std, r_squared=r2,
        )
        notes.append(
            f"baseline fit F={a:.3g}·Q^{b:.2f}+{c:.3g} over {n_fit} levels "
            f"(R²={r2:.3f}, residσ={residual_std:.3g} a.u.)"
        )
    else:
        notes.append("power-law baseline fit failed; relying on variance/collapse")

    # --- (1) upward-deviation onset ---
    # Band per level = max(n_sigma·residualσ, rel_tol·predicted). On a very
    # clean low-flow region the in-sample residual σ is tiny, so a pure
    # Nσ band false-triggers on ordinary power-law extrapolation error;
    # the relative-tolerance floor guards against that. Require TWO
    # consecutive exceedances so a single noisy level doesn't trip it, and
    # report the first of the pair as the onset.
    if analysis.fit is not None:
        prev_exceed = False
        for i in range(n_fit, len(valid)):
            predicted = float(analysis.fit.eval(q[i]))
            band = max(n_sigma * analysis.fit.residual_std, rel_tol * abs(predicted))
            exceed = (fmean[i] - predicted) > band
            if exceed and prev_exceed:
                analysis.deviation_flow = float(q[i - 1])
                break
            prev_exceed = exceed

    # --- (2) variance-rise onset ---
    quiet = fstd[:n_fit]
    quiet = quiet[quiet > 0]
    if quiet.size:
        baseline_noise = float(np.median(quiet))
        if baseline_noise > 0:
            thr = var_factor * baseline_noise
            for i in range(n_fit, len(valid)):
                if math.isfinite(fstd[i]) and fstd[i] > thr:
                    analysis.variance_onset_flow = float(q[i])
                    break
            notes.append(
                f"quiet-region force σ≈{baseline_noise:.3g} a.u.; "
                f"variance trigger at {var_factor:.0f}× = {thr:.3g} a.u."
            )

    # --- (3) force collapse (skip) ---
    peak_i = int(np.argmax(fmean))
    if peak_i < len(valid) - 1:
        peak = float(fmean[peak_i])
        for i in range(peak_i + 1, len(valid)):
            if peak > 0 and fmean[i] < peak * (1.0 - collapse_frac):
                analysis.collapse_flow = float(q[i])
                break

    # --- recommended max (earliest soft trigger, derated) ---
    soft = [x for x in (analysis.deviation_flow, analysis.variance_onset_flow)
            if x is not None]
    if soft:
        earliest = min(soft)
        analysis.recommended_max_flow = round(earliest * (1.0 - derate_frac), 2)
        notes.append(
            f"recommended max = {earliest:.2f} × (1−{derate_frac:.2f}) "
            f"= {analysis.recommended_max_flow:.2f} mm³/s"
        )
    elif analysis.collapse_flow is not None:
        analysis.recommended_max_flow = round(
            analysis.collapse_flow * (1.0 - derate_frac), 2
        )
        notes.append(
            "no soft breakdown seen before collapse; recommended max derated "
            f"from collapse point = {analysis.recommended_max_flow:.2f} mm³/s"
        )
    else:
        analysis.recommended_max_flow = float(q[-1])
        notes.append(
            "no breakdown detected across the swept range -- the real max "
            "flow is at or above the highest level tested; raise max flow "
            "and re-run to find it"
        )

    return analysis


def flow_analysis_to_dict(a: FlowAnalysis) -> dict[str, Any]:
    """JSON-safe dict for the API / frontend plot."""
    def sf(x: float | None) -> float | None:
        if x is None:
            return None
        try:
            return float(x) if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    return {
        "sweep_t0": sf(a.sweep_t0),
        "sweep_end": sf(a.sweep_end),
        "step_width": sf(a.step_width),
        "detect_method": a.detect_method,
        "tare": sf(a.tare),
        "sample_rate_hz": a.sample_rate_hz,
        "derate_frac": a.derate_frac,
        "n_sigma": a.n_sigma,
        "var_factor": a.var_factor,
        "deviation_flow": sf(a.deviation_flow),
        "variance_onset_flow": sf(a.variance_onset_flow),
        "collapse_flow": sf(a.collapse_flow),
        "recommended_max_flow": sf(a.recommended_max_flow),
        "fit": (
            {
                "a": sf(a.fit.a), "b": sf(a.fit.b), "c": sf(a.fit.c),
                "n_points": a.fit.n_points,
                "fit_max_flow": sf(a.fit.fit_max_flow),
                "residual_std": sf(a.fit.residual_std),
                "r_squared": sf(a.fit.r_squared),
            }
            if a.fit is not None else None
        ),
        "levels": [
            {
                "flow_mm3_s": sf(lv.flow_mm3_s),
                "feed_mm_s": sf(lv.feed_mm_s),
                "n_samples": lv.n_samples,
                "force_mean": sf(lv.force_mean),
                "force_std": sf(lv.force_std),
                "force_median": sf(lv.force_median),
                "t_window_start": sf(lv.t_window_start),
                "t_settle": sf(lv.t_settle),
                "t_end": sf(lv.t_end),
                "t": lv.t,
                "force": lv.force,
            }
            for lv in a.levels
        ],
        "notes": a.notes,
    }
