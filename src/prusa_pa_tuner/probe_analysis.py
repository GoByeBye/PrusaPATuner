"""Analyse a lateral touch-probe characterisation run.

Goal: from the captured loadcell + toolhead-position streams, decide whether
the loadcell sees lateral nozzle contact cleanly enough to build a probe on
-- and quantify it. Everything is plotted so the user judges it visually
(this is a v1 heuristic; thresholds will be re-tuned on real hardware).

Method (position-space, NOT time-anchored -- robust against the variable
heat-up/home delay and UDP latency):

  1. Interpolate the (often sparser) probe-axis position onto the (dense
     ~180 Hz) force timestamps, so every force sample carries a position.
  2. Convert to signed PROGRESS along the probe direction:
        progress = (pos - standoff) * dir_sign
     so each slow creep sweeps progress 0 -> creep_mm, and the fast
     retracts/approaches show up as large |velocity| excursions.
  3. Segment the N slow creeps: contiguous runs whose probe-axis speed sits
     in a band around the slow feed (well below travel speed), spanning a
     good fraction of creep_mm. (Mirrors flow's force-activity segmentation,
     but on velocity.)
  4. Per creep: tare the force to the PRE-CONTACT baseline (force over the
     first part of the creep, before the target), then find the CONTACT
     position -- the first progress where tared force rises above
     n_sigma·noise and stays there. Record the peak force + monotonicity.
  5. Aggregate across touches: contact-position mean/σ (repeatability) and a
     signal-to-noise ratio. The verdict ("usable" / "marginal" / "no clear
     signal") is a convenience label; the overlaid force-vs-position plot is
     the real evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .probe_gen import ProbePlan


@dataclass(slots=True)
class ProbeTouchResult:
    idx: int
    n_samples: int
    baseline_force: float      # pre-contact median (raw units)
    noise: float               # pre-contact MAD-based σ (raw units)
    contact_pos: float | None  # progress (mm) where contact was detected
    peak_force: float          # max tared force over the creep (raw units)
    monotonic_frac: float      # fraction of contact-region steps that rise
    # Decimated force-vs-progress trace for the overlay plot. `pos` is
    # progress in mm from the standoff (0..creep); `force` is tared.
    pos: list[float] = field(default_factory=list)
    force: list[float] = field(default_factory=list)


@dataclass(slots=True)
class ProbeAnalysis:
    axis: str
    dir_sign: float
    standoff: float                # probe-axis standoff coord (machine mm)
    creep_mm: float
    sample_rate_hz: float = 0.0
    touches: list[ProbeTouchResult] = field(default_factory=list)
    contact_pos_mean: float | None = None
    contact_pos_std: float | None = None     # repeatability (mm)
    n_contacts: int = 0
    signal_to_noise: float = 0.0   # median peak force / median noise
    verdict: str = ""              # "usable" | "marginal" | "no clear signal"
    n_sigma: float = 5.0
    notes: list[str] = field(default_factory=list)


def _mad_sigma(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    s = 1.4826 * mad
    if not np.isfinite(s) or s <= 0:
        s = float(np.std(x)) if x.size > 1 else 0.0
    return s


def _segment_creeps(
    progress: np.ndarray,
    t: np.ndarray,
    creep_mm: float,
    slow_mm_s: float,
    n_touches: int,
) -> list[tuple[int, int]]:
    """Return [(lo, hi), ...] index spans of the slow forward creeps.

    A creep sample is one whose forward speed sits in a band around the
    slow feed -- fast travel (retract/approach) is far above the upper edge,
    and idle dwell is below the lower edge. Contiguous creep samples are
    merged (bridging brief gaps); spans must cover a good fraction of
    creep_mm to count. The longest-span n_touches are returned, time-sorted.
    """
    n = len(t)
    if n < 5:
        return []
    # forward velocity (mm/s) along the probe direction
    dt = np.gradient(t)
    dt[dt <= 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    vel = np.gradient(progress) / dt
    # light smoothing so position-sample quantisation doesn't shred the band
    if n >= 5:
        k = max(1, int(round(0.05 * (1.0 / np.median(dt[dt > 0]))))) if np.any(dt > 0) else 1
        if k > 1:
            kernel = np.ones(k) / k
            vel = np.convolve(vel, kernel, mode="same")

    lo_band = 0.2 * slow_mm_s
    hi_band = max(4.0 * slow_mm_s, slow_mm_s + 1.0)
    creeping = (vel > lo_band) & (vel < hi_band)

    # bridge short gaps (< 0.3 s) inside a creep
    gap_max = 1
    if np.any(dt > 0):
        gap_max = max(1, int(round(0.3 * (1.0 / np.median(dt[dt > 0])))))
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not creeping[i]:
            i += 1
            continue
        j = i
        while j < n:
            if creeping[j]:
                j += 1
            else:
                k = j
                while k < n and not creeping[k]:
                    k += 1
                if k < n and (k - j) <= gap_max:
                    j = k
                else:
                    break
        runs.append((i, j - 1))
        i = j

    # keep runs that span a meaningful fraction of the creep distance
    good = []
    for lo, hi in runs:
        span = abs(progress[hi] - progress[lo])
        if span >= 0.4 * creep_mm and (hi - lo) >= 3:
            good.append((lo, hi))
    if not good:
        return []
    # the N largest-span creeps, returned in time order
    good.sort(key=lambda r: abs(progress[r[1]] - progress[r[0]]), reverse=True)
    good = good[: max(1, n_touches)]
    good.sort(key=lambda r: r[0])
    return good


def _detect_contact(
    pos: np.ndarray, f_tared: np.ndarray, noise: float, n_sigma: float,
) -> tuple[float | None, float]:
    """Find the contact progress + monotonic fraction of the contact region.

    Contact = first index where tared force exceeds n_sigma·noise and stays
    above for >= 3 consecutive samples (debounce). Returns (contact_pos,
    monotonic_frac) where monotonic_frac is the fraction of post-contact
    steps that increase (a clean knee rises steadily).
    """
    thr = n_sigma * noise if noise > 0 else (0.0 if f_tared.size == 0 else float(np.max(f_tared)) + 1.0)
    above = f_tared > thr
    contact_i: int | None = None
    run = 0
    for i in range(len(above)):
        if above[i]:
            run += 1
            if run >= 3:
                contact_i = i - 2
                break
        else:
            run = 0
    if contact_i is None:
        return None, 0.0
    contact_pos = float(pos[contact_i])
    tail = f_tared[contact_i:]
    if tail.size >= 2:
        d = np.diff(tail)
        monotonic_frac = float(np.mean(d > 0))
    else:
        monotonic_frac = 0.0
    return contact_pos, monotonic_frac


def _decimate(x: np.ndarray, y: np.ndarray, max_pts: int = 400) -> tuple[list[float], list[float]]:
    if x.size <= max_pts:
        return [float(v) for v in x], [float(v) for v in y]
    stride = int(np.ceil(x.size / max_pts))
    return [float(v) for v in x[::stride]], [float(v) for v in y[::stride]]


def analyse_probe(
    *,
    force_t: np.ndarray,
    force_y: np.ndarray,
    pos_t: np.ndarray | None,
    pos: np.ndarray | None,
    plan: ProbePlan,
    n_sigma: float = 5.0,
    baseline_frac: float = 0.2,
) -> ProbeAnalysis:
    """Characterise lateral contact from the captured streams.

    `pos_t`/`pos` are the PROBE-AXIS position stream (pos_x if axis==X, else
    pos_y). Without position we can't put force in position-space, so the
    analysis degrades to "no segmentation" with an explanatory note.
    """
    p = plan.params
    axis = plan.axis
    sign = plan.dir_sign
    standoff = p.start_x if axis == "X" else p.start_y

    force_t = np.asarray(force_t, dtype=float)
    force_y = np.asarray(force_y, dtype=float)
    notes: list[str] = []

    analysis = ProbeAnalysis(
        axis=axis,
        dir_sign=sign,
        standoff=standoff,
        creep_mm=p.creep_mm,
        n_sigma=n_sigma,
    )

    if force_t.size < 5:
        notes.append("no loadcell samples captured -- nothing to analyse")
        analysis.verdict = "no data"
        analysis.notes = notes
        return analysis

    dt = float(np.median(np.diff(force_t))) if force_t.size > 1 else 0.0
    analysis.sample_rate_hz = (1.0 / dt) if dt > 0 else 0.0

    if pos_t is None or pos is None or np.asarray(pos).size < 2:
        notes.append(
            f"no pos_{axis.lower()} telemetry -- cannot place force in "
            "position-space. Enable the position metric and re-run."
        )
        analysis.verdict = "no position data"
        analysis.notes = notes
        return analysis

    pos_t = np.asarray(pos_t, dtype=float)
    pos = np.asarray(pos, dtype=float)
    order = np.argsort(pos_t, kind="stable")
    pos_t, pos = pos_t[order], pos[order]

    # interpolate position onto the dense force time grid
    pos_at_f = np.interp(force_t, pos_t, pos)
    progress = (pos_at_f - standoff) * sign

    slow_mm_s = max(p.slow_feed_mm_min / 60.0, 1e-6)
    spans = _segment_creeps(progress, force_t, p.creep_mm, slow_mm_s, p.n_touches)
    if not spans:
        notes.append(
            "could not segment any slow creeps from the position stream "
            "(check that pos telemetry streamed and the creep actually ran)"
        )
        analysis.verdict = "no creeps detected"
        analysis.notes = notes
        return analysis

    notes.append(
        f"segmented {len(spans)} creep(s) of the planned {p.n_touches} "
        f"from the position stream"
    )

    contacts: list[float] = []
    peaks: list[float] = []
    noises: list[float] = []
    for k, (lo, hi) in enumerate(spans):
        seg_pos = progress[lo : hi + 1]
        seg_f = force_y[lo : hi + 1]
        # pre-contact baseline: the first `baseline_frac` of the creep span,
        # assumed to be before the target.
        cut = seg_pos[0] + baseline_frac * (seg_pos[-1] - seg_pos[0])
        base_mask = seg_pos <= cut
        if int(base_mask.sum()) >= 3:
            base = float(np.median(seg_f[base_mask]))
            noise = _mad_sigma(seg_f[base_mask])
        else:
            base = float(np.median(seg_f))
            noise = _mad_sigma(seg_f)
        f_tared = seg_f - base
        contact_pos, mono = _detect_contact(seg_pos, f_tared, noise, n_sigma)
        peak = float(np.max(f_tared)) if f_tared.size else 0.0
        dpos, dforce = _decimate(seg_pos, f_tared)
        analysis.touches.append(
            ProbeTouchResult(
                idx=k,
                n_samples=int(seg_f.size),
                baseline_force=base,
                noise=noise,
                contact_pos=contact_pos,
                peak_force=peak,
                monotonic_frac=mono,
                pos=dpos,
                force=dforce,
            )
        )
        if contact_pos is not None:
            contacts.append(contact_pos)
        peaks.append(peak)
        if noise > 0:
            noises.append(noise)

    analysis.n_contacts = len(contacts)
    if contacts:
        analysis.contact_pos_mean = float(np.mean(contacts))
        analysis.contact_pos_std = float(np.std(contacts)) if len(contacts) > 1 else 0.0
    median_peak = float(np.median(peaks)) if peaks else 0.0
    median_noise = float(np.median(noises)) if noises else 0.0
    analysis.signal_to_noise = (median_peak / median_noise) if median_noise > 0 else 0.0

    # --- verdict (convenience label; the plot is the real evidence) ---
    frac_contact = analysis.n_contacts / max(1, len(spans))
    std = analysis.contact_pos_std
    if (
        analysis.signal_to_noise >= 10.0
        and frac_contact >= 0.6
        and std is not None
        and std <= 0.1
    ):
        analysis.verdict = "usable"
        notes.append(
            f"clear, repeatable contact: SNR≈{analysis.signal_to_noise:.0f}, "
            f"{analysis.n_contacts}/{len(spans)} touches, "
            f"repeatability σ={std * 1000:.0f} µm"
        )
    elif analysis.signal_to_noise >= 4.0 and frac_contact >= 0.4:
        analysis.verdict = "marginal"
        notes.append(
            f"some signal but not clean enough to trust yet: "
            f"SNR≈{analysis.signal_to_noise:.1f}, "
            f"{analysis.n_contacts}/{len(spans)} touches contacted"
            + (f", repeatability σ={std * 1000:.0f} µm" if std is not None else "")
        )
    else:
        analysis.verdict = "no clear signal"
        notes.append(
            f"lateral contact did not register cleanly "
            f"(SNR≈{analysis.signal_to_noise:.1f}, "
            f"{analysis.n_contacts}/{len(spans)} touches). The loadcell may "
            "be too insensitive to lateral force for a probe."
        )

    analysis.notes = notes
    return analysis


def probe_analysis_to_dict(a: ProbeAnalysis) -> dict[str, Any]:
    """JSON-safe dict for the API / frontend plot."""
    import math

    def sf(x: float | None) -> float | None:
        if x is None:
            return None
        try:
            return float(x) if math.isfinite(x) else None
        except (TypeError, ValueError):
            return None

    return {
        "axis": a.axis,
        "dir_sign": a.dir_sign,
        "standoff": sf(a.standoff),
        "creep_mm": sf(a.creep_mm),
        "sample_rate_hz": a.sample_rate_hz,
        "contact_pos_mean": sf(a.contact_pos_mean),
        "contact_pos_std": sf(a.contact_pos_std),
        "n_contacts": a.n_contacts,
        "signal_to_noise": sf(a.signal_to_noise),
        "verdict": a.verdict,
        "n_sigma": a.n_sigma,
        "touches": [
            {
                "idx": t.idx,
                "n_samples": t.n_samples,
                "baseline_force": sf(t.baseline_force),
                "noise": sf(t.noise),
                "contact_pos": sf(t.contact_pos),
                "peak_force": sf(t.peak_force),
                "monotonic_frac": sf(t.monotonic_frac),
                "pos": t.pos,
                "force": t.force,
            }
            for t in a.touches
        ],
        "notes": a.notes,
    }
