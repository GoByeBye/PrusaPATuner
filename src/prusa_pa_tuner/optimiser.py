"""Weight optimiser: tune BD_DEFAULT_WEIGHTS against ground-truth K values.

A user runs a test print, measures the best K visually, and stores it on
a saved npz via the annotation UI (`user_k_opt`). Once several runs are
annotated, this module fits a weight vector so the analyser's K_opt
matches those ground truths under bootstrap resampling of the segments.

Loss = Σ_runs sqrt(n_segs_median) · [(mean(K_opt_boot) − K_user)² + α·var(K_opt_boot)]

The bootstrap variance term enforces a *stable* optimum -- weights whose
K_opt jumps around under small data perturbations are penalised, even if
their mean prediction happens to land on K_user.

Optimisation is scipy.optimize.differential_evolution (gradient-free,
handles non-smooth loss from the argmin + parabolic-interp branches).
Weights are gauge-fixed inside the objective via sum-normalisation, so
the search box is a unit hypercube. Metrics the caller wants to exclude
are bounded to [0, 0] so DE never touches them.

Bootstrap indices are PRE-DRAWN once before DE starts and re-used for
every weight evaluation, so the loss is deterministic in w (otherwise
DE perceives the loss surface as noisy and stalls).
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.optimize import differential_evolution

from .analysis import (
    BD_DEFAULT_WEIGHTS,
    BD_METRIC_NAMES,
    _argmin_with_parabolic,
)
from .replay import read_annotation, replay

log = logging.getLogger(__name__)

# Matches the analyser's own MIN_INCLUDED_SEGS gate; K values with fewer
# included segments aren't reliable enough to contribute to K_opt.
MIN_INCLUDED_SEGS = 4

# Subset of BD_METRIC_NAMES that's actually tunable: the metrics that
# appear in BD_DEFAULT_WEIGHTS (the existing UI sliders). The other six
# (baseline_median, baseline_noise_std, rise_slope, high_level,
# plateau_creep, fall_error_area) are diagnostic-only -- they're shown
# in the per-segment debug view but don't conceptually carry a "best at
# minimum" interpretation, so they stay out of the cost composition.
OPTIMISABLE_METRICS: tuple[str, ...] = tuple(BD_DEFAULT_WEIGHTS.keys())


@dataclass(slots=True)
class RunData:
    """Per-run cache used by the bootstrap inner loop.

    Built once from `replay(path)` + `read_annotation(path)`. The
    optimiser never re-touches the npz; the DE objective only re-uses
    `metrics`, `norm_scale`, `boot_indices`, and `user_k_opt`.
    """
    path: Path
    filename: str
    ks: np.ndarray                 # (n_K,) K values kept after seg-count gate
    metrics: np.ndarray            # (n_K, max_seg, n_metrics) -- NaN padded
    n_per_k: np.ndarray            # (n_K,) -- actual segments per K
    norm_scale: np.ndarray         # (n_metrics,) -- max(|deterministic-median|) per metric
    user_k_opt: float
    n_segments_median: int         # data-quality weight in the loss


@dataclass(slots=True)
class RunPrediction:
    """One row in the per-run prediction-error table."""
    filename: str
    user_k_opt: float
    pred_mean: float
    pred_std: float
    pred_p05: float
    pred_p95: float
    n_segments_median: int


@dataclass(slots=True)
class OptimiseResult:
    """Full optimiser output, ready to serialise to JSON for the UI."""
    weights: dict[str, float]
    weights_display: dict[str, float]  # normalised so max = 1.0 (matches BD_DEFAULT_WEIGHTS convention)
    excluded_metrics: list[str]
    alpha: float
    n_boot: int
    seed: int
    n_runs: int
    rms_error: float
    median_abs_error: float
    mean_bootstrap_std: float
    per_run: list[RunPrediction]
    de_message: str
    de_iterations: int
    duration_s: float
    timestamp_unix: float
    warnings: list[str] = field(default_factory=list)


# ---------- run-data preparation ----------

def prepare_run_data(path: str | Path) -> RunData | None:
    """Build a `RunData` from a saved npz, or return None if the run
    is unusable (no annotation, too few K's, no segments).
    """
    p = Path(path)
    user_k_opt, _notes = read_annotation(p)
    if user_k_opt is None:
        return None

    plan, analysis = replay(p)
    if not analysis.bd_segments:
        return None

    by_k: dict[float, list] = defaultdict(list)
    for s in analysis.bd_segments:
        if not s.excluded:
            by_k[s.k].append(s)
    ks_kept = sorted(k for k, segs in by_k.items() if len(segs) >= MIN_INCLUDED_SEGS)
    if len(ks_kept) < 3:
        return None

    n_K = len(ks_kept)
    n_metrics = len(BD_METRIC_NAMES)
    max_seg = max(len(by_k[k]) for k in ks_kept)
    metrics = np.full((n_K, max_seg, n_metrics), np.nan, dtype=float)
    n_per_k = np.zeros(n_K, dtype=int)
    for i, k in enumerate(ks_kept):
        segs = by_k[k]
        n_per_k[i] = len(segs)
        for j, s in enumerate(segs):
            for m_idx, name in enumerate(BD_METRIC_NAMES):
                metrics[i, j, m_idx] = s.metrics.get(name, np.nan)

    # Fixed normaliser: max(|deterministic-median|) per metric over the
    # KEPT K subset. Bootstrap inherits the same scale so the cost
    # surface doesn't jitter from re-normalising on every draw.
    ks_kept_set = set(ks_kept)
    bd_by_k = {r.k: r for r in analysis.bd_per_k}
    norm_scale = np.zeros(n_metrics, dtype=float)
    for m_idx, name in enumerate(BD_METRIC_NAMES):
        vals: list[float] = []
        for k in ks_kept:
            r = bd_by_k.get(k)
            if r is not None:
                v = r.medians.get(name, np.nan)
                if np.isfinite(v):
                    vals.append(v)
        if vals:
            norm_scale[m_idx] = float(np.max(np.abs(vals)))

    return RunData(
        path=p,
        filename=p.name,
        ks=np.asarray(ks_kept, dtype=float),
        metrics=metrics,
        n_per_k=n_per_k,
        norm_scale=norm_scale,
        user_k_opt=float(user_k_opt),
        n_segments_median=int(np.median(n_per_k)),
    )


def discover_annotated_runs(runs_dir: str | Path = "runs") -> list[Path]:
    """Return paths of every annotated npz in `runs_dir`, sorted.

    "Annotated" means `user_k_opt` is set and finite. Unannotated dumps
    can't contribute to the optimisation and are silently skipped.
    """
    p = Path(runs_dir)
    if not p.exists():
        return []
    out: list[Path] = []
    for f in sorted(p.glob("run_*.npz")):
        try:
            k, _notes = read_annotation(f)
            if k is not None and np.isfinite(k):
                out.append(f)
        except Exception:
            continue
    return out


# ---------- bootstrap inner loop ----------

def _draw_boot_indices(
    n_per_k: np.ndarray, n_boot: int, rng: np.random.Generator,
) -> list[np.ndarray]:
    """One (n_boot, n_segs_for_K) integer matrix per K, sampling with
    replacement. Pre-drawn once before DE starts.
    """
    return [rng.integers(0, int(n), size=(n_boot, int(n))) for n in n_per_k]


def _bootstrap_kopts(
    run: RunData,
    boot_idx: list[np.ndarray],
    w_arr: np.ndarray,
    over_idx: int,
    under_idx: int,
) -> np.ndarray:
    """Vectorised bootstrap of the K_opt distribution under weights `w_arr`.

    Returns a (n_boot,) array of K_opt; entries are NaN where the cost
    curve had no finite values (e.g. an excluded metric carrying NaN
    through everything).
    """
    n_K = len(run.ks)
    n_boot = boot_idx[0].shape[0] if n_K > 0 else 0
    n_metrics = run.metrics.shape[2]

    medians = np.empty((n_boot, n_K, n_metrics), dtype=float)
    for i in range(n_K):
        # metrics[i, :n_per_k[i], :] has shape (n_seg, n_metrics).
        # boot_idx[i] has shape (n_boot, n_seg). Fancy-indexing the
        # first axis yields (n_boot, n_seg, n_metrics).
        n_seg = int(run.n_per_k[i])
        gathered = run.metrics[i, :n_seg, :][boot_idx[i], :]
        # Median over the per-boot resampled segments. Suppress the
        # "all-NaN slice" warning -- when every segment of a K had a
        # NaN for some metric, we *want* the median to be NaN so it
        # falls out of the cost.
        with np.errstate(invalid="ignore"):
            medians[:, i, :] = np.nanmedian(gathered, axis=1)

    # Normalise with the FIXED scale (deterministic max). Divide-by-zero
    # for metrics whose scale is zero produces NaN, which the weighted
    # sum then propagates.
    safe_scale = np.where(run.norm_scale > 0, run.norm_scale, np.nan)
    normalised = medians / safe_scale[None, None, :]
    # Overshoot/undershoot: matches `_bd_compute_cost`'s clip-at-zero
    # treatment so a K with no spike isn't double-penalised by being
    # near-zero.
    normalised[..., over_idx] = np.maximum(0.0, normalised[..., over_idx])
    normalised[..., under_idx] = np.maximum(0.0, normalised[..., under_idx])

    # Weighted sum over metrics → (n_boot, n_K) cost. Any NaN in
    # normalised propagates -- so a K whose every-metric value was NaN
    # for a given bootstrap draw shows up as NaN cost and is skipped
    # by `_argmin_with_parabolic` (which only considers finite costs).
    cost = np.einsum("bkm,m->bk", normalised, w_arr)

    kopts = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ko = _argmin_with_parabolic(run.ks, cost[b])
        kopts[b] = ko if ko is not None else np.nan
    return kopts


# ---------- DE objective ----------

def _expand_to_full_weights(w_active: np.ndarray) -> np.ndarray:
    """Map an 8-d weight vector over OPTIMISABLE_METRICS into a 14-d
    vector over BD_METRIC_NAMES (zeros for diagnostic-only metrics).
    """
    w_full = np.zeros(len(BD_METRIC_NAMES), dtype=float)
    for i, name in enumerate(OPTIMISABLE_METRICS):
        full_idx = BD_METRIC_NAMES.index(name)
        w_full[full_idx] = w_active[i]
    return w_full


def _make_objective(
    runs: list[RunData],
    all_boot_idx: list[list[np.ndarray]],
    alpha: float,
) -> Callable[[np.ndarray], float]:
    """Closure over the bootstrap indices, returning a DE-compatible loss.

    `w` from DE is over OPTIMISABLE_METRICS (8-dim); expand to the full
    14-dim metric space before calling the bootstrap. Returns a large
    finite penalty (1e6) on degenerate weight vectors so DE doesn't get
    stuck on NaN losses.
    """
    over_idx = BD_METRIC_NAMES.index("overshoot")
    under_idx = BD_METRIC_NAMES.index("undershoot")
    PENALTY = 1e6

    def loss(w: np.ndarray) -> float:
        s = float(w.sum())
        if s <= 1e-9:
            return PENALTY  # all weights ~0 → no cost surface
        w_norm = w / s
        w_full = _expand_to_full_weights(w_norm)
        total = 0.0
        for run, bidx in zip(runs, all_boot_idx):
            kopts = _bootstrap_kopts(run, bidx, w_full, over_idx, under_idx)
            finite = kopts[np.isfinite(kopts)]
            if len(finite) < 10:
                return PENALTY  # weights produced mostly-undefined K_opts
            bias_sq = float((finite.mean() - run.user_k_opt) ** 2)
            var = float(finite.var())
            wt = float(np.sqrt(run.n_segments_median))
            total += wt * (bias_sq + alpha * var)
        return total

    return loss


# ---------- entry point ----------

def optimise_weights(
    annotated_paths: list[Path],
    excluded_metrics: list[str] | None = None,
    alpha: float = 1.0,
    n_boot: int = 200,
    seed: int = 0xC0FFEE,
    popsize: int = 15,
    maxiter: int = 50,
    progress_cb: Callable[[float, str], None] | None = None,
) -> OptimiseResult:
    """Run the full optimisation against the supplied annotated npz paths.

    `progress_cb(fraction_done, message)` is invoked from the DE callback
    so the GUI can show a progress bar. The function is CPU-bound (and
    DE is a long pure-Python loop); callers usually wrap it in
    `asyncio.to_thread`.
    """
    if excluded_metrics is None:
        excluded_metrics = []

    t0 = time.monotonic()
    warnings: list[str] = []

    # Prepare run data. Each path may produce None (no annotation, too
    # few K's, no segments) -- those are warned but the optimiser keeps
    # going on the rest.
    runs: list[RunData] = []
    if progress_cb:
        progress_cb(0.0, "loading annotated runs")
    for i, p in enumerate(annotated_paths):
        try:
            rd = prepare_run_data(p)
        except Exception as exc:
            warnings.append(f"{p.name}: load failed: {type(exc).__name__}: {exc}")
            continue
        if rd is None:
            warnings.append(f"{p.name}: no annotation or insufficient data, skipped")
            continue
        runs.append(rd)
        if progress_cb:
            progress_cb(
                0.1 * (i + 1) / max(len(annotated_paths), 1),
                f"loaded {rd.filename}",
            )
    if not runs:
        raise ValueError(
            "no usable annotated runs found "
            "(need at least one npz with user_k_opt set + >= 3 K's passing the segment-count gate)"
        )
    if len(runs) < 5:
        warnings.append(
            f"only {len(runs)} annotated run(s) available -- weights may overfit; "
            "annotate more runs for a robust optimum"
        )

    # Pre-draw bootstrap indices: same draws used for every DE evaluation
    # so the loss is deterministic in w.
    rng = np.random.default_rng(seed)
    all_boot_idx = [_draw_boot_indices(r.n_per_k, n_boot, rng) for r in runs]

    # Bounds over the 8 OPTIMISABLE_METRICS only. Diagnostic-only
    # metrics in BD_METRIC_NAMES \ OPTIMISABLE_METRICS get weight 0
    # baked in (never appear in the search). Excluded metrics passed
    # in by the caller also get weight 0.
    bounds: list[tuple[float, float]] = []
    for name in OPTIMISABLE_METRICS:
        if name in excluded_metrics:
            bounds.append((0.0, 1e-9))  # near-zero so DE doesn't trip the s<=1e-9 guard
        else:
            bounds.append((0.0, 1.0))

    n_active = sum(1 for b in bounds if b[1] > 1e-6)
    if n_active < 2:
        raise ValueError(
            "at least 2 metrics must remain (un-excluded) for the optimiser to have something to balance"
        )

    objective = _make_objective(runs, all_boot_idx, alpha)

    # DE callback: scipy passes `(xk, convergence)`. `convergence` is
    # [0, 1] roughly proportional to "DE thinks it's done". We use it as
    # the progress fraction directly, scaled to the [0.1, 0.95] band so
    # the load phase (0..0.1) and finalisation (0.95..1.0) don't conflict.
    de_iters = [0]

    def _cb(xk, convergence):
        de_iters[0] += 1
        if progress_cb:
            frac = min(0.95, 0.1 + 0.85 * float(convergence))
            progress_cb(frac, f"DE iter {de_iters[0]}")
        return False  # don't stop early

    if progress_cb:
        progress_cb(0.1, f"optimising over {n_active} active metric(s), {len(runs)} run(s)")

    result = differential_evolution(
        objective,
        bounds=bounds,
        seed=seed,
        popsize=popsize,
        maxiter=maxiter,
        tol=1e-4,
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        init="sobol",
        callback=_cb,
        # DE's `workers=-1` would parallelise but each worker re-imports
        # scipy + numpy and the bootstrap is already cheap; leave serial.
        workers=1,
    )

    # result.x is 8-dim over OPTIMISABLE_METRICS. Build both the
    # sum-normalised (Σ=1) and display-normalised (max=1) versions
    # restricted to the optimisable subset, then zero out excluded
    # metrics so the JSON file is exactly what the user requested.
    w_raw = np.asarray(result.x, dtype=float)
    for i, name in enumerate(OPTIMISABLE_METRICS):
        if name in excluded_metrics:
            w_raw[i] = 0.0
    s = w_raw.sum()
    w_norm_sum1 = w_raw / s if s > 0 else w_raw
    max_w = float(w_norm_sum1.max()) if w_norm_sum1.max() > 0 else 1.0
    w_display = (w_norm_sum1 / max_w) if max_w > 0 else w_norm_sum1

    weights: dict[str, float] = {
        name: float(w_norm_sum1[i]) for i, name in enumerate(OPTIMISABLE_METRICS)
    }
    weights_display: dict[str, float] = {
        name: float(w_display[i]) for i, name in enumerate(OPTIMISABLE_METRICS)
    }

    # Per-run prediction error table -- evaluate with the FINAL weights
    # (sum-normalised, in the 14-dim BD_METRIC_NAMES space the bootstrap
    # expects).
    if progress_cb:
        progress_cb(0.95, "computing per-run prediction error")
    over_idx = BD_METRIC_NAMES.index("overshoot")
    under_idx = BD_METRIC_NAMES.index("undershoot")
    w_full_arr = _expand_to_full_weights(np.array(
        [weights[name] for name in OPTIMISABLE_METRICS], dtype=float,
    ))
    sum_w = w_full_arr.sum()
    if sum_w > 0:
        w_arr = w_full_arr / sum_w
    else:
        w_arr = w_full_arr
    per_run: list[RunPrediction] = []
    abs_errors: list[float] = []
    boot_stds: list[float] = []
    for run, bidx in zip(runs, all_boot_idx):
        kopts = _bootstrap_kopts(run, bidx, w_arr, over_idx, under_idx)
        finite = kopts[np.isfinite(kopts)]
        if len(finite) == 0:
            mean = std = p05 = p95 = float("nan")
        else:
            mean = float(finite.mean())
            std = float(finite.std())
            p05 = float(np.percentile(finite, 5))
            p95 = float(np.percentile(finite, 95))
        per_run.append(
            RunPrediction(
                filename=run.filename,
                user_k_opt=run.user_k_opt,
                pred_mean=mean,
                pred_std=std,
                pred_p05=p05,
                pred_p95=p95,
                n_segments_median=run.n_segments_median,
            )
        )
        if np.isfinite(mean):
            abs_errors.append(abs(mean - run.user_k_opt))
            boot_stds.append(std)

    rms_error = float(np.sqrt(np.mean([e * e for e in abs_errors]))) if abs_errors else float("nan")
    median_abs_error = float(np.median(abs_errors)) if abs_errors else float("nan")
    mean_bootstrap_std = float(np.mean(boot_stds)) if boot_stds else float("nan")

    duration_s = time.monotonic() - t0

    if progress_cb:
        progress_cb(1.0, "done")

    return OptimiseResult(
        weights=weights,
        weights_display=weights_display,
        excluded_metrics=list(excluded_metrics),
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        n_runs=len(runs),
        rms_error=rms_error,
        median_abs_error=median_abs_error,
        mean_bootstrap_std=mean_bootstrap_std,
        per_run=per_run,
        de_message=str(result.message),
        de_iterations=int(de_iters[0]),
        duration_s=duration_s,
        timestamp_unix=time.time(),
        warnings=warnings,
    )


# ---------- persistence ----------

WEIGHTS_OPT_PATH = Path("runs") / "weights_opt.json"


def write_weights_json(result: OptimiseResult, path: str | Path = WEIGHTS_OPT_PATH) -> None:
    """Persist the optimised weights so the server can load them on startup."""
    p = Path(path)
    p.parent.mkdir(exist_ok=True)
    payload = {
        "weights": result.weights_display,  # display-normalised (max=1.0)
        "alpha": result.alpha,
        "n_runs": result.n_runs,
        "rms_error": result.rms_error,
        "median_abs_error": result.median_abs_error,
        "excluded_metrics": result.excluded_metrics,
        "timestamp_unix": result.timestamp_unix,
        "shipped_defaults": dict(BD_DEFAULT_WEIGHTS),  # for posterity
    }
    p.write_text(json.dumps(payload, indent=2))


def read_weights_json(
    path: str | Path = WEIGHTS_OPT_PATH,
) -> dict[str, Any] | None:
    """Read a previously-written weights file. Returns None if absent or invalid."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if not isinstance(data, dict) or "weights" not in data:
            return None
        return data
    except Exception:
        return None


def result_to_dict(r: OptimiseResult) -> dict[str, Any]:
    """JSON-serialisable dict for the FastAPI response + the apply endpoint."""
    return {
        "weights": r.weights,
        "weights_display": r.weights_display,
        "excluded_metrics": r.excluded_metrics,
        "alpha": r.alpha,
        "n_boot": r.n_boot,
        "seed": r.seed,
        "n_runs": r.n_runs,
        "rms_error": r.rms_error,
        "median_abs_error": r.median_abs_error,
        "mean_bootstrap_std": r.mean_bootstrap_std,
        "per_run": [
            {
                "filename": pr.filename,
                "user_k_opt": pr.user_k_opt,
                "pred_mean": pr.pred_mean,
                "pred_std": pr.pred_std,
                "pred_p05": pr.pred_p05,
                "pred_p95": pr.pred_p95,
                "n_segments_median": pr.n_segments_median,
                "delta": pr.pred_mean - pr.user_k_opt,
            }
            for pr in r.per_run
        ],
        "de_message": r.de_message,
        "de_iterations": r.de_iterations,
        "duration_s": r.duration_s,
        "timestamp_unix": r.timestamp_unix,
        "warnings": r.warnings,
        "optimisable_metrics": list(OPTIMISABLE_METRICS),
    }
