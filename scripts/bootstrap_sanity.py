"""Bootstrap sanity check for the weight-optimiser design.

Resamples segments-per-K with replacement, re-runs the cost minimisation
under BD_DEFAULT_WEIGHTS, and reports the distribution of K_opt across
draws. Run on an existing dump:

    python bootstrap_sanity.py runs/run_1779100636.npz

The output answers one question: is the bootstrap K_opt distribution
well-behaved (unimodal, tight) on real data, before we build an
optimiser on top of it?
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from prusa_pa_tuner.analysis import (
    BD_DEFAULT_WEIGHTS,
    BD_METRIC_NAMES,
    BdKResult,
    _argmin_with_parabolic,
    _bd_compute_cost,
    _bd_compute_normalised,
)
from prusa_pa_tuner.replay import replay


N_BOOT = 300
RNG_SEED = 0xC0FFEE


def _bootstrap_kopt(segments, rng: np.random.Generator) -> float | None:
    """One bootstrap draw: resample segments within each K with
    replacement, rebuild per-K medians, normalise, compute cost under
    BD_DEFAULT_WEIGHTS, return the parabolic-interp argmin K.
    """
    by_k: dict[float, list] = defaultdict(list)
    for s in segments:
        if not s.excluded:
            by_k[s.k].append(s)

    resampled: list[BdKResult] = []
    for k in sorted(by_k):
        segs = by_k[k]
        n = len(segs)
        if n == 0:
            continue
        idx = rng.integers(0, n, size=n)
        draw = [segs[i] for i in idx]
        medians: dict[str, float] = {}
        for name in BD_METRIC_NAMES:
            vals = np.asarray(
                [s.metrics.get(name, np.nan) for s in draw], dtype=float
            )
            if np.isnan(vals).all():
                medians[name] = float("nan")
            else:
                medians[name] = float(np.nanmedian(vals))
        resampled.append(
            BdKResult(
                k=float(k),
                n_segments_total=n,
                n_segments_included=n,
                medians=medians,
            )
        )

    _bd_compute_normalised(resampled)
    ks = np.array([r.k for r in resampled], dtype=float)
    cost = _bd_compute_cost(resampled, BD_DEFAULT_WEIGHTS)
    return _argmin_with_parabolic(ks, cost)


def _ascii_hist(values: np.ndarray, bins: int = 30, width: int = 60) -> str:
    counts, edges = np.histogram(values, bins=bins)
    peak = counts.max() if counts.max() > 0 else 1
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(round(width * c / peak))
        lines.append(f"  {edges[i]:.4f} | {bar} {c}")
    lines.append(f"  {edges[-1]:.4f} |")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        candidates = sorted(Path("runs").glob("run_*.npz"))
        if not candidates:
            print("usage: python bootstrap_sanity.py <path-to-npz>")
            return 1
        path = candidates[-1]
        print(f"(no path given, using newest dump: {path})")
    else:
        path = Path(sys.argv[1])

    print(f"loading + analysing {path} ...")
    plan, analysis = replay(path)
    n_seg_total = len(analysis.bd_segments)
    n_seg_included = sum(1 for s in analysis.bd_segments if not s.excluded)
    print(
        f"  bd_segments: {n_seg_included} included / "
        f"{n_seg_total} total across {len(analysis.bd_per_k)} K values"
    )
    print(f"  analysis bd_k_opt (deterministic): {analysis.bd_k_opt}")

    rng = np.random.default_rng(RNG_SEED)
    k_opts: list[float] = []
    n_failed = 0
    for _ in range(N_BOOT):
        ko = _bootstrap_kopt(analysis.bd_segments, rng)
        if ko is None or not np.isfinite(ko):
            n_failed += 1
            continue
        k_opts.append(float(ko))

    arr = np.asarray(k_opts)
    print()
    print(f"=== bootstrap K_opt (n={len(arr)}, failed={n_failed}) ===")
    print(f"  mean   = {arr.mean():.5f}")
    print(f"  median = {np.median(arr):.5f}")
    print(f"  std    = {arr.std(ddof=1):.5f}")
    print(f"  min    = {arr.min():.5f}")
    print(f"  max    = {arr.max():.5f}")
    print(f"  p05    = {np.percentile(arr, 5):.5f}")
    print(f"  p95    = {np.percentile(arr, 95):.5f}")
    print(f"  IQR    = {np.percentile(arr, 75) - np.percentile(arr, 25):.5f}")
    print()
    print("=== histogram ===")
    print(_ascii_hist(arr))

    out_dir = Path("diagnostics")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{path.stem}.bootstrap.npz"
    np.savez(out, k_opts=arr, deterministic_k_opt=np.array([analysis.bd_k_opt or np.nan]))
    print(f"\nraw samples saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
