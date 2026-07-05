"""Shared metric-sample extraction and stream-ordering helpers.

Used by all four job runners (PA sweep, max-flow, touch-probe, live-map)
to turn raw UDP MetricSamples into clean numeric time series. Moved out
of runner.py so the other runners no longer reach back into it for
private helpers.
"""
from __future__ import annotations

import math

import numpy as np

from .udp_metrics import MetricSample


def sort_by_time(
    t: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, y) sorted by t ascending. Uses argsort with kind='stable'
    so samples with equal timestamps keep their original order.

    Necessary because the UDP stream can deliver per-metric samples
    with overlapping timestamps when consecutive packets cover
    overlapping firmware-time spans (the firmware-offset spread in
    udp_metrics anchors each batch at its own host arrival time, so
    when host inter-packet gap is shorter than the batch's firmware
    span, the second batch's earliest samples land before the first
    batch's latest). udp_metrics also clips to enforce monotonicity,
    but a defensive sort here means any future bug or network reorder
    won't break the plots / segment slicing.
    """
    if len(t) <= 1:
        return t, y
    # Skip the sort when already monotonic (the common case).
    if bool(np.all(np.diff(t) >= 0)):
        return t, y
    order = np.argsort(t, kind="stable")
    return t[order], y[order]


def extract_numeric(sample: MetricSample) -> float | None:
    """Pull the first finite numeric value out of a sample.

    Used for position metrics (pos_x / pos_y / pos_z) and any other plain-
    value telemetry. Unlike _extract_force we look at `v` first (the canonical
    field name on this firmware), then fall through to any other numeric.
    """
    f = sample.fields
    v = f.get("v")
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        x = float(v)
        if math.isfinite(x):
            return x
    for val in f.values():
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            x = float(val)
            if math.isfinite(x):
                return x
    return None


def extract_force(sample: MetricSample) -> float | None:
    """Try the well-known field names from probe_load_line / loadcell metrics.

    The exact field name depends on what's actually streaming. We accept any of:
      - `v` (single value)
      - `load` / `force` / `z` (named field)
      - the first numeric field in the sample as a last resort.

    NaN values are rejected (return None). loadcell_hp streams `v=nan` while
    the loadcell is idle -- if we let those through, the "first metric to
    deliver wins" race would lock onto loadcell_hp and flood force_y with
    NaNs, breaking downstream analysis.
    """
    f = sample.fields
    for key in ("load", "force", "z", "v"):
        if key in f and isinstance(f[key], (int, float)) and not isinstance(f[key], bool):
            v = float(f[key])
            if math.isnan(v) or math.isinf(v):
                return None
            return v
    for v in f.values():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            x = float(v)
            if math.isnan(x) or math.isinf(x):
                continue
            return x
    return None
