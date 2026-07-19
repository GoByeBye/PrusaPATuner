"""Map a captured Live Map run onto its parsed gcode.

Each loadcell reading is placed at the **real toolhead position at its own
timestamp** -- pos_x/pos_y are streamed on the same recv_monotonic host clock
as the force, so interpolating them at the force time gives the true position
with no accumulation and no drift. (An earlier version re-derived position from
a single global arc-length scale; the streamed-vs-gcode distance ratio isn't
constant, so the error grew across the print and smeared the colours -- exactly
what we're avoiding here.)

Per sample we need: position (real, above), layer, whether it's a travel, and
which feature it belongs to.

  * layer  -- PRIMARY: the pos_z staircase. Raw pos_z carries junk spikes
              (MBL/encoder artefacts spanning ±168 mm), but ~2/3 of samples
              are the true z, so a rolling MEDIAN recovers a clean staircase
              (verified on livemap_1781710258: 0.31→0.70→…→5.10, i.e. the
              layer heights plus a constant MBL offset). We detect the
              staircase STEP TIMES (robust to that offset) and snap each one
              to the nearest travel -- a layer change is always a travel.
              FALLBACK: cumulative arc length (streamed XY distance ≈ gcode
              path distance). That estimate drifts by tens of mm within a
              layer (interp corner-cutting shortens the streamed path,
              unevenly across layers), which visibly bled the end of layer N
              into layer N+1 before the pos_z path existed.
  * travel -- from the real-position SPEED. Travels run faster than any
              extrusion feedrate; the threshold is derived from the gcode.
              Drift-free.
  * feature -- nearest gcode extrusion segment within the sample's layer.
               Drift-free, and robust in dense infill (neighbouring lines share
               the same feature anyway).

Pre-print moves (layer -1) are excluded by gcode_parse, so the purge / start
gcode never appear.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .gcode_parse import ParsedGcode, layer_polylines, layer_zs

_LAYER_POINT_CAP = 80_000
_FEATURE_SAMPLE_CAP = 200_000
_FOOTPRINT_MARGIN_MM = 2.0
_RUN_GAP_S = 0.4
_NEAREST_CHUNK = 2000
# Break the drawn line before a sample when it's separated from the previous
# kept sample by a real travel. A travel either dropped samples in between (the
# toolhead was sampled while moving fast -> index gap) OR, if those samples were
# lost too, leaves a jump bigger than this. Packet-loss gaps on a continuous
# extrusion are smaller than this and stay connected (no false dashing).
_BREAK_JUMP_MM = 12.0
# The position stream is ~78 Hz and also drops packets; we fill it by LINEAR
# interpolation. If a force sample falls inside a position gap wider than this,
# the interpolated point lies on a straight chord -- which cuts across the real
# path wherever the head turned or travelled during the gap (the stray
# wrong-angle lines). We can't trust that position, so we drop the sample and
# break the line there. ~2x the nominal 13 ms spacing (one missing pos sample).
_POS_GAP_MAX_S = 0.025
# Timestamp PILE-UP in a position stream: two samples closer than this are
# not a real 78 Hz cadence -- they're the seam where overlapping UDP batches
# were squeezed together (the value can step ~1 mm within ~0.1 ms). Positions
# interpolated near such a seam are laterally wrong (on 45° infill they land
# on the ADJACENT line), so we drop force samples within the radius below.
_POS_PILE_MIN_DT = 0.004
_POS_PILE_RADIUS_S = 0.02
# The force stream arrives in bursts (~3 ms spacing) separated by ~30 ms
# holes (firmware metric-buffer batching). A hole whose two bracketing kept
# samples sit on DIFFERENT infill lines used to be drawn as a straight
# connector cutting across the part. We break the line over a hole when the
# gcode arc-length advance between the two snaps disagrees with the straight
# chord (the head went around a turn during the hole); an along-line hole
# (advance ≈ chord) stays connected, so no false dashing.
_HOLE_BREAK_S = 0.015
_HOLE_MISMATCH_MM = 0.5
# pos_z staircase layer detection (see module docstring).
_Z_MED_WIN = 51            # rolling-median window (~0.65 s at 78 Hz)
_Z_MAD_MAX_FRAC = 0.45     # give up when residual MAD > this × layer pitch
_Z_TRAVEL_SNAP = 400       # snap a z boundary to a travel within ± this many samples


@dataclass(slots=True)
class MappedRun:
    parsed: ParsedGcode
    # one entry per kept (printing) loadcell sample, in time order
    s_x: np.ndarray          # snapped onto the nearest gcode line (clean/straight)
    s_y: np.ndarray
    s_xr: np.ndarray         # raw measured toolhead position (before snapping)
    s_yr: np.ndarray
    s_force: np.ndarray      # tared load
    s_layer: np.ndarray      # gcode layer index
    s_feat: np.ndarray       # feature index into parsed.features
    s_brk: np.ndarray        # bool: True = break the line before this sample (a travel)
    tare: float = 0.0
    force_lo: float = 0.0
    force_hi: float = 1.0
    feature_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    cross: dict[str, Any] = field(default_factory=dict)
    # pos_z staircase layer detection: source, per-boundary methods/shifts and
    # a downsampled staircase for the UI verification plot.
    layer_diag: dict[str, Any] = field(default_factory=dict)


def _interp_axis(qt, st, sv):
    if st is None or sv is None or len(st) == 0:
        return np.full(len(qt), np.nan)
    if len(st) == 1:
        return np.full(len(qt), float(sv[0]))
    return np.interp(qt, st, sv)


def _interp_gap(qt, st):
    """For each query time, the gap between the bracketing source-stream
    samples -- i.e. how far the linear interpolation reaches. Large => the
    interpolated value is untrustworthy."""
    if st is None or len(st) < 2:
        return np.zeros(len(qt))
    st = np.asarray(st, dtype=float)
    j = np.clip(np.searchsorted(st, qt), 1, len(st) - 1)
    return st[j] - st[j - 1]


def _pileup_mask(qt: np.ndarray, st) -> np.ndarray:
    """True where a query time sits within _POS_PILE_RADIUS_S of a timestamp
    pile-up in the source stream (two source samples < _POS_PILE_MIN_DT
    apart). Interpolation near such a seam is untrustworthy."""
    if st is None or len(np.atleast_1d(st)) < 2:
        return np.zeros(len(qt), dtype=bool)
    st = np.asarray(st, dtype=float)
    piles = st[1:][np.diff(st) < _POS_PILE_MIN_DT]
    if not piles.size:
        return np.zeros(len(qt), dtype=bool)
    j = np.searchsorted(piles, qt)
    near = np.full(len(qt), np.inf)
    m = j > 0
    near[m] = qt[m] - piles[j[m] - 1]
    m = j < len(piles)
    near[m] = np.minimum(near[m], piles[j[m]] - qt[m])
    return near < _POS_PILE_RADIUS_S


def _rolling_median(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or a.size < w:
        return a
    from numpy.lib.stride_tricks import sliding_window_view
    pad = w // 2
    ap = np.pad(a, pad, mode="edge")
    return np.median(sliding_window_view(ap, w), axis=1)


def _turn_fold_mask(qt: np.ndarray, sxt, sx_, syt, sy_) -> np.ndarray:
    """True where a query time falls inside a position-stream interval that
    hides a direction REVERSAL (a zigzag turnaround folded between two ~13 ms
    position samples). Interpolation across such an interval returns points
    on the fold's chord -- laterally between the two infill lines -- which
    the snap then flips onto either line, drawing little cross-links near
    the line ends. Detection: the directions entering and leaving interval j
    oppose (dot < -0.17·|a||b|) while the interval's own chord is short
    (the fold ate path length)."""
    if sxt is None or sx_ is None or len(np.atleast_1d(sxt)) < 4:
        return np.zeros(len(qt), dtype=bool)
    xt = np.asarray(sxt, dtype=float)
    X_ = np.asarray(sx_, dtype=float)
    Y_ = _interp_axis(xt, syt, sy_)
    dx = np.diff(X_)
    dy = np.diff(Y_)
    ln = np.hypot(dx, dy)
    # interval j is bounded by directions j-1 (in) and j+1 (out)
    d_in_x, d_in_y, l_in = dx[:-2], dy[:-2], ln[:-2]
    d_out_x, d_out_y, l_out = dx[2:], dy[2:], ln[2:]
    l_mid = ln[1:-1]
    dot = d_in_x * d_out_x + d_in_y * d_out_y
    fold = (
        (dot < -0.17 * l_in * l_out)
        & (l_in > 0.05) & (l_out > 0.05)
        & (l_mid < 0.7 * 0.5 * (l_in + l_out))
    )
    out = np.zeros(len(qt), dtype=bool)
    if not fold.any():
        return out
    j = np.clip(np.searchsorted(xt, qt, side="right") - 1, 0, len(xt) - 2)
    inner = (j >= 1) & (j <= len(xt) - 3)
    out[inner] = fold[j[inner] - 1]
    return out


def _layers_from_pos_z(
    wt: np.ndarray,
    wlayer: np.ndarray,
    pos_z_t, pos_z,
    layer_z: list[float],
    trav: np.ndarray,
    wx: np.ndarray | None = None,
    wy: np.ndarray | None = None,
    layer_start_xy: list[tuple[float, float]] | None = None,
    start_layer_index: int = 0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Per-window-sample layer indices from the pos_z staircase.

    Raw pos_z is ~1/3 junk spikes, but a rolling median recovers the true z
    staircase (layer heights + a constant MBL offset). The offset is estimated
    against the coarse arc-length layer (median residual -- robust because the
    coarse layer is right for the vast majority of samples), then each layer
    boundary is located as the offset-corrected median-z crossing of the
    midpoint between the two layers' z values, searched near the coarse
    transition, and finally snapped to the end of a nearby travel run (a
    layer change is always a travel, so no boundary can split an extrusion).
    The z crossing has ~0.3 s of median-filter slop, so among candidate
    travel runs we PREFER one whose exit position matches where the gcode
    says the next layer starts (`layer_start_xy`), and only fall back to the
    nearest run -- otherwise a mid-layer travel can steal the boundary.

    Returns (wlayer_refined, diag); (None, diag-with-reason) when pos_z can't
    be trusted, in which case the caller keeps the coarse arc-length layers.
    """
    n = len(wt)
    max_layer = max(0, len(layer_z) - 1)
    start_layer_index = int(np.clip(start_layer_index, 0, max_layer))
    diag: dict[str, Any] = {
        "available": False,
        "reason": "",
        "source": "arc_length",
        "start_layer_index": start_layer_index,
    }
    if (pos_z_t is None or pos_z is None
            or len(np.atleast_1d(pos_z)) < _Z_MED_WIN or len(layer_z) < 2):
        diag["reason"] = "no usable pos_z stream"
        return None, diag
    zs = np.asarray(layer_z, dtype=float)
    pitch = float(np.median(np.diff(zs)))
    if not pitch > 0.02:
        diag["reason"] = "layer pitch too small for pos_z banding"
        return None, diag

    zm = _rolling_median(np.asarray(pos_z, dtype=float), _Z_MED_WIN)
    z_s = np.interp(wt, np.asarray(pos_z_t, dtype=float), zm)
    resid = z_s - zs[np.clip(wlayer, 0, len(zs) - 1)]
    offset = float(np.median(resid))
    mad = float(np.median(np.abs(resid - offset)))
    diag.update({"offset": offset, "mad": mad, "pitch": pitch})
    if mad > _Z_MAD_MAX_FRAC * pitch:
        diag["reason"] = (
            f"pos_z too noisy for layer banding (median |resid| {mad:.3f} mm "
            f"vs layer pitch {pitch:.3f} mm)")
        return None, diag
    zc = z_s - offset

    counts = np.bincount(np.clip(wlayer, 0, len(zs) - 1), minlength=len(zs))
    bounds: list[int] = []
    binfo: list[dict[str, Any]] = []
    prev_b = 0
    # An attached capture may begin partway through the file. Boundaries below
    # its printer-reported starting layer have already happened and must not be
    # recreated at the beginning of the saved stream.
    for k in range(start_layer_index + 1, len(zs)):
        c_k = int(np.searchsorted(wlayer, k))          # coarse boundary
        W = int(np.clip(0.6 * max(1, min(counts[k - 1], counts[k])), 50, 20000))
        lo = max(prev_b + 1, c_k - W)
        hi = min(n - 1, c_k + W)
        method = "arc"
        b = min(max(c_k, prev_b + 1), n - 1)
        if hi > lo:
            mid_z = 0.5 * (zs[k - 1] + zs[k])
            above = zc[lo:hi + 1] >= mid_z
            edges = np.where(above[1:] & ~above[:-1])[0] + 1 + lo
            if edges.size:
                # the upward crossing nearest the coarse estimate
                b = int(edges[np.argmin(np.abs(edges - c_k))])
                method = "pos_z"
        if method == "pos_z":
            # snap to the END of a travel run near the z crossing: the layer
            # change is a travel, so the boundary must not split an
            # extrusion. Prefer the run whose exit lands where the gcode
            # says layer k starts; among equals take the one nearest the
            # z crossing.
            lo2 = max(0, b - _Z_TRAVEL_SNAP)
            hi2 = min(n - 1, b + _Z_TRAVEL_SNAP)
            tm = trav[lo2:hi2 + 1].astype(np.int8)
            e = np.diff(np.r_[0, tm, 0])
            r_starts = np.where(e == 1)[0] + lo2
            r_ends = np.where(e == -1)[0] + lo2      # exclusive end
            best_key = None
            best_b = None
            sxy = layer_start_xy[k] if layer_start_xy and k < len(layer_start_xy) else None
            for s0, e0 in zip(r_starts, r_ends):
                if e0 >= n:
                    continue
                match = 1
                if (sxy is not None and wx is not None
                        and np.isfinite(wx[e0]) and np.isfinite(wy[e0])
                        and np.hypot(wx[e0] - sxy[0], wy[e0] - sxy[1]) < 3.0):
                    match = 0                        # exits at the layer's start point
                key = (match, abs((s0 + e0) // 2 - b))
                if best_key is None or key < best_key:
                    best_key = key
                    best_b = int(e0)
            if best_b is not None:
                b = best_b
                method = "pos_z+travel" if best_key[0] else "pos_z+travel+start"
        b = min(max(b, prev_b + 1), n - 1)
        bounds.append(b)
        prev_b = b
        binfo.append({
            "into_layer": int(k),
            "t_s": float(wt[b] - wt[0]),
            "coarse_t_s": float(wt[min(c_k, n - 1)] - wt[0]),
            "shift_s": float(wt[b] - wt[min(c_k, n - 1)]),
            "method": method,
        })

    out = np.full(n, start_layer_index, dtype=int)
    for b in bounds:
        out[b:] += 1
    ds = np.linspace(0, n - 1, min(n, 1500)).astype(int)
    diag.update({
        "available": True,
        "source": "pos_z",
        "n_pos_z_boundaries": sum(1 for x in binfo if x["method"].startswith("pos_z")),
        "boundaries": binfo,
        "stair_t": [float(v) for v in (wt[ds] - wt[0])],
        "stair_z": [float(round(v, 4)) for v in zc[ds]],
        "layer_z": [float(v) for v in zs],
    })
    return out, diag


def _longest_run(mask: np.ndarray, fs: float) -> tuple[int, int] | None:
    if not mask.any():
        return None
    m = mask.copy()
    gap = max(1, int(_RUN_GAP_S * fs))
    edges = np.diff(np.r_[0, m.view(np.int8), 0])
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    for a, b in zip(ends[:-1], starts[1:]):
        if b - a <= gap:
            m[a:b] = True
    edges = np.diff(np.r_[0, m.view(np.int8), 0])
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    if not len(starts):
        return None
    j = int((ends - starts).argmax())
    return int(starts[j]), int(ends[j] - 1)


def _moving_avg(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or a.size < w:
        return a
    k = np.ones(w) / w
    return np.convolve(a, k, mode="same")


def _travel_threshold(parsed: ParsedGcode) -> float:
    """Speed (mm/s) above which a move is a travel, derived from the gcode's
    feedrates. Travels run faster than any extrusion; we set the cut 40% of the
    way from the fastest print speed to the fastest travel speed."""
    print_f = [m.feed_mm_s for m in parsed.moves if m.extruding and m.feed_mm_s > 0]
    travel_f = [
        m.feed_mm_s for m in parsed.moves
        if (not m.extruding) and m.feed_mm_s > 0
        and (abs(m.x1 - m.x0) > 0.1 or abs(m.y1 - m.y0) > 0.1)
    ]
    max_print = max(print_f) if print_f else 100.0
    max_travel = max(travel_f) if travel_f else max_print * 2.0
    if max_travel > max_print * 1.1:
        return max_print + 0.4 * (max_travel - max_print)
    return max_print * 1.5  # no clear separation -> permissive


def _nearest_seg(px, py, sx0, sy0, sx1, sy1):
    """Per point: nearest segment index + projection (cx, cy) + fraction along
    it. Chunked to bound memory."""
    n = len(px)
    k_out = np.zeros(n, dtype=int)
    cx_out = np.asarray(px, dtype=float).copy()
    cy_out = np.asarray(py, dtype=float).copy()
    fr_out = np.zeros(n)
    if not len(sx0):
        return k_out, cx_out, cy_out, fr_out
    dx = sx1 - sx0; dy = sy1 - sy0
    L2 = np.where((dx * dx + dy * dy) > 1e-12, dx * dx + dy * dy, 1.0)
    for i in range(0, n, _NEAREST_CHUNK):
        a = px[i:i + _NEAREST_CHUNK][:, None]
        b = py[i:i + _NEAREST_CHUNK][:, None]
        tt = np.clip(((a - sx0) * dx + (b - sy0) * dy) / L2, 0.0, 1.0)
        cx = sx0 + tt * dx; cy = sy0 + tt * dy
        k = ((a - cx) ** 2 + (b - cy) ** 2).argmin(axis=1)
        rows = np.arange(k.size)
        k_out[i:i + _NEAREST_CHUNK] = k
        cx_out[i:i + _NEAREST_CHUNK] = cx[rows, k]
        cy_out[i:i + _NEAREST_CHUNK] = cy[rows, k]
        fr_out[i:i + _NEAREST_CHUNK] = tt[rows, k]
    return k_out, cx_out, cy_out, fr_out


def _consistency_reject(snap_x, snap_y, snap_s, raw_x, raw_y, brk,
                        back_tol=1.0, fwd_tol=3.0):
    """Reject snaps that violate path continuity, falling back to raw position.

    Walking in time order, the arc-length position should advance by ~the XY
    distance the head moved since the last reading. If a snap's arc length jumps
    far ahead/behind that (it grabbed a parallel line), we keep the RAW measured
    point instead. `prev` still advances by the XY step on rejection, so a run of
    bad snaps doesn't derail the following good ones (no cascade). Layer is taken
    separately from the drift-tolerant arc-length, so it's unaffected."""
    n = len(snap_x)
    ox = snap_x.copy(); oy = snap_y.copy()
    prev = None
    for i in range(n):
        if prev is None or brk[i]:
            prev = snap_s[i]
            continue
        adv = float(np.hypot(raw_x[i] - raw_x[i - 1], raw_y[i] - raw_y[i - 1]))
        ds = snap_s[i] - prev
        if -back_tol <= ds <= adv + fwd_tol:
            prev = snap_s[i]
        else:
            ox[i] = raw_x[i]; oy[i] = raw_y[i]
            prev = prev + adv
    return ox, oy


def _empty(parsed: ParsedGcode, reason: str) -> "MappedRun":
    z = np.array([]); zi = np.array([], dtype=int); zb = np.array([], dtype=bool)
    return MappedRun(parsed=parsed, s_x=z, s_y=z, s_xr=z, s_yr=z, s_force=z,
                     s_layer=zi, s_feat=zi, s_brk=zb,
                     cross={"available": False, "reason": reason})


def map_run(
    force_t, force_y,
    pos_t, pos_x,
    pos_y_t, pos_y,
    pos_z_t, pos_z,          # rolling-median staircase -> layer boundaries
    parsed: ParsedGcode,
    start_layer_index: int = 0,
) -> MappedRun:
    ft = np.asarray(force_t, dtype=float)
    fy = np.asarray(force_y, dtype=float)
    n = int(min(len(ft), len(fy)))
    ft, fy = ft[:n], fy[:n]

    # REAL toolhead position at each force timestamp (same host clock).
    # The map places force by the streamed toolhead position. If pos_x/pos_y
    # never arrived (the metrics weren't streaming), there's nothing to place --
    # say so clearly rather than rendering an empty map.
    if (pos_x is None or np.asarray(pos_x).size == 0
            or pos_y is None or np.asarray(pos_y).size == 0):
        return _empty(
            parsed,
            "no position telemetry: pos_x/pos_y streamed 0 samples (force "
            f"streamed {n}). The map needs the toolhead position. Open the "
            "Diagnostics card during a print -- pos_x should read ~80 Hz; if "
            "it's 0, the pos_x/pos_y/pos_z metrics aren't enabled on the printer.",
        )

    X = _interp_axis(ft, pos_t, pos_x)
    Y = _interp_axis(ft, pos_y_t, pos_y)

    max_layer = max(0, len(parsed.layers) - 1)
    start_layer_index = int(np.clip(start_layer_index, 0, max_layer))
    ext = [
        i for i, m in enumerate(parsed.moves)
        if m.extruding and m.layer >= start_layer_index
    ]
    if not ext or n < 4:
        return _empty(parsed, "no extruding print moves")
    pm = parsed.moves[ext[0]: ext[-1] + 1]
    px0 = np.array([m.x0 for m in pm]); py0 = np.array([m.y0 for m in pm])
    px1 = np.array([m.x1 for m in pm]); py1 = np.array([m.y1 for m in pm])
    pm_layer = np.array([m.layer for m in pm], dtype=int)
    pm_feat = np.array([m.feature_idx for m in pm], dtype=int)
    seg_len = np.hypot(px1 - px0, py1 - py0)
    Sg = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = float(Sg[-1])
    if L <= 0:
        return _empty(parsed, "degenerate gcode path")

    exm = np.array([m.extruding for m in pm])
    xs = np.r_[px0[exm], px1[exm]]; ys = np.r_[py0[exm], py1[exm]]
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()

    span = float(ft[-1] - ft[0]) or 1.0
    fs = n / span
    mm = _FOOTPRINT_MARGIN_MM
    inb = (X >= bx0 - mm) & (X <= bx1 + mm) & (Y >= by0 - mm) & (Y <= by1 + mm)
    run = _longest_run(inb, fs)
    if run is None:
        return _empty(parsed, "no in-footprint print window")
    start, end = run

    # window slices (REAL positions + force in the print window)
    wx = X[start:end + 1]; wy = Y[start:end + 1]; wt = ft[start:end + 1]
    wf = fy[start:end + 1]

    # --- coarse LAYER from cumulative arc length (drift-tolerant: layer bands
    # are thousands of mm apart, so tens of mm of drift can't flip them) ---
    dS = np.hypot(np.diff(X), np.diff(Y))
    S = np.concatenate([[0.0], np.cumsum(dS)])
    stream_len = float(S[end] - S[start])
    scale = float(np.clip(L / stream_len, 0.8, 1.25)) if stream_len > 1e-6 else 1.0
    a = np.clip((S[start:end + 1] - S[start]) * scale, 0.0, L - 1e-9)
    mi = np.clip(np.searchsorted(Sg, a, side="right") - 1, 0, len(pm) - 1)
    wlayer = pm_layer[mi]

    # --- travel detection from REAL-position speed (drift-free) ---
    dt = np.diff(wt)
    dt = np.where(dt > 1e-6, dt, 1e-6)
    step = np.hypot(np.diff(wx), np.diff(wy))
    speed = np.empty(len(wx))
    speed[1:] = step / dt
    speed[0] = speed[1] if len(speed) > 1 else 0.0
    speed = _moving_avg(speed, 3)
    thresh = _travel_threshold(parsed)
    trav = speed >= thresh
    if trav.any():
        # Dilate the travel mask: the accel/decel ramps at a travel's ends dip
        # below the threshold, and they sit OUTSIDE the part -- dropping a few
        # neighbours of every fast sample removes those stubs so no line is
        # drawn across the part between extrusions.
        d = 2
        trav = np.convolve(trav.astype(np.int8), np.ones(2 * d + 1, np.int8), mode="same") > 0

    # --- travel detection (2): GCODE GEOMETRY (drift-free, catches the SLOW
    # travels the speed gate misses -- "avoid crossing perimeters" detours,
    # short hops, anything moving at print-ish speed). During real extrusion the
    # head sits ON an extrusion line (~0.01 mm away), so no travel line can be
    # closer; when a sample's NEAREST gcode move (over ALL moves in its coarse
    # arc-length layer) is a TRAVEL, the head is genuinely off the extrusion
    # path -> mask it. Candidates exclude zero-length moves (retract/z-hop). ---
    finite_w = np.isfinite(wx) & np.isfinite(wy)
    cand = seg_len > 1e-6
    trav_g = np.zeros(len(wx), dtype=bool)
    n_travel_gcode = 0
    for li in np.unique(wlayer):
        lm = np.where((pm_layer == li) & cand)[0]
        if not lm.size:
            continue
        sel = np.where((wlayer == li) & finite_w)[0]
        if not sel.size:
            continue
        k, _, _, _ = _nearest_seg(wx[sel], wy[sel],
                                  px0[lm], py0[lm], px1[lm], py1[lm])
        trav_g[sel] = ~exm[lm[k]]
    n_travel_gcode = int(trav_g.sum())
    trav = trav | trav_g

    # --- refine LAYER with the pos_z staircase (needs `trav` for the travel
    # snap, and each layer's gcode start point to pick the RIGHT travel).
    # Falls back to the coarse arc-length layers when pos_z is missing/too
    # noisy; `layer_diag` records which source won and why. ---
    layer_starts: list[tuple[float, float]] = []
    for lyr in parsed.layers:
        p0 = (float("nan"), float("nan"))
        for mv in parsed.moves[lyr.move_lo:lyr.move_hi]:
            if mv.extruding:
                p0 = (mv.x0, mv.y0)
                break
        layer_starts.append(p0)
    wlayer_z, layer_diag = _layers_from_pos_z(
        wt, wlayer, pos_z_t, pos_z, [lyr.z for lyr in parsed.layers], trav,
        wx, wy, layer_starts, start_layer_index=start_layer_index)
    if wlayer_z is not None:
        wlayer = wlayer_z

    # Drop samples whose REAL position was interpolated across a wide position
    # gap (untrustworthy -- straight chord across an unknown turn/travel),
    # near a timestamp pile-up seam (laterally wrong on diagonal moves), or
    # inside an interval that hides a zigzag turnaround (chord cuts the
    # corner between two infill lines).
    pos_gap = np.maximum(_interp_gap(wt, pos_t), _interp_gap(wt, pos_y_t))
    pos_pile = _pileup_mask(wt, pos_t) | _pileup_mask(wt, pos_y_t)
    pos_fold = _turn_fold_mask(wt, pos_t, pos_x, pos_y_t, pos_y)
    pos_ok = (pos_gap < _POS_GAP_MAX_S) & ~pos_pile & ~pos_fold

    is_print = (~trav) & pos_ok & np.isfinite(wf) & np.isfinite(wx) & np.isfinite(wy)

    # tare = static load before the print starts
    pre = fy[:start]
    if pre.size >= 200:
        tare = float(np.median(pre))
    elif fy.size:
        tare = float(np.percentile(fy, 10))
    else:
        tare = 0.0

    # raw measured positions of the kept (printing) samples
    s_xr = wx[is_print]; s_yr = wy[is_print]
    s_force = wf[is_print] - tare

    # Break the line only at real travels: samples were dropped between these
    # two kept ones (index gap > 1 -> the toolhead was moving fast), or the
    # jump is travel-sized. A packet-loss gap on a continuous extrusion leaves
    # the two bracketing samples ADJACENT in the stream (just a time gap) and
    # close in space, so it stays connected -- no false dashing.
    kept_idx = np.where(is_print)[0]
    s_brk = np.zeros(s_xr.size, dtype=bool)
    if s_xr.size:
        s_brk[0] = True
        jump = np.hypot(np.diff(s_xr), np.diff(s_yr))
        s_brk[1:] = (np.diff(kept_idx) > 1) | (jump > _BREAK_JUMP_MM)

    s_layer = wlayer[is_print]

    # --- snap onto the nearest gcode line in the layer, then SANITY-CHECK with
    # path continuity. The gcode is one continuous path the head only advances
    # along, so between two readings the arc length should advance by ~the XY
    # distance moved. A snap onto a PARALLEL line (close in XY but far ahead/
    # behind in arc length) violates that -- we reject it and fall back to the
    # raw measured position there. Degrades gracefully (small bend) instead of
    # rendering on the wrong line. ---
    snap_x = s_xr.copy(); snap_y = s_yr.copy()
    snap_s = np.zeros(s_xr.size)            # arc length of each snap (mm)
    s_feat = np.zeros(s_xr.size, dtype=int)
    for li in np.unique(s_layer) if s_layer.size else []:
        le = (pm_layer == li) & exm
        gj = np.where(le)[0]
        if not gj.size:
            continue
        sel = np.where(s_layer == li)[0]
        k, cx, cy, fr = _nearest_seg(s_xr[sel], s_yr[sel],
                                     px0[gj], py0[gj], px1[gj], py1[gj])
        gm = gj[k]                          # global move index per point
        snap_x[sel] = cx; snap_y[sel] = cy
        snap_s[sel] = Sg[gm] + fr * seg_len[gm]
        s_feat[sel] = pm_feat[gm]

    # --- break the line over force-stream HOLES that hide a turn. The force
    # stream arrives in ~3 ms bursts with ~30 ms holes; a hole spanning a
    # zigzag turnaround leaves its two bracketing samples adjacent in the
    # stream (no index gap, jump << _BREAK_JUMP_MM) but on DIFFERENT infill
    # lines -- drawn, that's a false connector cutting across the part. The
    # tell is the gcode: going around the turn advances the arc length far
    # more than the straight chord. Along-line holes (advance ≈ chord) stay
    # connected, so the line doesn't dash at every batch boundary. ---
    n_hole_breaks = 0
    kt = wt[is_print]
    if s_xr.size > 1:
        dt_k = np.diff(kt)
        chord = np.hypot(np.diff(s_xr), np.diff(s_yr))
        adv = np.diff(snap_s)
        # gate 1: arc advance disagrees with the chord (turn went around)
        mism = np.abs(adv - chord) > np.maximum(_HOLE_MISMATCH_MM, 0.35 * chord)
        # gate 2: the head should have covered ~dt × local speed of path
        # during the hole; a chord far shorter than that means it doubled
        # back (a turnaround the snap mis-attributed, so gate 1 missed it)
        v_loc = _rolling_median(chord / np.maximum(dt_k, 1e-6), 15)
        expected = dt_k * v_loc
        short = (expected - np.maximum(chord, np.abs(adv))) > np.maximum(1.0, 0.5 * expected)
        hole_brk = (dt_k > _HOLE_BREAK_S) & (mism | short) & ~s_brk[1:]
        n_hole_breaks = int(hole_brk.sum())
        s_brk[1:] |= hole_brk

    s_x, s_y = _consistency_reject(snap_x, snap_y, snap_s, s_xr, s_yr, s_brk)

    if s_force.size:
        force_lo, force_hi = (float(v) for v in np.percentile(s_force, [2, 98]))
        if force_hi <= force_lo:
            force_hi = force_lo + 1.0
    else:
        force_lo, force_hi = 0.0, 1.0

    feature_stats: dict[str, dict[str, float]] = {}
    if s_force.size:
        for fi in np.unique(s_feat):
            vals = s_force[s_feat == fi]
            if vals.size > _FEATURE_SAMPLE_CAP:
                vals = vals[:: vals.size // _FEATURE_SAMPLE_CAP]
            name = parsed.features[fi] if 0 <= fi < len(parsed.features) else "(none)"
            feature_stats[name] = {
                "n": int(vals.size),
                "mean": float(np.mean(vals)),
                "p50": float(np.median(vals)),
                "p90": float(np.percentile(vals, 90)),
            }

    kept_frac = float(is_print.mean()) if is_print.size else 0.0
    if abs(scale - 1.0) < 0.06 and kept_frac > 0.4:
        agreement = "good"
    elif abs(scale - 1.0) < 0.15:
        agreement = "warn"
    else:
        agreement = "poor"
    cross = {
        "available": True,
        "time_scale": scale,
        "kept_fraction": kept_frac,
        "agreement": agreement,
        "print_start_s": float(ft[start] - ft[0]),
        "print_end_s": float(ft[end] - ft[0]),
        "travel_speed_threshold": float(thresh),
        "n_travel_gcode": n_travel_gcode,
        "pos_gap_dropped": int((pos_gap >= _POS_GAP_MAX_S).sum()),
        "pos_pile_dropped": int((pos_pile & (pos_gap < _POS_GAP_MAX_S)).sum()),
        "pos_fold_dropped": int((pos_fold & ~pos_pile & (pos_gap < _POS_GAP_MAX_S)).sum()),
        "n_hole_breaks": n_hole_breaks,
        "layer_source": layer_diag.get("source", "arc_length"),
        "start_layer_index": start_layer_index,
        "n_samples": int(s_force.size),
    }

    return MappedRun(
        parsed=parsed, s_x=s_x, s_y=s_y, s_xr=s_xr, s_yr=s_yr, s_force=s_force,
        s_layer=s_layer, s_feat=s_feat, s_brk=s_brk,
        tare=tare, force_lo=force_lo, force_hi=force_hi,
        feature_stats=feature_stats, cross=cross, layer_diag=layer_diag,
    )


def mapped_summary(m: MappedRun, filename: str = "") -> dict[str, Any]:
    layers = []
    for lyr in m.parsed.layers:
        n_pts = int(np.count_nonzero(m.s_layer == lyr.index)) if m.s_layer.size else 0
        layers.append({"index": lyr.index, "z": lyr.z, "n_points": n_pts})
    return {
        "filename": filename,
        "n_layers": len(m.parsed.layers),
        "n_moves": len(m.parsed.moves),
        "n_extruding": int(m.parsed.n_extruding),
        "n_samples": int(m.s_force.size),
        "bbox": list(m.parsed.bbox),
        "tare": m.tare,
        "force_lo": m.force_lo,
        "force_hi": m.force_hi,
        "features": list(m.parsed.features),
        "layer_zs": layer_zs(m.parsed),
        "feature_stats": m.feature_stats,
        "cross_check": m.cross,
        "layer_diag": m.layer_diag,
        "layers": layers,
    }


def layer_detail(m: MappedRun, layer_idx: int, point_cap: int = _LAYER_POINT_CAP) -> dict[str, Any]:
    """Ordered real-position samples for one layer + the faint backdrop.

    `pts` are in print order at the true toolhead position; the frontend
    connects consecutive samples into coloured sub-segments (breaking on a
    travel-sized jump). `feat` lets it recolour by feature."""
    z = m.parsed.layers[layer_idx].z if 0 <= layer_idx < len(m.parsed.layers) else None
    sel = np.where(m.s_layer == layer_idx)[0] if m.s_layer.size else np.array([], dtype=int)
    n_total = int(sel.size)
    if sel.size > point_cap:
        sel = sel[np.linspace(0, sel.size - 1, point_cap).astype(int)]
    return {
        "index": layer_idx,
        "z": z,
        "polylines": layer_polylines(m.parsed, layer_idx),
        "pts": {
            "x": [float(v) for v in m.s_x[sel]],     # snapped to gcode (default)
            "y": [float(v) for v in m.s_y[sel]],
            "xr": [float(v) for v in m.s_xr[sel]],    # raw measured position
            "yr": [float(v) for v in m.s_yr[sel]],
            "f": [float(v) for v in m.s_force[sel]],
            "feat": [int(v) for v in m.s_feat[sel]],
            "brk": [bool(v) for v in m.s_brk[sel]],
        },
        "force_lo": m.force_lo,
        "force_hi": m.force_hi,
        "n_points": n_total,
    }
