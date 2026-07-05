"""Map a captured Live Map run onto its parsed gcode.

Each loadcell reading is placed at the **real toolhead position at its own
timestamp** -- pos_x/pos_y are streamed on the same recv_monotonic host clock
as the force, so interpolating them at the force time gives the true position
with no accumulation and no drift. (An earlier version re-derived position from
a single global arc-length scale; the streamed-vs-gcode distance ratio isn't
constant, so the error grew across the print and smeared the colours -- exactly
what we're avoiding here.)

Per sample we need: position (real, above), layer, whether it's a travel, and
which feature it belongs to. None of those may use pos_z (on this hardware it
carries MBL/raw-encoder noise spanning ±168 mm). So:

  * layer  -- from cumulative arc length (streamed XY distance ≈ gcode path
              distance). Drift-tolerant: layer bands are thousands of mm apart,
              so even tens of mm of arc-length error can't flip the layer.
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
    pos_z_t, pos_z,          # accepted for signature compat; NOT used (MBL noise)
    parsed: ParsedGcode,
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

    ext = [i for i, m in enumerate(parsed.moves) if m.extruding and m.layer >= 0]
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

    # Drop samples whose REAL position was interpolated across a wide position
    # gap (untrustworthy -- straight chord across an unknown turn/travel).
    pos_gap = np.maximum(_interp_gap(wt, pos_t), _interp_gap(wt, pos_y_t))
    pos_ok = pos_gap < _POS_GAP_MAX_S

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
        "pos_gap_dropped": int((~pos_ok).sum()),
        "n_samples": int(s_force.size),
    }

    return MappedRun(
        parsed=parsed, s_x=s_x, s_y=s_y, s_xr=s_xr, s_yr=s_yr, s_force=s_force,
        s_layer=s_layer, s_feat=s_feat, s_brk=s_brk,
        tare=tare, force_lo=force_lo, force_hi=force_hi,
        feature_stats=feature_stats, cross=cross,
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
