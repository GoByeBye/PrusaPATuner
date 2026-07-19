"""Parse ASCII G-code into a geometry + feature/layer model + a time-model.

Used by the Live Map module to:
  (a) draw the 2D per-layer backdrop (the planned toolhead path),
  (b) label streamed nozzle force by the feature / speed being printed,
  (c) build a trapezoidal time-model that the runner cross-checks against the
      streamed toolhead position (see livemap_map.cross_check).

We parse PrusaSlicer-flavoured ASCII gcode:
  * G0/G1 motion plus PrusaSlicer G2/G3 XY arcs with relative I/J centres;
    G90/G91 (XYZ abs/rel), M82/M83 (E abs/rel), G92 origin
  * `;TYPE:<feature>` feature comments (perimeter / infill / ...)
  * `;LAYER_CHANGE` + `;Z:<height>` layer markers (with a z-binning fallback
    for gcode that lacks them)
  * M204 P/T/S accel for the time-model

Binary .bgcode is intentionally NOT supported -- the module requires plain
ASCII so feature/layer/speed are parseable (see project_live_map_module memory).
The parser is deliberately tolerant: an unrecognised line is skipped, never
raised, so one odd token can't kill a whole upload.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# `;TYPE:External perimeter`  (PrusaSlicer) / `;TYPE:WALL-OUTER` (Cura).
_TYPE_RE = re.compile(r"^\s*;\s*TYPE:\s*(.+?)\s*$", re.IGNORECASE)
# `;Z:0.6` -- absolute layer height emitted right after `;LAYER_CHANGE`.
_Z_RE = re.compile(r"^\s*;\s*Z:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# Two extruding moves whose start/end z differ by less than this are the
# same layer (covers float noise + z-hop residue).
_Z_EPS = 0.02

# PrusaSlicer's ``arc_fitting = emit_center`` writes G2/G3 moves in the XY
# plane with relative I/J centre offsets.  Live Map needs those curves in its
# geometry model, but the rest of the mapper deliberately works with straight
# segments.  Tessellate finely enough to preserve the printed path without
# turning every arc into hundreds of points.  The angular cap matters for
# very small-radius arcs whose total length is below the linear cap.
_ARC_SEGMENT_MM = 0.5
_ARC_SEGMENT_RAD = math.radians(10.0)
_ARC_RADIUS_MISMATCH_MM = 0.05


@dataclass(slots=True)
class Move:
    """One G0/G1 segment with absolute start/end coords + model timing."""

    x0: float
    y0: float
    z0: float
    x1: float
    y1: float
    z1: float
    e: float          # extrusion delta (mm of filament); <=0 for travel/retract
    feed_mm_s: float  # commanded trajectory feedrate (F / 60)
    feature_idx: int  # index into ParsedGcode.features (-1 = none)
    layer: int        # layer index this move belongs to (-1 before first layer)
    extruding: bool
    length_mm: float  # dominant move length (xy, else z, else |e|)
    t_start: float    # model print-time at move start (s)
    t_end: float      # model print-time at move end (s)


@dataclass(slots=True)
class Layer:
    index: int
    z: float
    move_lo: int  # first move index in this layer (inclusive)
    move_hi: int  # last move index + 1 (exclusive)


@dataclass(slots=True)
class ParsedGcode:
    moves: list[Move] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # xmin,xmax,ymin,ymax
    total_time_s: float = 0.0
    n_extruding: int = 0


def _move_time_s(dist_mm: float, v_mm_s: float, accel: float) -> float:
    """Trapezoidal (stop-start) duration for one move.

    No junction/lookahead modelling -- each move is assumed to start and end
    at rest. This OVERESTIMATES wall-clock time, which is fine: the time-model
    is only a cross-check against the streamed position (the streamed position
    is what actually places the data). The divergence the cross-check surfaces
    is partly this very simplification, which is exactly what we want to flag.
    """
    if dist_mm <= 1e-9 or v_mm_s <= 1e-9:
        return 0.0
    a = accel if accel > 1e-6 else 2000.0
    d_acc = v_mm_s * v_mm_s / (2.0 * a)  # distance to reach v from rest
    if 2.0 * d_acc <= dist_mm:
        # accelerate, cruise, decelerate
        return 2.0 * (v_mm_s / a) + (dist_mm - 2.0 * d_acc) / v_mm_s
    # triangular profile: never reaches v
    v_peak = math.sqrt(a * dist_mm)
    return 2.0 * v_peak / a


def _axes(tokens: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for tok in tokens[1:]:
        if not tok:
            continue
        letter = tok[0].upper()
        if letter in ("X", "Y", "Z", "E", "F", "I", "J", "R"):
            try:
                out[letter] = float(tok[1:])
            except ValueError:
                continue
    return out


def _xy_arc_points(
    x0: float,
    y0: float,
    z0: float,
    x1: float,
    y1: float,
    z1: float,
    axes: dict[str, float],
    *,
    clockwise: bool,
) -> tuple[list[tuple[float, float, float]], float] | None:
    """Tessellate one PrusaSlicer-style XY arc.

    Buddy/Marlin and PrusaSlicer use incremental I/J centre offsets for the
    emitted G17 arcs, independent of G90/G91 endpoint mode.  ``None`` means
    the command is outside that supported shape (for example an R arc or an
    inconsistent radius); the caller then keeps a single endpoint chord so
    machine position still advances instead of poisoning every later move.

    The returned length is the true circular XY path length.  Callers use it
    once for the command-level time model, then divide that duration across
    the straight display/mapping segments.  Treating every tessellated chord
    as a separately stopping command would badly inflate estimated time.
    """
    if "I" not in axes and "J" not in axes:
        return None

    cx = x0 + axes.get("I", 0.0)
    cy = y0 + axes.get("J", 0.0)
    r0 = math.hypot(x0 - cx, y0 - cy)
    r1 = math.hypot(x1 - cx, y1 - cy)
    if r0 <= 1e-9:
        return None
    if abs(r0 - r1) > max(_ARC_RADIUS_MISMATCH_MM, 0.01 * max(r0, r1)):
        return None

    a0 = math.atan2(y0 - cy, x0 - cx)
    a1 = math.atan2(y1 - cy, x1 - cx)
    if clockwise:
        sweep = -((a0 - a1) % (2.0 * math.pi))
    else:
        sweep = (a1 - a0) % (2.0 * math.pi)
    # Identical endpoints with a valid non-zero radius denote a full circle.
    if abs(sweep) <= 1e-12 and math.hypot(x1 - x0, y1 - y0) <= 1e-9:
        sweep = -2.0 * math.pi if clockwise else 2.0 * math.pi
    if abs(sweep) <= 1e-12:
        return None

    radius = 0.5 * (r0 + r1)
    arc_len = radius * abs(sweep)
    n = max(
        1,
        math.ceil(arc_len / _ARC_SEGMENT_MM),
        math.ceil(abs(sweep) / _ARC_SEGMENT_RAD),
    )
    points: list[tuple[float, float, float]] = []
    for j in range(1, n + 1):
        f = j / n
        if j == n:
            # Preserve the exact commanded endpoint; slicer coordinates are
            # rounded, so it can differ from the reconstructed circle by a few
            # microns even though the command is valid.
            points.append((x1, y1, z1))
            continue
        a = a0 + sweep * f
        points.append(
            (
                cx + radius * math.cos(a),
                cy + radius * math.sin(a),
                z0 + (z1 - z0) * f,
            )
        )
    return points, arc_len


def parse_gcode(text: str, *, default_accel: float = 2000.0) -> ParsedGcode:
    """Parse ASCII gcode into a ParsedGcode model. Never raises on bad lines."""
    pg = ParsedGcode()
    feature_index: dict[str, int] = {}

    abs_xyz = True
    abs_e = True
    plane = "G17"
    arc_centers_relative = True
    x = y = z = 0.0
    last_e = 0.0
    feed_mm_min = 1200.0
    cur_feature = -1

    accel_print = default_accel
    accel_travel = default_accel

    has_layer_markers = ";LAYER_CHANGE" in text.upper()
    expect_new_layer = not has_layer_markers  # binning mode opens layer 0 lazily
    pending_z: float | None = None
    cur_layer = -1
    cur_layer_z: float | None = None

    xmin = ymin = math.inf
    xmax = ymax = -math.inf

    def _feature_idx(name: str) -> int:
        idx = feature_index.get(name)
        if idx is None:
            idx = len(pg.features)
            feature_index[name] = idx
            pg.features.append(name)
        return idx

    def _open_layer(zval: float) -> None:
        nonlocal cur_layer, cur_layer_z
        cur_layer = len(pg.layers)
        cur_layer_z = zval
        pg.layers.append(Layer(index=cur_layer, z=zval, move_lo=len(pg.moves), move_hi=len(pg.moves)))

    t_cursor = 0.0

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line[0] == ";":
            # comment line -- only the feature/layer markers matter
            m = _TYPE_RE.match(line)
            if m:
                cur_feature = _feature_idx(m.group(1))
                continue
            if has_layer_markers and line.upper().startswith(";LAYER_CHANGE"):
                expect_new_layer = True
                pending_z = None
                continue
            mz = _Z_RE.match(line)
            if mz and expect_new_layer:
                try:
                    pending_z = float(mz.group(1))
                except ValueError:
                    pending_z = None
            continue

        # strip any inline comment from the command
        semi = line.find(";")
        if semi >= 0:
            line = line[:semi].strip()
        if not line:
            continue
        tokens = line.split()
        code = tokens[0].upper()

        if code in ("G17", "G18", "G19"):
            plane = code
            continue
        if code == "G90.1":
            arc_centers_relative = False
            continue
        if code == "G91.1":
            arc_centers_relative = True
            continue
        if code in ("G90",):
            abs_xyz = True
            continue
        if code in ("G91",):
            abs_xyz = False
            continue
        if code in ("M82",):
            abs_e = True
            continue
        if code in ("M83",):
            abs_e = False
            continue
        if code == "G92":
            a = _axes(tokens)
            if "X" in a:
                x = a["X"]
            if "Y" in a:
                y = a["Y"]
            if "Z" in a:
                z = a["Z"]
            if "E" in a:
                last_e = a["E"]
            continue
        if code == "M204":
            a = _axes_mp(tokens)
            if "P" in a:
                accel_print = a["P"]
            if "S" in a:
                accel_print = a["S"]
                if "T" not in a:
                    accel_travel = a["S"]
            if "T" in a:
                accel_travel = a["T"]
            continue
        if code not in ("G0", "G1", "G00", "G01", "G2", "G3", "G02", "G03"):
            continue

        a = _axes(tokens)
        if "F" in a:
            feed_mm_min = a["F"]
        if abs_xyz:
            x1 = a.get("X", x)
            y1 = a.get("Y", y)
            z1 = a.get("Z", z)
        else:
            x1 = x + a.get("X", 0.0)
            y1 = y + a.get("Y", 0.0)
            z1 = z + a.get("Z", 0.0)

        if "E" in a:
            if abs_e:
                de = a["E"] - last_e
                last_e = a["E"]
            else:
                de = a["E"]
                last_e += a["E"]
        else:
            de = 0.0

        dx = x1 - x
        dy = y1 - y
        dz = z1 - z
        xy = math.hypot(dx, dy)
        is_arc = code in ("G2", "G3", "G02", "G03")
        arc = (
            _xy_arc_points(
                x, y, z, x1, y1, z1, a,
                clockwise=code in ("G2", "G02"),
            )
            if is_arc and plane == "G17" and arc_centers_relative else None
        )
        if arc is not None:
            points, path_xy = arc
        else:
            # Unsupported/malformed arcs degrade to their endpoint chord.  It
            # is more important to advance XYZ/E state than to let one command
            # shift every subsequent segment, which was the old failure mode.
            points = [(x1, y1, z1)]
            path_xy = xy

        if path_xy > 1e-9:
            length = path_xy
        elif abs(dz) > 1e-9:
            length = abs(dz)
        else:
            length = abs(de)
        extruding = de > 1e-6 and path_xy > 1e-9

        # --- layer assignment ---
        # Marker mode: a `;LAYER_CHANGE` armed `expect_new_layer`; open the
        # layer on the first extruding move after it (z from `;Z:` or the
        # move). Extruding moves BEFORE the first `;LAYER_CHANGE` (the custom
        # start gcode -- purge line, intro, MBL) stay at layer -1 and are
        # excluded from the map: the user only wants what's actually printed.
        # Binning mode (no markers): open layer 0 lazily, then a new layer
        # whenever an extruding move's z rises past the current one.
        if extruding:
            if expect_new_layer or (not has_layer_markers and cur_layer < 0):
                _open_layer(pending_z if pending_z is not None else z1)
                expect_new_layer = False
                pending_z = None
            elif (not has_layer_markers and cur_layer_z is not None
                  and z1 > cur_layer_z + _Z_EPS):
                _open_layer(z1)

        feed_mm_s = feed_mm_min / 60.0
        accel = accel_print if extruding else accel_travel
        dur = _move_time_s(length, feed_mm_s, accel)

        # Keep one command's stop/start time as one command even though arcs
        # are represented by several linear Move objects for geometry.  Arc
        # segments have equal angular/path spans, so equal time slices are the
        # appropriate command-level approximation.
        n_parts = len(points)
        px, py, pz = x, y, z
        for j, (qx, qy, qz) in enumerate(points, 1):
            t0 = t_cursor + dur * ((j - 1) / n_parts)
            t1 = t_cursor + dur * (j / n_parts)
            mv = Move(
                x0=px, y0=py, z0=pz, x1=qx, y1=qy, z1=qz,
                e=de / n_parts, feed_mm_s=feed_mm_s, feature_idx=cur_feature,
                layer=cur_layer, extruding=extruding,
                length_mm=length / n_parts,
                t_start=t0, t_end=t1,
            )
            pg.moves.append(mv)

            # bbox from real-print extrusion segments.  Arc tessellation adds
            # intermediate extrema that an endpoint-only chord would miss.
            if extruding and cur_layer >= 0:
                xmin = min(xmin, px, qx)
                xmax = max(xmax, px, qx)
                ymin = min(ymin, py, qy)
                ymax = max(ymax, py, qy)
            px, py, pz = qx, qy, qz

        if cur_layer >= 0:
            pg.layers[cur_layer].move_hi = len(pg.moves)
        t_cursor += dur

        # Count source extrusion commands, not tessellated display segments.
        if extruding and cur_layer >= 0:
            pg.n_extruding += 1

        x, y, z = x1, y1, z1

    pg.total_time_s = t_cursor
    if math.isfinite(xmin):
        pg.bbox = (xmin, xmax, ymin, ymax)
    elif pg.moves:
        # no extruding moves -- fall back to the travel envelope
        xs = [m.x1 for m in pg.moves]
        ys = [m.y1 for m in pg.moves]
        pg.bbox = (min(xs), max(xs), min(ys), max(ys))
    return pg


def _axes_mp(tokens: list[str]) -> dict[str, float]:
    """Like _axes but for M-code params (P/T/S/R) -- accel codes."""
    out: dict[str, float] = {}
    for tok in tokens[1:]:
        if not tok:
            continue
        letter = tok[0].upper()
        if letter in ("P", "T", "S", "R"):
            try:
                out[letter] = float(tok[1:])
            except ValueError:
                continue
    return out


def layer_polylines(pg: ParsedGcode, layer_idx: int) -> list[dict]:
    """Return the extruding path of one layer as feature-grouped polylines.

    Each polyline is a run of contiguous extruding moves sharing a feature;
    travels / feature changes / coordinate jumps break the run. Shape:
        [{"feature": <str>, "x": [...], "y": [...]}, ...]
    Used as the faint backdrop in the 2D view and (recoloured) as the
    feature overlay.
    """
    if layer_idx < 0 or layer_idx >= len(pg.layers):
        return []
    lyr = pg.layers[layer_idx]
    out: list[dict] = []
    cur: dict | None = None
    last_end: tuple[float, float] | None = None
    for i in range(lyr.move_lo, lyr.move_hi):
        m = pg.moves[i]
        if not m.extruding:
            cur = None
            last_end = None
            continue
        feat = pg.features[m.feature_idx] if 0 <= m.feature_idx < len(pg.features) else ""
        contiguous = last_end is not None and abs(m.x0 - last_end[0]) < 1e-6 and abs(m.y0 - last_end[1]) < 1e-6
        if cur is None or cur["feature"] != feat or not contiguous:
            cur = {"feature": feat, "x": [m.x0, m.x1], "y": [m.y0, m.y1]}
            out.append(cur)
        else:
            cur["x"].append(m.x1)
            cur["y"].append(m.y1)
        last_end = (m.x1, m.y1)
    return out


def layer_zs(pg: ParsedGcode) -> list[float]:
    return [lyr.z for lyr in pg.layers]
