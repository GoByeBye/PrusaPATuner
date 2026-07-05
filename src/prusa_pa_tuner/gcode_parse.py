"""Parse ASCII G-code into a geometry + feature/layer model + a time-model.

Used by the Live Map module to:
  (a) draw the 2D per-layer backdrop (the planned toolhead path),
  (b) label streamed nozzle force by the feature / speed being printed,
  (c) build a trapezoidal time-model that the runner cross-checks against the
      streamed toolhead position (see livemap_map.cross_check).

We parse PrusaSlicer-flavoured ASCII gcode:
  * G0/G1 motion with G90/G91 (XYZ abs/rel), M82/M83 (E abs/rel), G92 origin
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
        if letter in ("X", "Y", "Z", "E", "F"):
            try:
                out[letter] = float(tok[1:])
            except ValueError:
                continue
    return out


def parse_gcode(text: str, *, default_accel: float = 2000.0) -> ParsedGcode:
    """Parse ASCII gcode into a ParsedGcode model. Never raises on bad lines."""
    pg = ParsedGcode()
    feature_index: dict[str, int] = {}

    abs_xyz = True
    abs_e = True
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
        if code not in ("G0", "G1", "G00", "G01"):
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
        if xy > 1e-9:
            length = xy
        elif abs(dz) > 1e-9:
            length = abs(dz)
        else:
            length = abs(de)
        extruding = de > 1e-6 and xy > 1e-9

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

        mv = Move(
            x0=x, y0=y, z0=z, x1=x1, y1=y1, z1=z1,
            e=de, feed_mm_s=feed_mm_s, feature_idx=cur_feature,
            layer=cur_layer, extruding=extruding, length_mm=length,
            t_start=t_cursor, t_end=t_cursor + dur,
        )
        pg.moves.append(mv)
        if cur_layer >= 0:
            pg.layers[cur_layer].move_hi = len(pg.moves)
        t_cursor += dur

        # bbox + extruding count from real-print moves only (layer >= 0), so
        # the purge line / start gcode doesn't blow the footprint out to the
        # whole bed.
        if extruding and cur_layer >= 0:
            pg.n_extruding += 1
            xmin = min(xmin, x, x1)
            xmax = max(xmax, x, x1)
            ymin = min(ymin, y, y1)
            ymax = max(ymax, y, y1)

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
