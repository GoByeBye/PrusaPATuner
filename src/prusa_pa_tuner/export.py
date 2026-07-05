"""Export a captured run's loadcell trace to an .xlsx workbook.

The web UI's "Export XLSX" buttons (the saved-run dropdown and the
results-card button) call the matching endpoints in `app.py`, which
delegate here. We keep it deliberately minimal -- two time columns plus
the nozzle force -- because that's all that's needed to inspect a run in
a spreadsheet:

  * timestamp_local -- wall-clock local time (naive datetime). Empty
    when the run predates the monotonic->wall-clock anchor (legacy npz).
  * t_rel_s         -- seconds since the first sample. Always present,
    full float precision, unaffected by Excel's date rounding -- this is
    the reliable time axis for charting.
  * force_raw       -- the loadcell value as the printer reports it. This
    is an UNCALIBRATED, arbitrary-unit reading (no grams conversion), so
    only relative changes are meaningful.

`npz2xlsx.py` at the repo root is the heavier, multi-stream offline
converter (pandas, one sheet per stream); this module is the lean
in-process path the running server uses, so it depends only on openpyxl
and writes in write_only mode -- a 10-minute sweep is ~100 k samples and
materialising every cell object would be wasteful.
"""
from __future__ import annotations

import datetime as _dt
import io
import os
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell

# Sub-second precision matters: the loadcell streams at ~100 Hz, so a
# whole-second format would collapse ten-ish rows onto the same displayed
# timestamp. The stored value carries the fraction either way; this just
# makes Excel show it.
_TS_FORMAT = "yyyy-mm-dd hh:mm:ss.000"
_XLSX_MEDIA_TYPE = (
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)


def build_force_xlsx(
    force_t,
    force_y,
    *,
    mono_anchor_mono: float | None = None,
    mono_anchor_unix: float | None = None,
    sheet_name: str = "loadcell",
) -> bytes:
    """Return xlsx bytes with (timestamp_local, t_rel_s, force_g) columns.

    `force_t` are monotonic-clock seconds; `force_y` the loadcell values.
    The two streams are zipped (truncated to the shorter, defensively)
    and sorted by time -- the in-memory current-run lists are in append
    order, which the UDP reordering logic can leave non-monotonic.

    When both anchor scalars are supplied, each monotonic time is mapped
    to a wall-clock unix timestamp via
    `t - mono_anchor_mono + mono_anchor_unix` and rendered as a local-time
    datetime; otherwise the timestamp column is left blank.
    """
    t = np.asarray(force_t, dtype=float)
    y = np.asarray(force_y, dtype=float)
    n = int(min(len(t), len(y)))
    t = t[:n]
    y = y[:n]
    if n > 1 and not bool(np.all(np.diff(t) >= 0)):
        order = np.argsort(t, kind="stable")
        t = t[order]
        y = y[order]

    have_anchor = mono_anchor_mono is not None and mono_anchor_unix is not None
    t0 = float(t[0]) if n else 0.0

    wb = Workbook(write_only=True)
    ws = wb.create_sheet(sheet_name)
    ws.append(["timestamp_local", "t_rel_s", "force_raw"])
    for i in range(n):
        tm = float(t[i])
        rel = tm - t0
        if have_anchor:
            unix = tm - float(mono_anchor_mono) + float(mono_anchor_unix)
            ts_cell = WriteOnlyCell(ws, value=_dt.datetime.fromtimestamp(unix))
            ts_cell.number_format = _TS_FORMAT
            ws.append([ts_cell, rel, float(y[i])])
        else:
            ws.append([None, rel, float(y[i])])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def saved_run_to_xlsx(path: str | os.PathLike) -> tuple[bytes, str]:
    """Load a `runs/run_*.npz` and return (xlsx_bytes, suggested_filename).

    Raises ValueError when the run carries no loadcell samples -- an
    empty workbook is useless and the endpoint should report that
    instead of serving a header-only file.
    """
    p = Path(path)
    with np.load(p, allow_pickle=True) as d:
        force_t = d["force_t"] if "force_t" in d else np.array([])
        force_y = d["force_y"] if "force_y" in d else np.array([])
        anchor_mono = (
            float(d["mono_anchor_mono"][0])
            if "mono_anchor_mono" in d and len(d["mono_anchor_mono"])
            else None
        )
        anchor_unix = (
            float(d["mono_anchor_unix"][0])
            if "mono_anchor_unix" in d and len(d["mono_anchor_unix"])
            else None
        )
        # Eager-copy out of the lazy NpzFile before the handle closes.
        force_t = np.asarray(force_t, dtype=float)
        force_y = np.asarray(force_y, dtype=float)

    if force_t.size == 0:
        raise ValueError("run contains no loadcell samples")

    data = build_force_xlsx(
        force_t,
        force_y,
        mono_anchor_mono=anchor_mono,
        mono_anchor_unix=anchor_unix,
    )
    return data, p.stem + ".xlsx"
