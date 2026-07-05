// Plain JS — no build step. Talks to the FastAPI backend.

const FIELDS = [
  "printer_host", "printer_user", "printer_password", "printer_api_key", "udp_port",
  "filament_label", "nozzle_temp", "preheat_temp", "nozzle_diameter", "filament_diameter",
  "slow_flow_mm3_s", "slow_volume_mm3", "fast_flow_mm3_s", "fast_volume_mm3",
  "cycles_per_K", "accel_mm_s2",
  "k_min", "k_max", "k_step",
  "purge_x", "purge_y", "purge_z",
  "coupled_dx_mm", "coupled_dy_mm", "coupled_dz_mm",
  "first_slow_leg_factor",
  // Max Flow test
  "flow_min_mm3_s", "flow_max_mm3_s", "flow_step_mm3_s",
  "flow_dwell_s", "flow_settle_frac", "flow_warmup_s", "flow_tare_dwell_s",
  // Touch Probe
  "probe_axis", "probe_dir", "probe_start_x", "probe_start_y", "probe_z",
  "probe_creep_mm", "probe_slow_feed_mm_min", "probe_travel_feed_mm_min",
  "probe_n_touches", "probe_backoff_mm", "probe_settle_ms", "probe_temp",
];

function $(id) { return document.getElementById(id); }

function readForm() {
  const cfg = {};
  for (const f of FIELDS) {
    const el = $(f);
    if (!el) continue;
    const v = el.value;
    if (el.type === "number") cfg[f] = parseFloat(v);
    else cfg[f] = v;
  }
  return cfg;
}

function writeForm(cfg) {
  for (const f of FIELDS) {
    if (cfg[f] === undefined) continue;
    const el = $(f);
    if (el) el.value = cfg[f];
  }
}

async function loadConfig() {
  const r = await fetch("/api/config");
  if (r.ok) writeForm(await r.json());
}

async function saveConfig() {
  const cfg = readForm();
  const r = await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  if (!r.ok) {
    alert("Save failed: " + (await r.text()));
    return;
  }
  flash($("btn_save"), "Saved");
}

async function previewGcode() {
  await saveConfig();
  window.open("/api/preview", "_blank");
}

async function startRun() {
  await saveConfig();
  const r = await fetch("/api/run", { method: "POST" });
  if (!r.ok) alert("Start failed: " + (await r.text()));
}

async function cancelRun() {
  await fetch("/api/cancel", { method: "POST" });
}

// Track the previous packet count + timestamp so we can compute a true
// packets/sec rate (the server only exposes the running total).
const diagState = { lastPkts: null, lastT: null };

async function refreshDiagnostics() {
  let data;
  try {
    const r = await fetch("/api/diagnostics?window_s=5");
    if (!r.ok) return;
    data = await r.json();
  } catch (_) { return; }
  const s = data.stats || {};
  const rates = data.rates_hz || {};
  const now = performance.now() / 1000;
  let pktRate = 0;
  if (diagState.lastPkts !== null && diagState.lastT !== null) {
    const dp = (s.packets || 0) - diagState.lastPkts;
    const dt = now - diagState.lastT;
    if (dt > 0.1) pktRate = dp / dt;
  }
  diagState.lastPkts = s.packets || 0;
  diagState.lastT = now;

  $("diag_pkt_rate").textContent = pktRate ? pktRate.toFixed(1) : "—";
  $("diag_pkts").textContent = (s.packets ?? "—");
  $("diag_dropped").textContent = (s.dropped_backpressure ?? "—");
  $("diag_malformed").textContent = (s.malformed_lines ?? "—");
  $("diag_n_metrics").textContent = (s.metrics_seen ?? "—");

  // Sort metrics by rate descending; ties broken by total samples.
  const names = Object.keys(rates);
  // The /api/metrics_seen endpoint also surfaces names without rate; pull
  // them too so the table shows "0 Hz" rows for metrics that ARE arriving
  // but were emitted fewer than two times in the window.
  try {
    const r2 = await fetch("/api/metrics_seen");
    if (r2.ok) {
      const seen = await r2.json();
      for (const n of Object.keys(seen.names || {})) {
        if (!(n in rates)) rates[n] = 0;
      }
    }
  } catch (_) {}

  const tbody = $("diag_rates_body");
  if (!tbody) return;
  const all = Object.keys(rates).map((name) => [name, rates[name]]);
  all.sort((a, b) => (b[1] - a[1]) || a[0].localeCompare(b[0]));
  tbody.innerHTML = all
    .map(([name, hz]) => {
      const color = hz > 50 ? "#2ea043" : hz > 5 ? "#d29922" : "#7d8590";
      return `<tr>
        <td style="padding:3px 8px;color:#e6edf3;">${name}</td>
        <td style="padding:3px 8px;text-align:right;color:${color};">${hz.toFixed(1)}</td>
        <td style="padding:3px 8px;text-align:right;color:#7d8590;">—</td>
      </tr>`;
    })
    .join("");
}

let diagTimer = null;
function startDiagnosticsPoll() {
  if (diagTimer !== null) return;
  refreshDiagnostics();
  diagTimer = setInterval(refreshDiagnostics, 1000);
}

function flash(btn, text) {
  const old = btn.textContent;
  btn.textContent = text;
  setTimeout(() => (btn.textContent = old), 1200);
}

// ---- live plot ----
const live = {
  // Loadcell trace (left y-axis).
  t: [],
  y: [],
  // pos_x DIRECTION trace (right y-axis): ±1 latched on the sign of
  // (pos_x - prev_pos_x). +1 = X currently moving toward (purge_x + dx),
  // -1 = moving back toward purge_x. The user uses this as a clean
  // square-wave phase reference -- comparing the rising/falling edges of
  // this square wave against loadcell peaks tells them the PA-induced
  // phase shift at a glance. Plotting raw pos_x instead would show ramps
  // and is harder to read off cycle-by-cycle.
  posT: [],
  posY: [],
  // Time-windowed buffer (not sample-count-windowed). Both traces drop
  // samples older than (latest_t - windowSeconds). Without this, the
  // slower pos_x stream covers far more time than the loadcell stream
  // at the same sample budget and the x-axis stretches unevenly.
  windowSeconds: 20.0,
  initialized: false,
  // X-axis epoch. recv_monotonic is the Python process's monotonic clock
  // (seconds-since-boot, can be ~10⁵ s); subtract this so the axis starts
  // at 0 on page open and resets to 0 every time a run starts.
  t0: null,
  // Last accepted loadcell timestamp; used to drop out-of-order samples.
  lastT: -Infinity,
  // Direction-latch state for the pos_x → square-wave derivation.
  posLastValue: null,
  posLastDir: 0,
  // Significant-step threshold (mm). Smaller deltas are treated as noise
  // and the latch holds its prior direction.
  posDirEps: 0.005,
};

function resetLive() {
  live.t = [];
  live.y = [];
  live.posT = [];
  live.posY = [];
  live.t0 = null;
  live.lastT = -Infinity;
  live.posLastValue = null;
  live.posLastDir = 0;
}

function pruneLive() {
  // Time-window both traces to live.windowSeconds. Cutoff is anchored to
  // whichever trace has the more recent sample so a slow stream doesn't
  // anchor the window in the distant past.
  const lastLoad = live.t.length ? live.t[live.t.length - 1] : -Infinity;
  const lastPos = live.posT.length ? live.posT[live.posT.length - 1] : -Infinity;
  const last = Math.max(lastLoad, lastPos);
  if (!Number.isFinite(last)) return;
  const cutoff = last - live.windowSeconds;
  let i = 0;
  while (i < live.t.length && live.t[i] < cutoff) i++;
  if (i > 0) { live.t = live.t.slice(i); live.y = live.y.slice(i); }
  let j = 0;
  while (j < live.posT.length && live.posT[j] < cutoff) j++;
  if (j > 0) { live.posT = live.posT.slice(j); live.posY = live.posY.slice(j); }
}

function pushLive(t, v) {
  if (v === null || v === undefined) return;
  if (live.t0 === null) live.t0 = t;
  const rel = t - live.t0;
  // Drop samples that arrive earlier than the latest one (UDP re-ordering).
  if (rel < live.lastT) return;
  live.t.push(rel);
  live.y.push(v);
  live.lastT = rel;
  pruneLive();
}

function pushPos(t, v) {
  if (v === null || v === undefined) return;
  if (live.t0 === null) live.t0 = t;
  const rel = t - live.t0;
  // Latch a ±1 direction from the sign of (v - prev). Hold the prior
  // value when the step is below the noise threshold, so brief samples
  // at intermediate values during a transition don't toggle the latch.
  let dir = live.posLastDir;
  if (live.posLastValue !== null) {
    const delta = v - live.posLastValue;
    if (delta > live.posDirEps) dir = 1;
    else if (delta < -live.posDirEps) dir = -1;
  }
  live.posLastValue = v;
  live.posLastDir = dir;
  live.posT.push(rel);
  live.posY.push(dir);
  pruneLive();
}

function renderLive() {
  if (!live.t.length && !live.posT.length) return;
  const data = [
    {
      x: live.t, y: live.y,
      type: "scattergl", mode: "lines",
      line: { color: "#f7931e" },
      connectgaps: true,
      name: "loadcell",
      yaxis: "y",
    },
    {
      x: live.posT, y: live.posY,
      type: "scattergl", mode: "lines",
      line: { color: "#58a6ff", width: 1, shape: "hv" },
      connectgaps: true,
      name: "dir(pos_x)",
      yaxis: "y2",
    },
  ];
  const layout = {
    margin: { l: 50, r: 55, t: 10, b: 30 },
    paper_bgcolor: "#161b22",
    plot_bgcolor: "#161b22",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "t (s, from open / last run start)" },
    yaxis: {
      gridcolor: "#2a2f37",
      title: { text: "loadcell", font: { color: "#f7931e" } },
      tickfont: { color: "#f7931e" },
    },
    yaxis2: {
      overlaying: "y",
      side: "right",
      showgrid: false,
      title: { text: "dir(pos_x)", font: { color: "#58a6ff" } },
      tickfont: { color: "#58a6ff" },
      range: [-1.5, 1.5],
      tickvals: [-1, 0, 1],
      ticktext: ["−X", "0", "+X"],
    },
    showlegend: false,
  };
  if (!live.initialized) {
    Plotly.newPlot("live_plot", data, layout, { displayModeBar: false, responsive: true });
    live.initialized = true;
  } else {
    Plotly.react("live_plot", data, layout);
  }
}

let lastRender = 0;
function maybeRender() {
  const now = performance.now();
  if (now - lastRender > 250) {
    renderLive();
    lastRender = now;
  }
}

// ---- bd_pressure segment browser ----
// State for the two-pane viewer. `bdState.analysis` is whatever was last
// loaded (live sweep OR a replayed npz). `selectedK` is the K row in the
// left navigator; `selectedSegIdx` is the segment index inside that K.
// The right pane re-renders any time either of those changes.
const bdState = {
  analysis: null,        // SweepAnalysis dict from the API
  windowsByK: {},        // {k: KWindow} keyed by k for quick lookup
  segmentsByK: {},       // {k: [BdSegment, ...]}
  selectedK: null,
  selectedSegIdx: 0,
  weights: {},           // current cost weights (mirror of slider state)
};

// Region color scheme — matches the bd_pressure reference image so the
// stats sidebar swatches and the on-plot shading make instant mental
// sense to anyone who's read the writeup.
const REGION_COLORS = {
  R1: "#58a6ff",  // baseline (blue)
  R2: "#3fb950",  // rising edge (green)
  R3: "#a371f7",  // overshoot (purple)
  R4: "#d29922",  // high plateau (yellow)
  R5: "#f7931e",  // creep (orange)
  R6: "#d9534f",  // falling edge (red)
  R7: "#db61a2",  // undershoot (pink)
  R8: "#39c5cf",  // recovery (cyan)
};

// Map region → which metrics belong to it (mirrors BD_METRIC_NAMES on
// the Python side). Used by the stats sidebar to group numbers.
const REGION_METRICS = {
  R1: ["baseline_median", "baseline_noise_std"],
  R2: ["rise_delay", "rise_error_area", "rise_signed_area", "rise_slope"],
  R3: ["overshoot"],
  R4: ["high_level"],
  R5: ["plateau_slope", "plateau_creep"],
  R6: ["fall_delay", "fall_error_area", "fall_signed_area"],
  R7: ["undershoot"],
  R8: ["tail_area", "settling_time"],
};

const REGION_TITLES = {
  R1: "1. low baseline",
  R2: "2. rising edge",
  R3: "3. overshoot",
  R4: "4. high plateau",
  R5: "5. plateau creep",
  R6: "6. falling edge",
  R7: "7. undershoot",
  R8: "8. recovery tail",
};

function fmt(v, digits = 3) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  if (Math.abs(v) >= 1000 || (Math.abs(v) > 0 && Math.abs(v) < 0.001)) {
    return v.toExponential(2);
  }
  return v.toFixed(digits);
}

function downsampleXY(xs, ys, maxPoints) {
  const n = Math.min(xs.length, ys.length);
  if (n <= maxPoints) return [xs.slice(0, n), ys.slice(0, n)];
  const stride = Math.ceil(n / maxPoints);
  const outX = [];
  const outY = [];
  for (let i = 0; i < n; i += stride) {
    outX.push(xs[i]);
    outY.push(ys[i]);
  }
  return [outX, outY];
}

// Build Plotly fill polygons shading between a trace (xs, ys) and a
// constant reference level `ref`, coloured by SIGN: `colorPos` where the
// trace is ABOVE ref, `colorNeg` where BELOW. The trace is split into
// contiguous same-sign runs, and a point interpolated AT the reference
// level is inserted at every crossing so adjacent polygons meet exactly
// on the reference line (no overlap, no gap). This is what turns an
// always-positive `toself` band into a visible overshoot(+)/lag(−)
// distinction — see the signed_area metrics on the Python side.
function signedFillTraces(xs, ys, ref, colorPos, colorNeg, namePos, nameNeg) {
  const out = [];
  if (xs.length < 2 || !Number.isFinite(ref)) return out;
  let runX = [], runY = [], runSign = 0;
  const flush = () => {
    if (runX.length >= 2) {
      out.push({
        x: [...runX, ...runX.slice().reverse()],
        y: [...runY, ...runX.map(() => ref)],
        type: "scatter", fill: "toself",
        fillcolor: runSign >= 0 ? colorPos : colorNeg,
        line: { width: 0 },
        name: runSign >= 0 ? namePos : nameNeg,
        hoverinfo: "skip",
      });
    }
    runX = []; runY = []; runSign = 0;
  };
  for (let i = 0; i < xs.length; i++) {
    const s = ys[i] >= ref ? 1 : -1;
    if (runSign !== 0 && s !== runSign) {
      // Sign flip between i-1 and i: split exactly at the ref crossing.
      const x0 = xs[i - 1], y0 = ys[i - 1], x1 = xs[i], y1 = ys[i];
      const denom = (y1 - y0) || 1e-12;
      const xc = x0 + ((ref - y0) / denom) * (x1 - x0);
      runX.push(xc); runY.push(ref);
      flush();
      runX.push(xc); runY.push(ref);
      runSign = s;
    }
    if (runSign === 0) runSign = s;
    runX.push(xs[i]); runY.push(ys[i]);
  }
  flush();
  return out;
}

// ---- segment browser: K navigator (left pane) ----
function renderBdKList() {
  const list = $("bd_k_list");
  if (!list) return;
  list.innerHTML = "";
  if (!bdState.analysis || !bdState.analysis.bd_per_k) {
    list.innerHTML = '<div class="meta" style="padding:10px;">no run loaded</div>';
    return;
  }
  const kOpt = bdState.analysis.bd_k_opt;
  for (const r of bdState.analysis.bd_per_k) {
    const row = document.createElement("div");
    row.className = "bd-k-row";
    if (bdState.selectedK !== null && Math.abs(r.k - bdState.selectedK) < 1e-6) {
      row.classList.add("active");
    }
    if (kOpt !== null && kOpt !== undefined
        && Number.isFinite(kOpt)
        && Math.abs(r.k - kOpt) < 0.0026) {
      row.classList.add("k-opt");
    }
    // Color-code seg count: green ≥ 11, yellow 4..10, red < 4
    let segClass = "red";
    if (r.n_segments_included >= 11) segClass = "green";
    else if (r.n_segments_included >= 4) segClass = "yellow";
    row.innerHTML = `
      <span class="k">${r.k.toFixed(4)}</span>
      <span class="segs ${segClass}">${r.n_segments_included}/${r.n_segments_total}</span>
    `;
    row.onclick = () => {
      bdState.selectedK = r.k;
      bdState.selectedSegIdx = 0;
      renderBdKList();
      renderBdSegment();
    };
    list.appendChild(row);
  }
}

function _segmentsForSelectedK() {
  if (bdState.selectedK === null) return [];
  return (bdState.segmentsByK[bdState.selectedK] || []);
}

// ---- segment browser: single segment plot + stats (right pane) ----
function renderBdSegment() {
  const label = $("bd_segment_label");
  const plot = $("bd_segment_plot");
  const stats = $("bd_segment_stats");
  const banner = $("bd_excluded_banner");
  if (!label || !plot || !stats || !banner) return;
  if (bdState.selectedK === null) {
    label.textContent = "select a K on the left";
    Plotly.purge(plot);
    stats.innerHTML = "";
    banner.style.display = "none";
    return;
  }
  const segs = _segmentsForSelectedK();
  if (!segs.length) {
    label.textContent = `K=${bdState.selectedK.toFixed(4)}: no segments produced`;
    Plotly.purge(plot);
    stats.innerHTML = "";
    banner.style.display = "none";
    return;
  }
  const idx = Math.max(0, Math.min(bdState.selectedSegIdx, segs.length - 1));
  bdState.selectedSegIdx = idx;
  const seg = segs[idx];
  const window_ = bdState.windowsByK[bdState.selectedK];
  label.textContent = `K=${seg.k.toFixed(4)} · segment ${seg.seg_idx + 1}/${segs.length}`;
  if (seg.excluded) {
    banner.textContent = "EXCLUDED: " + (seg.exclusion_reasons || []).join("; ");
    banner.style.display = "";
  } else {
    banner.style.display = "none";
  }
  _drawBdSegmentPlot(seg, window_);
  _renderBdSegmentStats(seg);
}

function _drawBdSegmentPlot(seg, window_) {
  const plot = $("bd_segment_plot");
  if (!plot) return;
  // Pull the force trace from the per-K KWindow and slice to this
  // segment's [t_start, t_end] (sweep-rel times).
  let tFull = window_ && Array.isArray(window_.t) ? window_.t : [];
  let yFull = window_ && Array.isArray(window_.force) ? window_.force : [];
  let dropoutFull =
    window_ && Array.isArray(window_.dropout_t) ? window_.dropout_t : [];

  // Slice using binary-search. Use the server-computed display crop
  // [t_lo_display, t_hi_display] (inset from [t_start, t_end] by ~10%
  // of slow_half on each side) so the plot doesn't show neighbour-
  // cycle artifacts at the shared boundary. Falls back to t_start /
  // t_end when the segment was returned by an early-exit path that
  // didn't fill the display fields.
  const tDispLo = Number.isFinite(seg.t_lo_display) && seg.t_lo_display > 0
    ? seg.t_lo_display
    : seg.t_start;
  const tDispHi = Number.isFinite(seg.t_hi_display) && seg.t_hi_display > 0
    ? seg.t_hi_display
    : seg.t_end;
  const lo = _lower_bound(tFull, tDispLo);
  const hi = _lower_bound(tFull, tDispHi);
  // tFull[hi] is the first sample >= tDispHi; exclude it (slice end
  // exclusive) so a wide firmware-throttle gap can't drag a
  // next-cycle fast-leg sample into this plot.
  const tSeg = tFull.slice(lo, hi);
  const ySeg = yFull.slice(lo, hi);
  // segment-relative time so prev/next stepping keeps the x-axis aligned
  const t0 = seg.t_start;
  const t = tSeg.map((x) => x - t0);
  const yRaw = ySeg.slice();
  const [tF, yF] = downsampleXY(t, yRaw, 1500);

  const baseline = seg.metrics.baseline_median;
  const high = seg.metrics.high_level;
  const tRiseR = seg.t_rise - t0;
  const tFallR = seg.t_fall - t0;
  // Region boundaries: prefer the threshold-based t_rise_end / t_fall_start
  // / t_fall_end which mark where force crosses 90% / 10% of the leg
  // delta. Fall back to the argmax-based t_peak / t_trough only when
  // threshold detection failed (very low SNR or no fast-leg plateau).
  const tRiseEndR = seg.t_rise_end !== null && seg.t_rise_end !== undefined
    ? seg.t_rise_end - t0
    : (seg.t_peak !== null ? seg.t_peak - t0 : null);
  // R4 plateau / R6 fall boundary -- use the DETECTED fall start when
  // available (the actual force often begins falling slightly before
  // the commanded t_fall due to PA lag). Falls back to the commanded
  // t_fall when not detected.
  const tFallStartR = seg.t_fall_start !== null && seg.t_fall_start !== undefined
    ? seg.t_fall_start - t0
    : (seg.t_fall - t0);
  const tFallEndR = seg.t_fall_end !== null && seg.t_fall_end !== undefined
    ? seg.t_fall_end - t0
    : (seg.t_trough !== null ? seg.t_trough - t0 : null);
  // Peak / trough markers (R3, R7) keep their own location for the
  // overshoot/undershoot annotation pins.
  const tPeakR = seg.t_peak !== null ? seg.t_peak - t0 : null;
  const tTroughR = seg.t_trough !== null ? seg.t_trough - t0 : null;

  const traces = [
    {
      x: tF, y: yF,
      type: "scatter", mode: "lines+markers",
      line: { color: "#f7931e", width: 1.6 },
      marker: { color: "#f7931e", size: 3 },
      name: "force",
      hovertemplate: "t=%{x:.3f}s<br>y=%{y:.1f}<extra></extra>",
    },
  ];

  const showTransitions = $("bd_overlay_transitions").checked;
  const showLevels = $("bd_overlay_levels").checked;
  const showPeaks = $("bd_overlay_peaks").checked;
  const showRegions = $("bd_overlay_regions").checked;
  const showAreas = $("bd_overlay_areas").checked;
  const showSlope = $("bd_overlay_slope").checked;
  const showLabels = $("bd_overlay_labels").checked;

  const shapes = [];
  const annotations = [];

  if (showRegions) {
    // Shade R1..R8. Region boundaries follow the threshold-based
    // t_rise_end (90% of delta) and t_fall_end (10% of delta) — these
    // mark the END of the actual rising/falling transition, NOT the
    // argmax which can land deep into the creeping plateau and make
    // R2 swallow the whole plateau or R6 swallow the recovery.
    const fast = tFallR - tRiseR;
    const lowNext = (t[t.length - 1] || 0) - tFallR;
    const r2End = tRiseEndR !== null ? tRiseEndR : tRiseR + 0.1 * fast;
    // R4/R6 boundary: prefer the threshold-detected fall start; the
    // commanded t_fall is the fallback. This shifts the boundary by
    // a few tens of ms to match the actual force fall (PA lag), so
    // the early fall transient lives in R6 (where it belongs) instead
    // of corrupting R4 (plateau) and the rise_error_area.
    const r4End = tFallStartR;
    const r6End = tFallEndR !== null ? tFallEndR : tFallR + 0.1 * lowNext;
    const r3X = tPeakR !== null ? tPeakR : r2End;
    const r7X = tTroughR !== null ? tTroughR : r6End;
    const regionBands = [
      { id: "R1", x0: 0, x1: tRiseR, alpha: 0.10 },
      { id: "R2", x0: tRiseR, x1: r2End, alpha: 0.12 },
      { id: "R3", x0: r3X - 0.005, x1: r3X + 0.005, alpha: 0.30 },
      { id: "R4", x0: r2End, x1: r4End, alpha: 0.12 },
      { id: "R5", x0: r2End, x1: r4End, alpha: 0.06 },
      { id: "R6", x0: r4End, x1: r6End, alpha: 0.12 },
      { id: "R7", x0: r7X - 0.005, x1: r7X + 0.005, alpha: 0.30 },
      { id: "R8", x0: r6End, x1: t[t.length - 1] || (tFallR + lowNext), alpha: 0.10 },
    ];
    for (const b of regionBands) {
      shapes.push({
        type: "rect", xref: "x", yref: "paper",
        x0: b.x0, x1: b.x1, y0: 0, y1: 1,
        fillcolor: REGION_COLORS[b.id], opacity: b.alpha,
        line: { width: 0 }, layer: "below",
      });
    }
  }

  if (showTransitions) {
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: tRiseR, x1: tRiseR, y0: 0, y1: 1,
      line: { color: "#7d8590", width: 1, dash: "dash" }, layer: "below",
    });
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: tFallR, x1: tFallR, y0: 0, y1: 1,
      line: { color: "#7d8590", width: 1, dash: "dash" }, layer: "below",
    });
  }
  if (showLevels && Number.isFinite(baseline)) {
    shapes.push({
      type: "line", xref: "paper", yref: "y",
      x0: 0, x1: 1, y0: baseline, y1: baseline,
      line: { color: "#58a6ff", width: 1, dash: "dot" }, layer: "below",
    });
  }
  if (showLevels && Number.isFinite(high)) {
    shapes.push({
      type: "line", xref: "paper", yref: "y",
      x0: 0, x1: 1, y0: high, y1: high,
      line: { color: "#d29922", width: 1, dash: "dot" }, layer: "below",
    });
  }
  if (showPeaks && tPeakR !== null && Number.isFinite(baseline)) {
    const peakY = baseline + (seg.metrics.high_level - baseline) + seg.metrics.overshoot;
    traces.push({
      x: [tPeakR], y: [peakY],
      type: "scatter", mode: "markers",
      marker: { color: REGION_COLORS.R3, size: 11, symbol: "diamond", line: { width: 1, color: "#fff" } },
      name: "peak",
      hovertemplate: "peak<br>t=%{x:.3f}s<br>overshoot=" + fmt(seg.metrics.overshoot, 1) + "<extra></extra>",
    });
  }
  if (showPeaks && tTroughR !== null && Number.isFinite(baseline)) {
    const troughY = baseline - seg.metrics.undershoot;
    traces.push({
      x: [tTroughR], y: [troughY],
      type: "scatter", mode: "markers",
      marker: { color: REGION_COLORS.R7, size: 11, symbol: "diamond", line: { width: 1, color: "#fff" } },
      name: "trough",
      hovertemplate: "trough<br>t=%{x:.3f}s<br>undershoot=" + fmt(seg.metrics.undershoot, 1) + "<extra></extra>",
    });
  }
  if (showSlope && Number.isFinite(seg.metrics.plateau_slope) && Number.isFinite(high)) {
    // Draw the plateau-slope linear fit as a green segment across R4.
    const tPlatLo = (tPeakR !== null ? tPeakR : tRiseR + 0.5 * (tFallR - tRiseR)) + 0.05;
    const tPlatHi = tFallR - 0.02;
    const slope = seg.metrics.plateau_slope;
    // The fit passes through high_level at the plateau midpoint
    const mid = 0.5 * (tPlatLo + tPlatHi);
    traces.push({
      x: [tPlatLo, tPlatHi],
      y: [high + slope * (tPlatLo + t0 - (mid + t0)), high + slope * (tPlatHi + t0 - (mid + t0))],
      type: "scatter", mode: "lines",
      line: { color: REGION_COLORS.R5, width: 2 },
      name: "slope fit",
      hovertemplate: "slope=" + fmt(slope) + "<extra></extra>",
    });
  }
  if (showAreas && Number.isFinite(baseline) && Number.isFinite(high)) {
    // SIGNED area fills: shade between the trace and a reference level,
    // coloured by which direction the deviation goes. The colour encodes
    // PA direction consistently across BOTH edges:
    //   red  = over-PA  contribution (rise overshoots ABOVE the plateau,
    //                                 fall undershoots BELOW the baseline)
    //   blue = under-PA contribution (rise lags BELOW the plateau,
    //                                 fall drools ABOVE the baseline)
    // This is the visual twin of rise_signed_area / fall_signed_area: the
    // net coloured balance is what those integrals sum to, and it flips
    // sign through K_opt. (The old single-colour `toself` band summed the
    // absolute area, so it could never show the overshoot↔lag direction.)
    const OVER = "rgba(217, 83, 79, 0.28)";   // over-PA (red)
    const UNDER = "rgba(88, 166, 255, 0.25)"; // under-PA (blue)
    // R2 (rising): force vs high_level over [t_rise, t_fall]. Above the
    // plateau = overshoot (over-PA, red); below = lag (under-PA, blue).
    const r2X = [], r2Y = [];
    for (let i = 0; i < t.length; i++) {
      if (t[i] >= tRiseR && t[i] <= tFallR) { r2X.push(t[i]); r2Y.push(yRaw[i]); }
    }
    for (const tr of signedFillTraces(r2X, r2Y, high, OVER, UNDER,
                                      "rise overshoot", "rise lag")) {
      traces.push(tr);
    }
    // R6+R8 (falling + recovery): force vs baseline over [t_fall, t_end].
    // Above baseline = drool/slow bleed (under-PA, blue); below = undershoot
    // dip (over-PA, red) — note this is the OPPOSITE position-to-PA mapping
    // from the rise edge, which is exactly why a single fixed colour band
    // hides the signal.
    const r6X = [], r6Y = [];
    for (let i = 0; i < t.length; i++) {
      if (t[i] >= tFallR) { r6X.push(t[i]); r6Y.push(yRaw[i]); }
    }
    for (const tr of signedFillTraces(r6X, r6Y, baseline, UNDER, OVER,
                                      "fall drool", "fall undershoot")) {
      traces.push(tr);
    }
  }
  if (showLabels) {
    if (Number.isFinite(seg.metrics.overshoot) && tPeakR !== null) {
      annotations.push({
        x: tPeakR, y: baseline + (high - baseline) + seg.metrics.overshoot,
        text: `Δ=${fmt(seg.metrics.overshoot, 1)}`,
        showarrow: true, arrowhead: 0, arrowcolor: REGION_COLORS.R3,
        font: { color: REGION_COLORS.R3, size: 11 },
        xanchor: "left", yanchor: "bottom",
      });
    }
    if (Number.isFinite(seg.metrics.undershoot) && tTroughR !== null) {
      annotations.push({
        x: tTroughR, y: baseline - seg.metrics.undershoot,
        text: `Δ=${fmt(seg.metrics.undershoot, 1)}`,
        showarrow: true, arrowhead: 0, arrowcolor: REGION_COLORS.R7,
        font: { color: REGION_COLORS.R7, size: 11 },
        xanchor: "left", yanchor: "top",
      });
    }
  }
  // Mark dropouts inside this segment with red Xs.
  const segDrop = dropoutFull
    .filter((dt) => dt >= seg.t_start && dt <= seg.t_end)
    .map((dt) => dt - t0);
  if (segDrop.length) {
    const dropY = segDrop.map((dt) => {
      const i = _lower_bound(t, dt);
      return yRaw[Math.min(i, yRaw.length - 1)] ?? 0;
    });
    traces.push({
      x: segDrop, y: dropY,
      type: "scatter", mode: "markers",
      marker: { color: "#ff4040", size: 11, symbol: "x", line: { width: 2 } },
      name: "dropout",
      hovertemplate: "dropout<br>t=%{x:.3f}s<extra></extra>",
    });
  }
  const layout = {
    margin: { l: 60, r: 20, t: 8, b: 36 },
    paper_bgcolor: "#1f242c",
    plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "t (s, segment-relative)" },
    yaxis: { gridcolor: "#2a2f37", title: "force (raw)", zeroline: false },
    showlegend: false,
    shapes: shapes,
    annotations: annotations,
  };
  Plotly.react(plot, traces, layout, { displayModeBar: false, responsive: true });
}

function _renderBdSegmentStats(seg) {
  const stats = $("bd_segment_stats");
  if (!stats) return;
  const blocks = Object.keys(REGION_METRICS).map((rid) => {
    const metricRows = REGION_METRICS[rid]
      .map((m) => `<div class="metric"><span>${m}</span><span class="value">${fmt(seg.metrics[m])}</span></div>`)
      .join("");
    return `
      <div class="bd-region" style="border-left-color:${REGION_COLORS[rid]};">
        <div class="name" style="color:${REGION_COLORS[rid]};">${REGION_TITLES[rid]}</div>
        ${metricRows}
      </div>
    `;
  });
  stats.innerHTML = blocks.join("");
}

// Binary search: index of first element ≥ target.
function _lower_bound(arr, target) {
  let lo = 0, hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < target) lo = mid + 1; else hi = mid;
  }
  return lo;
}

// ---- weight sliders + per-K metric plots ----
// Metrics that contribute to the composite cost. Each gets a weight
// slider in the Results panel and a row in the "per-metric K_opt"
// table. Must mirror the keys in BD_DEFAULT_WEIGHTS on the Python side
// (analysis.py). Order here = display order in the slider panel.
//   - area metrics ({rise/fall}_error_area, overshoot, undershoot,
//     tail_area, plateau_slope) bound the RIGHT side of the cost
//     valley.
//   - timing metrics (rise_delay, fall_delay, settling_time) bound
//     the LEFT side: at low K the response is slow, so these are
//     large and the cost rises again -- pushing the minimum to the
//     actual elbow.
const COST_METRICS = [
  "rise_error_area", "overshoot", "undershoot", "tail_area", "plateau_slope",
  "rise_delay", "fall_delay", "settling_time",
];

// Every per-K metric the analyser exposes — drives the metric grid.
const ALL_DISPLAY_METRICS = [
  "overshoot", "undershoot", "rise_error_area", "fall_error_area",
  "rise_signed_area", "fall_signed_area",
  "tail_area", "plateau_slope", "plateau_creep", "high_level",
  "baseline_noise_std", "rise_delay", "fall_delay", "settling_time",
];

function _activeCostMetrics() {
  // Cost metrics in display order. Starts from the static COST_METRICS
  // list (which dictates ordering) but ALSO picks up any extras the
  // server shipped in bd_default_weights that we forgot to add here --
  // prevents the "Python ships a new weighted metric but the UI has
  // no slider for it" footgun the user hit when rise_delay /
  // fall_delay / settling_time were added.
  const seen = new Set();
  const out = [];
  for (const n of COST_METRICS) {
    if (!seen.has(n)) { seen.add(n); out.push(n); }
  }
  const serverWeights = (bdState.analysis && bdState.analysis.bd_default_weights) || {};
  for (const n of Object.keys(serverWeights)) {
    if (!seen.has(n)) { seen.add(n); out.push(n); }
  }
  return out;
}

function _renderBdWeightsSourceBadge() {
  // "defaults" -- shipped BD_DEFAULT_WEIGHTS; "optimised" -- weights_opt.json
  // produced by the optimiser. Empty when no analysis is loaded yet.
  const el = $("bd_weights_source_badge");
  if (!el) return;
  const src = bdState.analysis && bdState.analysis.bd_weights_source;
  if (!src) { el.textContent = ""; return; }
  if (src.kind === "optimised") {
    const parts = [`weights: optimised`];
    if (src.n_runs !== undefined && src.n_runs !== null) parts.push(`${src.n_runs} runs`);
    if (src.date) parts.push(src.date);
    if (src.rms_error !== undefined && src.rms_error !== null)
      parts.push(`RMS ${Number(src.rms_error).toFixed(4)}`);
    el.textContent = parts.join(" · ");
    el.style.color = "#f7931e";
  } else {
    el.textContent = "weights: shipped defaults";
    el.style.color = "";
  }
}

function renderBdWeightSliders() {
  _renderBdWeightsSourceBadge();
  const panel = $("bd_weight_sliders");
  if (!panel) return;
  panel.innerHTML = "";
  for (const name of _activeCostMetrics()) {
    const w = bdState.weights[name] ?? 1.0;
    const row = document.createElement("label");
    row.innerHTML = `
      <div class="slider-row"><span class="name">${name}</span><span class="value" data-name="${name}">${w.toFixed(2)}</span></div>
      <input type="range" min="0" max="5" step="0.05" value="${w}" data-name="${name}">
    `;
    const slider = row.querySelector("input");
    const display = row.querySelector(".value");
    slider.oninput = () => {
      const v = parseFloat(slider.value);
      bdState.weights[name] = v;
      display.textContent = v.toFixed(2);
      renderBdCostAndKOpt();
    };
    panel.appendChild(row);
  }
}

function _ksAndCost() {
  // Compute composite cost per K from the analyser's normalised metrics
  // and the current slider weights. Clip overshoot/undershoot at 0 on
  // the negative side (matches the Python implementation).
  const ks = [];
  const cost = [];
  if (!bdState.analysis || !bdState.analysis.bd_per_k) return { ks, cost };
  for (const r of bdState.analysis.bd_per_k) {
    ks.push(r.k);
    let total = 0;
    let nan = false;
    for (const [name, w] of Object.entries(bdState.weights)) {
      let v = r.normalised[name];
      if (v === null || v === undefined || !Number.isFinite(v)) { nan = true; break; }
      if (name === "overshoot" || name === "undershoot") v = Math.max(0, v);
      total += w * v;
    }
    cost.push(nan ? NaN : total);
  }
  return { ks, cost };
}

// JS port of _argmin_with_parabolic. Same clamping behaviour.
function jsArgminParabolic(ks, ys) {
  const finiteKs = [], finiteYs = [];
  for (let i = 0; i < ks.length; i++) {
    if (Number.isFinite(ys[i])) { finiteKs.push(ks[i]); finiteYs.push(ys[i]); }
  }
  if (!finiteKs.length) return null;
  let iMin = 0;
  for (let i = 1; i < finiteYs.length; i++) {
    if (finiteYs[i] < finiteYs[iMin]) iMin = i;
  }
  if (finiteYs.length < 3 || iMin === 0 || iMin === finiteYs.length - 1) {
    return finiteKs[iMin];
  }
  const x0 = finiteKs[iMin - 1], x1 = finiteKs[iMin], x2 = finiteKs[iMin + 1];
  const y0 = finiteYs[iMin - 1], y1 = finiteYs[iMin], y2 = finiteYs[iMin + 1];
  const denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
  if (denom === 0) return x1;
  const a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
  const b = (x2*x2*(y0 - y1) + x1*x1*(y2 - y0) + x0*x0*(y1 - y2)) / denom;
  if (a <= 0) return x1;
  const vx = -b / (2 * a);
  return Math.max(Math.min(vx, x2), x0);
}

// Optimum for a SIGNED metric: the K where it crosses ZERO, not the
// argmin. Signed step-response areas (rise_signed_area, fall_signed_area)
// start strongly saturated at low K (large lag) and cross zero once at the
// lag→overshoot transition; later wiggles are noise around zero. So scan
// from low K and return the FIRST sign change vs the first finite sample,
// linearly interpolated between the two bracketing K points. Returns null
// if the curve never crosses (stays one sign across the whole sweep).
function jsZeroCrossing(ks, ys) {
  const xs = [], vs = [];
  for (let i = 0; i < ks.length; i++) {
    if (Number.isFinite(ys[i])) { xs.push(ks[i]); vs.push(ys[i]); }
  }
  if (xs.length < 2) return null;
  const s0 = Math.sign(vs[0]);
  if (s0 === 0) return xs[0];
  for (let i = 1; i < xs.length; i++) {
    const s = Math.sign(vs[i]);
    if (s !== 0 && s !== s0) {
      const x0 = xs[i - 1], y0 = vs[i - 1], x1 = xs[i], y1 = vs[i];
      const denom = (y1 - y0) || 1e-12;
      return x0 + ((0 - y0) / denom) * (x1 - x0);
    }
  }
  return null;
}

// Metrics whose optimum is a zero crossing (signed), not an argmin.
const SIGNED_ZERO_CROSS_METRICS = new Set(["rise_signed_area", "fall_signed_area"]);

function renderBdCostAndKOpt() {
  const { ks, cost } = _ksAndCost();
  const kOpt = jsArgminParabolic(ks, cost);
  $("bd_k_opt_display").textContent = kOpt !== null ? kOpt.toFixed(4) : "—";
  // K-color the navigator if we re-picked a different K opt.
  if (bdState.analysis) {
    bdState.analysis.bd_k_opt = kOpt;
    renderBdKList();
  }
  _drawBdCostPlot(ks, cost, kOpt);
  _renderPerMetricTable();
}

function _drawBdCostPlot(ks, cost, kOpt) {
  const plot = $("bd_cost_plot");
  if (!plot) return;
  const finite = cost.map((v) => Number.isFinite(v) ? v : null);
  const traces = [{
    x: ks, y: finite,
    type: "scatter", mode: "lines+markers",
    line: { color: "#f7931e", width: 1.4 },
    marker: { color: "#f7931e", size: 6 },
    name: "cost",
  }];
  const shapes = [];
  if (kOpt !== null && Number.isFinite(kOpt)) {
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: kOpt, x1: kOpt, y0: 0, y1: 1,
      line: { color: "#2ea043", dash: "dash", width: 1.5 },
    });
  }
  const layout = {
    margin: { l: 50, r: 20, t: 8, b: 36 },
    paper_bgcolor: "#1f242c", plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "K" },
    yaxis: { gridcolor: "#2a2f37", title: "composite cost" },
    showlegend: false,
    shapes: shapes,
  };
  Plotly.react(plot, traces, layout, { displayModeBar: false, responsive: true });
}

function _renderPerMetricTable() {
  const table = $("bd_per_metric_table");
  if (!table || !bdState.analysis) return;
  const ks = bdState.analysis.bd_per_k.map((r) => r.k);
  const rows = [`<tr><th>metric</th><th>K_opt</th></tr>`];
  for (const name of _activeCostMetrics()) {
    let ys = bdState.analysis.bd_per_k.map((r) => r.normalised[name]);
    if (name === "overshoot" || name === "undershoot") {
      ys = ys.map((v) => Number.isFinite(v) ? Math.max(0, v) : v);
    }
    const k = jsArgminParabolic(ks, ys);
    rows.push(`<tr><td>${name}</td><td class="kopt">${k !== null ? k.toFixed(4) : "—"}</td></tr>`);
  }
  table.innerHTML = rows.join("");
}

function renderBdMetricGrid() {
  const grid = $("bd_metric_grid");
  if (!grid || !bdState.analysis) return;
  grid.innerHTML = "";
  const ks = bdState.analysis.bd_per_k.map((r) => r.k);
  for (const name of ALL_DISPLAY_METRICS) {
    const cell = document.createElement("div");
    cell.className = "bd-metric-cell";
    const plotId = `bd_metric_${name}`;
    const vals = bdState.analysis.bd_per_k.map((r) => r.medians[name]);
    // Error bars: median absolute deviation of this metric across the
    // included segments at each K (×1.4826 ≈ σ-equivalent). MADs are
    // computed server-side in `_bd_aggregate_per_k`. A tight bar means
    // the segment-to-segment variance is small relative to the median
    // -- the metric is reliable at this K. A huge bar means the median
    // is dominated by noise / outliers.
    const mads = bdState.analysis.bd_per_k.map(
      (r) => (r.mads && Number.isFinite(r.mads[name])) ? r.mads[name] : null,
    );
    const signed = SIGNED_ZERO_CROSS_METRICS.has(name);
    let kOpt;
    if (signed) {
      // Signed metric: optimum is the zero crossing, not the minimum.
      kOpt = jsZeroCrossing(ks, vals);
    } else {
      let valsForKopt = vals;
      if (name === "overshoot" || name === "undershoot") {
        valsForKopt = vals.map((v) => Number.isFinite(v) ? Math.max(0, v) : v);
      }
      kOpt = jsArgminParabolic(ks, valsForKopt);
    }
    cell.innerHTML = `
      <div class="label"><span>${name}</span><span class="kopt">${kOpt !== null ? "K_opt=" + kOpt.toFixed(4) : ""}</span></div>
      <div id="${plotId}" style="width:100%;height:160px;"></div>
    `;
    grid.appendChild(cell);
    const traces = [{
      x: ks, y: vals.map((v) => Number.isFinite(v) ? v : null),
      type: "scatter", mode: "lines+markers",
      line: { color: "#f7931e", width: 1 },
      marker: { color: "#f7931e", size: 4 },
      error_y: {
        type: "data",
        array: mads.map((v) => v !== null ? v : 0),
        visible: true,
        color: "#f7931e",
        thickness: 1,
        width: 3,
      },
      hovertemplate: "K=%{x:.4f}<br>median=%{y:.3g}<br>MAD=%{error_y.array:.3g}<extra></extra>",
    }];
    const shapes = [];
    if (kOpt !== null && Number.isFinite(kOpt)) {
      shapes.push({
        type: "line", xref: "x", yref: "paper",
        x0: kOpt, x1: kOpt, y0: 0, y1: 1,
        line: { color: "#2ea043", dash: "dash", width: 1.2 },
      });
    }
    const layout = {
      margin: { l: 36, r: 10, t: 4, b: 24 },
      paper_bgcolor: "#161b22", plot_bgcolor: "#161b22",
      font: { color: "#e6edf3", size: 10 },
      xaxis: { gridcolor: "#2a2f37" },
      // Signed metrics: show the y=0 line so the green K_opt bar visibly
      // sits on the zero crossing (the optimum), not at the curve minimum.
      yaxis: {
        gridcolor: "#2a2f37",
        zeroline: signed,
        zerolinecolor: "#888",
        zerolinewidth: 1,
      },
      showlegend: false,
      shapes: shapes,
    };
    Plotly.newPlot(plotId, traces, layout, { displayModeBar: false, responsive: true });
  }
}

// Take a full analysis dict from /api/status or /api/runs and prime the
// segment browser + sliders + metric grid. Called from renderRun() when
// a sweep finishes, and from the replay handlers when a saved run is
// selected.
function loadBdAnalysis(analysis) {
  bdState.analysis = analysis;
  // Build O(1) lookups.
  bdState.windowsByK = {};
  for (const w of analysis.windows || []) {
    bdState.windowsByK[w.k] = w;
  }
  bdState.segmentsByK = {};
  for (const s of analysis.bd_segments || []) {
    if (!bdState.segmentsByK[s.k]) bdState.segmentsByK[s.k] = [];
    bdState.segmentsByK[s.k].push(s);
  }
  for (const k of Object.keys(bdState.segmentsByK)) {
    bdState.segmentsByK[k].sort((a, b) => a.seg_idx - b.seg_idx);
  }
  // Initialise weights from server defaults (only on first load /
  // when keys are missing -- preserve user slider state across re-renders).
  for (const [name, w] of Object.entries(analysis.bd_default_weights || {})) {
    if (!(name in bdState.weights)) bdState.weights[name] = w;
  }
  // Pick a K to focus by default: bd_k_opt if available, else the
  // first one with included segments, else just the first K.
  if (bdState.selectedK === null
      || !bdState.segmentsByK[bdState.selectedK]) {
    let chosen = null;
    if (analysis.bd_k_opt !== null && analysis.bd_k_opt !== undefined) {
      // Find the closest K in the sweep to bd_k_opt
      let bestDelta = Infinity;
      for (const r of analysis.bd_per_k) {
        const d = Math.abs(r.k - analysis.bd_k_opt);
        if (d < bestDelta) { bestDelta = d; chosen = r.k; }
      }
    }
    if (chosen === null) {
      const ok = (analysis.bd_per_k || []).find((r) => r.n_segments_included > 0);
      chosen = ok ? ok.k : (analysis.bd_per_k[0] || {}).k ?? null;
    }
    bdState.selectedK = chosen;
    bdState.selectedSegIdx = 0;
  }
  renderBdWeightSliders();
  renderBdKList();
  renderBdSegment();
  renderBdCostAndKOpt();
  renderBdMetricGrid();
}

// ---- result plots ----
function plotFit(divId, k, y, fit, yTitle, customdata) {
  // `customdata` (optional): array of strings shown on hover as
  // "<value>\n<extra>". Used to surface per-K bd cycle inclusion
  // (e.g. "9/10 cycles") on the integral plot so users can see what
  // the shared-exclusion gate dropped.
  const trace = {
    x: k, y: y, mode: "markers", type: "scatter",
    marker: { color: "#f7931e", size: 9 },
    name: "measured",
  };
  if (customdata && customdata.length === k.length) {
    trace.customdata = customdata;
    trace.hovertemplate = "K=%{x:.4f}<br>y=%{y:.3g}<br>%{customdata}<extra></extra>";
  }
  const traces = [trace];
  if (fit && fit.k_opt !== null) {
    const xs = [Math.min(...k), Math.max(...k)];
    const ys = xs.map((x) => fit.slope * x + fit.intercept);
    traces.push({
      x: xs, y: ys, mode: "lines", type: "scatter",
      line: { color: "#2ea043", dash: "dash" },
      name: "fit",
    });
    traces.push({
      x: [fit.k_opt], y: [0], mode: "markers", type: "scatter",
      marker: { color: "#2ea043", size: 12, symbol: "x" },
      name: "K_opt",
    });
  }
  const layout = {
    margin: { l: 50, r: 10, t: 10, b: 40 },
    paper_bgcolor: "#1f242c",
    plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "K" },
    yaxis: { gridcolor: "#2a2f37", title: yTitle, zeroline: true, zerolinecolor: "#444" },
    showlegend: false,
  };
  Plotly.newPlot(divId, traces, layout, { displayModeBar: false, responsive: true });
}

// bd_pressure-style argmin plots: scatter of metric vs K with a vertical
// line at k_opt_argmin (the recommended K). No fitted slope -- argmin is
// the direct estimator, the curve shape is the diagnostic.
function plotArgmin(divId, k, y, kOpt, yTitle, zeroLine) {
  const finiteK = [];
  const finiteY = [];
  for (let i = 0; i < k.length; i++) {
    if (Number.isFinite(y[i])) {
      finiteK.push(k[i]);
      finiteY.push(y[i]);
    }
  }
  const traces = [{
    x: finiteK, y: finiteY, mode: "lines+markers", type: "scatter",
    marker: { color: "#f7931e", size: 7 },
    line: { color: "#f7931e", width: 1.2 },
    name: "measured",
  }];
  if (kOpt !== null && kOpt !== undefined && Number.isFinite(kOpt)) {
    const yMin = finiteY.length ? Math.min(...finiteY) : 0;
    const yMax = finiteY.length ? Math.max(...finiteY) : 1;
    const pad = (yMax - yMin) * 0.1 || 1;
    traces.push({
      x: [kOpt, kOpt], y: [yMin - pad, yMax + pad],
      mode: "lines", type: "scatter",
      line: { color: "#2ea043", dash: "dash", width: 1.5 },
      name: "K_opt",
    });
  }
  const layout = {
    margin: { l: 50, r: 10, t: 10, b: 40 },
    paper_bgcolor: "#1f242c",
    plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "K" },
    yaxis: {
      gridcolor: "#2a2f37",
      title: yTitle,
      zeroline: !!zeroLine,
      zerolinecolor: "#444",
    },
    showlegend: false,
  };
  Plotly.newPlot(divId, traces, layout, { displayModeBar: false, responsive: true });
}

let lastRunData = null;
let lastRunState = null;

function renderRun(run) {
  // Reset the live trace at the START of a run (idle/done/error → preparing
  // /running). The next incoming sample will set t0 fresh so the X axis
  // reads from 0 for the new test.
  const wasIdle = lastRunState === null
    || lastRunState === "idle"
    || lastRunState === "done"
    || lastRunState === "error";
  const nowActive = run.state === "preparing" || run.state === "running";
  if (wasIdle && nowActive) {
    resetLive();
    if (live.initialized) renderLive();
  }
  lastRunState = run.state;
  lastRunData = run;
  $("status_state").textContent = run.state;
  $("status_msg").textContent = run.message || "";
  // current_k is a float advance value (e.g. 0.04); show with 4 decimals so
  // both small (0.0000) and larger (0.1000) values render the same width.
  $("status_k").textContent =
    typeof run.current_k === "number" ? run.current_k.toFixed(4) : "—";
  $("status_pct").textContent = (run.progress_pct ?? 0).toFixed(0);

  const a = run.analysis;
  if (!a) return;
  const k = a.per_k.map((r) => r.k);
  const lag = a.per_k.map((r) => r.phase_lag_ms);
  const area = a.per_k.map((r) => r.integral_area);
  const areaLegacy = a.per_k.map((r) => r.integral_area_legacy);
  // Per-K bd-cycle inclusion for the integral plot's hover tooltip.
  // The bd_pressure analysis flags each cycle as included/excluded
  // (dropouts, low sample rate, signal-below-noise, ...); the integral
  // metric now consumes the SAME flags so both methods agree on which
  // cycles are good. Showing "9/10 cycles" tells the user how many
  // survived for each K.
  const integralCounts = a.per_k.map((r) => {
    const inc = r.integral_n_included, tot = r.integral_n_total;
    return tot > 0 ? `${inc}/${tot} cycles included` : "no bd cycles built";
  });

  plotFit("phase_plot", k, lag, a.phase_fit, "lag (ms)");
  plotFit("integral_plot", k, area, a.integral_fit, "area", integralCounts);
  plotFit("integral_legacy_plot", k, areaLegacy, a.integral_legacy_fit, "area (legacy)");

  // Signed fall-area zero-crossing estimator. The backing per-K medians
  // live in bd_per_k[*].medians.fall_signed_area (NaN→null), and the fit
  // is bd_signed_fall_fit. Positive = drool/under-PA, negative =
  // undershoot/over-PA; the zero crossing is K_opt.
  const bdRows = a.bd_per_k || [];
  const signedK = bdRows.map((r) => r.k);
  const signedFall = bdRows.map((r) => (r.medians ? r.medians.fall_signed_area : null));
  // Optimum = the data's zero crossing (NOT the linear-fit crossing, which a
  // noisy high-K tail can drag off the visible crossing). plotArgmin draws
  // the curve + a y=0 line + a vertical bar at the supplied K, so the bar
  // sits exactly where the measured curve crosses zero — same method as the
  // metric-grid panels.
  const signedFallCross = jsZeroCrossing(signedK, signedFall);
  plotArgmin("signed_plot", signedK, signedFall, signedFallCross, "fall signed area", true);

  // bd_pressure: hand the whole analysis to the segment browser / weight
  // sliders / metric grid. They keep their own state across renders so
  // user slider tweaks and segment selection survive live updates.
  loadBdAnalysis(a);

  const pf = a.phase_fit, ig = a.integral_fit, igL = a.integral_legacy_fit;
  $("phase_k_opt").textContent = pf && pf.k_opt !== null ? pf.k_opt.toFixed(4) : "—";
  $("phase_slope").textContent = pf ? pf.slope.toExponential(2) : "—";
  $("phase_r2").textContent = pf ? pf.r_squared.toFixed(3) : "—";
  $("integral_k_opt").textContent = ig && ig.k_opt !== null ? ig.k_opt.toFixed(4) : "—";
  $("integral_slope").textContent = ig ? ig.slope.toExponential(2) : "—";
  $("integral_r2").textContent = ig ? ig.r_squared.toFixed(3) : "—";
  $("integral_legacy_k_opt").textContent = igL && igL.k_opt !== null ? igL.k_opt.toFixed(4) : "—";
  $("integral_legacy_slope").textContent = igL ? igL.slope.toExponential(2) : "—";
  $("integral_legacy_r2").textContent = igL ? igL.r_squared.toFixed(3) : "—";
  // Headline K_opt = the measured zero crossing; slope/R² come from the
  // server's linear fit and are kept only as trend diagnostics (R² ≈ how
  // monotone/clean the signal is — a low R² means the crossing is shaky).
  const sf = a.bd_signed_fall_fit;
  $("signed_k_opt").textContent = signedFallCross !== null ? signedFallCross.toFixed(4) : "—";
  $("signed_slope").textContent = sf ? sf.slope.toExponential(2) : "—";
  $("signed_r2").textContent = sf ? sf.r_squared.toFixed(3) : "—";

  const b = a.baseline;
  $("baseline_mean").textContent = b && b.mean !== null ? b.mean.toFixed(2) : "—";
  $("baseline_std").textContent = b && b.std !== null ? b.std.toFixed(3) : "—";
  $("baseline_drift").textContent = b && b.drift !== null
    ? (b.drift >= 0 ? "+" : "") + b.drift.toFixed(3) : "—";
  $("baseline_n").textContent = b && b.n_samples !== null ? b.n_samples : "—";

  if (a.notes && a.notes.length) {
    $("notes").innerHTML = "<strong>Notes:</strong><br/>" + a.notes.map((n) => `• ${n}`).join("<br/>");
  } else {
    $("notes").textContent = "";
  }
}

async function copyPressureAdvance() {
  if (!lastRunData || !lastRunData.analysis) {
    flash($("btn_copy"), "No result yet");
    return;
  }
  const a = lastRunData.analysis;
  // Prefer the bd_pressure step-response K_opt (composite cost argmin
  // with the user's current slider weights). The argmin/parabolic-interp
  // K is in `bdState` once loadBdAnalysis ran. Fall back to phase-lag /
  // integral fits when bd has no usable K (e.g. all segments excluded).
  let k = null;
  let source = "";
  const bdK = bdState.analysis ? bdState.analysis.bd_k_opt : null;
  if (bdK !== null && bdK !== undefined && Number.isFinite(bdK)) {
    k = bdK;
    source = "bd_pressure";
  } else if (a.phase_fit && a.phase_fit.k_opt !== null) {
    k = a.phase_fit.k_opt;
    source = "phase";
  } else if (a.integral_fit && a.integral_fit.k_opt !== null) {
    k = a.integral_fit.k_opt;
    source = "integral";
  }
  if (k === null) {
    flash($("btn_copy"), "No K_opt");
    return;
  }
  // Prusa Buddy/Core One uses M572 S<value>, NOT Marlin's M900 K.
  const txt = `M572 S${k.toFixed(4)} ; PA tuner (${source})`;
  await navigator.clipboard.writeText(txt);
  $("copy_status").textContent = `Copied: ${txt}`;
  setTimeout(() => ($("copy_status").textContent = ""), 3000);
}

// Fetch an xlsx export and save it via a synthetic <a download>. We go
// through fetch (rather than just navigating to the URL) so a 404/500 --
// e.g. exporting the current run when none has finished -- surfaces as a
// readable alert instead of a silent failed navigation or a downloaded
// error blob. The server names the file via Content-Disposition; we
// honour it and fall back to `fallbackName` if the header is absent.
async function downloadXlsx(url, fallbackName) {
  try {
    const r = await fetch(url);
    if (!r.ok) {
      let msg = await r.text();
      try { msg = JSON.parse(msg).detail || msg; } catch (_) { /* plain text */ }
      alert("Export failed: " + msg);
      return;
    }
    const blob = await r.blob();
    let name = fallbackName;
    const cd = r.headers.get("Content-Disposition") || "";
    const m = cd.match(/filename="?([^"]+)"?/);
    if (m) name = m[1];
    const objUrl = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = objUrl;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(objUrl);
  } catch (e) {
    alert("Export error: " + e);
  }
}

// ---- Max Flow test ----
let lastFlowData = null;          // last flow run/replay data (copy/export)
let lastFlowState = null;
let loadedFlowReplayFilename = null;
const flowRunsByFilename = {};
// Per-step viewer state (mirrors bdState for the PA segment browser).
const flowViewer = { analysis: null, levels: [], selectedIdx: 0 };

function setMode(mode) {
  document.body.classList.toggle("mode-flow", mode === "flow");
  document.body.classList.toggle("mode-livemap", mode === "livemap");
  document.body.classList.toggle("mode-probe", mode === "probe");
  $("mode_pa").classList.toggle("active", mode === "pa");
  $("mode_flow").classList.toggle("active", mode === "flow");
  $("mode_livemap").classList.toggle("active", mode === "livemap");
  $("mode_probe").classList.toggle("active", mode === "probe");
  try { localStorage.setItem("ppat_mode", mode); } catch (_) { /* private mode */ }
  // Plotly can't size a display:none container; resize on un-hide.
  setTimeout(() => {
    try { if (live.initialized) Plotly.Plots.resize("live_plot"); } catch (_) {}
    try { Plotly.Plots.resize("flow_plot"); } catch (_) {}
    try { Plotly.Plots.resize("lm_plot"); } catch (_) {}
    try { Plotly.Plots.resize("probe_plot"); } catch (_) {}
    try { Plotly.Plots.resize("probe_touch_plot"); } catch (_) {}
    if (mode === "livemap") lmapRequestRender();
  }, 50);
}

async function flowPreview() {
  await saveConfig();
  window.open("/api/flow/preview", "_blank");
}
async function startFlow() {
  await saveConfig();
  const r = await fetch("/api/flow/run", { method: "POST" });
  if (!r.ok) alert("Start failed: " + (await r.text()));
}
async function cancelFlow() {
  await fetch("/api/flow/cancel", { method: "POST" });
}

function renderFlowRun(run) {
  // Reset the live trace at the start of a flow run (mirrors renderRun).
  const wasIdle = lastFlowState === null || lastFlowState === "idle"
    || lastFlowState === "done" || lastFlowState === "error";
  const nowActive = run.state === "preparing" || run.state === "running";
  if (wasIdle && nowActive) {
    resetLive();
    if (live.initialized) renderLive();
  }
  lastFlowState = run.state;
  lastFlowData = run;
  $("status_state").textContent = run.state;
  $("status_msg").textContent = run.message || "";
  $("status_k").textContent = "—";
  $("status_pct").textContent = (run.progress_pct ?? 0).toFixed(0);
  if (run.analysis) renderFlowResults(run.analysis);
}

function renderFlowResults(a) {
  const reco = a.recommended_max_flow;
  $("flow_reco_display").textContent =
    (reco !== null && reco !== undefined) ? reco.toFixed(2) : "—";

  const rows = [
    ["Recommended max", a.recommended_max_flow, "#2ea043"],
    ["Variance onset", a.variance_onset_flow, "#d29922"],
    ["Force deviation", a.deviation_flow, "#f7931e"],
    ["Collapse (skip)", a.collapse_flow, "#f85149"],
  ];
  $("flow_markers_table").innerHTML = rows.map(([label, v, color]) =>
    `<tr><td style="color:${color}">${label}</td>` +
    `<td style="text-align:right">${(v !== null && v !== undefined) ? v.toFixed(2) + " mm³/s" : "—"}</td></tr>`
  ).join("");

  if (a.notes && a.notes.length) {
    $("flow_notes").innerHTML = "<strong>Notes:</strong><br/>" +
      a.notes.map((n) => `• ${n}`).join("<br/>");
  } else {
    $("flow_notes").textContent = "";
  }

  // force-vs-flow plot
  const levels = (a.levels || []).filter(
    (lv) => lv.force_mean !== null && lv.force_mean !== undefined);
  const q = levels.map((lv) => lv.flow_mm3_s);
  const fmean = levels.map((lv) => lv.force_mean);
  // null (not 0) for missing σ so the line breaks instead of dipping.
  const fstd = levels.map((lv) =>
    (lv.force_std !== null && lv.force_std !== undefined && Number.isFinite(lv.force_std))
      ? lv.force_std : null);
  const traces = [{
    x: q, y: fmean, type: "scatter", mode: "lines+markers",
    name: "steady-state force",
    marker: { color: "#58a6ff", size: 7 },
    line: { color: "#58a6ff", width: 1.2 },
    hovertemplate: "Q=%{x:.2f} mm³/s<br>F=%{y:.1f} (raw)<extra></extra>",
  }];
  if (a.fit && q.length) {
    const qmin = Math.min(...q), qmax = Math.max(...q);
    const xs = [], ys = [];
    const N = 80;
    for (let i = 0; i <= N; i++) {
      const x = qmin + (qmax - qmin) * i / N;
      xs.push(x);
      ys.push(a.fit.a * Math.pow(x, a.fit.b) + a.fit.c);
    }
    traces.push({
      x: xs, y: ys, type: "scatter", mode: "lines",
      name: `fit a·Q^${a.fit.b.toFixed(2)}+c`,
      line: { color: "#8b949e", dash: "dash", width: 1.5 },
      hoverinfo: "skip",
    });
  }
  // Force σ (noise) on a secondary axis -- the earliest breakdown signal.
  // Once the extruder skips it explodes (>10x), which would flatten the
  // whole curve, so we anchor the y2 scale to the QUIET low-flow σ (median
  // of the first few finite values) and let the runaway clip off the top.
  traces.push({
    x: q, y: fstd, type: "scatter", mode: "lines+markers", yaxis: "y2",
    name: "force σ (noise)",
    marker: { color: "#d29922", size: 5 },
    line: { color: "#d29922", width: 1, dash: "dot" },
    connectgaps: false,
    hovertemplate: "Q=%{x:.2f} mm³/s<br>σ=%{y:.1f} (raw)<extra></extra>",
  });
  let y2range = null;
  const quietStd = fstd.filter((v) => v !== null && v > 0).slice(0, 5)
    .sort((u, v) => u - v);
  if (quietStd.length) {
    const ref = quietStd[Math.floor(quietStd.length / 2)];  // median of first few
    if (ref > 0) y2range = [0, ref * 8];
  }

  const shapes = [], annos = [];
  const marks = [
    ["variance", a.variance_onset_flow, "#d29922"],
    ["deviation", a.deviation_flow, "#f7931e"],
    ["collapse", a.collapse_flow, "#f85149"],
    ["max", a.recommended_max_flow, "#2ea043"],
  ];
  for (const [label, v, color] of marks) {
    if (v === null || v === undefined) continue;
    shapes.push({
      type: "line", x0: v, x1: v, yref: "paper", y0: 0, y1: 1,
      line: { color, width: 1.5, dash: label === "max" ? "solid" : "dot" },
    });
    annos.push({
      x: v, yref: "paper", y: 1, text: label, showarrow: false,
      font: { color, size: 10 }, yanchor: "bottom",
    });
  }
  const layout = {
    margin: { l: 55, r: 58, t: 16, b: 42 },
    paper_bgcolor: "#1f242c", plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "flow rate (mm³/s)" },
    yaxis: { gridcolor: "#2a2f37", title: "nozzle force (raw, tared)" },
    yaxis2: {
      overlaying: "y", side: "right", showgrid: false, rangemode: "tozero",
      title: { text: "force σ (noise, scaled to quiet)", font: { color: "#d29922" } },
      tickfont: { color: "#d29922" },
      ...(y2range ? { range: y2range } : {}),
    },
    showlegend: true,
    legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)", font: { size: 10 } },
    shapes, annotations: annos,
  };
  Plotly.react("flow_plot", traces, layout, { displayModeBar: false, responsive: true });

  renderFlowViewer(a);
}

// ---- per-step viewer ----
function _flowBreakdownOnset(a) {
  // earliest flow at which any breakdown marker fired (for tinting the list)
  const marks = [a.deviation_flow, a.variance_onset_flow, a.collapse_flow]
    .filter((v) => v !== null && v !== undefined);
  return marks.length ? Math.min(...marks) : Infinity;
}

function renderFlowViewer(a) {
  flowViewer.analysis = a;
  flowViewer.levels = (a.levels || []).slice()
    .sort((x, y) => x.flow_mm3_s - y.flow_mm3_s);
  if (flowViewer.selectedIdx >= flowViewer.levels.length) flowViewer.selectedIdx = 0;
  renderFlowLevelList();
  renderFlowLevel();
}

function renderFlowLevelList() {
  const list = $("flow_level_list");
  if (!list) return;
  list.innerHTML = "";
  if (!flowViewer.levels.length) {
    list.innerHTML = '<div class="meta" style="padding:10px;">no run loaded</div>';
    return;
  }
  const brk = _flowBreakdownOnset(flowViewer.analysis || {});
  flowViewer.levels.forEach((lv, i) => {
    const row = document.createElement("div");
    row.className = "bd-k-row";
    if (i === flowViewer.selectedIdx) row.classList.add("active");
    row.textContent = lv.flow_mm3_s.toFixed(1);
    if (lv.flow_mm3_s >= brk) row.style.color = "#f7931e";
    row.onclick = () => {
      flowViewer.selectedIdx = i;
      renderFlowLevelList();
      renderFlowLevel();
    };
    list.appendChild(row);
  });
}

function renderFlowLevel() {
  const lv = flowViewer.levels[flowViewer.selectedIdx];
  const label = $("flow_level_label");
  if (!lv) { if (label) label.textContent = "no run loaded"; return; }
  if (label) {
    label.textContent =
      `flow ${lv.flow_mm3_s.toFixed(1)} mm³/s  (feed ${lv.feed_mm_s.toFixed(2)} mm/s)` +
      `  —  step ${flowViewer.selectedIdx + 1}/${flowViewer.levels.length}`;
  }

  const traces = [{
    x: lv.t, y: lv.force, type: "scattergl", mode: "lines",
    line: { color: "#f7931e", width: 1 }, name: "force",
    hovertemplate: "t=%{x:.2f}s<br>F=%{y:.1f} (raw)<extra></extra>",
  }];
  if (lv.force_mean !== null && lv.force_mean !== undefined) {
    traces.push({
      x: [lv.t_settle, lv.t_end], y: [lv.force_mean, lv.force_mean],
      type: "scatter", mode: "lines",
      line: { color: "#2ea043", width: 2, dash: "dash" },
      name: "steady-state", hoverinfo: "skip",
    });
  }
  const shapes = [{
    type: "rect", xref: "x", yref: "paper",
    x0: lv.t_settle, x1: lv.t_end, y0: 0, y1: 1,
    fillcolor: "rgba(46,160,67,0.12)", line: { width: 0 },
  }];
  const layout = {
    margin: { l: 55, r: 10, t: 10, b: 36 },
    paper_bgcolor: "#1f242c", plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "t (s, from sweep start)" },
    yaxis: { gridcolor: "#2a2f37", title: "nozzle force (raw, tared)" },
    showlegend: true,
    legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)", font: { size: 10 } },
    shapes,
  };
  Plotly.react("flow_level_plot", traces, layout, { displayModeBar: false, responsive: true });

  const fmt = (v, d = 2) =>
    (v !== null && v !== undefined && Number.isFinite(v)) ? v.toFixed(d) : "—";
  $("flow_level_stats").innerHTML = [
    ["flow", `${fmt(lv.flow_mm3_s, 2)} mm³/s`],
    ["feed", `${fmt(lv.feed_mm_s, 3)} mm/s`],
    ["steady-state force", `${fmt(lv.force_mean, 1)} (raw)`],
    ["force σ (noise)", `${fmt(lv.force_std, 2)}`],
    ["measured samples", `${lv.n_samples}`],
    ["window", `${fmt(lv.t_window_start, 1)}–${fmt(lv.t_end, 1)} s (measure from ${fmt(lv.t_settle, 1)} s)`],
  ].map(([k, v]) => `<div><span class="meta">${k}</span><div>${v}</div></div>`).join("");
}

function flowStep(delta) {
  if (!flowViewer.levels.length) return;
  const i = Math.max(0, Math.min(
    flowViewer.levels.length - 1, flowViewer.selectedIdx + delta));
  flowViewer.selectedIdx = i;
  renderFlowLevelList();
  renderFlowLevel();
}

async function copyFlowMax() {
  const reco = lastFlowData && lastFlowData.analysis
    ? lastFlowData.analysis.recommended_max_flow : null;
  if (reco === null || reco === undefined) {
    flash($("btn_flow_copy"), "No result");
    return;
  }
  await navigator.clipboard.writeText(reco.toFixed(2));
  $("flow_copy_status").textContent = `Copied: ${reco.toFixed(2)} mm³/s`;
  setTimeout(() => ($("flow_copy_status").textContent = ""), 3000);
}

function _renderFlowRunMeta(filename) {
  const el = $("flow_run_meta");
  if (!el) return;
  const run = flowRunsByFilename[filename];
  if (!run) { el.textContent = ""; return; }
  const parts = [];
  if (run.filament_label) parts.push(run.filament_label);
  if (run.nozzle_temp > 0) parts.push(`${run.nozzle_temp.toFixed(0)} °C`);
  parts.push(`${run.min_flow}–${run.max_flow} mm³/s`);
  el.textContent = parts.join("  ·  ");
}

async function refreshFlowRunsList() {
  const sel = $("flow_runs_select");
  if (!sel) return;
  try {
    const r = await fetch("/api/flow/runs");
    if (!r.ok) return;
    const data = await r.json();
    const prev = sel.value;
    sel.innerHTML = '<option value="">— select —</option>';
    for (const key of Object.keys(flowRunsByFilename)) delete flowRunsByFilename[key];
    for (const run of (data.runs || [])) {
      flowRunsByFilename[run.filename] = run;
      const opt = document.createElement("option");
      opt.value = run.filename;
      const date = _formatLocalTimestamp(run.mtime_unix);
      opt.textContent = `${date}  ${run.min_flow}–${run.max_flow} mm³/s  (${run.filename})`;
      sel.appendChild(opt);
    }
    sel.value = prev;
    _renderFlowRunMeta(sel.value);
  } catch (_) { /* next refresh handles it */ }
}

async function loadFlowReplay(filename) {
  if (!filename) { loadedFlowReplayFilename = null; return; }
  try {
    const r = await fetch(
      `/api/flow/runs/${encodeURIComponent(filename)}/analyse`, { method: "POST" });
    if (!r.ok) { alert("Replay failed: " + (await r.text())); return; }
    const data = await r.json();
    loadedFlowReplayFilename = filename;
    lastFlowData = { analysis: data.analysis };
    renderFlowResults(data.analysis);
  } catch (e) {
    alert("Replay error: " + e);
  }
}

// ---- Touch Probe ----
let lastProbeData = null;
let lastProbeState = null;
let loadedProbeReplayFilename = null;
const probeRunsByFilename = {};
const probeViewer = { analysis: null, touches: [], selectedIdx: 0 };
// Distinguishable palette for the per-touch overlay.
const PROBE_TOUCH_COLORS = [
  "#58a6ff", "#2ea043", "#f7931e", "#d2a8ff", "#56d4dd",
  "#f778ba", "#e3b341", "#79c0ff", "#7ee787", "#ffa657",
];
const PROBE_VERDICT_COLOR = {
  "usable": "#2ea043",
  "marginal": "#d29922",
  "no clear signal": "#f85149",
};

async function probePreview() {
  await saveConfig();
  window.open("/api/probe/preview", "_blank");
}
async function startProbe() {
  await saveConfig();
  const r = await fetch("/api/probe/run", { method: "POST" });
  if (!r.ok) alert("Start failed: " + (await r.text()));
}
async function cancelProbe() {
  await fetch("/api/probe/cancel", { method: "POST" });
}

function renderProbeRun(run) {
  // Reset the live trace at the start of a probe run (mirrors renderFlowRun).
  const wasIdle = lastProbeState === null || lastProbeState === "idle"
    || lastProbeState === "done" || lastProbeState === "error";
  const nowActive = run.state === "preparing" || run.state === "running";
  if (wasIdle && nowActive) {
    resetLive();
    if (live.initialized) renderLive();
  }
  lastProbeState = run.state;
  lastProbeData = run;
  $("status_state").textContent = run.state;
  $("status_msg").textContent = run.message || "";
  $("status_k").textContent = "—";
  $("status_pct").textContent = (run.progress_pct ?? 0).toFixed(0);
  if (run.analysis) renderProbeResults(run.analysis);
}

function renderProbeResults(a) {
  const vEl = $("probe_verdict");
  if (vEl) {
    vEl.textContent = a.verdict || "—";
    vEl.style.color = PROBE_VERDICT_COLOR[a.verdict] || "#e6edf3";
  }

  const fmt = (v, d = 3, unit = "") =>
    (v !== null && v !== undefined && Number.isFinite(v)) ? v.toFixed(d) + unit : "—";
  const meanMm = a.contact_pos_mean;
  const stdUm = (a.contact_pos_std !== null && a.contact_pos_std !== undefined)
    ? a.contact_pos_std * 1000 : null;
  const nTouches = a.touches ? a.touches.length : 0;
  $("probe_stats_table").innerHTML = [
    ["contacts detected", `${a.n_contacts} / ${nTouches}`],
    ["contact position", meanMm != null ? `${fmt(meanMm, 3)} mm into creep` : "—"],
    ["repeatability σ", stdUm != null ? `${fmt(stdUm, 0)} µm` : "—"],
    ["signal / noise", fmt(a.signal_to_noise, 1)],
    ["sample rate", fmt(a.sample_rate_hz, 0, " Hz")],
  ].map(([k, v]) => `<tr><td>${k}</td><td style="text-align:right">${v}</td></tr>`).join("");

  if (a.notes && a.notes.length) {
    $("probe_notes").innerHTML = "<strong>Notes:</strong><br/>" +
      a.notes.map((n) => `• ${n}`).join("<br/>");
  } else {
    $("probe_notes").textContent = "";
  }

  // overlay: force-vs-position, one trace per touch + a contact marker line.
  const traces = [];
  const shapes = [];
  (a.touches || []).forEach((t, i) => {
    const color = PROBE_TOUCH_COLORS[i % PROBE_TOUCH_COLORS.length];
    traces.push({
      x: t.pos, y: t.force, type: "scattergl", mode: "lines",
      line: { color, width: 1 }, name: `touch ${i}`,
      hovertemplate: `touch ${i}<br>x=%{x:.3f} mm<br>F=%{y:.1f}<extra></extra>`,
    });
    if (t.contact_pos !== null && t.contact_pos !== undefined) {
      shapes.push({
        type: "line", xref: "x", yref: "paper",
        x0: t.contact_pos, x1: t.contact_pos, y0: 0, y1: 1,
        line: { color, width: 1, dash: "dash" },
      });
    }
  });
  const layout = {
    margin: { l: 55, r: 10, t: 10, b: 42 },
    paper_bgcolor: "#1f242c", plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "probe-axis position (mm from standoff)" },
    yaxis: { gridcolor: "#2a2f37", title: "nozzle force (raw, tared)" },
    showlegend: true,
    legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)", font: { size: 10 } },
    shapes,
  };
  Plotly.react("probe_plot", traces, layout, { displayModeBar: false, responsive: true });

  renderProbeViewer(a);
}

function renderProbeViewer(a) {
  probeViewer.analysis = a;
  probeViewer.touches = (a.touches || []).slice();
  if (probeViewer.selectedIdx >= probeViewer.touches.length) probeViewer.selectedIdx = 0;
  renderProbeTouchList();
  renderProbeTouch();
}

function renderProbeTouchList() {
  const list = $("probe_touch_list");
  if (!list) return;
  list.innerHTML = "";
  if (!probeViewer.touches.length) {
    list.innerHTML = '<div class="meta" style="padding:10px;">no run loaded</div>';
    return;
  }
  probeViewer.touches.forEach((t, i) => {
    const row = document.createElement("div");
    row.className = "bd-k-row";
    if (i === probeViewer.selectedIdx) row.classList.add("active");
    row.textContent = `touch ${i}`;
    if (t.contact_pos === null || t.contact_pos === undefined) row.style.color = "#8b949e";
    row.onclick = () => {
      probeViewer.selectedIdx = i;
      renderProbeTouchList();
      renderProbeTouch();
    };
    list.appendChild(row);
  });
}

function renderProbeTouch() {
  const t = probeViewer.touches[probeViewer.selectedIdx];
  const label = $("probe_touch_label");
  if (!t) { if (label) label.textContent = "no run loaded"; return; }
  const a = probeViewer.analysis || {};
  if (label) label.textContent = `touch ${t.idx + 1}/${probeViewer.touches.length}`;

  const traces = [{
    x: t.pos, y: t.force, type: "scattergl", mode: "lines",
    line: { color: "#58a6ff", width: 1.2 }, name: "force",
    hovertemplate: "x=%{x:.3f} mm<br>F=%{y:.1f}<extra></extra>",
  }];
  if (t.noise !== null && t.noise !== undefined && a.n_sigma && t.pos.length) {
    const thr = a.n_sigma * t.noise;
    traces.push({
      x: [t.pos[0], t.pos[t.pos.length - 1]], y: [thr, thr],
      type: "scatter", mode: "lines",
      line: { color: "#d29922", width: 1, dash: "dot" },
      name: `${a.n_sigma}σ threshold`, hoverinfo: "skip",
    });
  }
  const shapes = [];
  if (t.pos.length) {
    const x0 = t.pos[0], x1 = t.pos[t.pos.length - 1];
    const cut = x0 + 0.2 * (x1 - x0);  // baseline window (matches analyser)
    shapes.push({
      type: "rect", xref: "x", yref: "paper", x0, x1: cut, y0: 0, y1: 1,
      fillcolor: "rgba(139,148,158,0.12)", line: { width: 0 },
    });
  }
  if (t.contact_pos !== null && t.contact_pos !== undefined) {
    shapes.push({
      type: "line", xref: "x", yref: "paper",
      x0: t.contact_pos, x1: t.contact_pos, y0: 0, y1: 1,
      line: { color: "#2ea043", width: 2, dash: "dash" },
    });
  }
  const layout = {
    margin: { l: 55, r: 10, t: 10, b: 42 },
    paper_bgcolor: "#1f242c", plot_bgcolor: "#1f242c",
    font: { color: "#e6edf3", size: 11 },
    xaxis: { gridcolor: "#2a2f37", title: "probe-axis position (mm from standoff)" },
    yaxis: { gridcolor: "#2a2f37", title: "nozzle force (raw, tared)" },
    showlegend: true,
    legend: { x: 0.02, y: 0.98, bgcolor: "rgba(0,0,0,0)", font: { size: 10 } },
    shapes,
  };
  Plotly.react("probe_touch_plot", traces, layout, { displayModeBar: false, responsive: true });

  const fmt = (v, d = 3) =>
    (v !== null && v !== undefined && Number.isFinite(v)) ? v.toFixed(d) : "—";
  $("probe_touch_stats").innerHTML = [
    ["contact position", t.contact_pos != null ? `${fmt(t.contact_pos, 3)} mm` : "— (no contact)"],
    ["peak force", `${fmt(t.peak_force, 1)} (raw, tared)`],
    ["baseline force", `${fmt(t.baseline_force, 1)} (raw)`],
    ["noise σ", `${fmt(t.noise, 2)}`],
    ["monotonic rise", t.monotonic_frac != null ? `${fmt(t.monotonic_frac * 100, 0)} %` : "—"],
    ["samples", `${t.n_samples}`],
  ].map(([k, v]) => `<div><span class="meta">${k}</span><div>${v}</div></div>`).join("");
}

function probeStep(delta) {
  if (!probeViewer.touches.length) return;
  const i = Math.max(0, Math.min(
    probeViewer.touches.length - 1, probeViewer.selectedIdx + delta));
  probeViewer.selectedIdx = i;
  renderProbeTouchList();
  renderProbeTouch();
}

function _renderProbeRunMeta(filename) {
  const el = $("probe_run_meta");
  if (!el) return;
  const run = probeRunsByFilename[filename];
  if (!run) { el.textContent = ""; return; }
  el.textContent =
    `${run.axis}${run.dir}  ·  ${run.n_touches} touches  ·  ${run.creep_mm} mm creep`;
}

async function refreshProbeRunsList() {
  const sel = $("probe_runs_select");
  if (!sel) return;
  try {
    const r = await fetch("/api/probe/runs");
    if (!r.ok) return;
    const data = await r.json();
    const prev = sel.value;
    sel.innerHTML = '<option value="">— select —</option>';
    for (const key of Object.keys(probeRunsByFilename)) delete probeRunsByFilename[key];
    for (const run of (data.runs || [])) {
      probeRunsByFilename[run.filename] = run;
      const opt = document.createElement("option");
      opt.value = run.filename;
      const date = _formatLocalTimestamp(run.mtime_unix);
      opt.textContent = `${date}  ${run.axis}${run.dir} ${run.n_touches}×  (${run.filename})`;
      sel.appendChild(opt);
    }
    sel.value = prev;
    _renderProbeRunMeta(sel.value);
  } catch (_) { /* next refresh handles it */ }
}

async function loadProbeReplay(filename) {
  if (!filename) { loadedProbeReplayFilename = null; return; }
  try {
    const r = await fetch(
      `/api/probe/runs/${encodeURIComponent(filename)}/analyse`, { method: "POST" });
    if (!r.ok) { alert("Replay failed: " + (await r.text())); return; }
    const data = await r.json();
    loadedProbeReplayFilename = filename;
    lastProbeData = { analysis: data.analysis };
    renderProbeResults(data.analysis);
  } catch (e) {
    alert("Replay error: " + e);
  }
}

// ---- websocket ----
function openWs() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/live`);
  ws.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "force") {
      // legacy single-sample format -- kept for compatibility but the
      // server now batches.
      pushLive(msg.t, msg.v);
      maybeRender();
    } else if (msg.type === "force_batch") {
      const ts = msg.t, vs = msg.v;
      if (Array.isArray(ts) && Array.isArray(vs)) {
        const n = Math.min(ts.length, vs.length);
        for (let i = 0; i < n; i++) pushLive(ts[i], vs[i]);
        maybeRender();
        lmapIngestForce(ts, vs);
      }
    } else if (msg.type === "pos_batch") {
      // Secondary axis: toolhead X position. Overlay so the user can see
      // every X transition that triggers a burst (sweep_t0 anchor).
      const ts = msg.t, vs = msg.v;
      if (Array.isArray(ts) && Array.isArray(vs)) {
        const n = Math.min(ts.length, vs.length);
        for (let i = 0; i < n; i++) pushPos(ts[i], vs[i]);
        maybeRender();
        lmapIngestPos("x", ts, vs);
      }
    } else if (msg.type === "pos_y_batch") {
      lmapIngestPos("y", msg.t, msg.v);
    } else if (msg.type === "pos_z_batch") {
      lmapIngestPos("z", msg.t, msg.v);
    } else if (msg.type === "run") {
      renderRun(msg.data);
    } else if (msg.type === "flow_run") {
      renderFlowRun(msg.data);
    } else if (msg.type === "livemap_run") {
      renderLiveMapRun(msg.data);
    } else if (msg.type === "probe_run") {
      renderProbeRun(msg.data);
    }
  };
  ws.onclose = () => setTimeout(openWs, 1500);
  ws.onerror = () => ws.close();
}

// ---- replay dropdown (saved runs/ npz) ----
// Per-run metadata keyed by filename, populated each refresh so the
// `change` handler can look up filament + temp without re-fetching.
const runsByFilename = {};

function _formatLocalTimestamp(unixSec) {
  // toISOString() always emits UTC, which is 1-2 h off from CET/CEST on
  // the user's machine and was the cause of the "2 or 3 h difference"
  // complaint. Build the timestamp in local time instead.
  const d = new Date(unixSec * 1000);
  const pad = (n) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}

function _renderRunMeta(filename) {
  const el = $("run_meta");
  if (!el) return;
  const run = runsByFilename[filename];
  if (!run) {
    el.textContent = "";
    return;
  }
  const parts = [];
  if (run.filament_label) parts.push(run.filament_label);
  if (run.nozzle_temp > 0) parts.push(`${run.nozzle_temp.toFixed(0)} °C`);
  el.textContent = parts.join("  @  ");
}

async function refreshRunsList() {
  const sel = $("runs_select");
  if (!sel) return;
  try {
    const r = await fetch("/api/runs");
    if (!r.ok) return;
    const data = await r.json();
    const prev = sel.value;
    sel.innerHTML = '<option value="">— select —</option>';
    // Reset and re-fill the lookup table.
    for (const key of Object.keys(runsByFilename)) delete runsByFilename[key];
    for (const run of (data.runs || [])) {
      runsByFilename[run.filename] = run;
      const opt = document.createElement("option");
      opt.value = run.filename;
      const date = _formatLocalTimestamp(run.mtime_unix);
      opt.textContent = `${date}  ${run.n_K}K × ${run.cycles_per_K}cyc  (${run.filename})`;
      sel.appendChild(opt);
    }
    sel.value = prev;
    _renderRunMeta(sel.value);
  } catch (_) { /* network hiccup -- next refresh handles it */ }
}

// Tracks which saved npz is currently rendered (null = live run / nothing).
// The annotation card needs this to know which file POST /annotate targets.
let loadedReplayFilename = null;

function _showAnnotateCard(filename, userKOpt, notes) {
  // Display the annotation card and pre-fill from the server-shipped
  // values. Empty box for `null` -- the user is supposed to enter the
  // value themselves; pre-filling with the analyser's K_opt would bias
  // them toward leaving it as-is.
  const card = $("bd_annotate");
  if (!card) return;
  card.style.display = filename ? "" : "none";
  $("user_k_opt_input").value = userKOpt !== null && userKOpt !== undefined ? userKOpt : "";
  $("user_k_opt_notes").value = notes || "";
  $("user_k_status").textContent = "";
}

async function saveUserKOpt(clear) {
  if (!loadedReplayFilename) return;
  const status = $("user_k_status");
  const raw = $("user_k_opt_input").value;
  const notes = $("user_k_opt_notes").value || "";
  let k = null;
  if (!clear) {
    if (raw === "" || raw === null) {
      status.textContent = "enter a K value or press Clear";
      return;
    }
    const parsed = Number(raw);
    if (!Number.isFinite(parsed) || parsed < 0) {
      status.textContent = "K must be a non-negative number";
      return;
    }
    k = parsed;
  }
  status.textContent = "saving...";
  try {
    const r = await fetch(
      `/api/runs/${encodeURIComponent(loadedReplayFilename)}/annotate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ k, notes: clear ? "" : notes }),
      },
    );
    if (!r.ok) {
      status.textContent = "save failed: " + (await r.text());
      return;
    }
    const data = await r.json();
    $("user_k_opt_input").value =
      data.user_k_opt !== null && data.user_k_opt !== undefined ? data.user_k_opt : "";
    $("user_k_opt_notes").value = data.user_k_opt_notes || "";
    status.textContent = clear ? "cleared" : `saved K=${data.user_k_opt}`;
    // Keep the dropdown's per-run lookup in sync so the next replay
    // refresh doesn't blow away the just-saved value visually.
    if (runsByFilename[loadedReplayFilename]) {
      runsByFilename[loadedReplayFilename].user_k_opt = data.user_k_opt;
      runsByFilename[loadedReplayFilename].user_k_opt_notes =
        data.user_k_opt_notes;
    }
  } catch (e) {
    status.textContent = "save error: " + e;
  }
}

async function loadReplay(filename) {
  if (!filename) {
    loadedReplayFilename = null;
    _showAnnotateCard(null, null, "");
    return;
  }
  try {
    const r = await fetch(`/api/runs/${encodeURIComponent(filename)}/analyse`, { method: "POST" });
    if (!r.ok) {
      alert("Replay failed: " + (await r.text()));
      return;
    }
    const data = await r.json();
    loadedReplayFilename = filename;
    // Synthesize a minimal `run` shape the existing renderRun expects.
    const fakeRun = {
      state: "done",
      message: `replay: ${filename}`,
      progress_pct: 100,
      current_k: null,
      analysis: data.analysis,
    };
    renderRun(fakeRun);
    _showAnnotateCard(filename, data.user_k_opt, data.user_k_opt_notes);
  } catch (e) {
    alert("Replay error: " + e);
  }
}

// ---- weight optimiser ----
// Pulls metric list + annotated-run count from /api/optimise_weights/info,
// renders the checkbox row, kicks off the DE optimiser and polls for
// progress, then displays the per-run prediction table and lets the
// user push the recommended weights into the sliders + persist them.
const optState = {
  info: null,             // {optimisable_metrics, annotated_run_count}
  excluded: new Set(),    // metric names with checkbox unchecked
  jobId: null,
  pollHandle: null,
  result: null,           // last completed result for Apply
};

async function refreshOptInfo() {
  try {
    const r = await fetch("/api/optimise_weights/info");
    if (!r.ok) return;
    const data = await r.json();
    optState.info = data;
    $("opt_run_count").textContent = String(data.annotated_run_count ?? 0);
    const wrap = $("opt_metric_checkboxes");
    wrap.innerHTML = "";
    for (const name of data.optimisable_metrics || []) {
      const id = `opt_chk_${name}`;
      const lbl = document.createElement("label");
      lbl.style.display = "inline-flex";
      lbl.style.alignItems = "center";
      lbl.style.gap = "4px";
      lbl.innerHTML = `<input type="checkbox" id="${id}" data-metric="${name}" ${optState.excluded.has(name) ? "" : "checked"}> ${name}`;
      const cb = lbl.querySelector("input");
      cb.onchange = () => {
        if (cb.checked) optState.excluded.delete(name);
        else optState.excluded.add(name);
      };
      wrap.appendChild(lbl);
    }
  } catch (e) {
    /* leave UI as-is; next refresh will retry */
  }
}

async function startOptimiser() {
  if (!optState.info) await refreshOptInfo();
  const body = {
    excluded_metrics: [...optState.excluded],
    alpha: parseFloat($("opt_alpha").value) || 1.0,
    n_boot: parseInt($("opt_n_boot").value, 10) || 200,
    maxiter: parseInt($("opt_maxiter").value, 10) || 50,
  };
  $("opt_status").textContent = "starting...";
  $("opt_progress_wrap").style.display = "";
  $("opt_progress").value = 0;
  $("opt_result").style.display = "none";
  $("btn_opt_apply").disabled = true;
  $("btn_opt_run").disabled = true;
  try {
    const r = await fetch("/api/optimise_weights/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      $("opt_status").textContent = "start failed: " + (await r.text());
      $("btn_opt_run").disabled = false;
      return;
    }
    const data = await r.json();
    optState.jobId = data.job_id;
    if (optState.pollHandle) clearInterval(optState.pollHandle);
    optState.pollHandle = setInterval(pollOptJob, 800);
  } catch (e) {
    $("opt_status").textContent = "start error: " + e;
    $("btn_opt_run").disabled = false;
  }
}

async function pollOptJob() {
  if (!optState.jobId) return;
  try {
    const r = await fetch(`/api/optimise_weights/${optState.jobId}`);
    if (!r.ok) {
      $("opt_status").textContent = "poll failed: " + (await r.text());
      _stopOptPoll();
      return;
    }
    const job = await r.json();
    $("opt_status").textContent = job.message || job.status;
    $("opt_progress").value = job.progress || 0;
    if (job.status === "done") {
      _stopOptPoll();
      optState.result = job.result;
      renderOptResult(job.result);
      $("btn_opt_apply").disabled = false;
      $("btn_opt_run").disabled = false;
    } else if (job.status === "error") {
      _stopOptPoll();
      $("opt_status").textContent = "error: " + (job.error || "unknown");
      $("btn_opt_run").disabled = false;
    }
  } catch (e) {
    $("opt_status").textContent = "poll error: " + e;
  }
}

function _stopOptPoll() {
  if (optState.pollHandle) {
    clearInterval(optState.pollHandle);
    optState.pollHandle = null;
  }
}

function renderOptResult(result) {
  if (!result) return;
  $("opt_result").style.display = "";
  $("opt_rms").textContent = Number.isFinite(result.rms_error)
    ? result.rms_error.toFixed(5) : "—";
  $("opt_med").textContent = Number.isFinite(result.median_abs_error)
    ? result.median_abs_error.toFixed(5) : "—";
  $("opt_mean_sigma").textContent = Number.isFinite(result.mean_bootstrap_std)
    ? result.mean_bootstrap_std.toFixed(5) : "—";
  $("opt_nruns").textContent = String(result.n_runs);
  $("opt_duration").textContent = `${result.duration_s.toFixed(1)} s`;

  const perRunBody = $("opt_per_run_body");
  perRunBody.innerHTML = "";
  for (const pr of result.per_run) {
    const tr = document.createElement("tr");
    const fmt = (v, n) => Number.isFinite(v) ? v.toFixed(n) : "—";
    const dCell = Number.isFinite(pr.delta)
      ? (pr.delta >= 0 ? `+${pr.delta.toFixed(4)}` : pr.delta.toFixed(4))
      : "—";
    tr.innerHTML = `
      <td style="padding:4px 8px;border-bottom:1px solid #1a1d24;font-family:monospace;">${pr.filename}</td>
      <td style="padding:4px 8px;border-bottom:1px solid #1a1d24;text-align:right;">${fmt(pr.user_k_opt, 4)}</td>
      <td style="padding:4px 8px;border-bottom:1px solid #1a1d24;text-align:right;">${fmt(pr.pred_mean, 4)}</td>
      <td style="padding:4px 8px;border-bottom:1px solid #1a1d24;text-align:right;">${fmt(pr.pred_std, 4)}</td>
      <td style="padding:4px 8px;border-bottom:1px solid #1a1d24;text-align:right;">${dCell}</td>
      <td style="padding:4px 8px;border-bottom:1px solid #1a1d24;text-align:right;">${pr.n_segments_median}</td>
    `;
    perRunBody.appendChild(tr);
  }

  const wBody = $("opt_weights_body");
  wBody.innerHTML = "";
  // Pull shipped defaults from the live analysis if available (the
  // server preload also flows through that field). Fall back to the
  // optimiser's own snapshot if the user hasn't loaded a run yet.
  const shipped = (bdState.analysis && bdState.analysis.bd_default_weights) || {};
  for (const name of result.optimisable_metrics) {
    const tr = document.createElement("tr");
    const def = shipped[name];
    const opt = result.weights_display[name];
    tr.innerHTML = `
      <td style="padding:4px 12px 4px 0;border-bottom:1px solid #1a1d24;">${name}</td>
      <td style="padding:4px 12px;border-bottom:1px solid #1a1d24;text-align:right;">${def !== undefined ? def.toFixed(2) : "—"}</td>
      <td style="padding:4px 12px;border-bottom:1px solid #1a1d24;text-align:right;">${opt !== undefined ? opt.toFixed(3) : "—"}</td>
    `;
    wBody.appendChild(tr);
  }

  const warnEl = $("opt_warnings");
  if (result.warnings && result.warnings.length) {
    warnEl.innerHTML = result.warnings.map((w) => `<div>⚠ ${w}</div>`).join("");
  } else {
    warnEl.innerHTML = "";
  }
}

async function applyOptResult() {
  if (!optState.jobId || !optState.result) return;
  $("opt_status").textContent = "saving weights_opt.json...";
  try {
    const r = await fetch(`/api/optimise_weights/${optState.jobId}/apply`, { method: "POST" });
    if (!r.ok) {
      $("opt_status").textContent = "apply failed: " + (await r.text());
      return;
    }
    const data = await r.json();
    $("opt_status").textContent = `wrote ${data.written}`;
    // Push the recommended weights into the live sliders so the existing
    // K_opt readout + cost plot update immediately. Use display-normalised
    // (max=1.0) values -- they're in the same range the sliders expect.
    if (bdState.analysis && optState.result.weights_display) {
      for (const [name, v] of Object.entries(optState.result.weights_display)) {
        bdState.weights[name] = v;
      }
      renderBdWeightSliders();
      renderBdCostAndKOpt();
    }
  } catch (e) {
    $("opt_status").textContent = "apply error: " + e;
  }
}

// ============================ Live Map ============================
// Upload a sliced ASCII .gcode, print it, and map the live nozzle force
// onto a 2D per-layer preview. Placement is purely client-side: force
// samples (force_batch) interpolated onto streamed toolhead position
// (pos_*_batch) -- both ride the same recv_monotonic host clock, so they
// are inherently time-aligned (no print-start anchor, no lag guesswork).
// The gcode is parsed server-side for the geometry backdrop + feature
// labels; the time-model cross-check (telemetry vs plan) is computed on
// the saved run and shown as a badge.

// Force colour scale: blue (low load) -> grey -> red (high load).
const LM_FORCE_SCALE = [
  [0.0, "#3b4cc0"], [0.25, "#7a9cf5"], [0.5, "#dcdcdc"],
  [0.75, "#f49a7b"], [1.0, "#b40426"],
];

// Canonical colours for the common PrusaSlicer/Cura feature types. Anything
// not listed falls back to a stable hash colour.
const LM_FEATURE_COLORS = {
  "External perimeter": "#f7931e", "Perimeter": "#ffd166",
  "Internal perimeter": "#ffd166", "Overhang perimeter": "#e85d75",
  "Solid infill": "#58a6ff", "Internal infill": "#3fb950", "Infill": "#3fb950",
  "Top solid infill": "#a371f7", "Bridge infill": "#39c5cf",
  "Gap fill": "#db61a2", "Skirt/Brim": "#d29922", "Skirt": "#d29922",
  "Brim": "#d29922", "Support material": "#8b949e",
  "Support material interface": "#b0b8c0", "Custom": "#7d8590",
  "WALL-OUTER": "#f7931e", "WALL-INNER": "#ffd166", "FILL": "#3fb950",
  "SKIN": "#58a6ff", "SUPPORT": "#8b949e",
};

function lmFeatureColor(name) {
  if (LM_FEATURE_COLORS[name]) return LM_FEATURE_COLORS[name];
  // stable hash -> hue
  let h = 0;
  for (let i = 0; i < (name || "").length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360}, 60%, 60%)`;
}

const lmap = {
  mode: null,          // 'live' | 'review' | null
  summary: null,       // {n_layers, layer_zs, features, bbox, force_lo, force_hi, feature_stats, cross_check, ...}
  reviewFile: null,    // saved npz filename when mode === 'review'
  geomCache: {},       // layerIdx -> polylines (pending/live backdrop)
  reviewCache: {},     // layerIdx -> {polylines, points} (saved-run review)
  // Live accumulation: ONE time-ordered toolhead path (x,y,force). We no
  // longer split by pos_z -- on this hardware pos_z carries MBL/raw-encoder
  // noise (±168 mm) and mis-assigns layers. Live shows the full path as a
  // connected force-coloured line; per-layer detail comes from the saved-run
  // review (server arc-length matched to the gcode).
  livePath: { x: [], y: [], force: [] },
  posX: { t: [], v: [] }, posY: { t: [], v: [] }, posZ: { t: [], v: [] },
  posCap: 6000,        // rolling cap on each position buffer
  ptCap: 60000,        // cap on the accumulated live path
  curLayer: 0,
  liveLayer: 0,
  followLive: true,
  colorMode: "force",
  posMode: "snapped",  // 'snapped' (project onto gcode line) | 'raw' (measured)
  needRender: false,
  initialized: false,
};

function lmResetLive() {
  lmap.livePath = { x: [], y: [], force: [] };
  lmap.posX = { t: [], v: [] };
  lmap.posY = { t: [], v: [] };
  lmap.posZ = { t: [], v: [] };
  lmap.liveLayer = 0;
}

// Interpolate v at time t in a (sorted) {t,v} buffer; clamp at the ends.
function _lerpAt(t, ts, vs) {
  const n = ts.length;
  if (n === 0) return null;
  if (t <= ts[0]) return vs[0];
  if (t >= ts[n - 1]) return vs[n - 1];
  let lo = 0, hi = n;
  while (lo < hi) { const m = (lo + hi) >> 1; if (ts[m] < t) lo = m + 1; else hi = m; }
  const t0 = ts[lo - 1], t1 = ts[lo], v0 = vs[lo - 1], v1 = vs[lo];
  const a = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
  return v0 + (v1 - v0) * a;
}

function _percentile(arr, p) {
  const a = arr.filter((v) => Number.isFinite(v)).slice().sort((x, y) => x - y);
  if (!a.length) return null;
  const i = Math.min(a.length - 1, Math.max(0, Math.round((p / 100) * (a.length - 1))));
  return a[i];
}

function lmLayerOfZ(z) {
  const zs = (lmap.summary && lmap.summary.layer_zs) || [];
  if (!zs.length || z === null || z === undefined || !Number.isFinite(z)) return 0;
  let best = 0, bestD = Infinity;
  for (let i = 0; i < zs.length; i++) {
    const d = Math.abs(zs[i] - z);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function lmapIngestPos(axis, ts, vs) {
  if (lmap.mode !== "live" || !lmap.summary) return;
  if (!Array.isArray(ts) || !Array.isArray(vs)) return;
  const buf = axis === "x" ? lmap.posX : axis === "y" ? lmap.posY : lmap.posZ;
  const n = Math.min(ts.length, vs.length);
  for (let i = 0; i < n; i++) {
    // keep strictly increasing time (server already clips, but be safe)
    if (buf.t.length && ts[i] <= buf.t[buf.t.length - 1]) continue;
    buf.t.push(ts[i]); buf.v.push(vs[i]);
  }
  if (buf.t.length > lmap.posCap) {
    const drop = buf.t.length - lmap.posCap;
    buf.t.splice(0, drop); buf.v.splice(0, drop);
  }
}

function lmapIngestForce(ts, vs) {
  if (lmap.mode !== "live" || !lmap.summary) return;
  const path = lmap.livePath;
  const n = Math.min(ts.length, vs.length);
  for (let i = 0; i < n; i++) {
    const t = ts[i];
    const x = _lerpAt(t, lmap.posX.t, lmap.posX.v);
    const y = _lerpAt(t, lmap.posY.t, lmap.posY.v);
    if (x === null || y === null) continue;  // no position yet
    if (path.x.length < lmap.ptCap) {
      path.x.push(x); path.y.push(y); path.force.push(vs[i]);
    }
  }
  lmapRequestRender();
}

function lmapRequestRender() { lmap.needRender = true; }

function lmapTick() {
  if (!lmap.needRender) return;
  if (!document.body.classList.contains("mode-livemap")) return;
  lmap.needRender = false;
  lmapRender();
}

async function lmapGeometry(layerIdx) {
  // Returns {polylines, points?} for a layer. Review pulls geometry+points
  // from the server (paged); live pulls only the backdrop polylines.
  if (lmap.mode === "review") {
    if (lmap.reviewCache[layerIdx]) return lmap.reviewCache[layerIdx];
    try {
      const r = await fetch(
        `/api/livemap/runs/${encodeURIComponent(lmap.reviewFile)}/layer/${layerIdx}`);
      if (!r.ok) return null;
      const d = await r.json();
      lmap.reviewCache[layerIdx] = d;
      return d;
    } catch (_) { return null; }
  }
  if (lmap.geomCache[layerIdx]) return { polylines: lmap.geomCache[layerIdx] };
  try {
    const r = await fetch(`/api/livemap/pending/layer/${layerIdx}`);
    if (!r.ok) return { polylines: [] };
    const d = await r.json();
    lmap.geomCache[layerIdx] = d.polylines || [];
    return { polylines: d.polylines || [] };
  } catch (_) { return { polylines: [] }; }
}

function _lmBackdropTraces(polylines) {
  // In force mode, one grey trace (all polylines, null-separated). In feature
  // mode, one coloured trace per feature.
  if (lmap.colorMode === "feature") {
    const byFeat = {};
    for (const p of polylines) {
      const f = p.feature || "(none)";
      if (!byFeat[f]) byFeat[f] = { x: [], y: [] };
      const g = byFeat[f];
      for (let i = 0; i < p.x.length; i++) { g.x.push(p.x[i]); g.y.push(p.y[i]); }
      g.x.push(null); g.y.push(null);
    }
    return Object.keys(byFeat).map((f) => ({
      x: byFeat[f].x, y: byFeat[f].y, type: "scattergl", mode: "lines",
      line: { color: lmFeatureColor(f), width: 2 }, name: f,
      hoverinfo: "name",
    }));
  }
  const X = [], Y = [];
  for (const p of polylines) {
    for (let i = 0; i < p.x.length; i++) { X.push(p.x[i]); Y.push(p.y[i]); }
    X.push(null); Y.push(null);
  }
  return [{
    x: X, y: Y, type: "scattergl", mode: "lines",
    line: { color: "#3a4250", width: 1.4 }, hoverinfo: "skip", name: "gcode",
  }];
}

// ---- continuous force-coloured lines (gcode-preview style) ----
const LM_BUCKETS = 24;

function _lmHexToRgb(hex) {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

// Sample LM_FORCE_SCALE at fraction f in [0,1] -> "rgb(r,g,b)".
function _lmScaleColor(f) {
  f = Math.max(0, Math.min(1, f));
  const s = LM_FORCE_SCALE;
  for (let i = 1; i < s.length; i++) {
    if (f <= s[i][0]) {
      const p0 = s[i - 1][0], p1 = s[i][0];
      const a = p1 > p0 ? (f - p0) / (p1 - p0) : 0;
      const c0 = _lmHexToRgb(s[i - 1][1]), c1 = _lmHexToRgb(s[i][1]);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * a);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * a);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * a);
      return `rgb(${r},${g},${b})`;
    }
  }
  const last = _lmHexToRgb(s[s.length - 1][1]);
  return `rgb(${last[0]},${last[1]},${last[2]})`;
}

// Bucket force-coloured segments into LM_BUCKETS null-separated polyline
// traces -- a handful of traces, continuous lines, adjacent segments meet at
// shared endpoints (how slicer web previews draw gradient extrusion paths).
function _lmBucketLineTraces(x0, y0, x1, y1, fvals, cmin, cmax) {
  const span = (cmax > cmin) ? (cmax - cmin) : 1;
  const bx = [], by = [];
  for (let b = 0; b < LM_BUCKETS; b++) { bx.push([]); by.push([]); }
  for (let i = 0; i < x0.length; i++) {
    const v = fvals[i];
    if (v === null || v === undefined || !Number.isFinite(v)) continue;
    let b = Math.round(((v - cmin) / span) * (LM_BUCKETS - 1));
    b = Math.max(0, Math.min(LM_BUCKETS - 1, b));
    bx[b].push(x0[i], x1[i], null);
    by[b].push(y0[i], y1[i], null);
  }
  const traces = [];
  for (let b = 0; b < LM_BUCKETS; b++) {
    if (!bx[b].length) continue;
    traces.push({
      x: bx[b], y: by[b], type: "scattergl", mode: "lines",
      line: { color: _lmScaleColor(b / (LM_BUCKETS - 1)), width: 3 },
      hoverinfo: "skip", showlegend: false,
    });
  }
  return traces;
}

// Feature-coloured segments: one line trace per feature.
function _lmFeatureSegTraces(x0, y0, x1, y1, feat, features) {
  const byF = {};
  for (let i = 0; i < x0.length; i++) {
    const name = features[feat[i]] || "(none)";
    if (!byF[name]) byF[name] = { x: [], y: [] };
    byF[name].x.push(x0[i], x1[i], null);
    byF[name].y.push(y0[i], y1[i], null);
  }
  return Object.keys(byF).map((name) => ({
    x: byF[name].x, y: byF[name].y, type: "scattergl", mode: "lines",
    line: { color: lmFeatureColor(name), width: 3 }, name, hoverinfo: "name",
  }));
}

// Distinct palette cycled by segment index so consecutive segments always
// differ -- makes every individual gcode segment (and its length) visible, and
// is the way to confirm none are dropped.
const LM_SEG_PALETTE = (() => {
  const out = [];
  const N = 18;
  for (let i = 0; i < N; i++) {
    // bit-reverse the hue order so neighbouring indices are far apart in hue
    let r = 0, x = i;
    for (let b = 0; b < 5; b++) { r = (r << 1) | (x & 1); x >>= 1; }
    out.push(`hsl(${Math.round((r % N) * 360 / N)},72%,${i % 2 ? 63 : 47}%)`);
  }
  return out;
})();

function _lmRandomSegTraces(x0, y0, x1, y1, groupIds) {
  const P = LM_SEG_PALETTE.length;
  const bx = [], by = [];
  for (let b = 0; b < P; b++) { bx.push([]); by.push([]); }
  for (let i = 0; i < x0.length; i++) {
    // colour by gcode move (groupIds) so a whole move is one colour and
    // adjacent moves differ -- you see each gcode line's extent within the
    // dense per-sample path. Falls back to per-sub-segment index.
    const b = ((groupIds ? groupIds[i] : i) % P + P) % P;
    bx[b].push(x0[i], x1[i], null);
    by[b].push(y0[i], y1[i], null);
  }
  const traces = [];
  for (let b = 0; b < P; b++) {
    if (!bx[b].length) continue;
    traces.push({
      x: bx[b], y: by[b], type: "scattergl", mode: "lines",
      line: { color: LM_SEG_PALETTE[b], width: 3 }, hoverinfo: "skip", showlegend: false,
    });
  }
  return traces;
}

// Load colour-scale range: honour the user's min/max inputs, else the supplied
// auto (percentile) bounds.
function _lmScaleRange(autoMin, autoMax) {
  let cmin = autoMin, cmax = autoMax;
  const im = parseFloat($("lm_scale_min").value);
  const ix = parseFloat($("lm_scale_max").value);
  if (Number.isFinite(im) && Number.isFinite(ix) && ix > im) { cmin = im; cmax = ix; }
  if (!(cmax > cmin)) cmax = cmin + 1;
  return { cmin, cmax };
}

// Connect ordered per-sample points into sub-segments. The colour of each
// sub-segment is its END sample (so a single gcode line carries its full
// per-reading gradient). A jump > maxJump mm (a travel between extrusions)
// breaks the line.
function _lmPointSubsegs(pts, maxJump) {
  const n = pts.x.length;
  const x0 = [], y0 = [], x1 = [], y1 = [], f = [], feat = [], stroke = [];
  const mj2 = maxJump * maxJump;
  // Review: break where the server flagged a real travel (`brk`), so the line
  // stays connected across packet-loss gaps on a continuous extrusion. Live:
  // no flags, fall back to a distance jump.
  const hasBrk = Array.isArray(pts.brk);
  let s = 0;  // stroke id: increments at each break (a continuous run)
  for (let i = 1; i < n; i++) {
    let brk;
    if (hasBrk) {
      brk = pts.brk[i];
    } else {
      const dx = pts.x[i] - pts.x[i - 1], dy = pts.y[i] - pts.y[i - 1];
      brk = dx * dx + dy * dy > mj2;
    }
    if (brk) { s++; continue; }
    x0.push(pts.x[i - 1]); y0.push(pts.y[i - 1]);
    x1.push(pts.x[i]); y1.push(pts.y[i]);
    f.push(pts.f[i]);
    feat.push(pts.feat ? pts.feat[i] : 0);
    stroke.push(s);
  }
  return { x0, y0, x1, y1, f, feat, stroke };
}

// Invisible marker trace that exists only to render the force colorbar
// (scattergl line traces can't carry one themselves).
function _lmColorbarTrace(cmin, cmax) {
  return {
    x: [null], y: [null], type: "scattergl", mode: "markers",
    marker: {
      colorscale: LM_FORCE_SCALE, cmin, cmax, color: [cmin],
      showscale: true, size: 0.1,
      colorbar: { title: { text: "load", side: "right" }, thickness: 12, len: 0.9 },
    },
    hoverinfo: "skip", showlegend: false,
  };
}

// Faint grey backdrop of a layer's full extruding path, so moves that caught
// no sample are still visible under the coloured lines.
function _lmBackdropFaint(polylines) {
  const X = [], Y = [];
  for (const p of polylines) {
    for (let i = 0; i < p.x.length; i++) { X.push(p.x[i]); Y.push(p.y[i]); }
    X.push(null); Y.push(null);
  }
  if (!X.length) return [];
  return [{
    x: X, y: Y, type: "scattergl", mode: "lines",
    line: { color: "#2a313c", width: 1 }, hoverinfo: "skip", showlegend: false,
  }];
}

// Live path -> force-coloured connected line. Consecutive samples form
// segments; a jump > 3 mm (a travel) breaks the line so we don't draw
// straight lines across the part.
function _lmLivePathTraces(path, cmin, cmax) {
  const n = path.x.length;
  if (n < 2) return [];
  const x0 = [], y0 = [], x1 = [], y1 = [], fv = [];
  for (let i = 1; i < n; i++) {
    const dx = path.x[i] - path.x[i - 1], dy = path.y[i] - path.y[i - 1];
    if (dx * dx + dy * dy > 9) continue;  // travel jump -> break
    x0.push(path.x[i - 1]); y0.push(path.y[i - 1]);
    x1.push(path.x[i]); y1.push(path.y[i]);
    fv.push(path.force[i]);
  }
  return _lmBucketLineTraces(x0, y0, x1, y1, fv, cmin, cmax);
}

async function lmapRender() {
  if (!lmap.summary) return;
  const plotDiv = $("lm_plot");
  if (!plotDiv) return;
  const L = lmap.curLayer;
  const review = lmap.mode === "review";
  const traces = [];
  let legendTxt = "";
  const sw = $("lm_scale_wrap");
  if (sw) sw.style.display = (lmap.colorMode === "force") ? "" : "none";

  if (review) {
    const geo = await lmapGeometry(L);
    const polylines = (geo && geo.polylines) || [];
    traces.push(..._lmBackdropFaint(polylines));   // faint full-layer path
    const pts0 = (geo && geo.pts) || null;
    // 'measured' shows the raw toolhead position; 'snapped' (default) projects
    // each reading onto the nearest gcode line (straight, matches the preview).
    const pts = (pts0 && lmap.posMode === "raw" && pts0.xr)
      ? { x: pts0.xr, y: pts0.yr, f: pts0.f, feat: pts0.feat, brk: pts0.brk }
      : pts0;
    if (pts && pts.x && pts.x.length > 1) {
      const ss = _lmPointSubsegs(pts, 3.0);   // one sub-segment per consecutive sample
      if (lmap.colorMode === "feature") {
        traces.push(..._lmFeatureSegTraces(
          ss.x0, ss.y0, ss.x1, ss.y1, ss.feat, lmap.summary.features || []));
        legendTxt = "colour = feature type";
      } else if (lmap.colorMode === "random") {
        traces.push(..._lmRandomSegTraces(ss.x0, ss.y0, ss.x1, ss.y1, ss.stroke));
        legendTxt = `colour = random per stroke (run between travels) · ${pts.x.length} samples at the real toolhead position`;
      } else {
        const { cmin, cmax } = _lmScaleRange(lmap.summary.force_lo, lmap.summary.force_hi);
        traces.push(..._lmBucketLineTraces(ss.x0, ss.y0, ss.x1, ss.y1, ss.f, cmin, cmax));
        traces.push(_lmColorbarTrace(cmin, cmax));
        legendTxt = `colour = nozzle load (tared) · ${pts.x.length} samples at the real toolhead position`;
      }
    } else {
      const xc = lmap.summary.cross_check;
      legendTxt = (xc && !xc.available && xc.reason)
        ? "⚠ " + xc.reason
        : "no mapped samples on this layer.";
    }
  } else {
    // LIVE: full accumulating toolhead path, coloured by raw load.
    const path = lmap.livePath;
    let amin = _percentile(path.force, 2), amax = _percentile(path.force, 98);
    if (amin === null) { amin = 0; amax = 1; }
    const { cmin, cmax } = _lmScaleRange(amin, amax);
    if (path.x.length >= 2) {
      traces.push(..._lmLivePathTraces(path, cmin, cmax));
      traces.push(_lmColorbarTrace(cmin, cmax));
    }
    legendTxt = path.x.length
      ? "live: full toolhead path coloured by raw load — per-layer review opens when the run finishes"
      : "waiting for nozzle-force + position telemetry…";
  }

  const [bx0, bx1, by0, by1] = lmap.summary.bbox || [0, 1, 0, 1];
  const padX = Math.max((bx1 - bx0) * 0.05, 2);
  const padY = Math.max((by1 - by0) * 0.05, 2);
  const layout = {
    margin: { l: 44, r: 10, t: 8, b: 36 },
    paper_bgcolor: "#161b22", plot_bgcolor: "#0d1117",
    font: { color: "#e6edf3", size: 11 },
    // Constant per run/layer so Plotly KEEPS the user's zoom/pan across data
    // updates (live) and layer switches -- no more snapping back to full view.
    uirevision: review ? "rev:" + lmap.reviewFile : "live:" + (lmap.summary.name || ""),
    xaxis: { gridcolor: "#21262d", range: [bx0 - padX, bx1 + padX], title: "X (mm)", constrain: "domain" },
    yaxis: {
      gridcolor: "#21262d", range: [by0 - padY, by1 + padY], title: "Y (mm)",
      scaleanchor: "x", scaleratio: 1,
    },
    showlegend: review && lmap.colorMode === "feature",
    legend: { x: 1.02, y: 1, font: { size: 10 }, bgcolor: "rgba(0,0,0,0)" },
  };
  Plotly.react(plotDiv, traces, layout, { displayModeBar: false, responsive: true });

  const z = (lmap.summary.layer_zs && lmap.summary.layer_zs[L]);
  if (review) {
    const npts = (lmap.reviewCache[L] && lmap.reviewCache[L].n_points) || 0;
    $("lm_layer_label").textContent =
      `layer ${L + 1}/${lmap.summary.n_layers}` +
      (z !== undefined ? ` · z=${Number(z).toFixed(2)} mm` : "") + ` · ${npts} samples`;
  } else {
    $("lm_layer_label").textContent =
      `live · ${lmap.livePath.x.length} samples` + (lmap.followLive ? " · ● recording" : "");
  }
  $("lm_legend").textContent = legendTxt;
}

function _lmSyncSlider() {
  const s = $("lm_layer_slider");
  if (s) s.value = String(lmap.curLayer);
}

function lmapSetLayer(idx, fromUser) {
  if (!lmap.summary) return;
  lmap.curLayer = Math.max(0, Math.min(lmap.summary.n_layers - 1, idx));
  if (fromUser) lmap.followLive = false;
  _lmSyncSlider();
  lmapRender();
}

function lmapStep(d) { lmapSetLayer(lmap.curLayer + d, true); }

function lmapJumpLive() {
  lmap.followLive = true;
  lmapSetLayer(lmap.liveLayer, false);
}

function lmapSetupFromSummary(summary, mode, reviewFile) {
  lmap.summary = summary;
  lmap.mode = mode;
  lmap.reviewFile = reviewFile || null;
  lmap.geomCache = {};
  lmap.reviewCache = {};
  lmap.curLayer = 0;
  lmap.followLive = mode === "live";
  if (mode === "live") lmResetLive();
  const s = $("lm_layer_slider");
  if (s) { s.min = "0"; s.max = String(Math.max(0, summary.n_layers - 1)); s.value = "0"; }
  // Seed the load-scale inputs with the auto (percentile) bounds so the user
  // sees the current scale and can tweak it; cleared for live (no tare yet).
  const smin = $("lm_scale_min"), smax = $("lm_scale_max");
  if (smin && smax) {
    if (Number.isFinite(summary.force_lo) && Number.isFinite(summary.force_hi)) {
      smin.value = String(Math.round(summary.force_lo));
      smax.value = String(Math.round(summary.force_hi));
    } else { smin.value = ""; smax.value = ""; }
  }
  _lmRenderSummaryText(summary);
  _lmRenderFeatureTable(summary.feature_stats || null);
  _lmRenderDivergence(summary.cross_check || null);
  lmapRender();
}

function _lmRenderSummaryText(s) {
  const el = $("lm_summary");
  if (!el) return;
  const mins = s.est_print_s ? (s.est_print_s / 60).toFixed(0) : "?";
  const feats = (s.features || []).length;
  el.innerHTML =
    `<strong>${s.name || s.filename || "run"}</strong> — ` +
    `${s.n_layers} layers · ${s.n_extruding ?? s.n_moves ?? "?"} extruding moves · ` +
    `${feats} feature types · ~${mins} min est. print time`;
}

function _lmRenderFeatureTable(stats) {
  const tbl = $("lm_feature_table");
  if (!tbl) return;
  if (!stats || !Object.keys(stats).length) {
    tbl.innerHTML = '<tr><td class="meta">no feature stats yet — finish or open a run</td></tr>';
    return;
  }
  const rows = [`<tr><th>feature</th><th>median load</th><th>mean</th><th>p90</th><th>n</th></tr>`];
  const entries = Object.entries(stats).sort((a, b) => b[1].p50 - a[1].p50);
  for (const [name, st] of entries) {
    const sw = `<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${lmFeatureColor(name)};margin-right:6px;"></span>`;
    rows.push(
      `<tr><td>${sw}${name}</td>` +
      `<td style="text-align:right">${fmt(st.p50, 1)}</td>` +
      `<td style="text-align:right">${fmt(st.mean, 1)}</td>` +
      `<td style="text-align:right">${fmt(st.p90, 1)}</td>` +
      `<td style="text-align:right">${st.n}</td></tr>`);
  }
  tbl.innerHTML = rows.join("");
}

function _lmRenderDivergence(xc) {
  const el = $("lm_divergence");
  if (!el) return;
  if (!xc || !xc.available) {
    // Mapping produced nothing -- show WHY (e.g. no position telemetry).
    el.innerHTML = (xc && xc.reason)
      ? `<span style="color:#f85149">⚠ ${xc.reason}</span>` : "";
    return;
  }
  const colors = { good: "#2ea043", warn: "#d29922", poor: "#f85149" };
  const c = colors[xc.agreement] || "#8b949e";
  el.innerHTML =
    `<span style="color:${c}">plan vs telemetry: ${xc.agreement}</span> ` +
    `(scale ${fmt(xc.time_scale, 2)}×, kept ${fmt((xc.kept_fraction || 0) * 100, 0)}%` +
    (xc.n_travel_gcode != null ? `, ${xc.n_travel_gcode} travel pts masked` : "") +
    `)`;
}

async function lmapUpload() {
  const inp = $("lm_file");
  const f = inp && inp.files && inp.files[0];
  if (!f) { alert("Choose a sliced .gcode file first."); return; }
  $("lm_summary").textContent = "uploading & parsing…";
  $("lm_run").disabled = true;
  try {
    const fd = new FormData();
    fd.append("file", f);
    const r = await fetch("/api/livemap/upload", { method: "POST", body: fd });
    if (!r.ok) {
      let msg = await r.text();
      try { msg = JSON.parse(msg).detail || msg; } catch (_) {}
      $("lm_summary").textContent = "upload failed: " + msg;
      return;
    }
    const s = await r.json();
    lmapSetupFromSummary(s, "live", null);
    $("lm_run").disabled = false;
  } catch (e) {
    $("lm_summary").textContent = "upload error: " + e;
  }
}

async function startLiveMap() {
  const r = await fetch("/api/livemap/run", { method: "POST" });
  if (!r.ok) {
    let msg = await r.text();
    try { msg = JSON.parse(msg).detail || msg; } catch (_) {}
    alert("Start failed: " + msg);
  }
}

async function cancelLiveMap() {
  await fetch("/api/livemap/cancel", { method: "POST" });
}

let lmLastRunState = null;
let lmLastReviewed = null;  // saved_filename we've already auto-opened for review

function renderLiveMapRun(run) {
  if (!run) return;
  $("lm_state").textContent = run.state;
  $("lm_msg").textContent = run.message || "";
  $("lm_pct").textContent = (run.progress_pct ?? 0).toFixed(0);

  const wasIdle = lmLastRunState === null || lmLastRunState === "idle"
    || lmLastRunState === "done" || lmLastRunState === "error";
  const nowActive = run.state === "preparing" || run.state === "running";
  if (wasIdle && nowActive) {
    // A run just started. If we don't have the summary (e.g. page reloaded
    // mid-print), pull it from the pending endpoint so we can map live.
    lmResetLive();
    lmap.followLive = true;
    if (!lmap.summary) {
      fetch("/api/livemap/pending").then((r) => r.ok ? r.json() : null).then((s) => {
        if (s) lmapSetupFromSummary(s, "live", null);
      });
    } else {
      lmap.mode = "live";
    }
  }
  lmLastRunState = run.state;

  if (run.state === "done" && run.saved_filename && lmLastReviewed !== run.saved_filename) {
    // Run finished: switch straight into the per-layer REVIEW view (server
    // arc-length matched to the gcode -- the good, layer-resolved render),
    // and refresh the saved-runs dropdown.
    lmLastReviewed = run.saved_filename;
    refreshLiveMapRuns();
    fetch(`/api/livemap/runs/${encodeURIComponent(run.saved_filename)}/analyse`, { method: "POST" })
      .then((r) => r.ok ? r.json() : null)
      .then((s) => {
        if (!s) return;
        lmapSetupFromSummary(s, "review", run.saved_filename);
        if (lmap.summary) lmap.summary.name = run.name || s.filename;
      });
  }
}

// ---- review (saved runs) ----
const lmRunsByFilename = {};

async function refreshLiveMapRuns() {
  const sel = $("lm_runs_select");
  if (!sel) return;
  try {
    const r = await fetch("/api/livemap/runs");
    if (!r.ok) return;
    const data = await r.json();
    const prev = sel.value;
    sel.innerHTML = '<option value="">— select —</option>';
    for (const k of Object.keys(lmRunsByFilename)) delete lmRunsByFilename[k];
    for (const run of (data.runs || [])) {
      lmRunsByFilename[run.filename] = run;
      const opt = document.createElement("option");
      opt.value = run.filename;
      const date = _formatLocalTimestamp(run.mtime_unix);
      const nm = run.gcode_name ? ` ${run.gcode_name}` : "";
      opt.textContent = `${date}${nm}  ${run.n_layers} layers  (${run.filename})`;
      sel.appendChild(opt);
    }
    sel.value = prev;
  } catch (_) { /* next refresh handles it */ }
}

async function loadLiveMapReview(filename) {
  $("lm_run_meta").textContent = "";
  if (!filename) return;
  try {
    const r = await fetch(`/api/livemap/runs/${encodeURIComponent(filename)}/analyse`, { method: "POST" });
    if (!r.ok) { alert("Open failed: " + (await r.text())); return; }
    const s = await r.json();
    lmapSetupFromSummary(s, "review", filename);
    const meta = lmRunsByFilename[filename];
    if (meta) {
      $("lm_run_meta").textContent =
        `${meta.n_force} force samples · ${(meta.duration_s / 60).toFixed(1)} min captured`;
    }
  } catch (e) {
    alert("Open error: " + e);
  }
}

// ---- wire up ----
document.addEventListener("DOMContentLoaded", () => {
  $("btn_save").onclick = saveConfig;
  $("btn_preview").onclick = previewGcode;
  $("btn_run").onclick = startRun;
  $("btn_cancel").onclick = cancelRun;
  $("btn_copy").onclick = copyPressureAdvance;

  // Replay picker
  $("btn_replay_refresh").onclick = refreshRunsList;
  $("runs_select").onchange = (ev) => {
    _renderRunMeta(ev.target.value);
    loadReplay(ev.target.value);
  };
  refreshRunsList();

  // XLSX export. The dropdown button exports whatever saved run is
  // selected (no need to load it first); the results-card button exports
  // whatever the page is currently showing -- a loaded replay if there is
  // one, otherwise the just-finished live run held in server memory.
  const btnExportRun = $("btn_export_run");
  if (btnExportRun) btnExportRun.onclick = () => {
    const fn = $("runs_select").value;
    if (!fn) { alert("Select a saved run from the dropdown first."); return; }
    downloadXlsx(
      `/api/runs/${encodeURIComponent(fn)}/export.xlsx`,
      fn.replace(/\.npz$/, ".xlsx"),
    );
  };
  const btnExportCurrent = $("btn_export_current");
  if (btnExportCurrent) btnExportCurrent.onclick = () => {
    if (loadedReplayFilename) {
      downloadXlsx(
        `/api/runs/${encodeURIComponent(loadedReplayFilename)}/export.xlsx`,
        loadedReplayFilename.replace(/\.npz$/, ".xlsx"),
      );
    } else {
      downloadXlsx("/api/current_run/export.xlsx", "current_run.xlsx");
    }
  };

  // ---- Max Flow test ----
  $("mode_pa").onclick = () => setMode("pa");
  $("mode_flow").onclick = () => setMode("flow");
  $("mode_livemap").onclick = () => setMode("livemap");
  $("mode_probe").onclick = () => setMode("probe");

  // ---- Live Map ----
  $("lm_upload").onclick = lmapUpload;
  $("lm_run").onclick = startLiveMap;
  $("lm_cancel").onclick = cancelLiveMap;
  $("lm_prev").onclick = () => lmapStep(-1);
  $("lm_next").onclick = () => lmapStep(1);
  $("lm_live").onclick = lmapJumpLive;
  $("lm_layer_slider").oninput = (ev) => lmapSetLayer(parseInt(ev.target.value, 10), true);
  $("lm_color_mode").onchange = (ev) => { lmap.colorMode = ev.target.value; lmapRender(); };
  const lmPosMode = $("lm_pos_mode");
  if (lmPosMode) lmPosMode.onchange = (ev) => { lmap.posMode = ev.target.value; lmapRender(); };
  const lmScaleMin = $("lm_scale_min"), lmScaleMax = $("lm_scale_max");
  if (lmScaleMin) lmScaleMin.oninput = () => lmapRender();
  if (lmScaleMax) lmScaleMax.oninput = () => lmapRender();
  const lmScaleAuto = $("lm_scale_auto");
  if (lmScaleAuto) lmScaleAuto.onclick = () => {
    const s = lmap.summary;
    if (s && Number.isFinite(s.force_lo) && Number.isFinite(s.force_hi)) {
      lmScaleMin.value = String(Math.round(s.force_lo));
      lmScaleMax.value = String(Math.round(s.force_hi));
    } else { lmScaleMin.value = ""; lmScaleMax.value = ""; }
    lmapRender();
  };
  $("lm_runs_refresh").onclick = refreshLiveMapRuns;
  $("lm_runs_select").onchange = (ev) => loadLiveMapReview(ev.target.value);
  const lmExport = $("lm_export");
  if (lmExport) lmExport.onclick = () => {
    const fn = $("lm_runs_select").value;
    if (!fn) { alert("Select a saved Live Map run from the dropdown first."); return; }
    downloadXlsx(`/api/livemap/runs/${encodeURIComponent(fn)}/export.xlsx`,
      fn.replace(/\.npz$/, ".xlsx"));
  };
  refreshLiveMapRuns();
  setInterval(lmapTick, 200);
  $("btn_flow_save").onclick = saveConfig;
  $("btn_flow_preview").onclick = flowPreview;
  $("btn_flow_run").onclick = startFlow;
  $("btn_flow_cancel").onclick = cancelFlow;
  $("btn_flow_copy").onclick = copyFlowMax;
  $("btn_flow_replay_refresh").onclick = refreshFlowRunsList;
  $("flow_runs_select").onchange = (ev) => {
    _renderFlowRunMeta(ev.target.value);
    loadFlowReplay(ev.target.value);
  };
  const btnFlowExportRun = $("btn_flow_export_run");
  if (btnFlowExportRun) btnFlowExportRun.onclick = () => {
    const fn = $("flow_runs_select").value;
    if (!fn) { alert("Select a saved flow run from the dropdown first."); return; }
    downloadXlsx(
      `/api/flow/runs/${encodeURIComponent(fn)}/export.xlsx`,
      fn.replace(/\.npz$/, ".xlsx"));
  };
  const btnFlowExportCurrent = $("btn_flow_export_current");
  if (btnFlowExportCurrent) btnFlowExportCurrent.onclick = () => {
    if (loadedFlowReplayFilename) {
      downloadXlsx(
        `/api/flow/runs/${encodeURIComponent(loadedFlowReplayFilename)}/export.xlsx`,
        loadedFlowReplayFilename.replace(/\.npz$/, ".xlsx"));
    } else {
      downloadXlsx("/api/flow/current_run/export.xlsx", "flow_run.xlsx");
    }
  };
  $("flow_prev").onclick = () => flowStep(-1);
  $("flow_next").onclick = () => flowStep(1);
  refreshFlowRunsList();

  // ---- Touch Probe ----
  $("btn_probe_save").onclick = saveConfig;
  $("btn_probe_preview").onclick = probePreview;
  $("btn_probe_run").onclick = startProbe;
  $("btn_probe_cancel").onclick = cancelProbe;
  $("btn_probe_replay_refresh").onclick = refreshProbeRunsList;
  $("probe_runs_select").onchange = (ev) => {
    _renderProbeRunMeta(ev.target.value);
    loadProbeReplay(ev.target.value);
  };
  const btnProbeExportRun = $("btn_probe_export_run");
  if (btnProbeExportRun) btnProbeExportRun.onclick = () => {
    const fn = $("probe_runs_select").value;
    if (!fn) { alert("Select a saved probe run from the dropdown first."); return; }
    downloadXlsx(
      `/api/probe/runs/${encodeURIComponent(fn)}/export.xlsx`,
      fn.replace(/\.npz$/, ".xlsx"));
  };
  const btnProbeExportCurrent = $("btn_probe_export_current");
  if (btnProbeExportCurrent) btnProbeExportCurrent.onclick = () => {
    if (loadedProbeReplayFilename) {
      downloadXlsx(
        `/api/probe/runs/${encodeURIComponent(loadedProbeReplayFilename)}/export.xlsx`,
        loadedProbeReplayFilename.replace(/\.npz$/, ".xlsx"));
    } else {
      downloadXlsx("/api/probe/current_run/export.xlsx", "probe_run.xlsx");
    }
  };
  $("probe_prev").onclick = () => probeStep(-1);
  $("probe_next").onclick = () => probeStep(1);
  refreshProbeRunsList();

  try {
    const m = localStorage.getItem("ppat_mode");
    if (m === "flow") setMode("flow");
    else if (m === "livemap") setMode("livemap");
    else if (m === "probe") setMode("probe");
  } catch (_) {}

  // Annotation card (only visible after a replay is loaded).
  const btnSaveUserK = $("btn_save_user_k");
  if (btnSaveUserK) btnSaveUserK.onclick = () => saveUserKOpt(false);
  const btnClearUserK = $("btn_clear_user_k");
  if (btnClearUserK) btnClearUserK.onclick = () => saveUserKOpt(true);

  // Weight optimiser panel.
  const btnOptRun = $("btn_opt_run");
  if (btnOptRun) btnOptRun.onclick = startOptimiser;
  const btnOptApply = $("btn_opt_apply");
  if (btnOptApply) btnOptApply.onclick = applyOptResult;
  refreshOptInfo();
  // Refresh annotated-run count whenever the replay list is reloaded
  // (the user may have just annotated a new run in another tab / via API).
  const origRefresh = refreshRunsList;
  // Reassigning `refreshRunsList` would break the closure already captured
  // by the dropdown handler; instead, piggyback by polling refreshOptInfo
  // when the user clicks the replay refresh button.
  $("btn_replay_refresh").addEventListener("click", refreshOptInfo);

  // bd_pressure browser: prev/next + arrow keys, overlay toggles.
  // Stepping wraps over K boundaries: at the last segment of K[i],
  // "next" advances to K[i+1] segment 0; at segment 0 of K[i], "prev"
  // jumps to K[i-1]'s last segment. Stops at the absolute first/last
  // segment in the sweep (no circular wrap -- the user said "go to the
  // next or previous one", not "loop forever").
  $("bd_prev").onclick = () => {
    if (bdState.selectedK === null) return;
    if (bdState.selectedSegIdx > 0) {
      bdState.selectedSegIdx -= 1;
    } else {
      // Find the previous K that has at least one segment.
      const perK = (bdState.analysis && bdState.analysis.bd_per_k) || [];
      const curIdx = perK.findIndex(
        (r) => Math.abs(r.k - bdState.selectedK) < 1e-6,
      );
      for (let j = curIdx - 1; j >= 0; j--) {
        const prevSegs = bdState.segmentsByK[perK[j].k] || [];
        if (prevSegs.length > 0) {
          bdState.selectedK = perK[j].k;
          bdState.selectedSegIdx = prevSegs.length - 1;
          break;
        }
      }
    }
    renderBdKList();
    renderBdSegment();
  };
  $("bd_next").onclick = () => {
    if (bdState.selectedK === null) return;
    const segs = _segmentsForSelectedK();
    if (bdState.selectedSegIdx < segs.length - 1) {
      bdState.selectedSegIdx += 1;
    } else {
      // Find the next K that has at least one segment.
      const perK = (bdState.analysis && bdState.analysis.bd_per_k) || [];
      const curIdx = perK.findIndex(
        (r) => Math.abs(r.k - bdState.selectedK) < 1e-6,
      );
      for (let j = curIdx + 1; j < perK.length; j++) {
        const nextSegs = bdState.segmentsByK[perK[j].k] || [];
        if (nextSegs.length > 0) {
          bdState.selectedK = perK[j].k;
          bdState.selectedSegIdx = 0;
          break;
        }
      }
    }
    renderBdKList();
    renderBdSegment();
  };
  document.addEventListener("keydown", (ev) => {
    // Don't hijack arrows while focused in an input/select.
    const tag = (ev.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "select" || tag === "textarea") return;
    // Route arrows to whichever browser the active mode shows.
    const flowMode = document.body.classList.contains("mode-flow");
    const livemapMode = document.body.classList.contains("mode-livemap");
    const probeMode = document.body.classList.contains("mode-probe");
    if (ev.key === "ArrowLeft") {
      if (livemapMode) lmapStep(-1);
      else if (probeMode) probeStep(-1);
      else (flowMode ? $("flow_prev") : $("bd_prev")).click();
      ev.preventDefault();
    } else if (ev.key === "ArrowRight") {
      if (livemapMode) lmapStep(1);
      else if (probeMode) probeStep(1);
      else (flowMode ? $("flow_next") : $("bd_next")).click();
      ev.preventDefault();
    }
  });
  for (const id of [
    "bd_overlay_transitions", "bd_overlay_levels", "bd_overlay_peaks",
    "bd_overlay_regions", "bd_overlay_areas", "bd_overlay_slope",
    "bd_overlay_labels",
  ]) {
    const el = $(id);
    if (el) el.onchange = () => renderBdSegment();
  }

  loadConfig();
  openWs();
  startDiagnosticsPoll();
});
