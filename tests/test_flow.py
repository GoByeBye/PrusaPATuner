"""Tests for the Max Flow test: gcode generation + breakdown detection."""
import numpy as np

from prusa_pa_tuner.flow_analysis import (
    _align_grid_phase,
    analyse_flow,
    detect_flow_sweep,
    flow_analysis_to_dict,
)
from prusa_pa_tuner.flow_gen import FlowRampParams, build_flow_ramp, flow_levels


def test_flow_levels_inclusive_grid():
    p = FlowRampParams(min_flow_mm3_s=5, max_flow_mm3_s=10, flow_step_mm3_s=1)
    assert flow_levels(p) == (5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
    # non-integer step, end inclusive when it lands on the grid
    p2 = FlowRampParams(min_flow_mm3_s=5, max_flow_mm3_s=6, flow_step_mm3_s=0.5)
    assert flow_levels(p2) == (5.0, 5.5, 6.0)


def test_flow_gcode_structure():
    p = FlowRampParams(
        min_flow_mm3_s=5, max_flow_mm3_s=8, flow_step_mm3_s=1,
        udp_host="10.0.0.9", udp_port=8500,
    )
    plan = build_flow_ramp(p)
    assert len(plan.segments) == 4
    g = plan.gcode
    assert "M334 10.0.0.9 8500" in g
    assert "M331 loadcell_value" in g
    assert "M331 pos_z" in g
    # one M117 FLOW marker per level
    assert g.count("M117 FLOW=") == 4
    # Z-marker for sweep_t0 anchoring
    assert "sweep marker UP" in g and "sweep marker DOWN" in g
    # segments carry increasing flow + a feed velocity
    flows = [s.flow_mm3_s for s in plan.segments]
    assert flows == [5.0, 6.0, 7.0, 8.0]
    assert all(s.feed_mm_s > 0 for s in plan.segments)
    # feed scales with flow / filament area
    area = np.pi * (p.filament_diameter / 2.0) ** 2
    assert abs(plan.segments[0].feed_mm_s - 5.0 / area) < 1e-9


def _synthesize(plan, *, dev_at, var_at, collapse_at, seed=0, offset=0.0):
    """Fabricate a force timeseries: power-law good region + upward break +
    rising variance + collapse, sampled at 180 Hz. `offset` adds a constant
    static-load baseline (the tare must remove it). The no-flow gap before
    the first level is filled at `offset` so the tare window has samples."""
    rng = np.random.default_rng(seed)
    a, b, c = 3.0, 0.55, 4.0
    fs = 180.0
    t_list, f_list = [], []
    # no-flow gap (sweep_t0 .. first level start): static load only
    gap = plan.segments[0].start_offset_s
    for i in range(int(gap * fs)):
        t_list.append(i / fs)
        f_list.append(offset + rng.normal(0, 0.3))
    for seg in plan.segments:
        q = seg.flow_mm3_s
        base = offset + a * q ** b + c
        if q >= dev_at:
            base += 0.8 * (q - dev_at) ** 2
        noise = 0.3 + (1.5 * (q - var_at) if q >= var_at else 0.0)
        if q >= collapse_at:
            base = offset + (base - offset) * 0.55
        t0 = seg.start_offset_s
        n = int(seg.duration_s * fs)
        for i in range(n):
            t_list.append(t0 + i / fs)
            f_list.append(base + rng.normal(0, noise))
    return np.array(t_list), np.array(f_list)


def test_flow_analysis_detects_breakdown():
    p = FlowRampParams(
        min_flow_mm3_s=5, max_flow_mm3_s=30, flow_step_mm3_s=1,
        dwell_s=2.0, settle_frac=0.5,
    )
    plan = build_flow_ramp(p)
    ft, fy = _synthesize(plan, dev_at=24, var_at=23, collapse_at=28)
    a = analyse_flow(sweep_t0=0.0, force_t=ft, force_y=fy, plan=plan)

    # power-law baseline recovered
    assert a.fit is not None
    assert abs(a.fit.b - 0.55) < 0.1
    assert a.fit.r_squared > 0.99

    # all three markers fire, in the research-predicted order:
    # variance onset <= mean deviation < collapse
    assert a.variance_onset_flow is not None
    assert a.deviation_flow is not None
    assert a.collapse_flow is not None
    assert a.variance_onset_flow <= a.deviation_flow < a.collapse_flow

    # recommended max is the earliest soft trigger, derated below it
    earliest = min(a.variance_onset_flow, a.deviation_flow)
    assert a.recommended_max_flow < earliest

    # dict is JSON-safe (no NaN/inf leaking through)
    d = flow_analysis_to_dict(a)
    assert d["recommended_max_flow"] == a.recommended_max_flow
    assert len(d["levels"]) == len(plan.segments)


def test_flow_analysis_no_breakdown_reports_top():
    """A clean run with no breakdown should report the top level, not crash."""
    p = FlowRampParams(min_flow_mm3_s=5, max_flow_mm3_s=15, flow_step_mm3_s=1)
    plan = build_flow_ramp(p)
    ft, fy = _synthesize(plan, dev_at=999, var_at=999, collapse_at=999)
    a = analyse_flow(sweep_t0=0.0, force_t=ft, force_y=fy, plan=plan)
    assert a.deviation_flow is None
    assert a.recommended_max_flow == 15.0


def test_flow_gcode_has_tare_hold():
    p = FlowRampParams(min_flow_mm3_s=5, max_flow_mm3_s=8, flow_step_mm3_s=1,
                       tare_dwell_s=1.5)
    plan = build_flow_ramp(p)
    assert ";FLOW_TUNER TARE" in plan.gcode
    assert "no-flow tare hold" in plan.gcode
    # first level starts after pre-roll (0.5) + tare hold (1.5)
    assert abs(plan.segments[0].start_offset_s - 2.0) < 1e-9


def test_flow_tare_removes_static_offset():
    """A large constant static load must be tared off so forces start near 0."""
    p = FlowRampParams(min_flow_mm3_s=5, max_flow_mm3_s=12, flow_step_mm3_s=1,
                       dwell_s=3.0, settle_frac=0.5, tare_dwell_s=1.5)
    plan = build_flow_ramp(p)
    ft, fy = _synthesize(plan, dev_at=999, var_at=999, collapse_at=999, offset=1500.0)
    a = analyse_flow(sweep_t0=0.0, force_t=ft, force_y=fy, plan=plan)
    # tare ~ the injected static load
    assert abs(a.tare - 1500.0) < 2.0
    # lowest-flow level's tared force is the small back-pressure, not ~1500
    lv0 = min(a.levels, key=lambda lv: lv.flow_mm3_s)
    expected = 3.0 * 5.0 ** 0.55 + 4.0  # ~11.3
    assert abs(lv0.force_mean - expected) < 1.5


def _synthesize_full_capture(plan, *, heat_s=40.0, tail_s=20.0, offset=1500.0,
                             seed=0):
    """A realistic capture: long idle heat-up head, a homing spike, the
    rising+collapsing sweep, an idle cooldown tail, and a post-sweep spike.
    The sweep is placed at t=heat_s; nothing about the planned timing /
    sweep_t0 tells you where it is -- detection must find it from the force."""
    rng = np.random.default_rng(seed)
    a, b, c = 3.0, 0.55, 4.0
    fs = 180.0
    segs = plan.segments
    n_steps = len(segs)
    dwell = plan.params.dwell_s
    t_list, f_list = [], []

    def emit(t_from, t_to, level_fn):
        n = int((t_to - t_from) * fs)
        for i in range(n):
            tt = t_from + i / fs
            t_list.append(tt)
            f_list.append(level_fn(tt))

    # idle heat-up head (with one homing-like spike)
    emit(0, heat_s, lambda tt: offset + rng.normal(0, 2)
         + (8000 if abs(tt - 8.0) < 0.05 else 0))
    # the sweep: N back-to-back steps
    for i, seg in enumerate(segs):
        q = seg.flow_mm3_s
        base = offset + a * q ** b + c
        if q >= 24:
            base += 0.8 * (q - 24) ** 2
        noise = 0.3 + (1.5 * (q - 23) if q >= 23 else 0.0)
        if q >= 28:
            base = offset + (base - offset) * 0.55
        t_from = heat_s + i * dwell
        emit(t_from, t_from + dwell, lambda tt, base=base, noise=noise:
             base + rng.normal(0, noise))
    # idle cooldown tail
    sweep_end = heat_s + n_steps * dwell
    emit(sweep_end, sweep_end + tail_s, lambda tt: offset + rng.normal(0, 2))
    # a brief post-sweep spike that must NOT be mistaken for the sweep
    spike_t = sweep_end + tail_s * 0.5
    t_arr = np.array(t_list)
    f_arr = np.array(f_list)
    f_arr[np.abs(t_arr - spike_t) < 0.1] += 9000
    return t_arr, f_arr, heat_s, sweep_end


def test_detect_sweep_in_realistic_capture():
    """Detection must find the sweep in a full capture and be immune to a
    wrong sweep_t0 and to idle/spike regions around it."""
    p = FlowRampParams(min_flow_mm3_s=4, max_flow_mm3_s=30, flow_step_mm3_s=2,
                       dwell_s=5.0, settle_frac=0.5)
    plan = build_flow_ramp(p)
    ft, fy, true_start, true_end = _synthesize_full_capture(plan, heat_s=40.0)

    det = detect_flow_sweep(ft, fy, len(plan.segments), p.dwell_s)
    assert det is not None
    start, end, width = det
    # end-anchored detection lands within a step of the truth
    assert abs(end - true_end) < p.dwell_s
    assert abs(start - true_start) < p.dwell_s
    assert abs(width - p.dwell_s) < 1e-6

    # full analysis, given a DELIBERATELY WRONG sweep_t0, still aligns +
    # tares off the big static offset
    a = analyse_flow(sweep_t0=0.0, force_t=ft, force_y=fy, plan=plan)
    assert a.detect_method.startswith("force-activity")
    assert abs(a.tare - 1500.0) < 5.0
    lv0 = min(a.levels, key=lambda lv: lv.flow_mm3_s)
    assert abs(lv0.force_mean - (3.0 * 4 ** 0.55 + 4.0)) < 2.0


def test_phase_alignment_recovers_rise_boundaries():
    """Given a staircase with rise ramps at known boundaries and a grid that
    is deliberately shifted LATE, the phase aligner must pull it back so the
    rises land at cell boundaries (out of the measured window)."""
    fs = 180.0
    W = 5.0
    N = 10
    rise_s = 0.6
    true_start = 20.0
    t = np.arange(0.0, true_start + N * W + 5.0, 1.0 / fs)
    y = np.zeros_like(t)
    plateaus = [400.0 * (i + 1) for i in range(N)]  # rising staircase
    for i in range(N):
        b = true_start + i * W
        prev = plateaus[i - 1] if i > 0 else 0.0
        cur = plateaus[i]
        ramp = (t >= b) & (t < b + rise_s)
        y[ramp] = prev + (cur - prev) * ((t[ramp] - b) / rise_s)
        plat = (t >= b + rise_s) & (t < b + W)
        y[plat] = cur
    y[t >= true_start + N * W] = 0.0
    rng = np.random.default_rng(0)
    y = y + rng.normal(0, 1.0, len(y))

    # grid handed in 0.3 s late (mimics the end-anchor catching decay tail)
    aligned = _align_grid_phase(t, y, true_start + 0.3, N, W, 0.5, fs)
    assert abs(aligned - true_start) < 0.25


def test_flow_levels_carry_raw_windows():
    """Each level exposes its raw (tared) trace + settle boundary for the viewer."""
    p = FlowRampParams(min_flow_mm3_s=5, max_flow_mm3_s=10, flow_step_mm3_s=1,
                       dwell_s=3.0, settle_frac=0.5)
    plan = build_flow_ramp(p)
    ft, fy = _synthesize(plan, dev_at=999, var_at=999, collapse_at=999)
    a = analyse_flow(sweep_t0=0.0, force_t=ft, force_y=fy, plan=plan)
    for lv in a.levels:
        assert len(lv.t) > 0 and len(lv.t) == len(lv.force)
        # settle boundary sits inside the window: start <= settle < end
        assert lv.t_window_start <= lv.t_settle < lv.t_end
        # settle is ~settle_frac of the dwell after the window start
        assert abs((lv.t_settle - lv.t_window_start) - 0.5 * 3.0) < 0.2
