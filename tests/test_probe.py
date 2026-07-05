"""Tests for the lateral touch-probe: gcode generation + contact detection."""
import json

import numpy as np

from prusa_pa_tuner.probe_analysis import analyse_probe, probe_analysis_to_dict
from prusa_pa_tuner.probe_gen import ProbeParams, build_probe_test


def test_probe_gcode_structure():
    p = ProbeParams(
        probe_axis="X", probe_dir="+", start_x=100.0, start_y=110.0,
        creep_mm=1.0, n_touches=3, udp_host="10.0.0.9", udp_port=8500,
    )
    plan = build_probe_test(p)
    g = plan.gcode
    assert plan.axis == "X" and plan.dir_sign == 1.0
    assert "M334 10.0.0.9 8500" in g
    assert "M331 loadcell_value" in g
    assert "M331 pos_x" in g and "M331 pos_y" in g
    # one M117 PROBE marker + one TOUCH_START tag per touch
    assert g.count("M117 PROBE=") == 3
    assert g.count(";PROBE_TUNER TOUCH_START") == 3
    assert ";PROBE_TUNER SWEEP_START" in g and ";PROBE_TUNER SWEEP_END" in g
    assert len(plan.touches) == 3
    # creep goes from standoff (100) to standoff + creep (101) for X+
    assert all(abs(t.axis_start - 100.0) < 1e-9 for t in plan.touches)
    assert all(abs(t.axis_end - 101.0) < 1e-9 for t in plan.touches)
    # there is no extrusion in a probe test (no E moves, no E-axis setup)
    assert "G1 E" not in g and "G92 E" not in g and "M82" not in g


def test_probe_axis_y_and_negative_dir():
    p = ProbeParams(probe_axis="Y", probe_dir="-", start_x=120.0, start_y=90.0,
                    creep_mm=2.0, n_touches=2)
    plan = build_probe_test(p)
    assert plan.axis == "Y" and plan.dir_sign == -1.0
    # Y+ moves Y, X pinned to standoff
    assert "G1 X120.000 Y" in plan.gcode
    # negative direction: creep end is below the standoff
    assert all(t.axis_start == 90.0 for t in plan.touches)
    assert all(abs(t.axis_end - 88.0) < 1e-9 for t in plan.touches)  # 90 - 2


def test_probe_cold_by_default():
    cold = build_probe_test(ProbeParams(probe_temp=0.0)).gcode
    assert "M104" not in cold and "M109" not in cold
    hot = build_probe_test(ProbeParams(probe_temp=150.0)).gcode
    assert "M104 S150" in hot and "M109 S150" in hot


def _synthesize_probe(plan, *, contact_at=0.6, stiffness=250.0, offset=1500.0,
                      noise_amp=0.4, seed=0):
    """Fabricate a realistic capture: idle head, then n_touches of
    fast-approach / slow-creep (force rises past `contact_at` mm) / retract.
    `stiffness`=0 simulates a loadcell that DOESN'T see lateral contact.
    Returns (force_t, force_y, pos_t, pos) on the probe axis.
    """
    p = plan.params
    sign = plan.dir_sign
    standoff = p.start_x if plan.axis == "X" else p.start_y
    far = standoff + sign * p.creep_mm
    pre = standoff - sign * p.backoff_mm
    slow = p.slow_feed_mm_min / 60.0
    travel = p.travel_feed_mm_min / 60.0
    fs = 180.0
    rng = np.random.default_rng(seed)
    ft, fy, pt, px = [], [], [], []
    t = [0.0]

    def force_at(x):
        prog = (x - standoff) * sign
        base = offset + rng.normal(0, noise_amp)
        if stiffness > 0 and prog > contact_at:
            base += stiffness * (prog - contact_at)
        return base

    def move(x0, x1, feed):
        dist = abs(x1 - x0)
        n = max(1, int((dist / feed) * fs))
        for i in range(n):
            xi = x0 + (x1 - x0) * i / n
            ft.append(t[0]); fy.append(force_at(xi)); pt.append(t[0]); px.append(xi)
            t[0] += 1.0 / fs

    def dwell(x, d):
        for _ in range(int(d * fs)):
            ft.append(t[0]); fy.append(force_at(x)); pt.append(t[0]); px.append(x)
            t[0] += 1.0 / fs

    dwell(pre, 3.0)  # idle heat/home head
    for _ in range(p.n_touches):
        move(pre, standoff, travel)
        dwell(standoff, p.settle_ms / 1000.0)
        move(standoff, far, slow)
        dwell(far, p.settle_ms / 1000.0)
        move(far, pre, travel)
    return np.array(ft), np.array(fy), np.array(pt), np.array(px)


def test_probe_analysis_detects_clean_contact():
    plan = build_probe_test(ProbeParams(
        probe_axis="X", probe_dir="+", start_x=100.0, start_y=100.0,
        creep_mm=1.0, slow_feed_mm_min=30.0, n_touches=5, settle_ms=300))
    ft, fy, pt, px = _synthesize_probe(plan, contact_at=0.6, stiffness=250.0)
    a = analyse_probe(force_t=ft, force_y=fy, pos_t=pt, pos=px, plan=plan)

    assert a.verdict == "usable"
    assert a.n_contacts == 5
    assert len(a.touches) == 5
    # contact recovered near the injected 0.6 mm, very repeatable
    assert abs(a.contact_pos_mean - 0.6) < 0.05
    assert a.contact_pos_std < 0.02   # < 20 µm spread
    assert a.signal_to_noise > 20
    # JSON-safe dict, no NaN/inf leaking through
    d = probe_analysis_to_dict(a)
    assert json.dumps(d)
    assert len(d["touches"]) == 5
    assert d["verdict"] == "usable"


def test_probe_analysis_no_lateral_signal():
    """A loadcell that doesn't register lateral contact -> 'no clear signal'."""
    plan = build_probe_test(ProbeParams(
        probe_axis="X", probe_dir="+", start_x=100.0, start_y=100.0,
        creep_mm=1.0, slow_feed_mm_min=30.0, n_touches=4))
    ft, fy, pt, px = _synthesize_probe(plan, stiffness=0.0)  # flat: no contact
    a = analyse_probe(force_t=ft, force_y=fy, pos_t=pt, pos=px, plan=plan)
    # creeps still segment from position, but no contact is detected
    assert len(a.touches) == 4
    assert a.n_contacts == 0
    assert a.verdict == "no clear signal"


def test_probe_analysis_negative_direction():
    """Contact detection must work when probing toward lower coordinates."""
    plan = build_probe_test(ProbeParams(
        probe_axis="Y", probe_dir="-", start_x=100.0, start_y=100.0,
        creep_mm=1.5, slow_feed_mm_min=30.0, n_touches=4))
    ft, fy, pt, px = _synthesize_probe(plan, contact_at=0.9, stiffness=300.0)
    a = analyse_probe(force_t=ft, force_y=fy, pos_t=pt, pos=px, plan=plan)
    assert a.verdict == "usable"
    assert a.n_contacts == 4
    assert abs(a.contact_pos_mean - 0.9) < 0.06


def test_probe_analysis_requires_position():
    plan = build_probe_test(ProbeParams(n_touches=3))
    ft, fy, _, _ = _synthesize_probe(plan, stiffness=250.0)
    a = analyse_probe(force_t=ft, force_y=fy, pos_t=None, pos=None, plan=plan)
    assert a.verdict == "no position data"
    assert a.n_contacts == 0
    # still JSON-safe
    assert json.dumps(probe_analysis_to_dict(a))


def test_probe_touch_traces_present():
    """Each touch exposes a decimated force-vs-position trace for the plot."""
    plan = build_probe_test(ProbeParams(creep_mm=1.0, n_touches=3))
    ft, fy, pt, px = _synthesize_probe(plan, stiffness=250.0)
    a = analyse_probe(force_t=ft, force_y=fy, pos_t=pt, pos=px, plan=plan)
    for t in a.touches:
        assert len(t.pos) == len(t.force) and len(t.pos) > 0
        # progress runs from ~0 at the standoff to ~creep_mm
        assert min(t.pos) < 0.2 and max(t.pos) > 0.8
