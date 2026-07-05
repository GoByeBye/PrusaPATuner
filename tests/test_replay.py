"""Tests for replay.py NPZ loading — in particular the sweep_params_json
round-trip that lets replay rebuild the exact plan a run used."""
import dataclasses
import json

import numpy as np
import pytest

from prusa_pa_tuner.gcode_gen import SweepParams
from prusa_pa_tuner.replay import load_run


def _base_arrays() -> dict:
    """Minimal stream arrays load_run needs."""
    return {
        "sweep_t0": np.array([100.0]),
        "force_t": np.linspace(100.0, 110.0, 50),
        "force_y": np.zeros(50),
        "pos_t": np.array([]),
        "pos_x": np.array([]),
    }


def test_load_run_prefers_sweep_params_json(tmp_path):
    """An NPZ carrying sweep_params_json must reproduce the run's exact
    SweepParams — including knobs the legacy scalar fields never stored
    — with no heuristic reconstruction."""
    params = SweepParams(
        K_values=(0.0, 0.004, 0.008),
        cycles_per_K=7,
        slow_half_s=1.5,
        fast_half_s=0.3,
        slow_feed_mm_s=0.9,
        fast_feed_mm_s=7.5,
        coupled_dx_mm=0.7,
        first_slow_leg_factor=6.0,
        purge_x=42.0,
        purge_y=33.0,
        purge_z=55.0,
        z_marker_lift_mm=3.0,
        inter_k_dwell_s=0.25,       # not in the legacy scalar fields
        baseline_dwell_s=4.5,       # not in the legacy scalar fields
        accel_mm_s2=4321.0,         # not in the legacy scalar fields
    )
    path = tmp_path / "run_1.npz"
    np.savez(
        path,
        sweep_params_json=np.array(
            [json.dumps(dataclasses.asdict(params))], dtype="U8192"
        ),
        **_base_arrays(),
    )

    plan, kwargs = load_run(path)
    p = plan.params
    assert p.K_values == params.K_values
    assert p.cycles_per_K == 7
    assert p.slow_half_s == 1.5
    assert p.coupled_dx_mm == 0.7
    assert p.first_slow_leg_factor == 6.0
    assert p.inter_k_dwell_s == 0.25
    assert p.baseline_dwell_s == 4.5
    assert p.accel_mm_s2 == 4321.0
    assert kwargs["z_marker_lift_mm"] == 3.0
    assert kwargs["sweep_t0"] == 100.0


def test_load_run_json_ignores_unknown_future_keys(tmp_path):
    """A future SweepParams revision may add fields; loading such a dump
    with today's code must drop the unknown keys, not crash."""
    params = SweepParams(K_values=(0.02,), cycles_per_K=3)
    raw = dataclasses.asdict(params)
    raw["some_future_knob"] = 123.4
    path = tmp_path / "run_2.npz"
    np.savez(
        path,
        sweep_params_json=np.array([json.dumps(raw)], dtype="U8192"),
        **_base_arrays(),
    )
    plan, _ = load_run(path)
    assert plan.params.K_values == (0.02,)
    assert plan.params.cycles_per_K == 3


def test_load_run_legacy_scalar_fields_still_work(tmp_path):
    """Old dumps without sweep_params_json must load via the legacy
    scalar-field reconstruction path."""
    path = tmp_path / "run_3.npz"
    np.savez(
        path,
        k_values=np.array([0.0, 0.01]),
        cycles_per_K=np.array([4]),
        slow_half_s=np.array([1.0]),
        fast_half_s=np.array([0.25]),
        slow_feed_mm_s=np.array([0.8]),
        fast_feed_mm_s=np.array([8.0]),
        coupled_dx_mm=np.array([0.3]),
        z_marker_lift_mm=np.array([2.0]),
        **_base_arrays(),
    )
    plan, kwargs = load_run(path)
    assert plan.params.K_values == (0.0, 0.01)
    assert plan.params.cycles_per_K == 4
    assert plan.params.coupled_dx_mm == 0.3
    assert kwargs["z_marker_lift_mm"] == 2.0
