"""CORE One INDX profile, tool-selection, and persistence regressions."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import prusa_pa_tuner.app as app_module
from prusa_pa_tuner.app import ConfigModel
from prusa_pa_tuner.config import AppConfig
from prusa_pa_tuner.flow_gen import FlowRampParams, build_flow_ramp
from prusa_pa_tuner.flow_runner import flow_params_from_config
from prusa_pa_tuner.gcode_gen import SweepParams, build_sweep
from prusa_pa_tuner.probe_gen import ProbeParams, build_probe_test
from prusa_pa_tuner.probe_runner import ProbeRunInfo, probe_params_from_config
from prusa_pa_tuner.replay import list_runs
from prusa_pa_tuner.runner import params_from_config


def _assert_ordered_prefixes(gcode: str, prefixes: list[str]) -> None:
    lines = gcode.splitlines()
    cursor = -1
    for prefix in prefixes:
        cursor = next(
            i for i, line in enumerate(lines[cursor + 1 :], cursor + 1) if line.startswith(prefix)
        )


def _referenced_tools(gcode: str) -> set[int]:
    """Return G-code T-word tool IDs, ignoring comments and M204's T accel."""
    tools: set[int] = set()
    for line in gcode.splitlines():
        code = line.split(";", 1)[0]
        for match in re.finditer(r"(?:^|\s)T([0-7])(?:\s|$)", code):
            tools.add(int(match.group(1)))
    return tools


@pytest.mark.parametrize("tool", [0, 7])
def test_indx_pa_uses_exact_profile_and_selected_tool(tool: int):
    gcode = build_sweep(
        SweepParams(
            printer_model="COREONEINDX",
            tool_index=tool,
            K_values=(0.0,),
            cycles_per_K=1,
        )
    ).gcode

    _assert_ordered_prefixes(
        gcode,
        [
            "; printer_model = COREONEINDX",
            f"M862.1 T{tool} P0.4 A0 F1",
            'M862.3 P "COREONEINDX"',
            'M862.6 P"INDX lock"',
            "M115 U6.6.0+15528",
            "G28 XY",
            f"T{tool} S1 L2 D0",
            "M104 S120",
            "G28 Z",
            "G0 Z40 F10000",
            "M104 S225",
            "M109 S225",
        ],
    )
    _assert_ordered_prefixes(
        gcode,
        [
            f"M104 T{tool} S0",
            "P0 S1",
            "G1 X242 Y205 F10200",
            "G4",
            "M84 X Y E",
        ],
    )
    assert _referenced_tools(gcode) == {tool}
    assert "G28 ; home" not in gcode


@pytest.mark.parametrize(
    "gcode",
    [
        build_flow_ramp(
            FlowRampParams(
                printer_model="COREONEINDX",
                tool_index=7,
                min_flow_mm3_s=5,
                max_flow_mm3_s=5,
            )
        ).gcode,
        build_probe_test(ProbeParams(printer_model="COREONEINDX", tool_index=7, n_touches=1)).gcode,
    ],
    ids=["max-flow", "touch-probe"],
)
def test_other_generated_tests_use_the_same_indx_tool_lifecycle(gcode: str):
    _assert_ordered_prefixes(
        gcode,
        [
            "; printer_model = COREONEINDX",
            "M862.1 T7 P0.4 A0 F1",
            'M862.6 P"INDX lock"',
            "G28 XY",
            "T7 S1 L2 D0",
            "M104 S120",
            "G28 Z",
            "G0 Z40 F10000",
            "P0 S1",
            "G1 X242 Y205 F10200",
        ],
    )
    assert _referenced_tools(gcode) == {7}


def test_cold_indx_probe_waits_for_a_plastic_safe_temperature():
    gcode = build_probe_test(
        ProbeParams(
            printer_model="COREONEINDX",
            tool_index=7,
            probe_temp=0,
            n_touches=1,
        )
    ).gcode
    _assert_ordered_prefixes(
        gcode,
        [
            "M104 S120",
            "G28 Z",
            "M104 T7 S0",
            "M109 T7 R40",
            "M104 T7 S0",
            ";PROBE_TUNER SWEEP_START",
        ],
    )


def test_indx_probe_waits_for_cooling_to_a_nonzero_temp_below_z_home_temp():
    gcode = build_probe_test(
        ProbeParams(
            printer_model="COREONEINDX",
            tool_index=3,
            probe_temp=50,
            n_touches=1,
        )
    ).gcode
    _assert_ordered_prefixes(
        gcode,
        [
            "M104 S120",
            "G28 Z",
            "M104 T3 S50",
            "M109 T3 R50",
            ";PROBE_TUNER SWEEP_START",
        ],
    )
    assert "M109 T3 S50" not in gcode


def test_plain_core_one_keeps_the_single_tool_lifecycle():
    gcode = build_sweep(SweepParams(K_values=(0.0,), cycles_per_K=1, printer_model="COREONE")).gcode
    assert "; printer_model = COREONE" in gcode
    assert "M862.1 P0.4 A0 F1" in gcode
    assert 'M862.6 P"INDX lock"' not in gcode
    assert "G28 ; home" in gcode
    assert "P0 S1" not in gcode
    assert _referenced_tools(gcode) == set()


@pytest.mark.parametrize("bad_tool", [-1, 8, True, 1.5, "3"])
def test_indx_generator_rejects_invalid_tool_ids(bad_tool: object):
    with pytest.raises(ValueError, match="tool_index|INDX"):
        build_sweep(
            SweepParams(
                printer_model="COREONEINDX",
                tool_index=bad_tool,  # type: ignore[arg-type]
                K_values=(0.0,),
            )
        )


@pytest.mark.parametrize("bad_tool", [-1, 8, True, 1.5, "3"])
def test_config_api_rejects_invalid_tool_ids(bad_tool: object):
    with pytest.raises(ValidationError):
        ConfigModel(printer_model="COREONEINDX", tool_index=bad_tool)


def test_config_propagates_indx_profile_to_every_generated_test():
    cfg = AppConfig(printer_model="COREONEINDX", tool_index=6)
    params = [
        params_from_config(cfg, "10.0.0.2"),
        flow_params_from_config(cfg, "10.0.0.2"),
        probe_params_from_config(cfg, "10.0.0.2"),
    ]
    for generated in params:
        assert generated.printer_model == "COREONEINDX"
        assert generated.tool_index == 6
        assert "T6" in generated.label


def test_config_api_normalises_a_friendly_persisted_indx_alias():
    model = ConfigModel.from_appconfig(AppConfig(printer_model="CORE_ONE_INDX", tool_index=7))
    assert model.printer_model == "COREONEINDX"
    assert model.tool_index == 7


def test_run_listing_preserves_indx_tool_but_not_a_legacy_phantom_t0(tmp_path):
    common = {
        "force_t": np.array([1.0, 2.0]),
        "pos_t": np.array([]),
        "k_values": np.array([0.0]),
        "cycles_per_K": np.array([1]),
    }
    np.savez(
        tmp_path / "run_t7_2.npz",
        printer_model=np.array(["COREONEINDX"]),
        tool_index=np.array([7]),
        **common,
    )
    np.savez(tmp_path / "run_1.npz", **common)
    np.savez(
        tmp_path / "run_t99_3.npz",
        printer_model=np.array(["COREONEINDX"]),
        tool_index=np.array([99]),
        **common,
    )

    runs = {run.filename: run for run in list_runs(tmp_path)}
    assert runs["run_t7_2.npz"].printer_model == "COREONEINDX"
    assert runs["run_t7_2.npz"].tool_index == 7
    assert runs["run_1.npz"].printer_model == "COREONE"
    assert runs["run_1.npz"].tool_index is None
    assert runs["run_t99_3.npz"].tool_index is None


@pytest.mark.asyncio
async def test_probe_runs_api_exposes_indx_tool_metadata(monkeypatch):
    info = ProbeRunInfo(
        filename="probe_t7_2.npz",
        path="ignored",
        mtime_unix=2.0,
        n_force=10,
        axis="X",
        dir="+",
        n_touches=5,
        creep_mm=1.0,
        duration_s=4.0,
        printer_model="COREONEINDX",
        tool_index=7,
    )
    monkeypatch.setattr(app_module, "list_probe_runs", lambda _path: [info])
    payload = await app_module.get_probe_runs()
    assert payload["runs"][0]["printer_model"] == "COREONEINDX"
    assert payload["runs"][0]["tool_index"] == 7


def test_run_and_preview_actions_abort_when_config_save_fails():
    js_path = Path(__file__).parents[1] / "src" / "prusa_pa_tuner" / "static" / "app.js"
    javascript = js_path.read_text(encoding="utf-8")
    assert "return false;" in javascript
    assert "return true;" in javascript
    assert javascript.count("if (!(await saveConfig())) return;") == 6
