"""Shared G-code preamble blocks for the four test-gcode generators.

gcode_gen (PA sweep), flow_gen (max flow), probe_gen (touch probe) and
livemap_gen (live map) all bring the printer to the same known state before
their test-specific motion: forge a PrusaSlicer-compatible header so
Buddy/Core One accepts the file without compatibility prompts, assert the
firmware features we rely on, point the UDP metric stream at this host and
prune it down to the streams we actually consume, then heat/home/park.

Every emitter below APPENDS lines to a caller-owned list -- the generators
stay in full control of ordering and of any module-specific lines in
between. The emitted bytes are exactly what each generator produced before
this module existed; where two modules genuinely diverge (a comment, an
extra key) the difference is a parameter, not a fork.
"""
from __future__ import annotations

from typing import Iterable, Sequence


COREONE_MODEL = "COREONE"
COREONE_INDX_MODEL = "COREONEINDX"
INDX_TOOL_MIN = 0
INDX_TOOL_MAX = 7


def normalise_printer_model(value: str) -> str:
    """Return the firmware/slicer model token used in generated G-code.

    Config files and callers may use a friendly ``INDX`` spelling, but the
    compatibility checks on Prusa CORE One INDX firmware require the exact
    ``COREONEINDX`` token emitted by PrusaSlicer.
    """
    compact = (
        str(value or COREONE_MODEL)
        .strip()
        .upper()
        .replace(" ", "")
        .replace("_", "")
    )
    if compact in {"COREONE", "CORE1"}:
        return COREONE_MODEL
    if compact in {"COREONEINDX", "CORE1INDX", "INDX"}:
        return COREONE_INDX_MODEL
    raise ValueError(
        f"unsupported printer_model {value!r}; expected COREONE or COREONEINDX"
    )


def validated_tool_index(printer_model: str, tool_index: int) -> int:
    """Validate and return the selected INDX G-code tool (T0..T7)."""
    model = normalise_printer_model(printer_model)
    if type(tool_index) is not int:
        raise ValueError("tool_index must be an integer from 0 through 7")
    tool = tool_index
    if model == COREONE_INDX_MODEL and not INDX_TOOL_MIN <= tool <= INDX_TOOL_MAX:
        raise ValueError("CORE One INDX tool_index must be from 0 through 7")
    return tool


def is_indx(printer_model: str) -> bool:
    return normalise_printer_model(printer_model) == COREONE_INDX_MODEL


# Per-module comment-marker prefixes. The generators tag baseline / sweep /
# segment boundaries with these so the runner + analyser can slice the
# timeseries against the gcode plan.
#
# NOTE: the tests assert against the literal strings (tests/test_gcode_gen.py,
# tests/test_flow.py, tests/test_probe.py), so treat these values as a wire
# format -- changing them breaks recorded runs and the test suite.
PA_MARKER_PREFIX = ";PA_TUNER"
FLOW_MARKER_PREFIX = ";FLOW_TUNER"
PROBE_MARKER_PREFIX = ";PROBE_TUNER"

# Metrics to disable (M332) before enabling the few streams we consume.
# These names match what THIS firmware build emits (verified via
# /api/diagnostics on the user's printer). Buddy's `M332 <name>` is a
# strict case-sensitive strcmp -- it silently no-ops on miss and the
# reply goes only to serial, which PrusaLink discards. So this list is
# curated against observed live names; do NOT add speculative names.
# Shared by build_sweep (PA tuning) and flow_gen.build_flow_ramp (max
# flow) so the two test gcodes bring the printer to the same minimal,
# low-UDP-load state.
METRICS_TO_SILENCE: tuple[str, ...] = (
    # Default-on noisy stuff we observed at runtime.
    "cmdcnt", "stp_stall",
    "runtime", "stack",
    "fan", "input_current", "heater_current",
    "oc_inp", "oc_nozz",
    "bed_voltage", "heater_voltage",
    "door_sensor", "chamber_temp",
    "points_dropped", "cur_mmu_imp",
    "esp_in", "esp_out", "eth_out",
    "heater_enabled",
    # Loadcell variants we don't use (loadcell_value is the primary).
    # `loadcell` is the processed/grams stream; we consume the raw
    # `loadcell_value` instead, so silence `loadcell` to save bandwidth.
    # NOTE: M332 is an EXACT case-sensitive match, so "loadcell" does NOT
    # touch "loadcell_value" -- the primary stream stays enabled.
    "loadcell",
    "loadcell_hysteresis",
    "loadcell_threshold", "loadcell_threshold_cont",
    # Gcode-queue telemetry -- not text echoes, just queue state.
    "gcd_que_sz", "ftch_cmds", "ftch_dur", "ftch_occ",
    "ftch_status", "ftch_tstatus",
    # Printer state we don't need.
    "is_printing", "home", "home_diff",
    "phxy_home", "phxy_meas", "phxy_probe",
    "fw_version", "buddy_bom", "filament",
    "classified", "detection",
    "metrics", "hit:", "opened:",
)


def slicer_header(
    lines: list[str],
    *,
    label: str,
    nozzle_diameter: float,
    filament_diameter: float,
    filament_label: str,
    nozzle_temp: float,
    printer_notes: str,
    printer_model: str = COREONE_MODEL,
    extra_comment_lines: Iterable[str] = (),
) -> None:
    """Forged PrusaSlicer-style header block.

    Buddy/Core One firmware shows a "gcode is not fully compatible" warning
    unless it recognises a slicer signature plus a minimal config block at
    the top. The values are advisory; what matters is presence of the keys.
    `extra_comment_lines` carries the module-specific trailing comments
    (K grid, flow grid, probe geometry). Ends with a blank line.
    """
    lines.append("; generated by PrusaSlicer 2.8.1+win64")
    lines.append(f"; {label}")
    lines.append(f"; nozzle_diameter = {nozzle_diameter}")
    lines.append(f"; filament_diameter = {filament_diameter}")
    lines.append(f"; filament_type = {filament_label}")
    lines.append(f"; first_layer_temperature = {nozzle_temp:.0f}")
    lines.append(f"; temperature = {nozzle_temp:.0f}")
    lines.append("; first_layer_bed_temperature = 0")
    lines.append("; bed_temperature = 0")
    lines.append("; layer_height = 0.2")
    lines.append("; max_print_height = 280")
    model = normalise_printer_model(printer_model)
    lines.append(f"; printer_model = {model}")
    lines.append(f"; printer_notes = {printer_notes}")
    for extra in extra_comment_lines:
        lines.append(extra)
    lines.append("")


def firmware_asserts(
    lines: list[str],
    *,
    nozzle_diameter: float,
    printer_model: str = COREONE_MODEL,
    tool_index: int = 0,
    input_shaper_comment: str = "FW feature check",
) -> None:
    """Prusa firmware feature assertions.

    M862.6 P"Input shaper" is the specific check that turns OFF the "not
    sliced for input shaping" warning on Buddy 6.5.3+. The others match what
    PrusaSlicer emits so the firmware accepts the gcode without prompting.
    Ends with a blank line.
    """
    model = normalise_printer_model(printer_model)
    tool = validated_tool_index(model, tool_index)
    lines.append("M17 ; enable steppers")
    if model == COREONE_INDX_MODEL:
        lines.append(
            f"M862.1 T{tool} P{nozzle_diameter} A0 F1 ; selected INDX tool nozzle check (HF)"
        )
    else:
        lines.append(f"M862.1 P{nozzle_diameter} A0 F1 ; nozzle check (HF)")
    lines.append(f'M862.3 P "{model}" ; printer model check')
    lines.append("M862.5 P2 ; g-code level check")
    lines.append(f'M862.6 P"Input shaper" ; {input_shaper_comment}')
    if model == COREONE_INDX_MODEL:
        # Buddy 6.6.x treats this exact feature declaration as proof that the
        # file was produced with the current INDX tool-change profile.  Without
        # it, the firmware pauses at 0% with "Sliced with old INDX profiles".
        lines.append('M862.6 P"INDX lock" ; current INDX profile feature check')
        lines.append("M115 U6.6.0+15528 ; require INDX-capable Buddy firmware")
    else:
        lines.append("M115 U6.5.3+12780 ; require Buddy firmware >= 6.5.3")
    lines.append("")


def home_selected_tool(
    lines: list[str],
    *,
    printer_model: str = COREONE_MODEL,
    tool_index: int = 0,
    indx_z_home_temp: float = 120.0,
) -> bool:
    """Home the machine and explicitly pick the requested INDX tool.

    Returns ``True`` when the helper heated an INDX nozzle for Z homing.
    The sequence mirrors the local PrusaSlicer COREONEINDX profile:
    home XY, pick ``Tn S1 L2 D0``, heat to 120 C, then home Z. A plain
    CORE One retains the historical all-axis ``G28`` path.
    """
    model = normalise_printer_model(printer_model)
    tool = validated_tool_index(model, tool_index)
    if model != COREONE_INDX_MODEL:
        lines.append("G28 ; home")
        return False
    lines.append("G28 XY ; home XY before INDX pickup")
    lines.append(f"T{tool} S1 L2 D0 ; pick INDX tool T{tool} for Z homing")
    lines.append(f"M104 S{indx_z_home_temp:.0f} ; INDX Z-home temperature")
    lines.append("G28 Z ; home Z with selected INDX tool")
    lines.append("G0 Z40 F10000 ; INDX post-home clearance")
    return True


def finish_selected_tool(
    lines: list[str],
    *,
    printer_model: str = COREONE_MODEL,
    tool_index: int = 0,
    turn_off_nozzle: bool = True,
) -> None:
    """Cool down and finish without leaving an INDX tool on the carriage."""
    model = normalise_printer_model(printer_model)
    tool = validated_tool_index(model, tool_index)
    if model == COREONE_INDX_MODEL:
        if turn_off_nozzle:
            lines.append(f"M104 T{tool} S0 ; selected INDX nozzle off")
        lines.append("P0 S1 ; park INDX tool")
        lines.append("G1 X242 Y205 F10200 ; clear the INDX dock area")
        lines.append("G4 ; wait for park moves")
        lines.append("M84 X Y E ; disable motion/extruder motors")
        return
    if turn_off_nozzle:
        lines.append("M104 S0 ; nozzle off")
    lines.append("M84 ; disable motors")


def metric_setup(
    lines: list[str],
    *,
    udp_host: str,
    udp_port: int,
    enables: Sequence[tuple[str, str]],
    silence_header: str = "; --- silence non-essential metrics to lower UDP load ---",
    blank_after_silence: bool = True,
    blank_after_enables: bool = True,
) -> None:
    """UDP metric-stream setup: M334 target, M332 silence loop, M331 enables.

    `enables` is (metric, comment) for each M331 command -- each module
    consumes a different stream set and annotates it differently.

    Keep the M331 command itself bare. Buddy's handler compares
    ``parser.string_arg`` with the metric name using ``strcmp``. Marlin's
    parser stops scanning at ``;`` but does not terminate the string there,
    so an inline comment becomes part of the argument and the enable silently
    fails (``loadcell_value ; comment`` is not ``loadcell_value``).
    livemap_gen uses the plain `silence_header` variant and no blank
    separators (its preamble is a single compact block).
    """
    lines.append(f"M334 {udp_host} {udp_port} ; stream metrics to host")
    lines.append(silence_header)
    for m in METRICS_TO_SILENCE:
        lines.append(f"M332 {m}")
    if blank_after_silence:
        lines.append("")
    for metric, comment in enables:
        if comment:
            lines.append(f"; enable {metric}: {comment}")
        lines.append(f"M331 {metric}")
    if blank_after_enables:
        lines.append("")


def heat_home_setup(
    lines: list[str],
    *,
    preheat_temp: float,
    nozzle_temp: float,
    accel_mm_s2: float,
    accel_comment: str,
    purge_x: float,
    purge_y: float,
    purge_z: float,
    printer_model: str = COREONE_MODEL,
    tool_index: int = 0,
) -> float:
    """Stuck-detection off, two-stage preheat + home, absolute-E mode,
    accel pinning, park at the purge position. Returns the preheat setpoint
    (max of preheat_temp / nozzle_temp) so the caller can decide whether a
    later "drop to test temp" M104 is needed.

    Two-stage warmup: heat to `preheat_temp` (typically nozzle_temp + 10)
    during homing + park. The preheat overshoot guarantees any residual
    filament from the previous run is fully molten before we measure; the
    caller drops the setpoint to `nozzle_temp` right before its first purge
    so the small temperature drop overlaps the prime / pre-roll.

    Absolute-E rationale: Buddy/Core One's G28 + heat sequence can silently
    flip the extruder back to absolute even after we asked for relative, so
    the generators use a monotonic cumulative E counter under M82 -- correct
    regardless of which E mode is actually active. See build_sweep for the
    full write-up (accel classes, M201 E vs M204 R interplay).
    """
    # Our slow legs look like a stalled feed to Buddy's stuck-filament
    # sensor; disable it for the duration (matches PrusaSlicer's pattern
    # during slow purge moves).
    lines.append("M591 S0 ; disable filament stuck detection")
    lines.append("G90 ; absolute XYZ")

    preheat = max(preheat_temp, nozzle_temp)
    model = normalise_printer_model(printer_model)
    if model == COREONE_INDX_MODEL:
        home_selected_tool(
            lines,
            printer_model=model,
            tool_index=tool_index,
            indx_z_home_temp=120.0,
        )
        lines.append(f"M104 S{preheat:.0f} ; preheat selected INDX tool")
    else:
        lines.append(f"M104 S{preheat:.0f} ; preheat (test temp + headroom)")
        home_selected_tool(lines, printer_model=model, tool_index=tool_index)
    lines.append(f"M109 S{preheat:.0f} ; wait for preheat")

    lines.append("G90 ; absolute XYZ (reassert after home)")
    lines.append("M82 ; ABSOLUTE E -- cumulative E targets follow")
    lines.append("G92 E0 ; reset E counter to zero")
    lines.append(f"M201 E{accel_mm_s2:.0f} ; raise E-axis max accel cap")
    lines.append(
        f"M204 P{accel_mm_s2:.0f} R{accel_mm_s2:.0f} T{accel_mm_s2:.0f} "
        f"; {accel_comment}"
    )

    # Park at the purge position -- well above the bed so dripping filament
    # doesn't stick.
    lines.append(f"G1 X{purge_x:.2f} Y{purge_y:.2f} Z{purge_z:.2f} F6000")
    lines.append("G4 P500 ; settle")
    return preheat


def baseline_dwell(
    lines: list[str],
    *,
    m117_prefix: str,
    marker_prefix: str,
    dwell_s: float,
) -> None:
    """Pre-prime loadcell baseline window: hot nozzle, no extrusion yet --
    static head load only. The runner watches for the M117 markers in the
    gcode-event stream and slices `baseline_t_start`/`_end` to pass to the
    analyser as a tare reference / drift diagnostic. Anything that's not a
    G4 here would corrupt the zero (M-codes that provoke motion, fan
    toggles, etc.) -- keep it strictly static. Callers skip the whole block
    when the dwell is zero.
    """
    lines.append(f"M117 {m117_prefix}_BASELINE_START")
    lines.append(f"{marker_prefix} BASELINE_START")
    lines.append(f"G4 P{int(dwell_s * 1000)} ; hold for tare/baseline")
    lines.append(f"M117 {m117_prefix}_BASELINE_END")
    lines.append(f"{marker_prefix} BASELINE_END")


def z_marker(lines: list[str], *, purge_z: float, lift_mm: float) -> None:
    """Sweep-start Z marker: lift the toolhead by `lift_mm`, brief hold,
    drop back to `purge_z`. This produces a one-shot, unique signature in
    the pos_z stream that the analyser locks onto as `sweep_t0` -- the
    pos_z return-to-baseline is the exact sweep start. Callers skip the
    block when the lift is zero (analyser falls back to pos_x periodicity).
    """
    lines.append("; --- sweep_t0 marker: Z lift then drop ---")
    lines.append(f"G1 Z{purge_z + lift_mm:.3f} F1800 ; sweep marker UP")
    lines.append("G4 P100 ; brief hold at top")
    lines.append(
        f"G1 Z{purge_z:.3f} F1800 ; sweep marker DOWN -- pos_z return = sweep_t0"
    )


def sweep_start_marker(
    lines: list[str], *, m117_prefix: str, marker_prefix: str
) -> None:
    """SWEEP_START pair (M117 for the gcode log + comment marker for the
    plan), followed by the blank line every module emits before its motion.
    """
    lines.append(f"M117 {m117_prefix}_SWEEP_START")
    lines.append(f"{marker_prefix} SWEEP_START")
    lines.append("")


def sweep_end_marker(
    lines: list[str], *, m117_prefix: str, marker_prefix: str
) -> None:
    """SWEEP_END pair. No surrounding blanks -- the callers' spacing before
    this block differs (loop-trailing blank vs explicit)."""
    lines.append(f"M117 {m117_prefix}_SWEEP_END")
    lines.append(f"{marker_prefix} SWEEP_END")
