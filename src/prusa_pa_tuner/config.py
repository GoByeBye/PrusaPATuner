"""Persistent user config — printer IP, API key, last-used sweep parameters."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def config_dir() -> Path:
    base = os.environ.get("PRUSA_PA_TUNER_CONFIG_DIR")
    if base:
        return Path(base)
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", str(Path.home()))) / "PrusaPATuner"
    return Path.home() / ".prusa_pa_tuner"


def config_path() -> Path:
    return config_dir() / "config.json"


@dataclass(slots=True)
class AppConfig:
    printer_host: str = ""
    printer_api_key: str = ""
    printer_user: str = "maker"
    printer_password: str = ""
    udp_port: int = 8514  # Prusa stock metrics port
    server_port: int = 8765

    # last-used sweep params (defaults match SweepParams)
    nozzle_temp: float = 215.0  # test temperature -- what bursts run at
    # Preheat target: held during homing + parking + baseline dwell, then
    # the gcode switches the setpoint to `nozzle_temp` at the start of the
    # first purge. Running the warm-up ~10 °C hot accelerates homing and
    # ensures any residual filament is fully molten before priming.
    preheat_temp: float = 225.0
    nozzle_diameter: float = 0.4
    filament_diameter: float = 1.75
    # Volumetric burst spec. The runner converts these into the SweepParams
    # time-domain pair (feed_mm_s, half_s) using filament_diameter to derive
    # cross-section area:
    #     feed_mm_s = flow_mm3_s / (pi/4 * filament_diameter^2)
    #     half_s    = volume_mm3 / flow_mm3_s
    # Defaults reproduce the previous Snapmaker-U1-style 0.8 mm/s × 1.0 s
    # slow + 8.0 mm/s × 0.25 s fast for 1.75 mm filament:
    #   area_1.75 ≈ 2.405 mm²
    #   slow flow = 0.8 * 2.405 ≈ 1.92 mm³/s; slow volume ≈ 1.92 mm³
    #   fast flow = 8.0 * 2.405 ≈ 19.24 mm³/s; fast volume ≈ 4.81 mm³
    slow_flow_mm3_s: float = 1.92
    fast_flow_mm3_s: float = 19.24
    slow_volume_mm3: float = 1.92
    fast_volume_mm3: float = 4.81
    cycles_per_K: int = 14
    # Raised from 200 → 5000 mm/s² after the bd_pressure comparison: at 200
    # the velocity transition takes ~36 ms and the resulting pressure
    # transient is barely K-dependent (both metrics came back flat with
    # R²≈0). At 5000 the transition is ~1.4 ms, dp/dt grows ~25×, and PA's
    # effect on the transient grows with it. Buddy may silently clamp below
    # 5000 -- the gcode-echo stream in the runner surfaces the actual value
    # that was applied.
    accel_mm_s2: float = 5000.0
    # Fine sweep matching bd_pressure's granularity (0..0.10 in 0.002
    # steps, inclusive of k_max). The previous coarse 9-step 0..0.40 sweep
    # relied on a clean linear trend through phase-lag-vs-K; at low SNR
    # the slope estimate was dominated by noise and `k_opt = -intercept/slope`
    # swung wildly. The new fine grid feeds the bd_pressure-style argmin
    # extraction over `amplitude + |asymmetry|`, which doesn't need
    # extrapolation -- it picks the K where the pressure transient is
    # smallest. K values are derived in the runner as
    # `k_min + i*k_step for i in 0..n` where n = round((k_max-k_min)/k_step).
    k_min: float = 0.0
    k_max: float = 0.10
    k_step: float = 0.002
    purge_x: float = 30.0
    purge_y: float = 30.0
    purge_z: float = 50.0
    # Per-axis coupling amplitudes. At least one of dx/dy must be > 0 for
    # Buddy/Marlin to apply M572 PA (Z+E does not trigger PA because Z is
    # on its own stepper, decoupled from the A/B/CoreXY pair). dz is
    # exposed so the user can experiment with what couples least into the
    # loadcell signal.
    coupled_dx_mm: float = 0.05
    coupled_dy_mm: float = 0.0
    coupled_dz_mm: float = 0.0
    # First slow-leg warm-up factor. K[0]'s very first slow extrusion
    # is `slow_half_s × first_slow_leg_factor` long -- so with the
    # default factor=10 and slow_half=2 s, the sweep opens with 20 s
    # of slow flow which both purges old filament and establishes
    # steady-state melt pressure before the first slow→fast
    # transition. Replaces the legacy 2 mm prime + 500 ms dwell.
    first_slow_leg_factor: float = 10.0
    filament_label: str = "PLA"

    # --- Max Flow test parameters ---
    # Free-air stepped flow sweep: extrude at increasing volumetric flow
    # rates and watch the loadcell back-pressure. min/max/step are the
    # user-facing knobs; dwell + settle_frac + warmup are tuning defaults
    # surfaced behind an "advanced" toggle. See flow_gen.FlowRampParams.
    flow_min_mm3_s: float = 5.0
    flow_max_mm3_s: float = 30.0
    flow_step_mm3_s: float = 1.0
    # Seconds extruded per flow level. The first `flow_settle_frac` of it is
    # discarded so melt pressure + the accel transient settle before we
    # measure the steady-state force + its variance. Longer = more time for
    # the hotend to add heat and reach true thermal steady state at each
    # level (at the cost of more filament slung into the air).
    flow_dwell_s: float = 3.0
    flow_settle_frac: float = 0.5
    # Pre-sweep warm-up extrusion at min flow (s) to purge old filament and
    # establish steady-state melt pressure before the first measured level.
    flow_warmup_s: float = 3.0
    # No-flow hold after the sweep-start marker, before the first level.
    # The loadcell reading here (extruder idle) is the static head load; the
    # analyser tares it off so the plotted force starts near 0 and reads as
    # back-pressure above baseline.
    flow_tare_dwell_s: float = 1.5

    # --- Touch-probe (lateral characterisation) parameters ---
    # Step 1 of the touch-probe project: push the nozzle TIP sideways into a
    # rigid target and see whether the Z loadcell registers lateral contact.
    # Open-loop-safe: the creep is short (`probe_creep_mm`) and slow
    # (`probe_slow_feed_mm_min`), so even with nothing halting the move the
    # worst case is a gentle lean, not a crash. See probe_gen.ProbeParams.
    probe_axis: str = "X"          # "X" or "Y" -- axis the head moves along
    probe_dir: str = "+"           # "+" or "-" -- direction toward the part
    probe_start_x: float = 125.0   # standoff X where the slow creep begins
    probe_start_y: float = 110.0   # standoff Y where the slow creep begins
    probe_z: float = 5.0           # probe height (also travel Z to standoff)
    probe_creep_mm: float = 1.0    # SLOW creep distance == hard overtravel cap
    probe_slow_feed_mm_min: float = 30.0    # creep speed (0.5 mm/s); gentle
    probe_travel_feed_mm_min: float = 3000.0  # rapid repositioning speed
    probe_n_touches: int = 5       # repeats from the same standoff
    probe_backoff_mm: float = 1.0  # retract past the standoff between touches
    probe_settle_ms: int = 300     # hold at the standoff and at the far end
    probe_temp: float = 0.0        # 0 => COLD probe (no heat); >0 heats first

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AppConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


def load_config() -> AppConfig:
    path = config_path()
    if not path.exists():
        return AppConfig()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AppConfig.from_dict(data)
    except Exception:
        return AppConfig()


def save_config(cfg: AppConfig) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
