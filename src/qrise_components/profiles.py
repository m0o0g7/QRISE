from qrise_math.physics import OpticalModel
from qrise_math.routing import Config
from .config import DeviceConfig


def profile_settings(name="mechanism"):
    if name != "mechanism":
        raise ValueError("Unknown experiment profile: " + name)
    config = Config(
        alpha=10.0,
        qrise_guard_mode="chain",
        qrise_lookahead_slots=20.0,
        baseline_fidelity_filter=False,
        baseline_guard_floor=True,
        qrise_reachability_filter=True,
        qrise_floor_base=0.965,
        qrise_floor_target_weight=0.015,
    )
    optical = OpticalModel(
        source_fidelity_offset=0.005,
        degraded_link_fraction=0.20,
        degraded_offset_min=-0.05,
        degraded_offset_max=-0.02,
        phase_fidelity_amplitude=0.012,
        quality_period_slots=120.0,
    )
    return config, DeviceConfig(memory_rate_scale=0.02), optical
