"""Declared input scenarios, not fitted output corrections."""
from qrise_math.physics import OpticalModel
from qrise_math.routing import Config
from .config import DeviceConfig


def profile_settings(name):
    """Return policy-independent physics and declared routing weights.

    Hardware inputs and planning floors are reconstruction assumptions.
    Profiles apply the same devices to every policy; relative performance
    depends on these inputs and the chosen baseline adapters.
    """
    if name == "recon":
        return Config(), DeviceConfig(), OpticalModel()
    if name in ("fig2", "hardware_only"):
        # High-quality sources and long-lived memories are shared device inputs.
        # Increasing alpha gives fidelity more weight relative to lane capacity.
        # These calibration choices are not values disclosed by the poster.
        config = Config(alpha=10.0 if name == "fig2" else 1.0)
        return config, DeviceConfig(memory_rate_scale=.02), OpticalModel(source_fidelity_offset=.005)
    if name in ("mechanism", "mechanism_e2e"):
        # Heterogeneous links and chain guards expose quality-aware planning.
        # The optional link floor is an additional service requirement shared
        # by all policies. The E2E-only control removes that requirement.
        common_floor = name == "mechanism"
        config = Config(alpha=10.0, qrise_guard_mode="chain", qrise_lookahead_slots=20.0,
                        baseline_fidelity_filter=False, baseline_guard_floor=common_floor,
                        qrise_reachability_filter=True,
                        qrise_floor_base=.965 if common_floor else 0.0,
                        qrise_floor_target_weight=.015 if common_floor else 0.0)
        optical = OpticalModel(source_fidelity_offset=.005, degraded_link_fraction=.20,
                               degraded_offset_min=-.05, degraded_offset_max=-.02,
                               phase_fidelity_amplitude=.012, quality_period_slots=120.0)
        return config, DeviceConfig(memory_rate_scale=.02), optical
    raise ValueError("Unknown experiment profile: " + name)
