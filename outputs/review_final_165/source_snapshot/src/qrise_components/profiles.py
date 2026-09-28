"""Declared input scenarios, not fitted output corrections."""
from qrise_math.physics import OpticalModel
from qrise_math.routing import Config
from .config import DeviceConfig


def profile_settings(name):
    """Return policy-independent physics and declared routing weights.

    The poster does not disclose these hardware inputs. The fig2 scenario is
    exploratory calibration on seeds 1701/1702 and 1801-1803, not an author setup.
    Neither profile guarantees the paper's relative performance.
    """
    if name == "recon":
        return Config(), DeviceConfig(), OpticalModel()
    if name in ("fig2", "hardware_only"):
        #v1 recon_Qu F~.920 and the previous component memory rate .5/s caused
        #v1 storage losses during real purification ACKs. Use F~.925 and .01/s
        #v1 for EVERY policy; source states and processor noise actually change.
        #v1 Original Eq.(3) had alpha=1, making its small fidelity differences
        #v1 weak against buffer-width differences. Alpha=10 gives link quality
        #v1 more influence without changing the equation or baseline adapters.
        #v1 This weight is a calibration choice, not a disclosed poster value.
        config = Config(alpha=10.0 if name == "fig2" else 1.0)
        return config, DeviceConfig(memory_rate_scale=.02), OpticalModel(source_fidelity_offset=.005)
    if name in ("mechanism", "mechanism_e2e"):
        #v1 Review of Claude: retain common link heterogeneity and predictive
        #v1 chain guards. No baseline min5, no invented setup delay, no .5/s
        #v1 expiry imposed on memories actually running at .01/s.
        #v1 A common service floor is an explicit unpublished assumption;
        #v1 mechanism_e2e removes it for EVERY policy as a stronger control.
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
