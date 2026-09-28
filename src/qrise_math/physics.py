"""Scalar physical equations used by the Q-RISE reconstruction.

The disclosed Q-RISE equations are kept separate from reconstruction
assumptions.  In particular, :func:`effective_routing_fidelity` is an explicit
calibration hypothesis because the poster does not publish the elementary-link
fidelity configuration needed to reproduce its reported operating point.

This module represents fidelity as scalar values.  It does not create qubits,
density matrices, quantum channels, or NetSquid components.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, exp, log, isfinite


# Speed of light in vacuum, expressed in kilometres per second.
C_KM_S = 299_792.458


@dataclass(frozen=True)
class OpticalModel:
    """Fixed inputs for the Gao-inspired LEO optical-link model."""

    divergence_rad: float = 5e-6
    aperture_m: float = 0.15
    pointing_sigma_rad: float = 0.5e-6
    qber: float = 0.01
    damping_rate_s: float = 0.5
    rotation_sigma_rad: float = 0.7e-6
    satellite_speed_km_s: float = 7.589
    earth_radius_km: float = 6371.0
    altitude_km: float = 550.0
    # Shared source calibration changes emitted states, not measured outputs.
    source_fidelity_offset: float = 0.0
    # Persistent quality offsets model a declared fraction of degraded links.
    # The defaults disable heterogeneity.
    degraded_link_fraction: float = 0.0
    degraded_offset_min: float = -0.10
    degraded_offset_max: float = -0.04
    # Synthetic quality amplitude and geometric trace period (not an orbit).
    phase_fidelity_amplitude: float = 0.004
    dynamics_period_slots: float = 420.0
    # Quality drift and geometric distance can have independent periods.
    # Fast quality changes are a synthetic pointing/noise scenario, not a LEO orbit.
    quality_period_slots: float | None = None

    def __post_init__(self):
        if self.quality_period_slots is not None and (
            not isfinite(self.quality_period_slots) or self.quality_period_slots <= 0
        ):
            raise ValueError("Quality period must be finite and positive")
        if not all(
            isfinite(v)
            for v in (
                self.source_fidelity_offset,
                self.degraded_link_fraction,
                self.degraded_offset_min,
                self.degraded_offset_max,
                self.phase_fidelity_amplitude,
                self.dynamics_period_slots,
            )
        ):
            raise ValueError("Optical calibration inputs must be finite")
        if not -0.1 <= self.source_fidelity_offset <= 0.05:
            raise ValueError("Source fidelity offset must be between -.1 and .05")
        if not 0 <= self.degraded_link_fraction <= 1:
            raise ValueError("Degraded link fraction must be in [0, 1]")
        if not -0.3 <= self.degraded_offset_min <= self.degraded_offset_max <= 0:
            raise ValueError("Degraded offsets must satisfy -.3 <= min <= max <= 0")
        if not 0 <= self.phase_fidelity_amplitude <= 0.03:
            raise ValueError("Phase fidelity amplitude must be in [0, .03]")
        if self.dynamics_period_slots <= 0:
            raise ValueError("Dynamics period must be positive")


def capture_probability(distance_km: float, model: OpticalModel) -> float:
    """Return Gaussian-beam capture probability with pointing error.

    This is the implementation of Gao et al. v1 Eq. (21) used to parameterize
    stochastic EPR generation.  It is a pair-generation probability, not a
    quantum-channel simulation.
    """

    distance_m = distance_km * 1000.0
    geometric_capture = 1.0 - exp(
        -2.0 * model.aperture_m**2 / (model.divergence_rad**2 * distance_m**2)
    )
    pointing_penalty = 4.0 * model.pointing_sigma_rad**2 / model.divergence_rad**2
    return geometric_capture / (1.0 + pointing_penalty)


def gao_initial_fidelity(distance_km: float, model: OpticalModel) -> float:
    """Evaluate the unmodified Gao v1 elementary-link fidelity equations.

    Retained for diagnostics. With the declared optical inputs, this predicts
    lower initial fidelity and substantially higher purification requirements
    than the high-quality source hypothesis used by the mechanism profile.
    """

    distance_m = distance_km * 1000.0
    capture = 1.0 - exp(
        -2.0 * model.aperture_m**2 / (model.divergence_rad**2 * distance_m**2)
    )
    transmission_s = distance_km / C_KM_S
    rotation_angle = (
        model.satellite_speed_km_s
        * transmission_s
        / (model.earth_radius_km + model.altitude_km)
    )
    rotation = 0.5 * (
        1.0 + cos(2.0 * rotation_angle) * exp(-2.0 * model.rotation_sigma_rad**2)
    )
    channel = (capture + (1.0 - capture) * model.qber) / (
        1.0 + 3.0 * (1.0 - capture) * model.qber
    )
    return rotation * channel


def effective_routing_fidelity(
    distance_km: float, phase: float, phase_amplitude: float = 0.004
) -> float:
    """Return the declared elementary-fidelity reconstruction hypothesis.

    The Q-RISE poster omits its source/channel calibration.  This function uses
    a high-quality elementary-pair ensemble centred near 0.920 at 45 km and
    preserves a bounded distance/phase trend.  It is an input assumption; no
    output value is rescaled after simulation.
    """

    normalized_distance = (45.0 / distance_km) ** 2
    value = 0.920 + 0.012 * (normalized_distance - 1.0) + phase_amplitude * cos(phase)
    return max(0.89, min(0.945, value))


def stored_fidelity(
    initial: float,
    age_slots: float,
    slot_s: float,
    damping_rate_s: float,
) -> float:
    """Apply Gao v1 Eq. (40): ``F(t) = F0 exp(-2 Gamma t)``."""

    return initial * exp(-2.0 * damping_rate_s * age_slots * slot_s)


def cutoff_seconds(
    initial: float,
    threshold: float,
    distance_km: float,
    damping_rate_s: float,
) -> float:
    """Return the Gao v1 Eq. (42) lifetime, clipped for invalid input pairs."""

    if initial <= threshold:
        return 0.0
    propagation_s = distance_km / C_KM_S
    decoherence_s = -log(threshold / initial) / (2.0 * damping_rate_s)
    return propagation_s + decoherence_s


def bbpssw_once(fidelity: float) -> tuple[float, float]:
    """Apply one equal-input Werner-state BBPSSW recurrence.

    Returns ``(output_fidelity, keep_probability)``.  The mathematical
    reconstruction uses the output-fidelity recurrence; it does not sample a
    component-level purification circuit.
    """

    error = (1.0 - fidelity) / 3.0
    keep_probability = fidelity**2 + 2.0 * fidelity * error + 5.0 * error**2
    output_fidelity = (fidelity**2 + error**2) / keep_probability
    return output_fidelity, keep_probability


def minimum_rounds(
    initial: float,
    guard: float,
    limit: int = 12,
) -> tuple[int, float]:
    """Implement Q-RISE MPR: first BBPSSW iterate at or above ``F_guard``."""

    value = initial
    for rounds in range(limit + 1):
        if value >= guard:
            return rounds, value
        value, _ = bbpssw_once(value)
    raise ValueError("purification guard is unreachable within the limit")


def mcsfm(left: float, right: float, swap_fidelity: float) -> float:
    """Compose adjacent link fidelities using Q-RISE Eq. (4)."""

    ideal_term = left * right
    error_term = (1.0 - left) * (1.0 - right) / 3.0
    return (ideal_term + error_term) * swap_fidelity


def equal_link_guard(
    target: float,
    hops: int,
    swap_fidelity: float,
    margin: float = 0.004,
) -> float:
    """Infer a common per-link guard from the disclosed MCSFM recurrence.

    Q-RISE does not publish how ``F_guard`` is chosen.  This monotone bisection
    is therefore a declared reconstruction assumption.  The fixed margin is an
    input and is not adjusted after observing a trial result.
    """

    wanted = min(0.999, target + margin)
    lower = max(0.5, wanted)
    upper = 0.999999
    for _ in range(70):
        midpoint = (lower + upper) / 2.0
        accumulated = midpoint
        for _ in range(hops - 1):
            accumulated = mcsfm(accumulated, midpoint, swap_fidelity)
        if accumulated >= wanted:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def chain_link_guard(
    chain_fidelity: float | None,
    target: float,
    hops: int,
    swap_fidelity: float,
    margin: float = 0.004,
) -> float:
    """Per-link guard that lets the *predicted* chain reach the E2E target.

    ``chain_fidelity`` is the MCSFM prediction of the already swapped
    source-to-frontier pair (None before the first hop) and ``hops`` counts the
    next link plus the estimated remaining hops.  Links that were purified above
    their guard (discrete rounds overshoot) therefore relax later guards.  This
    is the reconstruction's reading of Q-RISE's E2E-aware MPR, not poster code.
    """

    if hops < 1 or not 0 < target < 1 or not 0 < swap_fidelity <= 1:
        raise ValueError("Invalid chain guard request")
    if chain_fidelity is not None and not 0.25 <= chain_fidelity <= 1:
        raise ValueError("Invalid predicted chain fidelity")
    wanted = min(0.999, target + margin)

    def reach(link: float) -> float:
        accumulated = link if chain_fidelity is None else chain_fidelity
        for _ in range(hops - (1 if chain_fidelity is None else 0)):
            accumulated = mcsfm(accumulated, link, swap_fidelity)
        return accumulated

    if reach(0.999999) < wanted:
        raise ValueError("predicted chain cannot reach the E2E target")
    lower, upper = 0.5, 0.999999
    if reach(lower) >= wanted:
        return lower
    for _ in range(70):
        midpoint = (lower + upper) / 2.0
        if reach(midpoint) >= wanted:
            upper = midpoint
        else:
            lower = midpoint
    return upper


def stored_werner_fidelity(initial: float, elapsed_s: float, rate_s: float) -> float:
    """Prediction for two equal-rate memory channels used by the processors.

    This is a planner estimate, never assigned to the actual quantum state.
    Unlike Gao's F exp(-2 Gamma t), depolarization tends to the mixed-state .25.
    """
    if (
        not all(isfinite(v) for v in (initial, elapsed_s, rate_s))
        or elapsed_s < 0
        or rate_s < 0
    ):
        raise ValueError("Invalid memory prediction parameters")
    return 0.25 + (initial - 0.25) * exp(-2 * rate_s * elapsed_s)
