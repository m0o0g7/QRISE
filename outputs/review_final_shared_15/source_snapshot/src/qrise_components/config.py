"""Shared physical assumptions, never selected by routing policy."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceConfig:
    """Unpublished device inputs fixed before running the experiment."""

    # recon_Qu used slot costs; here durations belong to physical instructions.
    single_gate_ns: float = 1000.0
    cnot_ns: float = 10000.0
    measurement_ns: float = 10000.0
    # Ideal local gates isolate storage/communication costs in the default run.
    # These assumptions are shared by ALL policies and are not poster values.
    gate_depolar_probability: float = 0.0
    memory_rate_scale: float = 1.0
    swap_noise: bool = True
    twirl: bool = True

    def __post_init__(self):
        if min(self.single_gate_ns, self.cnot_ns, self.measurement_ns) <= 0:
            raise ValueError("Instruction durations must be positive")
        if self.memory_rate_scale < 0:
            raise ValueError("Memory rate scale must be nonnegative")
        if not 0 <= self.gate_depolar_probability <= 1:
            raise ValueError("Invalid gate depolarization probability")
