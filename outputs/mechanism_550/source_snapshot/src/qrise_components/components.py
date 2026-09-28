"""Actual sources, lossy links, finite processors, and CPTP noise maps."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from netsquid.components import QSource, QuantumChannel, ClassicalChannel, QuantumProcessor, PhysicalInstruction
from netsquid.components.qsource import SourceStatus
from netsquid.components.models.qerrormodels import QuantumErrorModel, DepolarNoiseModel
from netsquid.components.instructions import INSTR_CNOT, INSTR_H, INSTR_MEASURE, INSTR_X, INSTR_Z
from netsquid.qubits import qubitapi as qapi, ketstates as ks
from netsquid.qubits.operators import Operator
from netsquid.qubits.state_sampler import StateSampler
import netsquid as ns
from qrise_math.physics import C_KM_S


@dataclass(frozen=True)
class Pair:
    """Two allocated memory positions, one at each endpoint."""
    u: int
    a: int
    v: int
    b: int


def werner(fidelity):
    """Initial-state hypothesis consistent with planner BBPSSW inputs."""
    bell = np.outer(ks.b00, ks.b00.conj())
    return fidelity * bell + (1 - fidelity) / 3 * (np.eye(4) - bell)


class CaptureLoss(QuantumErrorModel):
    """Apply capture probability once, to the travelling half."""
    def __init__(self):
        super().__init__()
        self.probability = 1.0

    def error_operation(self, qubits, **kwargs):
        for index in range(len(qubits)):
            self.lose_qubit(qubits, index, prob_loss=1 - self.probability)


def make_processor(node, positions, optical, device):
    memory = [DepolarNoiseModel(depolar_rate=optical.damping_rate_s * device.memory_rate_scale)
              for _ in range(positions)]
    instructions = []
    for instruction, duration in ((INSTR_CNOT, device.cnot_ns), (INSTR_H, device.single_gate_ns),
                                  (INSTR_X, device.single_gate_ns), (INSTR_Z, device.single_gate_ns),
                                  (INSTR_MEASURE, device.measurement_ns)):
        noise = (DepolarNoiseModel(depolar_rate=device.gate_depolar_probability, time_independent=True)
                 if device.gate_depolar_probability else None)
        instructions.append(PhysicalInstruction(instruction, duration=duration, quantum_noise_model=noise))
    # recon_Qu only stored memory counts. Here they cap actual qubit positions.
    return QuantumProcessor("node-{}".format(node), num_positions=positions,
                            mem_noise_models=memory, phys_instructions=instructions, fallback_to_nonphysical=False)


class Lane:
    """One source and quantum channel, with classical herald return."""
    def __init__(self, network, u, v, index):
        self.net, self.u, self.v = network, u, v
        self.pair = None
        self.done = True
        self.success = False
        self.available_ns = 0.0
        name = "{}-{}-{}".format(u, v, index)
        self.loss = CaptureLoss()
        self.source = QSource("source-" + name, state_sampler=StateSampler([werner(1)], [1]),
                              num_ports=2, status=SourceStatus.EXTERNAL)
        self.channel = QuantumChannel("quantum-" + name, delay=1, transmit_empty_items=True,
                                      models={"quantum_loss_model": self.loss})
        self.herald = ClassicalChannel("herald-" + name, delay=1)
        self.source.ports["qout0"].bind_output_handler(self.local_arrival)
        self.source.ports["qout1"].connect(self.channel.ports["send"])
        self.channel.ports["recv"].bind_output_handler(self.remote_arrival)
        self.herald.ports["recv"].bind_output_handler(self.herald_arrival)

    def local_arrival(self, message):
        self.net.processors[self.u].put(message.items[0], positions=self.pair.a)

    def remote_arrival(self, message):
        qubit = message.items[0] if message.items else None
        self.success = qubit is not None
        if self.success:
            self.net.processors[self.v].put(qubit, positions=self.pair.b)
        self.net.stats["delivered_pairs" if self.success else "generation_losses"] += 1
        self.net.stats["classical_messages"] += 1
        self.herald.send(self.success)

    def herald_arrival(self, message):
        self.done = True
        self.net.wake()

    def trigger(self, pair, state):
        self.pair, self.done, self.success = pair, False, False
        delay = state["distance_km"] / C_KM_S * 1e9
        self.channel.models["delay_model"].properties["delay"] = delay
        self.herald.models["delay_model"].properties["delay"] = delay
        self.channel.properties["length"] = state["distance_km"]
        self.loss.probability = state["generation_p"]
        # recon_Qu's one-slot forecast age remains in planning only. Actual
        # memory aging starts at emission, avoiding double-counted aging.
        self.source.state_sampler = StateSampler([werner(state["fidelity"])], [1])
        self.available_ns = ns.sim_time() + self.net.cfg.slot_s * 1e9 / self.net.cfg.attempts_per_lane_slot
        self.net.stats["generation_attempts"] += 1
        self.source.trigger()


def _twirl_operators():
    """24 coordinated local Cliffords U tensor U*, preserving Phi+."""
    h = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    s = np.diag([1, 1j])
    pending, found = [np.eye(2, dtype=complex)], []
    while pending:
        u = pending.pop()
        entry = u.ravel()[np.flatnonzero(np.abs(u.ravel()) > 1e-8)[0]]
        u = u / (entry / abs(entry))
        if any(np.allclose(u, old) for old in found):
            continue
        found.append(u)
        pending.extend((h @ u, s @ u))
    assert len(found) == 24
    return tuple(Operator("twirl-{}".format(i), np.kron(u, u.conj())) for i, u in enumerate(found))


TWIRL_OPERATORS = _twirl_operators()


class BilateralTwirl(QuantumErrorModel):
    """CPTP ensemble of coordinated local rotations, not a fidelity reset.

    Ideal instantaneous twirling with shared randomness is an explicit common
    device assumption. No scalar fidelity is read to construct the output.
    """
    def error_operation(self, qubits, **kwargs):
        qapi.multi_operate(qubits, TWIRL_OPERATORS, weights=(1 / 24,) * 24)
