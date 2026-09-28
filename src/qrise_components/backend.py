"""Execute planning decisions using real NetSquid quantum/classical events."""
from __future__ import annotations
from dataclasses import dataclass, field
import networkx as nx
import netsquid as ns
from netsquid.protocols import Protocol
from netsquid.components import ClassicalChannel
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.qubits import qubitapi as qapi, ketstates as ks
from qrise_math.physics import C_KM_S, mcsfm, stored_werner_fidelity
from qrise_math.routing import (
    Config,
    _cost,
    _valid_state,
    _observed_state,
    _makes_forward_progress,
    qrise_rounds,
    baseline_plan,
    predicted_state,
    forward_hops,
)
from .config import DeviceConfig
from .components import Pair, Lane, make_processor, BilateralTwirl
from .programs import PurifyProgram, SwapProgram, CorrectProgram


@dataclass
class Result:
    """Component outcomes; epr_pairs retains the original attempt-count unit."""

    policy: str
    success: bool
    fidelity: float | None
    epr_pairs: int
    latency_slots: float
    path: list[int]
    rounds: list[int]
    reason: str
    rollback_count: int
    elapsed_ns: float
    predicted_fidelity: float | None
    counters: dict = field(default_factory=dict)
    link_records: list = field(default_factory=list)


class ResourceFailure(Exception):
    """A finite-memory schedule cannot prepare the requested pair."""


class Trial(Protocol):
    """Shared physical executor; policy only chooses paths and round counts."""

    def __init__(self, policy, topology, target, cfg, device):
        super().__init__(name="component-trial")
        self.add_signal("io")
        self.policy, self.topology, self.target = policy, topology, target
        self.cfg, self.device = cfg, device
        self.processors = {
            node: make_processor(node, size, topology.optical, device)
            for node, size in topology.memories.items()
        }
        self.allocated = {node: set() for node in topology.graph}
        self.lanes, self.classical = {}, {}
        self.stats = dict(
            generation_attempts=0,
            delivered_pairs=0,
            generation_losses=0,
            purification_attempts=0,
            purification_failures=0,
            purification_discarded_pairs=0,
            swaps=0,
            classical_messages=0,
            peak_memory_positions=0,
            peak_parallel_emissions=0,
            source_wait_ns=0.0,
            generation_wait_ns=0.0,
            purification_wait_ns=0.0,
            swapping_wait_ns=0.0,
        )
        # Keep one source-to-frontier pair instead of storing every elementary link.
        self.path, self.rounds = [topology.source], []
        self.chain = None
        self.predicted_chain = None
        self.predicted_chain_ns = float(ns.sim_time())
        self.stats["restart_discarded_pairs"] = 0
        self.records, self.predictions = [], []
        self.rollbacks = 0
        self.result = None
        self.twirl = BilateralTwirl()

    @property
    def slot(self):
        """Current simulation time expressed in configured slots."""
        return float(ns.sim_time()) / (self.cfg.slot_s * 1e9)

    def wake(self):
        self.send_signal("io")

    def wait_until(self, predicate):
        """Suspend the protocol until an I/O event makes the condition true."""
        while not predicate():
            yield self.await_signal(self, "io")

    def reserve(self, u, v):
        """Reserve one free memory position at each endpoint."""
        positions = []
        for node in (u, v):
            free = sorted(
                set(range(self.topology.memories[node])) - self.allocated[node]
            )
            if not free:
                raise ResourceFailure("memory")
            positions.append(free[0])
        for node, position in zip((u, v), positions):
            self.allocated[node].add(position)
            self.stats["peak_memory_positions"] = max(
                self.stats["peak_memory_positions"], len(self.allocated[node])
            )
        return Pair(u, positions[0], v, positions[1])

    def release_position(self, node, position):
        """Discard any stored qubit and release its allocated position."""
        if position not in self.allocated[node]:
            return
        (qubit,) = self.processors[node].pop(positions=[position])
        if qubit is not None:
            qapi.discard(qubit)
        self.allocated[node].remove(position)

    def release(self, pair):
        self.release_position(pair.u, pair.a)
        self.release_position(pair.v, pair.b)

    def qubits(self, pair):
        return [
            self.processors[pair.u].peek(positions=[pair.a])[0],
            self.processors[pair.v].peek(positions=[pair.b])[0],
        ]

    def fidelity(self, pair):
        """Read squared Bell-state fidelity for diagnostics and final reporting."""
        return float(qapi.fidelity(self.qubits(pair), ks.b00, squared=True))

    def messages(self, specifications):
        """Wait for actual classical-channel deliveries, not a slot estimate."""
        boxes = []
        for u, v, payload in specifications:
            key = (u, v)
            if key not in self.classical:
                self.classical[key] = ClassicalChannel(
                    "classical-{}-{}".format(u, v), delay=1
                )
            channel = self.classical[key]
            channel.models["delay_model"].properties["delay"] = (
                self.topology.edge(u, v).distance(self.slot) / C_KM_S * 1e9
            )
            box = []

            def receive(message, box=box):
                box.append(message.items[0])
                self.wake()

            channel.ports["recv"].bind_output_handler(receive)
            channel.send([payload])
            self.stats["classical_messages"] += 1
            boxes.append(box)
        yield from self.wait_until(lambda: all(boxes))
        return [box[0] for box in boxes]

    def generate(self, u, v, count):
        """Trigger up to link.width independent lanes within reserved memory."""
        start = ns.sim_time()
        key = (u, v)
        if key not in self.lanes:
            self.lanes[key] = [
                Lane(self, u, v, i) for i in range(self.topology.edge(u, v).width)
            ]
        selected = sorted(self.lanes[key], key=lambda lane: lane.available_ns)[:count]
        available = max(lane.available_ns for lane in selected)
        if available > ns.sim_time():
            wait = available - ns.sim_time()
            self.stats["source_wait_ns"] += wait
            yield self.await_timer(wait)
        self.stats["peak_parallel_emissions"] = max(
            self.stats["peak_parallel_emissions"], len(selected)
        )
        for lane in selected:
            pair = self.reserve(u, v)
            lane.trigger(pair, self.topology.state(u, v, self.slot))
        yield from self.wait_until(lambda: all(lane.done for lane in selected))
        pairs = []
        for lane in selected:
            if lane.success:
                pairs.append(lane.pair)
            else:
                self.release(lane.pair)
        self.stats["generation_wait_ns"] += ns.sim_time() - start
        return pairs

    def purify_once(self, left, right):
        """Bilateral CNOT, measurement exchange, postselection and twirling."""
        start = ns.sim_time()
        self.stats["purification_attempts"] += 1
        a, b = self.processors[left.u], self.processors[left.v]
        pa, pb = PurifyProgram(), PurifyProgram()
        a.execute_program(pa, qubit_mapping=[left.a, right.a], error_on_fail=True)
        b.execute_program(pb, qubit_mapping=[left.b, right.b], error_on_fail=True)
        yield self.await_program(a) & self.await_program(b)
        ma, mb = int(pa.output["m"][0]), int(pb.output["m"][0])
        received = yield from self.messages(
            [(left.u, left.v, ma), (left.v, left.u, mb)]
        )
        self.release(right)
        if received[0] != mb or received[1] != ma:
            self.release(left)
            self.stats["purification_failures"] += 1
            self.stats["purification_discarded_pairs"] += 2
            pair = None
        else:
            # Coordinated local rotations restore the Werner-state premise
            # through a physical CPTP map rather than assigning a scalar F.
            if self.device.twirl:
                self.twirl(self.qubits(left))
            self.stats["purification_discarded_pairs"] += 1
            pair = left
        self.stats["purification_wait_ns"] += ns.sim_time() - start
        return pair

    def prepare_link(self, u, v, rounds):
        """Balanced recurrence tree with bounded live storage and real retries."""
        levels = [[] for _ in range(rounds + 1)]
        while not levels[rounds]:
            ready = next(
                (level for level in reversed(range(rounds)) if len(levels[level]) >= 2),
                None,
            )
            if ready is not None:
                right, left = levels[ready].pop(), levels[ready].pop()
                survivor = yield from self.purify_once(left, right)
                if survivor is not None:
                    levels[ready + 1].append(survivor)
                continue
            free = min(
                self.topology.memories[n] - len(self.allocated[n]) for n in (u, v)
            )
            if free <= 0:
                raise ResourceFailure("memory")
            remaining = 2**rounds - sum(
                (2**level) * len(pairs) for level, pairs in enumerate(levels)
            )
            count = min(free, remaining, self.topology.edge(u, v).width)
            levels[0].extend((yield from self.generate(u, v, count)))
        result = levels[rounds].pop()
        for pairs in levels:
            for pair in pairs:
                self.release(pair)
        return result

    def swap(self, left, right):
        """Consume relay qubits, transmit BSM outcomes and correct the new endpoint."""
        start = ns.sim_time()
        assert left.v == right.u
        processor = self.processors[left.v]
        program = SwapProgram()
        processor.execute_program(
            program, qubit_mapping=[left.b, right.a], error_on_fail=True
        )
        yield self.await_program(processor)
        payload = dict(x=int(program.output["x"][0]), z=int(program.output["z"][0]))
        (received,) = yield from self.messages([(right.u, right.v, payload)])
        destination = self.processors[right.v]
        if received["x"] or received["z"]:
            destination.execute_program(
                CorrectProgram(),
                qubit_mapping=[right.b],
                error_on_fail=True,
                **received,
            )
            yield self.await_program(destination)
        self.release_position(left.v, left.b)
        self.release_position(right.u, right.a)
        result = Pair(left.u, left.a, right.v, right.b)
        if self.device.swap_noise:
            # Depolarization is affine: F'=(1-p)F+p/4. Match swap_fidelity
            # at the perfect-Bell input; the scalar planner is an approximation.
            p = 4 * (1 - self.cfg.swap_fidelity) / 3
            DepolarNoiseModel(depolar_rate=p, time_independent=True)(
                [self.qubits(result)[1]]
            )
        self.stats["swaps"] += 1
        self.stats["swapping_wait_ns"] += ns.sim_time() - start
        return result

    def record_link(self, pair, rounds, predicted, guard, start):
        self.rounds.append(rounds)
        self.predictions.append(predicted)
        self.records.append(
            dict(
                u=pair.u,
                v=pair.v,
                rounds=rounds,
                guard=guard,
                predicted_fidelity=predicted,
                prepared_fidelity=self.fidelity(pair),
                prepared_ns=float(ns.sim_time()),
                elapsed_ns=float(ns.sim_time() - start),
            )
        )

    def planning_chain(self, extra_slots=0.0):
        """Age a prediction at the physical memory rate without inspecting qubits."""
        if self.predicted_chain is None:
            return None
        if self.cfg.qrise_guard_mode != "chain":
            return self.predicted_chain
        age_s = max(0.0, float(ns.sim_time()) - self.predicted_chain_ns) * 1e-9
        age_s += extra_slots * self.cfg.slot_s
        rate = self.topology.optical.damping_rate_s * self.device.memory_rate_scale
        return stored_werner_fidelity(self.predicted_chain, age_s, rate)

    def extend_chain(self, pair):
        """Connect a newly prepared link before planning/preparing the next hop."""
        # MCSFM supplies planning estimates and diagnostics. Measured fidelity
        # comes from the state after BSM, classical delivery and correction.
        record = self.records[-1]
        if self.chain is None:
            self.chain = pair
            self.predicted_chain = record["predicted_fidelity"]
        else:
            start = float(ns.sim_time())
            record["swap_started_ns"] = start
            record["pre_swap_fidelity"] = self.fidelity(pair)
            record["storage_wait_ns"] = start - record["prepared_ns"]
            record["chain_pre_swap_fidelity"] = self.fidelity(self.chain)
            # The first elementary pair waits for hop two. Later swaps consume
            # an accumulated chain, so do not inspect already measured old pairs.
            if len(self.rounds) == 2:
                first = next(r for r in self.records if not r.get("rolled_back"))
                first["pre_swap_fidelity"] = record["chain_pre_swap_fidelity"]
                first["storage_wait_ns"] = start - first["prepared_ns"]
                first["swap_started_ns"] = start
            self.chain = yield from self.swap(self.chain, pair)
            # Age the accumulated estimate over waiting and BSM time at the
            # actual processor rate. This never overwrites the quantum state.
            self.predicted_chain = mcsfm(
                self.planning_chain(),
                record["predicted_fidelity"],
                self.cfg.swap_fidelity,
            )
            record["swap_completed_ns"] = float(ns.sim_time())
        self.predicted_chain_ns = float(ns.sim_time())
        record["chain_ready_ns"] = self.predicted_chain_ns
        record["chain_fidelity"] = self.fidelity(self.chain)
        record["predicted_chain_fidelity"] = self.predicted_chain

    def restart_chain(self, blocked):
        """Discard an irreversible chain and retry routing from the source."""
        # BSM consumes the intermediate qubits irreversibly. Block the failed
        # branch and regenerate from the source, retaining all previous costs.
        blocked.setdefault(self.path[-2], set()).add(self.path[-1])
        self.release(self.chain)
        self.stats["restart_discarded_pairs"] += 1
        for record in self.records:
            if not record.get("rolled_back"):
                record["rolled_back"] = True
        self.chain = None
        self.predicted_chain = None
        self.predicted_chain_ns = float(ns.sim_time())
        self.path = [self.topology.source]
        self.rounds.clear()
        self.predictions.clear()
        self.rollbacks += 1

    def run_qrise(self):
        """Select, prepare and connect one hop at a time until destination or failure."""
        blocked = {}
        while self.path[-1] != self.topology.destination:
            current = self.path[-1]
            candidates = []
            hop_map = (
                forward_hops(self.topology, self.slot, self.cfg, blocked)
                if self.cfg.qrise_reachability_filter
                else None
            )
            for peer in self.topology.graph.neighbors(current):
                if peer in self.path or peer in blocked.get(current, set()):
                    continue
                if hop_map is not None and peer not in hop_map:
                    continue
                if not _makes_forward_progress(self.topology, current, peer):
                    continue
                state = predicted_state(
                    self.topology, current, peer, self.slot, self.cfg
                )
                if _valid_state(state, self.cfg):
                    candidates.append((_cost(state, self.cfg), peer, state))
            if not candidates:
                if len(self.path) == 1:
                    return "no_admissible_neighbor"
                self.restart_chain(blocked)
                # Retain the one-slot retry control delay; regeneration is physical.
                yield self.await_timer(self.cfg.slot_s * 1e9)
                continue
            _, peer, state = min(candidates, key=lambda item: (item[0], item[1]))
            try:
                chain_estimate = self.planning_chain(self.cfg.qrise_lookahead_slots)
                rounds, predicted, guard = qrise_rounds(
                    self.topology,
                    self.path,
                    peer,
                    state,
                    self.target,
                    self.cfg,
                    chain_estimate,
                    None if hop_map is None else hop_map[peer],
                )
            except (nx.NetworkXNoPath, ValueError):
                blocked.setdefault(current, set()).add(peer)
                yield self.await_timer(self.cfg.slot_s * 1e9)
                continue
            start = ns.sim_time()
            pair = yield from self.prepare_link(current, peer, rounds)
            self.path.append(peer)
            self.record_link(pair, rounds, predicted, guard, start)
            self.records[-1].update(
                planning_initial_fidelity=state["fidelity"],
                planning_chain_fidelity=chain_estimate,
                forecast_slots=self.cfg.qrise_lookahead_slots,
            )
            # Finish swapping this hop before observing/selecting the next one.
            yield from self.extend_chain(pair)
        return None

    def run_baseline(self):
        """Execute the fixed path and purification plan on the shared components."""
        try:
            plan = baseline_plan(self.topology, self.policy, self.target, self.cfg)
        except ValueError:
            return "guard"
        if plan is None:
            self.path = []
            return "no_path"
        self.path, rounds, guard = plan
        for u, v in zip(self.path, self.path[1:]):
            from qrise_math.physics import bbpssw_once

            predicted = _observed_state(self.topology, u, v, self.slot, self.cfg)[
                "fidelity"
            ]
            for _ in range(rounds):
                predicted, _ = bbpssw_once(predicted)
            start = ns.sim_time()
            pair = yield from self.prepare_link(u, v, rounds)
            self.record_link(pair, rounds, predicted, guard, start)
            # Baselines keep their fixed plans but share the same immediate BSM
            # schedule as Q-RISE, without policy-specific physical penalties.
            yield from self.extend_chain(pair)
        # Count only source triggers and elapsed simulation time.
        return None

    def finish(self, reason, fidelity=None):
        """Capture measured resource counts, time and the terminal request status."""
        predicted = None
        if self.predictions:
            predicted = self.predictions[0]
            for value in self.predictions[1:]:
                predicted = mcsfm(predicted, value, self.cfg.swap_fidelity)
        if self.cfg.qrise_guard_mode == "chain":
            predicted = self.planning_chain()
        self.stats["in_flight_emissions"] = (
            self.stats["generation_attempts"]
            - self.stats["delivered_pairs"]
            - self.stats["generation_losses"]
        )
        self.result = Result(
            self.policy,
            reason == "success",
            fidelity,
            self.stats["generation_attempts"],
            self.slot,
            list(self.path),
            list(self.rounds),
            reason,
            self.rollbacks,
            float(ns.sim_time()),
            predicted,
            dict(self.stats),
            list(self.records),
        )

    def run(self):
        try:
            reason = yield from (
                self.run_qrise() if self.policy == "Q-RISE" else self.run_baseline()
            )
            if reason is None:
                # The E2E pair already exists; each hop was swapped during
                # construction. There is no deferred swapping pass here.
                pair = self.chain
                # Confirm end-to-end availability back to the source over the
                # selected physical path before reporting completion/fidelity.
                for u, v in zip(reversed(self.path[1:]), reversed(self.path[:-1])):
                    yield from self.messages([(u, v, "end_to_end_ready")])
                fidelity = self.fidelity(pair)
                self.finish(
                    "success" if fidelity >= self.target else "fidelity", fidelity
                )
            else:
                self.finish(reason)
        except ResourceFailure as exc:
            self.finish(str(exc))
        ns.sim_stop()


def run(policy, topology, target, seed, config=None, device=None):
    """Run one independent density-matrix simulation until completion/deadline."""
    cfg, physical = config or Config(), device or DeviceConfig()
    if policy not in ("GREEDY", "HSP", "NOPP", "Q-CAST", "Q-RISE"):
        raise ValueError("Unknown policy: " + policy)
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    ns.set_random_state(seed=seed)
    trial = Trial(policy, topology, target, cfg, physical)
    trial.start()
    # Apply one deadline to every policy, including final confirmation messages.
    # A rejected first hop may already have finished synchronously in start().
    if trial.result is None:
        ns.sim_run(end_time=cfg.cutoff_slots * cfg.slot_s * 1e9)
    if trial.result is None:
        trial.finish("cutoff")
        trial.stop()
    return trial.result
