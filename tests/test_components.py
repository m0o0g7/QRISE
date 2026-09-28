"""Circuit, resource, and accounting checks using actual NetSquid events."""
from dataclasses import replace
import networkx as nx
import numpy as np
import pytest
import netsquid as ns
from netsquid.qubits import qubitapi as qapi, ketstates as ks
from qrise_components.backend import Trial, run
from qrise_components.components import werner, BilateralTwirl
from qrise_components.config import DeviceConfig
from qrise_math.routing import Config
from qrise_math.topology import make_topology
from qrise_math.physics import bbpssw_once, mcsfm


def probe(action, device=None, topology=None):
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    ns.set_random_state(seed=812)

    class Probe(Trial):
        def run(self):
            self.value = yield from action(self)
            ns.sim_stop()

    t = Probe(
        "Q-RISE",
        topology or make_topology(20, 1701),
        0.9,
        Config(),
        device or DeviceConfig(memory_rate_scale=0, swap_noise=False),
    )
    t.start()
    ns.sim_run()
    return t


def inject(t, u, v, fidelity):
    """Prepare test inputs only; production inputs must come from QSource."""
    pair = t.reserve(u, v)
    qubits = qapi.create_qubits(2)
    qapi.assign_qstate(qubits, werner(fidelity))
    t.processors[u].put(qubits[0], positions=pair.a)
    t.processors[v].put(qubits[1], positions=pair.b)
    return pair


def test_twirl_is_trace_preserving_and_keeps_bell_overlap():
    ns.set_qstate_formalism(ns.QFormalism.DM)
    qubits = qapi.create_qubits(2)
    before = qapi.fidelity(qubits, ks.b00, squared=True)
    BilateralTwirl()(qubits)
    assert np.allclose(qapi.reduced_dm(qubits), werner(before), atol=1e-12)
    for q in qubits:
        qapi.discard(q)


def test_bbpssw_conditional_fidelity_and_keep_probability():
    expected_f, expected_p = bbpssw_once(0.8)

    def action(t):
        u, v = next(iter(t.topology.graph.edges))
        kept = 0
        for _ in range(250):
            left, right = inject(t, u, v, 0.8), inject(t, u, v, 0.8)
            pair = yield from t.purify_once(left, right)
            if pair is not None:
                kept += 1
                assert t.fidelity(pair) == pytest.approx(expected_f, abs=1e-10)
                t.release(pair)
        return kept / 250

    trial = probe(action)
    assert abs(trial.value - expected_p) < 4 * np.sqrt(
        expected_p * (1 - expected_p) / 250
    )
    assert not any(trial.allocated.values())
    assert trial.stats["purification_attempts"] == 250


@pytest.mark.parametrize("left_f,right_f", [(1, 1), (0.9, 0.8)])
def test_bsm_actual_corrections_match_ideal_werner_composition(left_f, right_f):
    outcomes = set()

    def action(t):
        original_messages = t.messages

        def capture(specifications):
            for _, _, payload in specifications:
                if isinstance(payload, dict) and "x" in payload:
                    outcomes.add((payload["x"], payload["z"]))
            return (yield from original_messages(specifications))

        t.messages = capture
        u, v, w = nx.shortest_path(
            t.topology.graph, t.topology.source, t.topology.destination
        )
        for _ in range(24):
            pair = yield from t.swap(inject(t, u, v, left_f), inject(t, v, w, right_f))
            assert t.fidelity(pair) == pytest.approx(
                mcsfm(left_f, right_f, 1), abs=1e-10
            )
            t.release(pair)

    trial = probe(action)
    assert not any(trial.allocated.values())
    assert trial.stats["swaps"] == 24
    assert outcomes == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_swap_noise_is_physical_affine_channel():
    def action(t):
        u, v, w = nx.shortest_path(
            t.topology.graph, t.topology.source, t.topology.destination
        )
        pair = yield from t.swap(inject(t, u, v, 0.9), inject(t, v, w, 0.8))
        return t.fidelity(pair)

    trial = probe(action, DeviceConfig(memory_rate_scale=0, swap_noise=True))
    ideal = mcsfm(0.9, 0.8, 1)
    p = 4 * (1 - Config().swap_fidelity) / 3
    assert trial.value == pytest.approx((1 - p) * ideal + p / 4)


def test_source_loss_herald_and_clock():
    topology = make_topology(20, 1701)
    original_state = topology.state

    def state(u, v, slot):
        values = original_state(u, v, slot)
        values.update(generation_p=0.5, fidelity=0.85)
        return values

    topology.state = state

    def action(t):
        u, v = next(iter(t.topology.graph.edges))
        for _ in range(100):
            pairs = yield from t.generate(u, v, t.topology.edge(u, v).width)
            for pair in pairs:
                assert t.fidelity(pair) == pytest.approx(0.85, abs=1e-10)
                t.release(pair)
        return ns.sim_time()

    trial = probe(action, topology=topology)
    attempts = 100 * topology.edge(*next(iter(topology.graph.edges))).width
    assert trial.stats["generation_attempts"] == attempts
    assert trial.stats["delivered_pairs"] + trial.stats["generation_losses"] == attempts
    assert abs(trial.stats["delivered_pairs"] - attempts / 2) < 4 * np.sqrt(
        attempts / 4
    )
    assert trial.value >= 99 * Config().slot_s * 1e9 / 3
    assert not any(trial.allocated.values())


def test_memory_decay_applies_to_held_pairs():
    def action(t):
        u, v = next(iter(t.topology.graph.edges))
        pair = inject(t, u, v, 0.9)
        yield t.await_timer(1e8)
        return t.fidelity(pair)

    trial = probe(action, DeviceConfig(swap_noise=False))
    assert trial.value == pytest.approx(0.25 + (0.9 - 0.25) * np.exp(-2 * 0.5 * 0.1))


def test_finite_memory_fails_without_hidden_extra_storage():
    topology = make_topology(20, 1701)
    topology.memories = {n: 1 for n in topology.graph}
    result = run("Q-RISE", topology, 0.9, 812)
    assert result.reason == "memory"
    assert result.counters["peak_memory_positions"] == 1


@pytest.mark.parametrize("policy", ["Q-RISE", "GREEDY", "HSP", "NOPP", "Q-CAST"])
def test_common_deadline_and_attempt_accounting(policy):
    result = run(
        policy, make_topology(20, 1701), 0.9, 812, replace(Config(), cutoff_slots=1)
    )
    assert result.reason == "cutoff"
    assert result.elapsed_ns == pytest.approx(1e6)
    assert result.latency_slots == pytest.approx(1)
    assert result.epr_pairs == result.counters["generation_attempts"]


def test_full_request_reports_qubit_fidelity_and_fractional_time():
    result = run("Q-RISE", make_topology(20, 1701), 0.9, 170104)
    assert result.fidelity is not None
    assert result.success == (result.fidelity >= 0.9)
    assert result.counters["swaps"] == len(result.path) - 2
    assert result.predicted_fidelity != result.fidelity
    assert result.epr_pairs == result.counters["generation_attempts"]
    assert (
        result.counters["delivered_pairs"] + result.counters["generation_losses"]
        == result.epr_pairs
    )
    assert result.latency_slots == result.elapsed_ns / 1e6


def test_consecutive_dead_ends_restart_and_regenerate_consumed_prefix(monkeypatch):
    import qrise_components.backend as backend
    from qrise_math.topology import Link, Topology
    from qrise_math.physics import OpticalModel

    graph = nx.Graph([(0, 1), (1, 2), (2, 3)])
    links = {(u, v): Link(u, v, 45, 0, 3) for u, v in graph.edges}
    topology = Topology(
        graph,
        links,
        {n: 10 for n in graph},
        {n: (45 * n, 0) for n in graph},
        0,
        3,
        OpticalModel(),
    )
    original = topology.state

    def state(u, v, slot):
        value = original(u, v, slot)
        if set((u, v)) == {2, 3}:
            value["fidelity"] = 0.5
        return value

    topology.state = state
    monkeypatch.setattr(backend, "qrise_rounds", lambda *args: (0, 0.92, 0.9))
    result = backend.run("Q-RISE", topology, 0.9, 812)
    assert result.reason == "no_admissible_neighbor"
    assert result.rollback_count == 2
    assert result.path == [0]
    assert result.rounds == []
    # After the first BSM, retrying the earlier frontier requires a fresh 0-1 pair.
    assert len(result.link_records) == 3
    assert result.counters["swaps"] == 1
    assert result.counters["restart_discarded_pairs"] == 2
    assert all(r.get("rolled_back") for r in result.link_records)


def test_nonzero_gate_noise_changes_the_actual_state():
    def action(t):
        u, v, w = nx.shortest_path(
            t.topology.graph, t.topology.source, t.topology.destination
        )
        pair = yield from t.swap(inject(t, u, v, 1), inject(t, v, w, 1))
        return t.fidelity(pair)

    trial = probe(
        action,
        DeviceConfig(
            memory_rate_scale=0, swap_noise=False, gate_depolar_probability=0.02
        ),
    )
    assert 0.5 < trial.value < 0.999


# Perfect elementary pairs isolate scheduling and memory ownership from MPR.
def perfect_topology(edges, destination, positions):
    from qrise_math.topology import Link, Topology
    from qrise_math.physics import OpticalModel

    graph = nx.Graph(edges)
    links = {tuple(sorted((u, v))): Link(u, v, 45, 0, 3) for u, v in graph.edges}
    topology = Topology(
        graph, links, {n: 10 for n in graph}, positions, 0, destination, OpticalModel()
    )
    original = topology.state

    def state(u, v, slot):
        values = original(u, v, slot)
        values.update(fidelity=1.0, generation_p=1.0)
        return values

    topology.state = state
    return topology


@pytest.mark.parametrize("policy", ["Q-RISE", "GREEDY", "HSP", "NOPP", "Q-CAST"])
def test_swap_completes_before_next_link_and_releases_relay_memory(monkeypatch, policy):
    import qrise_components.backend as backend

    topology = perfect_topology(
        [(0, 1), (1, 2), (2, 3)], 3, {n: (45 * n, 0) for n in range(4)}
    )
    monkeypatch.setattr(backend, "qrise_rounds", lambda *args: (0, 1.0, 0.9))
    monkeypatch.setattr(backend, "baseline_plan", lambda *args: ([0, 1, 2, 3], 0, 0.9))
    events, instances = [], []

    class ObservedTrial(Trial):
        def __init__(self, *args):
            super().__init__(*args)
            instances.append(self)

        def prepare_link(self, u, v, rounds):
            if self.chain is not None:
                assert (self.chain.u, self.chain.v) == (0, u)
                assert sum(len(positions) for positions in self.allocated.values()) == 2
                assert all(
                    not positions
                    for node, positions in self.allocated.items()
                    if node not in (0, u)
                )
            events.append(("prepare", u, v))
            return (yield from super().prepare_link(u, v, rounds))

        def swap(self, left, right):
            pair = yield from super().swap(left, right)
            events.append(("swapped", pair.u, pair.v))
            return pair

    monkeypatch.setattr(backend, "Trial", ObservedTrial)
    result = backend.run(
        policy,
        topology,
        0.9,
        812,
        device=DeviceConfig(memory_rate_scale=0, swap_noise=False),
    )
    assert events == [
        ("prepare", 0, 1),
        ("prepare", 1, 2),
        ("swapped", 0, 2),
        ("prepare", 2, 3),
        ("swapped", 0, 3),
    ]
    assert result.success and result.fidelity == pytest.approx(1)
    assert result.epr_pairs == 3 and result.counters["swaps"] == 2
    records = result.link_records
    assert records[1]["swap_started_ns"] == records[1]["prepared_ns"]
    assert (
        records[1]["swap_completed_ns"]
        <= records[2]["prepared_ns"] - records[2]["elapsed_ns"]
    )
    assert records[0]["storage_wait_ns"] > 0
    assert records[2]["storage_wait_ns"] == 0
    assert records[-1]["predicted_chain_fidelity"] == pytest.approx(
        result.predicted_fidelity
    )
    assert result.elapsed_ns > records[-1]["chain_ready_ns"]
    trial = instances[0]
    assert sum(map(len, trial.allocated.values())) == 2
    trial.release(trial.chain)
    assert not any(trial.allocated.values())


def test_dead_end_after_bsm_restarts_then_succeeds_without_free_resource_reuse(
    monkeypatch,
):
    import qrise_components.backend as backend

    topology = perfect_topology(
        [(0, 1), (1, 2), (2, 4), (1, 3), (3, 4)],
        4,
        {0: (0, 0), 1: (45, 0), 2: (90, 0), 3: (90, 0), 4: (135, 0)},
    )
    original = topology.state

    def state(u, v, slot):
        values = original(u, v, slot)
        if set((u, v)) == {2, 4}:
            values["fidelity"] = 0.5
        return values

    topology.state = state
    monkeypatch.setattr(backend, "qrise_rounds", lambda *args: (0, 1.0, 0.9))
    result = backend.run(
        "Q-RISE",
        topology,
        0.9,
        812,
        device=DeviceConfig(memory_rate_scale=0, swap_noise=False),
    )
    assert result.success and result.path == [0, 1, 3, 4]
    assert result.rollback_count == result.counters["restart_discarded_pairs"] == 1
    assert result.epr_pairs == 5 and result.counters["swaps"] == 3
    assert [r.get("rolled_back", False) for r in result.link_records] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert (
        result.counters["delivered_pairs"]
        - result.counters["swaps"]
        - result.counters["restart_discarded_pairs"]
        == 1
    )
    assert len(result.rounds) == 3


@pytest.mark.parametrize("deadline", [0.62, 0.70])
def test_deadline_during_incremental_bsm_or_correction_delivery(monkeypatch, deadline):
    import qrise_components.backend as backend

    topology = perfect_topology([(0, 1), (1, 2)], 2, {n: (45 * n, 0) for n in range(3)})
    monkeypatch.setattr(backend, "qrise_rounds", lambda *args: (0, 1.0, 0.9))
    result = backend.run(
        "Q-RISE",
        topology,
        0.9,
        812,
        replace(Config(), cutoff_slots=deadline),
        DeviceConfig(memory_rate_scale=0, swap_noise=False),
    )
    assert result.reason == "cutoff" and result.fidelity is None
    assert result.latency_slots == pytest.approx(deadline)
    assert len(result.link_records) == 2
    assert "swap_started_ns" in result.link_records[-1]
    assert "swap_completed_ns" not in result.link_records[-1]
    assert result.counters["swaps"] == 0 and result.epr_pairs == 2


def test_profile_fidelity_is_emitted_by_qsource_not_only_planner():
    from qrise_components.profiles import profile_settings

    _, _, optical = profile_settings("mechanism")
    topology = make_topology(20, 1701, optical)

    def action(t):
        u, v = next(iter(t.topology.graph.edges))
        expected = topology.state(u, v, 0)["fidelity"]
        pairs = yield from t.generate(u, v, topology.edge(u, v).width)
        assert pairs
        for pair in pairs:
            assert t.fidelity(pair) == pytest.approx(expected, abs=1e-10)
            t.release(pair)

    probe(action, topology=topology)
