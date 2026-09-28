"""Physical consistency of predictive guards, link quality and shared controls."""
from dataclasses import replace
from math import exp
import networkx as nx
import pytest
import netsquid as ns
from qrise_math.physics import (
    OpticalModel,
    chain_link_guard,
    equal_link_guard,
    mcsfm,
    minimum_rounds,
)
from qrise_math.routing import (
    Config,
    baseline_plan,
    qrise_rounds,
    predicted_state,
    _observed_state,
)
from qrise_math.topology import make_topology
from qrise_components.config import DeviceConfig
from qrise_components.profiles import profile_settings
from qrise_components.backend import Trial, run


def test_heterogeneity_does_not_change_graph_memory_loss_or_motion():
    cfg, device, optical = profile_settings("mechanism")
    base = make_topology(
        50, 1701, replace(optical, degraded_link_fraction=0, quality_period_slots=None)
    )
    varied = make_topology(50, 1701, optical)
    assert base.memories == varied.memories and set(base.links) == set(varied.links)
    assert any(e.quality_offset < 0 for e in varied.links.values())
    for u, v in base.links:
        a, b = base.edge(u, v), varied.edge(u, v)
        assert (a.width, a.base_distance_km, a.phase, a.period_slots) == (
            b.width,
            b.base_distance_km,
            b.phase,
            b.period_slots,
        )
        for slot in (0, 20, 100):
            before, after = base.state(u, v, slot), varied.state(u, v, slot)
            assert before["distance_km"] == after["distance_km"]
            assert before["generation_p"] == after["generation_p"]


def test_chain_guard_respects_residual_budget_and_rejects_impossible_target():
    assert chain_link_guard(None, 0.9, 4, 0.99) == pytest.approx(
        equal_link_guard(0.9, 4, 0.99)
    )
    loose = chain_link_guard(0.99, 0.9, 3, 0.99)
    tight = chain_link_guard(0.95, 0.9, 3, 0.99)
    assert loose < tight
    value = 0.95
    for _ in range(3):
        value = mcsfm(value, tight, 0.99)
    assert value == pytest.approx(0.904)
    with pytest.raises(ValueError):
        chain_link_guard(0.6, 0.9, 3, 0.99)
    with pytest.raises(ValueError):
        chain_link_guard(None, 0.99, 6, 0.99)


def test_predictive_window_is_optional_and_does_not_mutate_topology():
    cfg, _, optical = profile_settings("mechanism")
    t = make_topology(50, 1701, optical)
    u, v = next(iter(t.links))
    observed = _observed_state(t, u, v, 0, cfg)
    result = predicted_state(t, u, v, 0, cfg)
    window = [_observed_state(t, u, v, k, cfg)["fidelity"] for k in range(21)]
    assert result["fidelity"] == min(window)
    assert result["fidelity"] <= observed["fidelity"]
    assert _observed_state(t, u, v, 0, cfg) == observed
    assert (
        predicted_state(t, u, v, 0, replace(cfg, qrise_lookahead_slots=0)) == observed
    )


def constant_fidelity_topology(fidelity):
    t = make_topology(20, 1701)
    original = t.state

    def state(u, v, slot):
        values = original(u, v, slot)
        values["fidelity"] = fidelity
        return values

    t.state = state
    return t


@pytest.mark.parametrize("policy", ["GREEDY", "HSP", "NOPP", "Q-CAST"])
def test_common_service_floor_does_not_impose_minimum_five_rounds(policy):
    cfg, _, _ = profile_settings("mechanism")
    t = constant_fidelity_topology(0.94)
    path, rounds, guard = baseline_plan(t, policy, 0.6, cfg)
    state = _observed_state(t, path[0], path[1], 0, cfg)
    qr_rounds, _, qr_guard = qrise_rounds(t, [path[0]], path[1], state, 0.6, cfg)
    assert (
        guard == qr_guard == cfg.qrise_floor_base + 0.6 * cfg.qrise_floor_target_weight
    )
    assert 0 < rounds == qr_rounds < 5
    assert rounds == minimum_rounds(state["fidelity"], guard)[0]


def test_e2e_only_control_removes_floor_for_all_policies():
    cfg = replace(
        profile_settings("mechanism")[0],
        baseline_guard_floor=False,
        qrise_floor_base=0.0,
        qrise_floor_target_weight=0.0,
    )
    t = constant_fidelity_topology(0.94)
    for p in ("GREEDY", "HSP", "NOPP", "Q-CAST"):
        path, rounds, _ = baseline_plan(t, p, 0.6, cfg)
        state = _observed_state(t, path[0], path[1], 0, cfg)
        assert rounds == qrise_rounds(t, [path[0]], path[1], state, 0.6, cfg)[0] == 0


def test_shared_filter_control_admits_no_low_fidelity_route():
    cfg, _, _ = profile_settings("mechanism")
    t = constant_fidelity_topology(0.88)
    assert baseline_plan(t, "NOPP", 0.6, cfg) is not None
    assert (
        baseline_plan(t, "NOPP", 0.6, replace(cfg, baseline_fidelity_filter=True))
        is None
    )


@pytest.mark.parametrize(
    "policy,reason", [("Q-RISE", "no_admissible_neighbor"), ("NOPP", "no_path")]
)
def test_synchronous_failure_consumes_no_epr_and_no_deadline(policy, reason):
    result = run(policy, constant_fidelity_topology(0.5), 0.9, 812)
    assert result.reason == reason
    assert result.epr_pairs == 0 and result.elapsed_ns == 0
    assert result.fidelity is None


@pytest.mark.parametrize("scale", [0, 0.02, 1])
def test_planner_chain_ages_at_processor_rate_without_reading_qubits(scale):
    ns.sim_reset()
    cfg, device, optical = profile_settings("mechanism")
    trial = Trial(
        "Q-RISE",
        make_topology(20, 1701, optical),
        0.9,
        cfg,
        replace(device, memory_rate_scale=scale),
    )
    trial.predicted_chain, trial.predicted_chain_ns = 0.99, 0

    def forbidden(*args):
        raise AssertionError("Planner must not inspect actual quantum fidelity")

    trial.fidelity = forbidden
    ns.sim_run(duration=100e6)
    expected = 0.25 + 0.74 * exp(-2 * optical.damping_rate_s * scale * 0.1)
    assert trial.planning_chain() == pytest.approx(expected)
    assert trial.predicted_chain == 0.99
    extra = 0.25 + 0.74 * exp(-2 * optical.damping_rate_s * scale * 0.12)
    assert trial.planning_chain(20) == pytest.approx(extra)


def test_degraded_source_fidelity_is_emitted_by_actual_qsource():
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    cfg, device, optical = profile_settings("mechanism")
    t = make_topology(20, 1701, optical)
    u, v = next(k for k, link in t.links.items() if link.quality_offset < 0)
    original = t.state

    def state(a, b, slot):
        values = original(a, b, slot)
        values["generation_p"] = 1.0
        return values

    t.state = state
    expected = t.state(u, v, 0)["fidelity"]

    class Probe(Trial):
        def run(self):
            pairs = yield from self.generate(u, v, 1)
            assert len(pairs) == 1
            self.measured = self.fidelity(pairs[0])
            self.release(pairs[0])
            ns.sim_stop()

    trial = Probe("Q-RISE", t, 0.9, cfg, replace(device, memory_rate_scale=0))
    trial.start()
    ns.sim_run()
    assert trial.measured == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(quality_period_slots=0),
        dict(quality_period_slots=float("nan")),
        dict(degraded_link_fraction=1.1),
        dict(phase_fidelity_amplitude=float("nan")),
    ],
)
def test_invalid_quality_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        OpticalModel(**kwargs)


def test_forward_reachability_excludes_known_dead_end_not_just_bad_next_link():
    from qrise_math.routing import forward_hops
    from qrise_math.topology import Topology, Link

    g = nx.Graph([(0, 1), (0, 2), (1, 3), (2, 3)])
    links = {tuple(sorted(e)): Link(*e, 45.0, 0.0, 3) for e in g.edges}
    t = Topology(
        g,
        links,
        {n: 10 for n in g},
        {0: (0, 0), 1: (45, -1), 2: (45, 1), 3: (90, 0)},
        0,
        3,
        OpticalModel(),
    )
    original = t.state

    def state(u, v, slot):
        values = original(u, v, slot)
        values["fidelity"] = 0.85 if set((u, v)) == {1, 3} else 0.95
        return values

    t.state = state
    cfg = Config(qrise_reachability_filter=True)
    assert forward_hops(t, 0, cfg) == {3: 0, 2: 1, 0: 2}
    assert forward_hops(t, 0, cfg, {2: {3}}) == {3: 0}
    result = run("Q-RISE", t, 0.6, 812, cfg, DeviceConfig(memory_rate_scale=0))
    assert result.path == [0, 2, 3]
    assert result.success and result.rollback_count == 0
    assert all(record["v"] != 1 for record in result.link_records)
