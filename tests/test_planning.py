"""Validate component planning paths, purification rules and input profiles."""
import networkx as nx
import pytest
from qrise_math import routing
from qrise_math.topology import make_topology
from qrise_math.physics import equal_link_guard, minimum_rounds


@pytest.mark.parametrize("policy", ["GREEDY", "HSP", "NOPP", "Q-CAST"])
@pytest.mark.parametrize("n,target", [(20, 0.6), (50, 0.9), (90, 0.9)])
def test_baseline_path_is_valid_and_rounds_have_no_penalty(policy, n, target):
    topology = make_topology(n, 1701)
    path, rounds, _ = routing.baseline_plan(topology, policy, target, routing.Config())
    assert path[0] == topology.source
    assert path[-1] == topology.destination
    assert len(set(path)) == len(path)
    assert all(topology.graph.has_edge(u, v) for u, v in zip(path, path[1:]))
    cfg = routing.Config()
    nominal = min(
        routing._observed_state(topology, u, v, 0, cfg)["fidelity"]
        for u, v in zip(path, path[1:])
    )
    guard = equal_link_guard(target, len(path) - 1, cfg.swap_fidelity)
    assert rounds == minimum_rounds(nominal, guard)[0]


def test_qrise_equal_guard_respects_local_floor():
    topology = make_topology(50, 1701)
    cfg = routing.Config()
    peer = next(iter(topology.graph.neighbors(topology.source)))
    state = routing._observed_state(topology, topology.source, peer, 3.2, cfg)
    horizon = 1 + nx.shortest_path_length(topology.graph, peer, topology.destination)
    guard = max(0.960 + 0.025 * 0.9, equal_link_guard(0.9, horizon, cfg.swap_fidelity))
    rounds, predicted, actual_guard = routing.qrise_rounds(
        topology, [topology.source], peer, state, 0.9, cfg
    )
    assert (rounds, predicted) == minimum_rounds(state["fidelity"], guard)
    assert actual_guard == guard


@pytest.mark.parametrize("fixed", [0, 4, 6])
def test_fixed_round_control_does_not_add_mpr(fixed):
    cfg = routing.Config(baseline_mode="fixed", baseline_fixed_rounds=fixed)
    for policy in ("GREEDY", "HSP", "NOPP", "Q-CAST"):
        for n, target in ((20, 0.6), (90, 0.9)):
            _, rounds, _ = routing.baseline_plan(
                make_topology(n, 1701), policy, target, cfg
            )
            assert rounds == fixed


def test_mechanism_profile_preserves_topology_and_generation():
    from qrise_components.profiles import profile_settings

    config, device, optical = profile_settings("mechanism")
    old = make_topology(50, 1701)
    new = make_topology(50, 1701, optical)
    assert set(new.links) == set(old.links) and new.memories == old.memories
    for key, link in new.links.items():
        before = old.links[key]
        assert (link.width, link.base_distance_km, link.phase) == (
            before.width,
            before.base_distance_km,
            before.phase,
        )
    assert device.memory_rate_scale * optical.damping_rate_s == 0.01
    for u, v in old.graph.edges:
        assert new.state(u, v, 0)["generation_p"] == old.state(u, v, 0)["generation_p"]


def test_only_mechanism_profile_exists():
    from qrise_components.profiles import profile_settings

    assert profile_settings() == profile_settings("mechanism")
    for name in ("recon", "fig2", "hardware_only", "mechanism_e2e"):
        with pytest.raises(ValueError):
            profile_settings(name)
