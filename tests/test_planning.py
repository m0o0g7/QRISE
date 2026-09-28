"""Ensure physical execution did not change the reference planner rules."""
from pathlib import Path
import ast
import importlib.util
import sys
import numpy as np
import pytest
from qrise_math import routing
from qrise_math.topology import make_topology
from qrise_math.physics import equal_link_guard, minimum_rounds

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("qrise_math.scalar_reference", ROOT / "reference/scalar_routing.py")
reference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = reference
spec.loader.exec_module(reference)


def test_original_helpers_have_identical_syntax_trees():
    def definitions(path):
        tree = ast.parse(path.read_text())
        return {node.name: ast.dump(node) for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    old = definitions(ROOT / "reference/scalar_routing.py")
    new = definitions(ROOT / "src/qrise_math/routing.py")
    for name in ("_valid_state", "_cost", "_observed_state", "_makes_forward_progress",
                 "_path_candidates", "_qcast_ext", "_select_baseline_path"):
        assert new[name] == old[name], name


@pytest.mark.parametrize("policy", ["GREEDY", "HSP", "NOPP", "Q-CAST"])
@pytest.mark.parametrize("n,target", [(20, .6), (50, .9), (90, .9)])
def test_baseline_path_matches_reference_and_rounds_have_no_penalty(policy, n, target):
    topology = make_topology(n, 1701)
    path, rounds, _ = routing.baseline_plan(topology, policy, target, routing.Config())
    old = reference.run(policy, topology, target, np.random.default_rng(1701), reference.Config())
    assert path == old.path
    cfg = routing.Config()
    nominal = min(routing._observed_state(topology, u, v, 0, cfg)["fidelity"]
                  for u, v in zip(path, path[1:]))
    guard = equal_link_guard(target, len(path) - 1, cfg.swap_fidelity)
    assert rounds == minimum_rounds(nominal, guard)[0]


def test_qrise_local_floor_and_guard_are_unchanged():
    topology = make_topology(50, 1701)
    cfg = routing.Config()
    peer = next(iter(topology.graph.neighbors(topology.source)))
    state = routing._observed_state(topology, topology.source, peer, 3.2, cfg)
    import networkx as nx
    horizon = 1 + nx.shortest_path_length(topology.graph, peer, topology.destination)
    guard = max(.960 + .025 * .9, equal_link_guard(.9, horizon, cfg.swap_fidelity))
    rounds, predicted, actual_guard = routing.qrise_rounds(topology, [topology.source], peer, state, .9, cfg)
    assert (rounds, predicted) == minimum_rounds(state["fidelity"], guard)
    assert actual_guard == guard


@pytest.mark.parametrize("fixed", [0, 4, 6])
def test_fixed_round_control_does_not_add_mpr(fixed):
    cfg = routing.Config(baseline_mode="fixed", baseline_fixed_rounds=fixed)
    for policy in ("GREEDY", "HSP", "NOPP", "Q-CAST"):
        for n, target in ((20, .6), (90, .9)):
            _, rounds, _ = routing.baseline_plan(make_topology(n, 1701), policy, target, cfg)
            assert rounds == fixed


def test_shared_source_profile_preserves_topology_and_planner():
    from qrise_components.profiles import profile_settings
    config, device, optical = profile_settings("fig2")
    old = make_topology(50, 1701)
    new = make_topology(50, 1701, optical)
    assert new.links == old.links and new.memories == old.memories
    from dataclasses import replace
    assert config == replace(routing.Config(), alpha=10.0)
    assert device.memory_rate_scale * optical.damping_rate_s == .01
    for u, v in old.graph.edges:
        before, after = old.state(u, v, 0), new.state(u, v, 0)
        assert after['generation_p'] == before['generation_p']
        assert after['fidelity'] == pytest.approx(before['fidelity'] + .005)


def test_recon_profile_retains_prechange_component_inputs():
    from qrise_components.profiles import profile_settings
    from qrise_components.config import DeviceConfig
    from qrise_math.physics import OpticalModel
    assert profile_settings("recon") == (routing.Config(), DeviceConfig(), OpticalModel())
