"""Q-RISE routing logic and four scoped baseline adapters.

Q-RISE follows the poster's observable control flow: filter neighbours by
physical constraints, rank valid neighbours using Eq. (3), reserve one hop,
apply minimum purification rounds (MPR), and accumulate E2E fidelity with the
MCSFM swapping equation.  Missing simulator details are isolated as named
configuration values or comments instead of being presented as paper facts.

GREEDY, HSP, NOPP, and Q-CAST are paper-informed adapters used for a common
comparison environment.  They are not the Q-RISE authors' unpublished source.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import islice
from math import log

try:
    from math import comb
except ImportError:  # Python 3.7, used by the existing NetSquid environment.
    from math import factorial

    def comb(n: int, k: int) -> int:
        """Exact binomial coefficient for the nonnegative Q-CAST counts."""
        return factorial(n) // (factorial(k) * factorial(n - k))

import networkx as nx

from .physics import (
    bbpssw_once,
    chain_link_guard,
    equal_link_guard,
    mcsfm,
    minimum_rounds,
    stored_fidelity,
)


@dataclass(frozen=True)
class Config:
    """All experiment constants that affect routing and resource accounting."""

    slot_s: float = 0.001  # One millisecond per slot.
    cutoff_slots: int = 224  # Whole-request deadline, not a per-link lifetime.
    f_base: float = 0.90  # Candidate fidelity threshold.
    swap_fidelity: float = 0.99  # MCSFM swap factor.
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1000.0
    delta: float = 0.02
    attempts_per_lane_slot: int = 3  # Source rate limit; herald waiting also applies.
    fixed_extra_rounds: int = 3  # Legacy scalar reference only.
    recovery_resource_fraction: float = 0.25  # Legacy scalar recovery accounting only.

    # Path MPR is the default. Fixed rounds are an explicit sensitivity control.
    baseline_mode: str = "path_mpr"
    baseline_fixed_rounds: int = 5
    qrise_floor_base: float = 0.960
    qrise_floor_target_weight: float = 0.025
    guard_margin: float = 0.004
    # Profiles can enable shared floors, predictive guards and reachability.
    # The defaults preserve the reference planning assumptions.
    baseline_fidelity_filter: bool = True
    baseline_guard_floor: bool = False
    qrise_guard_mode: str = "equal"
    qrise_lookahead_slots: float = 0.0
    # Reject branches with no forward path under the current forecast.
    # This uses the already available graph, not a pre-reserved complete path.
    qrise_reachability_filter: bool = False

    def __post_init__(self):
        if self.qrise_guard_mode not in ("equal", "chain"):
            raise ValueError("Unknown Q-RISE guard mode")
        if not 0 <= self.qrise_lookahead_slots <= self.cutoff_slots:
            raise ValueError("Lookahead must be within the request deadline")
        if self.baseline_mode not in ("path_mpr", "fixed"):
            raise ValueError("Unknown baseline purification mode")
        if not 0 <= self.baseline_fixed_rounds <= 12:
            raise ValueError("Fixed rounds must be between 0 and 12")
        if not 0 <= self.qrise_floor_base <= 1 or not 0 <= self.qrise_floor_target_weight <= 1 - self.qrise_floor_base:
            raise ValueError("Q-RISE local floor must stay within [0, 1]")
        if not 0 <= self.guard_margin < 0.1:
            raise ValueError("Guard margin must be in [0, .1)")


@dataclass
class Result:
    """One routing trial and its directly measured scalar outcomes."""

    policy: str
    success: bool
    fidelity: float | None
    epr_pairs: int
    latency_slots: int
    path: list[int]
    rounds: list[int]
    reason: str
    rollback_count: int = 0


def _valid_state(state: dict, cfg: Config) -> bool:
    """Apply Q-RISE candidate constraints before computing link cost."""

    return (
        state["fidelity"] >= cfg.f_base
        and state["buffer"] >= 2
        and 40.0 <= state["distance_km"] <= 50.0
        and state["latency_s"] <= cfg.cutoff_slots * cfg.slot_s
    )


def _cost(state: dict, cfg: Config) -> float:
    """Evaluate Q-RISE Eq. (3) without normalization or extra penalties."""

    return (
        cfg.alpha / state["fidelity"]
        + cfg.beta / state["buffer"]
        + cfg.gamma * state["latency_s"]
        + cfg.delta * state["distance_km"]
    )


def _observed_state(topology, u: int, v: int, slot: int, cfg: Config) -> dict:
    """Read a link and apply one-slot expected storage aging.

    The one-slot age corresponds to the Gao-model ``lambda=1`` expected
    consumption age used by this reconstruction.
    """

    state = topology.state(u, v, slot)
    state["fidelity"] = stored_fidelity(
        state["fidelity"],
        age_slots=1,
        slot_s=cfg.slot_s,
        damping_rate_s=topology.optical.damping_rate_s,
    )
    return state


def _makes_forward_progress(topology, current: int, peer: int) -> bool:
    """Approximate the poster's bounded-direction loop-avoidance rule."""

    current_x, _ = topology.positions[current]
    peer_x, _ = topology.positions[peer]
    destination_x, _ = topology.positions[topology.destination]
    return abs(destination_x - peer_x) < abs(destination_x - current_x)


def _path_candidates(topology, slot: int, cfg: Config, limit: int = 40):
    """Return up to ``limit`` shortest paths over currently admissible links."""

    admissible = nx.Graph()
    for u, v in topology.graph.edges:
        state = _observed_state(topology, u, v, slot, cfg)
        if _valid_state(state, cfg):
            admissible.add_edge(u, v)
    try:
        paths = nx.shortest_simple_paths(
            admissible,
            topology.source,
            topology.destination,
        )
        return list(islice(paths, limit))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def _qcast_ext(topology, path: list[int], slot: int, cfg: Config) -> float:
    """Evaluate the Q-CAST expected-throughput (EXT) path metric."""

    width = min(
        topology.edge(u, v).width for u, v in zip(path, path[1:])
    )
    probabilities = [
        _observed_state(topology, u, v, slot, cfg)["generation_p"]
        for u, v in zip(path, path[1:])
    ]
    expectation = 0.0
    for at_least in range(1, width + 1):
        joint_probability = 1.0
        for probability in probabilities:
            joint_probability *= sum(
                comb(width, generated)
                * probability**generated
                * (1.0 - probability) ** (width - generated)
                for generated in range(at_least, width + 1)
            )
        expectation += joint_probability
    return expectation * 0.9 ** (len(path) - 2)


def _select_baseline_path(topology, policy: str, slot: int, cfg: Config):
    """Select one path using the named baseline's principal path rule."""

    candidates = _path_candidates(topology, slot, cfg)
    if not candidates:
        return None

    if policy == "NOPP":
        return min(candidates, key=lambda path: (len(path), path))

    if policy == "HSP":
        return max(
            candidates,
            key=lambda path: (
                sum(
                    log(
                        max(
                            _observed_state(topology, u, v, slot, cfg)[
                                "generation_p"
                            ],
                            1e-12,
                        )
                    )
                    for u, v in zip(path, path[1:])
                ),
                -len(path),
            ),
        )

    if policy == "Q-CAST":
        return max(
            candidates,
            key=lambda path: (_qcast_ext(topology, path, slot, cfg), -len(path)),
        )

    if policy == "GREEDY":
        path = [topology.source]
        while path[-1] != topology.destination:
            current = path[-1]
            peers = [
                peer
                for peer in topology.graph.neighbors(current)
                if peer not in path
                and _makes_forward_progress(topology, current, peer)
                and _valid_state(
                    _observed_state(topology, current, peer, slot, cfg), cfg
                )
            ]
            if not peers:
                return None
            peer = min(
                peers,
                key=lambda candidate: (
                    abs(
                        topology.positions[topology.destination][1]
                        - topology.positions[candidate][1]
                    ),
                    candidate,
                ),
            )
            path.append(peer)
        return path

    raise ValueError(f"unknown policy: {policy}")


def predicted_state(topology, u: int, v: int, slot: float, cfg: Config) -> dict:
    """Observed state whose fidelity is the worst forecast over the lookahead.

    This synthetic trace is assumed known. It is not a measured ephemeris or
    a forecast of random quantum outcomes; forecast error is not modeled. With zero lookahead this is the v1 state.
    """
    state = _observed_state(topology, u, v, slot, cfg)
    if cfg.qrise_lookahead_slots > 0:
        steps = max(1, int(round(cfg.qrise_lookahead_slots)))
        state["fidelity"] = min(
            _observed_state(topology, u, v, slot + cfg.qrise_lookahead_slots * k / steps, cfg)["fidelity"]
            for k in range(steps + 1))
    return state


def forward_hops(topology, slot: float, cfg: Config, blocked=None) -> dict:
    """Destination hop distances on the currently admissible forward graph.

    This optional reconstruction assumption avoids spending pairs on a known
    dead end. It does not guarantee future availability or choose/reserve the
    remaining path; Eq.(3) still ranks the current next-hop candidates.
    """
    admitted = nx.DiGraph()
    admitted.add_nodes_from(topology.graph)
    blocked = blocked or {}
    for left, right in topology.graph.edges:
        for u, v in ((left, right), (right, left)):
            if v in blocked.get(u, ()) or not _makes_forward_progress(topology, u, v):
                continue
            if _valid_state(predicted_state(topology, u, v, slot, cfg), cfg):
                admitted.add_edge(u, v)
    return dict(nx.single_source_shortest_path_length(
        admitted.reverse(copy=False), topology.destination))


def qrise_rounds(topology, path, peer, state, target, cfg, chain_fidelity=None, remaining_hops=None):
    """Plan local MPR using the declared floor and E2E guard assumptions."""
    if remaining_hops is None:
        remaining_hops = nx.shortest_path_length(topology.graph, peer, topology.destination)
    estimated_horizon = len(path) + remaining_hops
    local_floor = cfg.qrise_floor_base + cfg.qrise_floor_target_weight * target
    if cfg.qrise_guard_mode == "chain":
        inferred_guard = chain_link_guard(chain_fidelity, target, 1 + remaining_hops,
                                          cfg.swap_fidelity, margin=cfg.guard_margin)
    else:
        inferred_guard = equal_link_guard(target, max(1, estimated_horizon), cfg.swap_fidelity, margin=cfg.guard_margin)
    guard = max(local_floor, inferred_guard)
    rounds, predicted = minimum_rounds(state["fidelity"], guard)
    return rounds, predicted, guard

def baseline_plan(topology, policy, target, cfg):
    """Select the original adapter path, then apply the declared round rule."""
    # A fidelity-unaware adapter may admit purifiable low-F links.
    # This is an adapter assumption, not a claim about every original baseline.
    # A shared-filter control can restore the stronger previous baseline.
    select_cfg = cfg if cfg.baseline_fidelity_filter else replace(cfg, f_base=0.0)
    path = _select_baseline_path(topology, policy, slot=0, cfg=select_cfg)
    if path is None:
        return None
    guard = equal_link_guard(target, len(path) - 1, cfg.swap_fidelity, margin=cfg.guard_margin)
    if cfg.baseline_guard_floor:
        guard = max(guard, cfg.qrise_floor_base + cfg.qrise_floor_target_weight * target)
    nominal_fidelity = min(_observed_state(topology, u, v, 0, cfg)["fidelity"]
                           for u, v in zip(path, path[1:]))
    # Apply MPR without additive rounds or a mandatory minimum.
    fixed_rounds = (cfg.baseline_fixed_rounds if cfg.baseline_mode == "fixed"
                    else minimum_rounds(nominal_fidelity, guard)[0])
    return path, fixed_rounds, guard
