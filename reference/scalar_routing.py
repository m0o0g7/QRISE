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

from dataclasses import dataclass
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
    equal_link_guard,
    mcsfm,
    minimum_rounds,
    stored_fidelity,
)


@dataclass(frozen=True)
class Config:
    """All experiment constants that affect routing and resource accounting."""

    slot_s: float = 0.001 # 1슬롯 = 0.001초 = 1ms
    cutoff_slots: int = 224 # 요청에서 걸리는 시간 상한선 = 0.224초
    f_base: float = 0.90 # 다음 홉 링크의 최소 fidelity
    swap_fidelity: float = 0.99 # (4)번 식의 F_swap으로 스와핑 품질 감소 계수
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1000.0
    delta: float = 0.02
    attempts_per_lane_slot: int = 3 # 채널 하나당 1슬롯에 가능한 생성 시도 수
    fixed_extra_rounds: int = 3 # 비교 알고리즘들에 추가되는 purification 라운드 수
    recovery_resource_fraction: float = 0.25 # Q-CAST 


@dataclass
class Result:
    """One routing trial and its directly measured scalar outcomes."""

    policy: str  # 실행한 알고리즘 이름
    success: bool # 요청 전체에 대한 성공 여부
    fidelity: float | None # 최종 fidelity
    epr_pairs: int # 요청 전체에 사용된 EPR 쌍 수
    latency_slots: int # 요청 전체에 걸린 시간(슬롯 단위)
    path: list[int] # 최종 경로의 노드 번호
    rounds: list[int] # 각 링크에서 사용된 purification 라운드 수
    reason: str # 성공 또는 실패 이유
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


def qrise(topology, target: float, rng, cfg: Config) -> Result:
    """Run the reconstructed constraint-aware Q-RISE procedure.

    The function preserves the existing Q-RISE metric.  It does not add an
    external score or tune output values to the poster after execution.
    """

    current = topology.source
    path = [current]
    purified_links: list[float] = []
    rounds_used: list[int] = []
    blocked: dict[int, set[int]] = {}
    latency = 0
    epr_pairs = 0
    rollbacks = 0

    while current != topology.destination and latency <= cfg.cutoff_slots:
        candidates = []
        for peer in topology.graph.neighbors(current): 
            if peer in path or peer in blocked.get(current, set()):
                continue # 현재 경로에 포함 되어있거나 시도했다가 제외한 이웃은 제외
            if not _makes_forward_progress(topology, current, peer):
                continue # x방향 거리 줄이지 못하면 제외

            state = _observed_state(topology, current, peer, latency, cfg)
            if _valid_state(state, cfg): # 라우팅 조건 만족하면 후보로 넣음
                candidates.append((_cost(state, cfg), peer, state))

        if not candidates: # 후보가 하나도 없으면 되돌아감
            # Local rollback releases only the latest reservation.  The source
            # cannot roll back, so exhausting it terminates the request.
            if len(path) == 1: # 현재 경로가 출발 노드만 있다면 알고리즘 종료
                return Result(
                    "Q-RISE",
                    False,
                    None,
                    epr_pairs,
                    latency,
                    path,
                    rounds_used,
                    "no_admissible_neighbor",
                    rollbacks,
                )
            # 마지막 노드를 제거
            removed = path.pop()
            purified_links.pop()
            rounds_used.pop()
            current = path[-1]
            blocked.setdefault(current, set()).add(removed)
            latency += 1
            rollbacks += 1
            continue

        # Eq. (3) is minimized exactly as disclosed.  Node id only breaks ties
        # deterministically and has no effect when costs differ.
        _, peer, state = min(candidates, key=lambda item: (item[0], item[1])) # 가장 저렴한 이웃 선택

        try: # 선택한 링크의 노드에서 목적지까지 남은 최단 홉수를 구함
            remaining_hops = nx.shortest_path_length(
                topology.graph,
                peer,
                topology.destination,
            )
            estimated_horizon = len(path) + remaining_hops

            # The poster does not define F_guard.  We combine MCSFM inversion
            # with one fixed local service floor.  Both are pre-run inputs.
            local_floor = 0.960 + 0.025 * target # 링크의 purification 하한, 최종 목표(target)이 높아지면 하한도 높아짐
            inferred_guard = equal_link_guard( # target을 만족하려면 각 링크를 어느 fidelity까지 purification 해야하는가를 추정한 값
                target,
                max(1, estimated_horizon), # 예상 전체 홉 수
                cfg.swap_fidelity, 
            )
            guard = max(local_floor, inferred_guard)
            purification_rounds, purified_fidelity = minimum_rounds(
                state["fidelity"], guard # 최소 라운드 수, 최소 라운드 수만큼 purification 했을 때 계산상 fidelity
            )
        except (nx.NetworkXNoPath, ValueError): # 남은 경로가 없거나 제한된 라운드 수 내에 purification이 불가능하면 해당 후보 제외하고 slot +1
            blocked.setdefault(current, set()).add(peer)
            latency += 1
            continue

        # Two input pairs are needed at every purification level.  The negative
        # binomial draw models the failed generation attempts needed to obtain
        # 2^rounds successful elementary pairs.
        required_pairs = 2**purification_rounds # purification 이 5라운드 필요하면 2^5 = 32개의 epr pair 필요
        generation_attempts = int( # pair를 얻으려고 할 때 확률은 음이항 분포를 따름
            rng.negative_binomial(required_pairs, state["generation_p"])
            + required_pairs
        )
        epr_pairs += generation_attempts

        # Link width supplies parallel lanes.  The remaining terms represent
        # purification and fixed link-operation slots in the scalar schedule.
        parallel_lanes = state["buffer"] * cfg.attempts_per_lane_slot # 슬롯당 가능한 총 생성 시도 수
        generation_slots = (
            generation_attempts + parallel_lanes - 1
        ) // parallel_lanes
        hop_slots = generation_slots + 2 * purification_rounds + 2
        latency += hop_slots

        path.append(peer)
        purified_links.append(purified_fidelity)
        rounds_used.append(purification_rounds)
        current = peer
    # 목적지 도착 못하면 시간초과 실패 반환
    if current != topology.destination:
        return Result(
            "Q-RISE",
            False,
            None,
            epr_pairs,
            latency,
            path,
            rounds_used,
            "cutoff",
            rollbacks,
        )
    # 위에서 계산을 통해 링크 purification 후 fidelity가 purified_links에 저장되어있음
    # 스와핑으로 E2E link를 만듦
    end_to_end = purified_links[0]
    for link_fidelity in purified_links[1:]:
        end_to_end = mcsfm(end_to_end, link_fidelity, cfg.swap_fidelity)

    success = end_to_end >= target
    return Result(
        "Q-RISE",
        success,
        end_to_end,
        epr_pairs,
        latency,
        path,
        rounds_used,
        "success" if success else "fidelity",
        rollbacks,
    )


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


def baseline(topology, policy: str, target: float, rng, cfg: Config) -> Result:
    """Run a scoped GREEDY/HSP/NOPP/Q-CAST comparison adapter."""

    path = _select_baseline_path(topology, policy, slot=0, cfg=cfg)
    if path is None:
        return Result(policy, False, None, 0, 0, [], [], "no_path")

    hops = len(path) - 1
    guard = equal_link_guard(target, hops, cfg.swap_fidelity)
    nominal_fidelity = min(
        _observed_state(topology, u, v, 0, cfg)["fidelity"]
        for u, v in zip(path, path[1:])
    )
    try:
        fixed_rounds = (
            minimum_rounds(nominal_fidelity, guard)[0]
            + cfg.fixed_extra_rounds
        )
    except ValueError:
        return Result(policy, False, None, 0, 0, path, [], "guard")

    # Q-RISE only describes these comparators as fixed-round baselines.  The
    # exact adaptation is unpublished, so five rounds is explicitly treated as
    # a reconstruction input rather than an author-supplied value.
    fixed_rounds = max(5, min(fixed_rounds, 10))

    epr_pairs = 0
    latency = 0
    link_fidelities: list[float] = []
    for u, v in zip(path, path[1:]):
        state = _observed_state(topology, u, v, latency, cfg)
        required_pairs = 2**fixed_rounds
        generation_attempts = int(
            rng.negative_binomial(required_pairs, state["generation_p"])
            + required_pairs
        )
        epr_pairs += generation_attempts

        lanes = state["buffer"] * cfg.attempts_per_lane_slot
        generation_slots = (generation_attempts + lanes - 1) // lanes
        latency += generation_slots + 2 * fixed_rounds + 2

        purified = state["fidelity"]
        for _ in range(fixed_rounds):
            purified, _ = bbpssw_once(purified)
        link_fidelities.append(purified)

    if policy == "Q-CAST":
        # Q-CAST reserves contention-free recovery resources.  Its paper
        # reports a 25% channel overhead when recovery is enabled.
        recovery_pairs = int(round(epr_pairs * cfg.recovery_resource_fraction))
        epr_pairs += recovery_pairs
        latency += hops + 2
    elif policy == "NOPP":
        latency += hops * 2
    elif policy == "HSP":
        latency += hops * 3
    else:
        latency += hops

    if latency > cfg.cutoff_slots:
        return Result(
            policy,
            False,
            None,
            epr_pairs,
            latency,
            path,
            [fixed_rounds] * hops,
            "cutoff",
        )

    end_to_end = link_fidelities[0]
    for link_fidelity in link_fidelities[1:]:
        end_to_end = mcsfm(end_to_end, link_fidelity, cfg.swap_fidelity)

    success = end_to_end >= target
    return Result(
        policy,
        success,
        end_to_end,
        epr_pairs,
        latency,
        path,
        [fixed_rounds] * hops,
        "success" if success else "fidelity",
    )


def run(policy, topology, target: float, rng, cfg: Config | None = None) -> Result:
    """Dispatch one trial to Q-RISE or a named baseline adapter."""

    config = cfg or Config()
    if policy == "Q-RISE":
        return qrise(topology, target, rng, config)
    return baseline(topology, policy, target, rng, config)

