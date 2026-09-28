"""Dynamic layered-grid LEO abstraction used in the reconstruction.

The Q-RISE poster does not publish its topology generator.  Gao et al. describe
a dynamic line/grid physical graph, while Q-CAST publishes useful resource
ranges.  This module therefore uses a layered grid with dynamic link distances,
Q-CAST's 10-14 node memories, and Q-CAST's 3-7 edge widths.  It does not relabel
Q-CAST's Waxman topology as a LEO constellation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, pi, sin

import networkx as nx
import numpy as np

from .physics import (
    OpticalModel,
    capture_probability,
    effective_routing_fidelity,
)


@dataclass(frozen=True)
class Link:
    """Static link parameters plus a deterministic contact-window trace."""

    u: int # 위성 ID
    v: int # 연결 위성 ID
    base_distance_km: float  
    phase: float
    width: int # 링크 폭 = 채널 개수
    #v2 Persistent per-link fidelity offset (0 = v1 behaviour) and the period
    #v2 of the contact-window trace (420 slots = v1 behaviour).
    quality_offset: float = 0.0
    period_slots: float = 420.0

    def distance(self, slot: float) -> float:
        """Return distance at one slot; state is frozen inside the slot."""

        # The 420-slot period and 1.7 km amplitude are reconstruction inputs,
        # not values recovered from the unpublished Q-RISE simulator.
        return self.base_distance_km + 1.7 * sin(
            self.phase + 2.0 * pi * slot / self.period_slots
        )


@dataclass
class Topology:
    """Network graph and all scalar resource state required by routing."""

    graph: nx.Graph
    links: dict[tuple[int, int], Link]
    memories: dict[int, int]
    positions: dict[int, tuple[float, float]]
    source: int
    destination: int
    optical: OpticalModel

    def edge(self, u: int, v: int) -> Link:
        """Return the undirected link between two nodes."""

        return self.links[tuple(sorted((u, v)))]

    def state(self, u: int, v: int, slot: float) -> dict[str, float | int]:
        """Observe the scalar state used by Q-RISE at a simulation slot."""

        link = self.edge(u, v)
        distance = link.distance(slot)
        #v1 Review: preserve the original geometric trace when accelerating
        #v1 synthetic quality drift; do not call a 120ms cycle orbital motion.
        quality_period = self.optical.quality_period_slots or link.period_slots
        dynamic_phase = link.phase + 2.0 * pi * slot / quality_period
        #v1 recon_Qu used a fixed curve; the shared source offset is an
        #v1 explicit assumption used by both planning and actual QSource.
        #v2 The per-link offset is applied after the curve's clamp so degraded
        #v2 links can fall below f_base; Werner F stays above .55 (purifiable).
        fidelity = effective_routing_fidelity(
            distance, dynamic_phase, self.optical.phase_fidelity_amplitude
        ) + self.optical.source_fidelity_offset
        if link.quality_offset:
            fidelity = max(0.55, fidelity + link.quality_offset)
        return {
            "distance_km": distance,
            "fidelity": min(1.0, max(0.0, fidelity)),
            "generation_p": capture_probability(distance, self.optical),
            "latency_s": distance / 299_792.458,
            "buffer": link.width,
        }


def make_topology(n: int, seed: int, optical: OpticalModel = None) -> Topology:
    """Create singleton endpoints joined by several parallel relay layers.

    Increasing ``n`` raises both the number of available relays and the
    source-destination depth.  The fixed mapping gives shortest paths of 2, 4,
    and 6 hops at 20, 50, and 90 nodes respectively.
    """

    if n < 20:
        raise ValueError("the reconstruction scope starts at 20 nodes")

    rng = np.random.default_rng(seed)
    hop_count = max(2, ceil(n / 15))
    relay_layer_count = hop_count - 1 # 중계노드

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    source = 0
    destination = n - 1

    # Divide all non-endpoint satellites as evenly as possible among layers.
    groups: list[list[int]] = []
    next_node = 1 
    for layer_index in range(relay_layer_count):
        remaining_nodes = n - 1 - next_node
        remaining_layers = relay_layer_count - layer_index
        layer_width = remaining_nodes // remaining_layers
        groups.append(list(range(next_node, next_node + layer_width)))
        next_node += layer_width
    assert next_node == destination and groups

    positions: dict[int, tuple[float, float]] = {
        source: (0.0, 0.0),
        destination: (45.0 * hop_count, 0.0),
    }
    for layer_index, nodes in enumerate(groups, start=1):
        for rank, node in enumerate(nodes):
            y_position = (
                0.0
                if len(nodes) == 1
                else -15.0 + 30.0 * rank / (len(nodes) - 1)
            )
            positions[node] = (45.0 * layer_index, y_position)

    # Connect each node to at most three nearest relays in the next layer.
    links: dict[tuple[int, int], Link] = {}
    layers = [[source], *groups, [destination]]
    for left_layer, right_layer in zip(layers, layers[1:]):
        for u in left_layer:
            nearest = sorted(
                right_layer,
                key=lambda v: abs(positions[u][1] - positions[v][1]),
            )[: min(3, len(right_layer))]
            for v in nearest:
                dx = positions[u][0] - positions[v][0]
                dy = positions[u][1] - positions[v][1]
                geometric_distance = (dx**2 + dy**2) ** 0.5
                base_distance = float(
                    max(42.2, min(47.8, geometric_distance + rng.uniform(-0.7, 0.7)))
                )
                link = Link(
                    u=u,
                    v=v,
                    base_distance_km=base_distance,
                    phase=float(rng.uniform(0.0, 2.0 * pi)),
                    width=int(rng.integers(3, 8)),
                )
                key = tuple(sorted((u, v)))
                links[key] = link
                graph.add_edge(*key)

    memories = {node: int(rng.integers(10, 15)) for node in graph}

    #v2 Draw link heterogeneity from an independent stream so graph, widths,
    #v2 distances and memories stay identical to v1 for the same seed.
    optical = OpticalModel() if optical is None else optical
    if optical.degraded_link_fraction > 0 or optical.dynamics_period_slots != 420.0:
        quality_rng = np.random.default_rng([seed, 2])
        for key in sorted(links):
            degraded = quality_rng.random() < optical.degraded_link_fraction
            offset = float(quality_rng.uniform(optical.degraded_offset_min,
                                               optical.degraded_offset_max))
            links[key] = replace(links[key],
                                 quality_offset=offset if degraded else 0.0,
                                 period_slots=optical.dynamics_period_slots)
    if not nx.has_path(graph, source, destination):
        raise AssertionError("constructed grid must connect both endpoints")

    return Topology(
        graph=graph,
        links=links,
        memories=memories,
        positions=positions,
        source=source,
        destination=destination,
        optical=optical,
    )

