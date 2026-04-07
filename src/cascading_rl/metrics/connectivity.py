from __future__ import annotations

from collections.abc import Hashable, Iterable

import networkx as nx

Node = Hashable


def connected_component_sizes(graph: nx.Graph, active_nodes: Iterable[Node]) -> list[int]:
    """Return connected component sizes for the currently active subgraph."""
    active_set = set(active_nodes)
    if not active_set:
        return []
    subgraph = graph.subgraph(active_set)
    return [len(component) for component in nx.connected_components(subgraph)]


def normalized_connectivity(graph: nx.Graph, active_nodes: Iterable[Node]) -> float:
    """NC at a single point in time."""
    total_nodes = graph.number_of_nodes()
    if total_nodes == 0:
        return 0.0
    component_sizes = connected_component_sizes(graph, active_nodes)
    return sum((s / total_nodes) ** 2 for s in component_sizes)

def anc_fixed_horizon(nc_by_round: list[float], max_rounds: int) -> float:
    """ANC normalized by fixed horizon: pad solved episodes with 1.0, divide by max_rounds. Result in [0, 1]."""
    if not nc_by_round:
        return 1.0
    padded = nc_by_round + [1.0] * (max_rounds - len(nc_by_round))
    return sum(padded) / max_rounds


def anc_adaptive_horizon(nc_by_round: list[float]) -> float:
    """ANC normalized by actual rounds used: divide by number of rounds taken. Result in [0, 1]."""
    if not nc_by_round:
        return 1.0
    return sum(nc_by_round) / len(nc_by_round)


def largest_component_ratio(graph: nx.Graph, active_nodes: Iterable[Node]) -> float:
    """Return the share of all nodes in the largest active component."""
    total_nodes = graph.number_of_nodes()
    component_sizes = connected_component_sizes(graph, active_nodes)
    if total_nodes == 0 or not component_sizes:
        return 0.0
    return max(component_sizes) / total_nodes
