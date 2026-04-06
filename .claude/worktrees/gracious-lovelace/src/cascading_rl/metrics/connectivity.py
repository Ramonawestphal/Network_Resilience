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


def accumulated_normalized_connectivity(
    graph: nx.Graph, active_nodes: Iterable[Node]
) -> float:
    """Compute ANC as the sum of squared component shares."""
    total_nodes = graph.number_of_nodes()
    if total_nodes == 0:
        return 0.0
    component_sizes = connected_component_sizes(graph, active_nodes)
    return sum((component_size / total_nodes) ** 2 for component_size in component_sizes)


def largest_component_ratio(graph: nx.Graph, active_nodes: Iterable[Node]) -> float:
    """Return the share of all nodes in the largest active component."""
    total_nodes = graph.number_of_nodes()
    component_sizes = connected_component_sizes(graph, active_nodes)
    if total_nodes == 0 or not component_sizes:
        return 0.0
    return max(component_sizes) / total_nodes
