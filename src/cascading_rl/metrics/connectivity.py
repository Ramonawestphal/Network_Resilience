from __future__ import annotations

from collections.abc import Hashable, Iterable
import warnings

import networkx as nx

Node = Hashable


def connected_component_sizes(graph: nx.Graph, active_nodes: Iterable[Node]) -> list[int]:
    """Return connected component sizes for the currently active subgraph."""
    active_set = set(active_nodes)
    if not active_set:
        return []
    subgraph = graph.subgraph(active_set)
    return [len(component) for component in nx.connected_components(subgraph)]


def pairwise_connectivity(graph: nx.Graph, active_nodes: Iterable[Node]) -> float:
    """Pairwise connectivity relative to the full graph node set ``V``.

    Counts unordered pairs of *active* nodes that belong to the same connected component
    in the active induced subgraph, divided by ``|V|(|V|-1)`` (all unordered pairs in ``V``).
    If there are fewer than two active nodes, returns ``0``.
    """
    active_set = set(active_nodes)
    if len(active_set) < 2:
        return 0.0
    n_total = graph.number_of_nodes()
    if n_total < 2:
        return 0.0
    component_sizes = connected_component_sizes(graph, active_set)
    connected_pairs = sum(s * (s - 1) for s in component_sizes)
    return connected_pairs / (n_total * (n_total - 1))

# Needs change
def accumulated_normalized_connectivity(
    graph: nx.Graph, active_nodes: Iterable[Node]
) -> float:
    """Historical name; now an alias for :func:`pairwise_connectivity`."""
    return pairwise_connectivity(graph, active_nodes)


def largest_component_ratio(graph: nx.Graph, active_nodes: Iterable[Node]) -> float:
    """Return the share of all nodes in the largest active component."""
    total_nodes = graph.number_of_nodes()
    component_sizes = connected_component_sizes(graph, active_nodes)
    if total_nodes == 0 or not component_sizes:
        return 0.0
    return max(component_sizes) / total_nodes
