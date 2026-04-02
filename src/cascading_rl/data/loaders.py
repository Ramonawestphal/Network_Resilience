from __future__ import annotations

from pathlib import Path
from typing import Union

import networkx as nx

PreparedGraph = Union[nx.Graph, nx.DiGraph]


def load_edge_list_graph(
    path: str | Path, delimiter: str | None = None, nodetype: type = int
) -> nx.Graph:
    """Load an undirected graph from a plain edge-list file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    return nx.read_edgelist(file_path, delimiter=delimiter, nodetype=nodetype)


def load_graphml_graph(path: str | Path) -> nx.Graph:
    """Load a graph from GraphML for real-world evaluation datasets."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    return nx.read_graphml(file_path)


def prepare_graph(
    graph: nx.Graph | nx.DiGraph,
    *,
    undirected: bool = True,
    largest_component_only: bool = True,
    drop_self_loops: bool = True,
) -> PreparedGraph:
    """Normalize a graph for downstream use.

    When ``undirected`` is True (default), ``graph`` is converted with
    :meth:`~networkx.Graph.to_undirected` and stored as a simple
    :class:`~networkx.Graph`, matching the undirected cascade model.

    When ``undirected`` is False, the result is a :class:`~networkx.DiGraph`
    built from ``graph`` (direction preserved; an undirected input becomes a
    digraph with both orientations per edge). Self-loops can be dropped and
    isolates removed the same as in the undirected path.

    If ``largest_component_only`` is True, the largest *connected* component is
    kept for undirected graphs, and the largest *weakly connected* component for
    directed graphs.
    """
    if undirected:
        prepared: PreparedGraph = nx.Graph(graph.to_undirected())
    else:
        prepared = nx.DiGraph(graph)

    if drop_self_loops:
        prepared.remove_edges_from(nx.selfloop_edges(prepared))

    prepared.remove_nodes_from(list(nx.isolates(prepared)))

    if largest_component_only and prepared.number_of_nodes() > 0:
        if undirected:
            components = nx.connected_components(prepared)
        else:
            components = nx.weakly_connected_components(prepared)
        largest_component = max(components, key=len)
        prepared = prepared.subgraph(largest_component).copy()

    return prepared


def load_prepared_graphml_graph(path: str | Path) -> nx.Graph:
    """Load and normalize a GraphML graph for evaluation."""
    return prepare_graph(load_graphml_graph(path))


def load_prepared_edge_list_graph(
    path: str | Path, delimiter: str | None = None, nodetype: type = int
) -> nx.Graph:
    """Load and normalize an edge-list graph for evaluation."""
    return prepare_graph(load_edge_list_graph(path, delimiter=delimiter, nodetype=nodetype))
