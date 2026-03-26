from __future__ import annotations

from pathlib import Path

import networkx as nx


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
    graph: nx.Graph,
    *,
    undirected: bool = True,
    largest_component_only: bool = True,
    drop_self_loops: bool = True,
) -> nx.Graph:
    """Normalize raw graphs into the simple undirected form used by the project."""
    prepared = graph.to_undirected() if undirected else graph.copy()
    prepared = nx.Graph(prepared)

    if drop_self_loops:
        prepared.remove_edges_from(nx.selfloop_edges(prepared))

    prepared.remove_nodes_from(list(nx.isolates(prepared)))

    if largest_component_only and prepared.number_of_nodes() > 0:
        largest_component = max(nx.connected_components(prepared), key=len)
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
