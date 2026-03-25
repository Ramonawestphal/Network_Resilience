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
