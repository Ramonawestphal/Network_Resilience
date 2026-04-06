from __future__ import annotations

from collections.abc import Iterable
from random import Random

import networkx as nx


def make_ba_graph(n: int = 40, m: int = 2, seed: int | None = None) -> nx.Graph:
    """Generate a Barabasi-Albert graph used for synthetic training data."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    if m < 1:
        raise ValueError("m must be at least 1.")
    if m >= n:
        raise ValueError("m must be smaller than n for a BA graph.")
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def make_graph_batch(
    num_graphs: int = 32,
    n_range: tuple[int, int] = (30, 50),
    m: int = 2,
    seed: int | None = None,
) -> list[nx.Graph]:
    """Generate a batch of synthetic BA graphs with varying sizes."""
    if num_graphs < 1:
        raise ValueError("num_graphs must be at least 1.")
    min_n, max_n = n_range
    if min_n > max_n:
        raise ValueError("n_range must be ordered as (min_n, max_n).")

    rng = Random(seed)
    graphs: list[nx.Graph] = []
    for graph_index in range(num_graphs):
        graph_size = rng.randint(min_n, max_n)
        graph_seed = rng.randint(0, 10**9)
        graph = make_ba_graph(n=graph_size, m=m, seed=graph_seed)
        graph.graph["graph_index"] = graph_index
        graphs.append(graph)
    return graphs


def relabel_graph_with_prefix(graph: nx.Graph, prefix: str) -> nx.Graph:
    """Return a copy with node names prefixed for easier dataset composition."""
    return nx.relabel_nodes(graph, {node: f"{prefix}{node}" for node in graph.nodes()})


def merge_graphs(graphs: Iterable[nx.Graph]) -> nx.Graph:
    """Compose several graphs into one disconnected test graph."""
    merged = nx.Graph()
    for graph in graphs:
        merged = nx.compose(merged, graph)
    return merged
