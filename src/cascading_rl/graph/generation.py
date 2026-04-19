from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path
from random import Random

import networkx as nx


def load_real_world_graph(name: str, data_dir: Path | str | None = None) -> nx.Graph:
    """Load a pre-downloaded real-world network from data/processed/.

    Parameters
    ----------
    name : "ieee300" or "watts_strogatz"
        Which dataset to load.
    data_dir : path to the data/processed/ directory. Defaults to the repo's
        data/processed/ folder resolved relative to this file.

    Returns
    -------
    A connected, undirected NetworkX graph with 0-indexed integer nodes.
    Raises FileNotFoundError if the CSV has not been downloaded yet —
    run scripts/download_real_world_data.py first.
    """
    filenames = {
        "ieee300": "ieee300_edges.csv",
        "watts_strogatz": "watts_strogatz_edges.csv",
    }
    if name not in filenames:
        raise ValueError(f"Unknown real-world graph '{name}'. Choose from: {list(filenames)}")

    if data_dir is None:
        # Resolve relative to this file: src/cascading_rl/graph/ -> repo root -> data/processed/
        data_dir = Path(__file__).resolve().parents[3] / "data" / "processed"
    csv_path = Path(data_dir) / filenames[name]

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Real-world graph file not found: {csv_path}\n"
            "Run:  python scripts/download_real_world_data.py"
        )

    edges: list[tuple[int, int]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            u, v = int(row["from"]), int(row["to"])
            if u != v:
                edges.append((u, v))

    g = nx.Graph()
    g.add_edges_from(edges)

    if g.number_of_nodes() == 0:
        raise ValueError(
            f"Real-world graph '{name}' loaded from {csv_path} contains no valid edges. "
            "The CSV may be empty or contain only self-loops."
        )

    # Ensure connectivity — keep largest component and re-index 0..N-1
    if not nx.is_connected(g):
        largest_cc = max(nx.connected_components(g), key=len)
        g = g.subgraph(largest_cc).copy()
        mapping = {old: new for new, old in enumerate(sorted(g.nodes()))}
        g = nx.relabel_nodes(g, mapping)

    g.graph["name"] = name
    return g


def make_ba_graph(n: int = 40, m: int = 2, seed: int | None = None) -> nx.Graph:
    """Generate a Barabasi-Albert graph used for synthetic training data."""
    if n < 2:
        raise ValueError("n must be at least 2.")
    if m < 1:
        raise ValueError("m must be at least 1.")
    if m >= n:
        raise ValueError("m must be smaller than n for a BA graph.")
    return nx.barabasi_albert_graph(n=n, m=m, seed=seed)


def _sample_graph(
    graph_type: str,
    n: int,
    m: int,
    graph_seed: int,
) -> nx.Graph:
    """Instantiate one random graph of the requested topology."""
    gt = graph_type.lower()
    if gt == "ba":
        return make_ba_graph(n=n, m=m, seed=graph_seed)
    if gt == "er":
        if n < 2:
            raise ValueError("n must be at least 2.")
        # Mean degree scale comparable to BA attachment parameter m.
        p = min(1.0, (2.0 * m) / max(n - 1, 1))
        return nx.fast_gnp_random_graph(n, p, seed=graph_seed)
    if gt == "ws":
        if n < 3:
            raise ValueError("Watts-Strogatz requires n >= 3; increase --n-low or use ba/er.")
        k = 2 * min(m, (n - 1) // 2)
        if k < 2:
            k = 2
        if k >= n:
            k = n - 1 if (n - 1) % 2 == 0 else n - 2
        k = max(2, k)
        if k % 2:
            k -= 1
        k = max(2, min(k, n - 1))
        return nx.connected_watts_strogatz_graph(n, k, 0.1, seed=graph_seed)
    raise ValueError(f"Unknown graph_type {graph_type!r}. Use 'ba', 'er', or 'ws'.")


def make_graph_batch(
    num_graphs: int = 32,
    n_range: tuple[int, int] = (30, 50),
    m: int = 2,
    seed: int | None = None,
    *,
    graph_type: str = "ba",
) -> list[nx.Graph]:
    """Generate a batch of synthetic graphs with varying sizes.

    ``graph_type``: ``ba`` (Barabasi-Albert), ``er`` (Erdos-Renyi G(n,p)),
    or ``ws`` (Watts-Strogatz). Default ``ba`` matches historical behaviour.
    """
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
        graph = _sample_graph(graph_type, graph_size, m, graph_seed)
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
