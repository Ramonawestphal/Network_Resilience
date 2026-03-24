import networkx as nx
import numpy as np

def make_ba_graph(n=40, m=2, seed=None):
    """
    Generate a Barabási–Albert graph.
    n: number of nodes (proposal: 30–50)
    m: edges per new node (controls density)
    """
    G = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    return G

def augment_load_capacity(G, alpha=0.2, load_fn=None):
    """
    Augment graph nodes with load and capacity as a function of degree.
    
    Default: L_i^(0) = degree(i)   [most common in literature]
    Capacity: C_i = (1 + alpha) * L_i^(0)   [Motter & Lai, 2003]
    
    load_fn: optional custom function load_fn(G, node) -> float
             e.g. lambda G, n: G.degree(n)**1.5  for superlinear scaling
    """
    for node in G.nodes():
        if load_fn is not None:
            load = load_fn(G, node)
        else:
            load = float(G.degree(node))   # L_i^(0) = degree_i
        
        capacity = (1 + alpha) * load      # C_i = (1 + alpha) * L_i^(0)

        G.nodes[node]['load']     = load
        G.nodes[node]['capacity'] = capacity
        G.nodes[node]['active']   = True

    return G

def make_augmented_ba_graph(n=40, m=2, alpha=0.2, seed=None, load_fn=None):
    """Single convenience function: generate + augment."""
    G = make_ba_graph(n=n, m=m, seed=seed)
    G = augment_load_capacity(G, alpha=alpha, load_fn=load_fn)
    return G

def make_graph_batch(num_graphs=500, n_range=(30, 50), m=2,
                     alpha=0.2, seed_offset=0):
    """Generate a training batch of augmented BA graphs."""
    graphs = []
    for i in range(num_graphs):
        n = np.random.randint(n_range[0], n_range[1] + 1)
        G = make_augmented_ba_graph(n=n, m=m, alpha=alpha, seed=seed_offset + i)
        graphs.append(G)
    return graphs

def make_sensitivity_batch(alphas=[0.1, 0.2, 0.3, 0.4, 0.5], num_graphs=100, seed_offset=0):
    """Returns dict of {alpha: [graphs]} for sensitivity analysis."""
    return {
        alpha: make_graph_batch(num_graphs=num_graphs, alpha=alpha, seed_offset=seed_offset)
        for alpha in alphas
    }
