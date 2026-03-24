import numpy as np
import networkx as nx

# Load-Capacity Model + LWFRR Redistribution
def init_loads_capacities(G, alpha=0.2):
    """L_i^(0) = degree(i), C_i = (1 + alpha) * L_i"""
    degrees = dict(G.degree())
    loads = {n: float(degrees[n]) for n in G.nodes()}
    caps  = {n: (1 + alpha) * loads[n] for n in G.nodes()}
    return loads, caps

def redistribute_load(G, failed_node, loads, caps, active):
    """LWFRR: redistribute load of failed_node to active neighbors."""
    neighbors = [n for n in G.neighbors(failed_node) if n in active]
    total_cap = sum(caps[n] for n in neighbors)
    if total_cap == 0:
        return loads  # isolated node, no redistribution
    share = loads[failed_node] / total_cap
    for n in neighbors:
        loads[n] += share * caps[n]
    loads[failed_node] = 0.0
    return loads

def run_cascade(G, loads, caps, active):
    """Iteratively fail overloaded nodes until no more failures."""
    failed_this_round = []
    changed = True
    while changed:
        changed = False
        for n in list(active):
            if loads[n] > caps[n]:
                active.remove(n)
                loads = redistribute_load(G, n, loads, caps, active)
                failed_this_round.append(n)
                changed = True
    return loads, active, failed_this_round
