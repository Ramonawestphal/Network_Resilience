import random, networkx as nx
from cascade_env import compute_anc

def random_recovery(failed, B):
    return random.sample(list(failed), min(B, len(failed)))

def degree_first(failed, G, B):
    return sorted(failed, key=lambda n: G.degree(n), reverse=True)[:B]

def overload_risk(failed, G, loads, caps, active, B):
    """Prioritize nodes whose active neighbors are closest to capacity."""
    def risk(n):
        neighbors = [m for m in G.neighbors(n) if m in active]
        if not neighbors: return 0
        return max(loads[m] / caps[m] for m in neighbors)
    return sorted(failed, key=risk, reverse=True)[:B]

def greedy_anc(failed, G, active, B):
    chosen = []
    active_copy = set(active)
    for _ in range(min(B, len(failed))):
        best, best_gain = None, -1
        for n in failed - set(chosen):
            gain = compute_anc(G, active_copy | {n}) - compute_anc(G, active_copy)
            if gain > best_gain:
                best, best_gain = n, gain
        chosen.append(best)
        active_copy.add(best)
    return chosen
