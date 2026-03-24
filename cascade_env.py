import gymnasium as gym
import numpy as np
import networkx as nx
from cascade_model import init_loads_capacities, run_cascade

# Custom MDP (s,a,r)

def compute_anc(G, active):
    """Accumulated Normalized Connectivity."""
    subgraph = G.subgraph(active)
    components = list(nx.connected_components(subgraph))
    n = len(G.nodes())
    return sum((len(c) / n) ** 2 for c in components)

class CascadeRecoveryEnv(gym.Env):
    def __init__(self, G, alpha=0.2, pfail=0.1, budget=3):
        self.G_base = G
        self.alpha = alpha
        self.pfail = pfail
        self.budget = budget

    def reset(self):
        self.G = self.G_base.copy()
        self.loads, self.caps = init_loads_capacities(self.G, self.alpha)
        self.active = set(self.G.nodes())
        # Initial random failures
        for n in list(self.active):
            if np.random.rand() < self.pfail:
                self.active.discard(n)
        self.loads, self.active, _ = run_cascade(
            self.G, self.loads, self.caps, self.active)
        self.failed = set(self.G.nodes()) - self.active
        self.remaining_budget = self.budget
        return self._get_state()

    def step(self, action):  # action = node to reactivate
        assert action in self.failed
        prev_anc = compute_anc(self.G, self.active)
        # Reactivate: restore capacity, zero load
        self.active.add(action)
        self.failed.discard(action)
        self.loads[action] = 0.0
        # Cascade may still propagate
        self.loads, self.active, _ = run_cascade(
            self.G, self.loads, self.caps, self.active)
        self.failed = set(self.G.nodes()) - self.active
        self.remaining_budget -= 1
        reward = compute_anc(self.G, self.active) - prev_anc
        done = self.remaining_budget == 0 or len(self.failed) == 0
        return self._get_state(), reward, done, {}

    def _get_state(self):
        return {
            "active": self.active,
            "failed": self.failed,
            "loads": self.loads,
            "caps": self.caps,
            "budget": self.remaining_budget
        }
