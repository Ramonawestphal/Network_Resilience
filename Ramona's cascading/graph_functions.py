from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx


@dataclass
class CascadeStepResult:
    observation: dict
    done: bool
    info: dict


class CascadingFailureProcess:
    """Cascade-only process on BA graphs.

    Dynamics:
    - t=0: all nodes are active, then an initial random failure set is sampled.
    - t -> t+1: only the nodes that failed at t redistribute their load.
    - New overloads fail, and will redistribute in the next step.
    """

    def __init__(
        self,
        n: int = 50,
        m: int = 2,
        alpha: float = 0.2,
        seed: Optional[int] = None,
        max_steps: int = 100,
        redistribution_mode: str = "capacity_weighted",
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        if m <= 0 or m >= n:
            raise ValueError("m must satisfy 0 < m < n")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if redistribution_mode != "capacity_weighted":
            raise ValueError("redistribution_mode must be 'capacity_weighted'")

        self.n = n
        self.m = m
        self.alpha = float(alpha)
        self.seed = seed
        self.max_steps = max_steps
        self.redistribution_mode = redistribution_mode

        self.rng = random.Random(seed)
        self.graph = nx.barabasi_albert_graph(n=n, m=m, seed=seed)

        self.initial_load: Dict[int, float] = {}
        self.current_load: Dict[int, float] = {}
        self.capacity: Dict[int, float] = {}
        self.active: Dict[int, bool] = {}

        self.t = 0
        self.frontier: List[int] = []
        self.done = False

    def reset(
        self,
        p_fail: float = 0.1,
        initial_failures: Optional[Sequence[int]] = None,
    ) -> dict:
        """Initialize state with all active nodes, then apply initial failures."""
        if not (0.0 <= p_fail <= 1.0):
            raise ValueError("p_fail must be in [0, 1]")

        self.t = 0
        self.done = False

        self.initial_load = {i: float(self.graph.degree(i)) for i in self.graph.nodes()}
        self.current_load = self.initial_load.copy()
        self.capacity = {i: (1.0 + self.alpha) * self.initial_load[i] for i in self.graph.nodes()}
        self.active = {i: True for i in self.graph.nodes()}

        if initial_failures is not None:
            failed_now: List[int] = []
            seen = set()
            for i in initial_failures:
                try:
                    node = int(i)
                except (TypeError, ValueError):
                    continue

                # Accept values like 1.0, but reject non-integer-like values such as 1.7.
                if isinstance(i, float) and not i.is_integer():
                    continue

                if node in self.active and node not in seen:
                    seen.add(node)
                    failed_now.append(node)
        else:
            failed_now = [i for i in self.graph.nodes() if self.rng.random() < p_fail]

        for node in failed_now:
            self.active[node] = False

        self.frontier = failed_now
        self.done = len(self.frontier) == 0
        obs = self._get_observation()
        obs["new_failures"] = failed_now.copy()
        return obs

    def step(self) -> CascadeStepResult:
        """Advance cascade by exactly one wave."""
        if self.done:
            return CascadeStepResult(
                observation=self._get_observation(),
                done=True,
                info={
                    "t": self.t,
                    "processed_failures": [],
                    "new_failures": [],
                    "active_count": sum(1 for v in self.active.values() if v),
                },
            )

        processed = self.frontier.copy()
        self._redistribute_load(processed)

        new_failures: List[int] = []
        for node, is_active in self.active.items():
            if is_active and self.current_load[node] > self.capacity[node] + 1e-12:
                self.active[node] = False
                new_failures.append(node)

        self.frontier = new_failures
        self.t += 1

        active_count = sum(1 for v in self.active.values() if v)
        self.done = (len(self.frontier) == 0) or (active_count == 0) or (self.t >= self.max_steps)

        info = {
            "t": self.t,
            "processed_failures": processed,
            "new_failures": new_failures,
            "active_count": active_count,
        }
        obs = self._get_observation()
        obs["new_failures"] = new_failures.copy()
        return CascadeStepResult(observation=obs, done=self.done, info=info)

    def _redistribute_load(self, failed_nodes: Sequence[int]) -> None:
        """Redistribute load only from the current frontier."""
        for node in failed_nodes:
            outgoing = self.current_load[node]
            self.current_load[node] = 0.0

            neighbors = [nbr for nbr in self.graph.neighbors(node) if self.active[nbr]]
            if not neighbors or outgoing <= 0:
                continue

            capacities = [self.capacity[nbr] for nbr in neighbors]
            total_capacity = sum(capacities)
            if total_capacity <= 1e-12:
                continue

            for nbr, cap in zip(neighbors, capacities):
                self.current_load[nbr] += outgoing * (cap / total_capacity)

    def reactivate_node(self, node: int) -> bool:
        """Reactivate a failed node for RL control actions.

        Returns True only when a failed node is successfully reactivated.
        """
        if node not in self.active:
            return False
        if self.active[node]:
            return False

        self.active[node] = True
        self.current_load[node] = 0.0
        self.capacity[node] = (1.0 + self.alpha) * self.initial_load[node]
        self.frontier = [n for n in self.frontier if n != node]
        return True

    def _load_ratio(self, node: int) -> float:
        cap = self.capacity[node]
        return self.current_load[node] / cap if cap > 0 else float("inf")

    def _get_observation(self) -> dict:
        return {
            "t": self.t,
            "active": self.active.copy(),
            "current_load": self.current_load.copy(),
            "capacity": self.capacity.copy(),
            "load_ratio": {i: self._load_ratio(i) for i in self.graph.nodes()},
            "frontier": self.frontier.copy(),
        }


def draw_cascade_state(
    G: nx.Graph,
    observation: dict,
    new_failures: Optional[Sequence[int]] = None,
    pos: Optional[dict] = None,
    ax=None,
) -> dict:
    """Draw one cascade timestep."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    active = observation["active"]
    load_ratio = observation["load_ratio"]
    new_failures = set(new_failures or [])

    node_colors = []
    node_sizes = []
    for node in G.nodes():
        if node in new_failures:
            node_colors.append("#ffb000")
        elif not active[node]:
            node_colors.append("#d1495b")
        else:
            r = min(max(load_ratio[node], 0.0), 1.4)
            if r < 0.7:
                node_colors.append("#2a9d8f")
            elif r < 1.0:
                node_colors.append("#e9c46a")
            else:
                node_colors.append("#f4a261")

        node_sizes.append(220)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#aaaaaa", width=0.8, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, linewidths=0.5, edgecolors="#333333")
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)

    t = observation.get("t", 0)
    active_count = sum(1 for v in active.values() if v)
    ax.set_title(f"Cascade t={t} | active={active_count}/{len(active)}")
    ax.set_axis_off()
    return pos


def plot_cascade_history(
    G: nx.Graph,
    history: Sequence[dict],
    cols: int = 4,
    figsize_per_subplot: Tuple[float, float] = (4.2, 3.6),
) -> None:
    """Plot all recorded timesteps in a grid."""
    if not history:
        return

    total = len(history)
    rows = math.ceil(total / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_subplot[0] * cols, figsize_per_subplot[1] * rows))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[a] for a in axes]

    pos = nx.spring_layout(G, seed=42)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < total:
                entry = history[idx]
                draw_cascade_state(G, entry["observation"], new_failures=entry.get("new_failures", []), pos=pos, ax=ax)
            else:
                ax.axis("off")
            idx += 1

    fig.tight_layout()
    plt.show()


def initialize_network(
    G: nx.Graph,
    alpha: float = 0.2,
    p_fail: float = 0.1,
    seed: Optional[int] = None,
) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float], Dict[int, bool]]:
    """Compatibility wrapper matching the original function signature."""
    rng = random.Random(seed)

    initial_load = {i: float(G.degree(i)) for i in G.nodes()}
    current_load = initial_load.copy()
    capacity = {i: (1.0 + alpha) * initial_load[i] for i in G.nodes()}
    active = {i: True for i in G.nodes()}

    for i in G.nodes():
        if rng.random() < p_fail:
            active[i] = False

    return initial_load, current_load, capacity, active
