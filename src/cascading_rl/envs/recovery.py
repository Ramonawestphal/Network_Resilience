from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from random import Random

import networkx as nx

from cascading_rl.dynamics.cascade import (
    CascadeState,
    LoadFunction,
    build_initial_state,
    reactivate_node,
)
from cascading_rl.metrics.connectivity import accumulated_normalized_connectivity

Node = Hashable


@dataclass(frozen=True)
class RecoveryObservation:
    graph: nx.Graph
    loads: dict[Node, float]
    capacities: dict[Node, float]
    active: frozenset[Node]
    failed: frozenset[Node]
    remaining_budget: int

    @property
    def valid_actions(self) -> tuple[Node, ...]:
        return tuple(self.failed)


class RecoveryEnv:
    """Budget-constrained recovery environment with post-action cascades."""

    def __init__(
        self,
        graph: nx.Graph,
        alpha: float = 0.2,
        pfail: float = 0.1,
        budget: int = 3,
        seed: int | None = None,
        load_fn: LoadFunction | None = None,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be at least 1.")

        self.base_graph = graph.copy()
        self.alpha = alpha
        self.pfail = pfail
        self.budget = budget
        self.load_fn = load_fn
        self._rng = Random(seed)
        self.state: CascadeState | None = None
        self.remaining_budget = budget

    def reset(self, seed: int | None = None) -> RecoveryObservation:
        if seed is not None:
            self._rng.seed(seed)
        self.state = build_initial_state(
            self.base_graph,
            alpha=self.alpha,
            pfail=self.pfail,
            rng=self._rng,
            load_fn=self.load_fn,
        )
        self.remaining_budget = self.budget
        return self.observe()

    def observe(self) -> RecoveryObservation:
        if self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        return RecoveryObservation(
            graph=self.state.graph,
            loads=dict(self.state.loads),
            capacities=dict(self.state.capacities),
            active=frozenset(self.state.active),
            failed=frozenset(self.state.failed),
            remaining_budget=self.remaining_budget,
        )

    def current_anc(self) -> float:
        if self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        return accumulated_normalized_connectivity(self.state.graph, self.state.active)

    def step(self, action: Node) -> tuple[RecoveryObservation, float, bool, dict[str, float | int]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        if self.remaining_budget <= 0:
            raise RuntimeError("No recovery budget remains.")

        previous_anc = accumulated_normalized_connectivity(self.state.graph, self.state.active)
        self.state = reactivate_node(self.state, action)
        self.remaining_budget -= 1

        next_anc = accumulated_normalized_connectivity(self.state.graph, self.state.active)
        reward = next_anc - previous_anc
        done = self.remaining_budget == 0 or not self.state.failed
        info = {
            "anc": next_anc,
            "failed_nodes": len(self.state.failed),
            "active_nodes": len(self.state.active),
        }
        return self.observe(), reward, done, info
