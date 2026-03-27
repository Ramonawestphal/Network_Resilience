from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from random import Random

import networkx as nx

from cascading_rl.dynamics.cascade import (
    advance_cascade_round,
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
    frontier: frozenset[Node]
    remaining_budget: int
    budget: int  
    current_round: int
    max_rounds: int

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
        max_rounds: int | None = None,
        seed: int | None = None,
        load_fn: LoadFunction | None = None,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be at least 1.")
        if max_rounds is not None and max_rounds < 1:
            raise ValueError("max_rounds must be at least 1 when provided.")

        self.base_graph = graph.copy()
        self.alpha = alpha
        self.pfail = pfail
        self.budget = budget
        self.max_rounds = max_rounds if max_rounds is not None else self.base_graph.number_of_nodes()
        self.load_fn = load_fn
        self._rng = Random(seed)
        self.state: CascadeState | None = None
        self.remaining_budget = budget
        self.current_round = 1

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
        self.current_round = 1
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
            frontier=frozenset(self.state.frontier),
            remaining_budget=self.remaining_budget,
            budget=self.budget,
            current_round=self.current_round,
            max_rounds=self.max_rounds,
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

        action_round = self.current_round
        action_index_in_round = self.budget - self.remaining_budget + 1
        previous_anc = accumulated_normalized_connectivity(self.state.graph, self.state.active)
        self.state = reactivate_node(self.state, action)
        self.remaining_budget -= 1

        round_complete = self.remaining_budget == 0
        newly_failed: list[Node] = []
        if round_complete and self.state.failed:
            newly_failed = advance_cascade_round(self.state)

        next_anc = accumulated_normalized_connectivity(self.state.graph, self.state.active)
        reward = next_anc - previous_anc
        exhausted_rounds = action_round >= self.max_rounds
        if not self.state.failed:
            done = True
        elif round_complete and exhausted_rounds:
            done = True
        else:
            done = False

        if round_complete and not done:
            self.current_round += 1
            self.remaining_budget = self.budget

        info = {
            "anc": next_anc,
            "failed_nodes": len(self.state.failed),
            "active_nodes": len(self.state.active),
            "frontier_nodes": len(self.state.frontier),
            "action_round": action_round,
            "action_index_in_round": action_index_in_round,
            "next_round": self.current_round,
            "remaining_budget": self.remaining_budget,
            "round_complete": round_complete,
            "max_rounds_reached": round_complete and exhausted_rounds,
            "reactivated_node": action,
            "newly_failed_nodes": newly_failed,
            "cascade_executed": round_complete,
        }
        return self.observe(), reward, done, info

    def step_batch(self, actions: list[Node]) -> tuple[RecoveryObservation, float, bool, dict]:
        """Reactivate up to B nodes at once, then fire one cascade."""
        if self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        if len(actions) > self.budget:
            raise ValueError(f"Cannot reactivate more than {self.budget} nodes per round.")
        if len(set(actions)) != len(actions):
            raise ValueError("Duplicate actions in batch.")

        previous_anc = accumulated_normalized_connectivity(self.state.graph, self.state.active)

        for action in actions:
            self.state = reactivate_node(self.state, action)

        newly_failed: list[Node] = []
        if self.state.failed:
            newly_failed = advance_cascade_round(self.state)

        next_anc = accumulated_normalized_connectivity(self.state.graph, self.state.active)
        reward = next_anc - previous_anc

        exhausted_rounds = self.current_round >= self.max_rounds
        done = not self.state.failed or exhausted_rounds

        if not done:
            self.current_round += 1
            self.remaining_budget = self.budget

        info = {
            "anc": next_anc,
            "failed_nodes": len(self.state.failed),
            "active_nodes": len(self.state.active),
            "newly_failed_nodes": newly_failed,
            "action_round": self.current_round,
            "max_rounds_reached": exhausted_rounds,
            "cascade_executed": True,
        }
        return self.observe(), reward, done, info