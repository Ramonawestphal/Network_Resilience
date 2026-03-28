from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from dataclasses import dataclass
from random import Random

import networkx as nx  # type: ignore[import-untyped]

from cascading_rl.dynamics.cascade import (
    advance_cascade_round,
    CascadeState,
    LoadFunction,
    build_initial_state,
    reactivate_node,
)
from cascading_rl.metrics.connectivity import accumulated_normalized_connectivity

Node = Hashable


def _nodes_within_hops_of_failed(graph: nx.Graph, failed: frozenset[Node], hops: int) -> set[Node]:
    """Nodes whose shortest-path distance to some failed node is <= hops (failed nodes included)."""
    if hops < 1:
        return set(graph.nodes())
    if not failed:
        return set(graph.nodes())
    dist: dict[Node, int] = {}
    queue: deque[Node] = deque()
    for node in failed:
        dist[node] = 0
        queue.append(node)
    while queue:
        current = queue.popleft()
        d = dist[current]
        if d == hops:
            continue
        for neighbor in graph.neighbors(current):
            if neighbor not in dist:
                dist[neighbor] = d + 1
                queue.append(neighbor)
    return set(dist.keys())


def _apply_obs_hops_mask(
    graph: nx.Graph,
    loads: dict[Node, float],
    capacities: dict[Node, float],
    failed: frozenset[Node],
    obs_hops: int,
) -> tuple[dict[Node, float], dict[Node, float]]:
    visible = _nodes_within_hops_of_failed(graph, failed, obs_hops)
    masked_loads = dict(loads)
    masked_capacities = dict(capacities)
    for node in graph.nodes():
        if node not in visible:
            masked_loads[node] = 0.0
            masked_capacities[node] = 0.0
    return masked_loads, masked_capacities


@dataclass(frozen=True)
class RecoveryObservation:
    graph: nx.Graph
    loads: dict[Node, float]
    capacities: dict[Node, float]
    active: frozenset[Node]
    failed: frozenset[Node]
    frontier: frozenset[Node]
    remaining_budget: int
    current_round: int
    action_space: str = "failed"
    obs_hops: int | None = None

    @property
    def valid_actions(self) -> tuple[Node, ...]:
        failed = tuple(self.failed)
        if not failed:
            return ()

        mode = str(self.action_space or "failed").lower()
        if mode == "failed":
            return failed
        if mode == "frontier":
            constrained = tuple(node for node in self.frontier if node in self.failed)
            return constrained or failed
        if mode in {"adjacent", "adjacent_to_active"}:
            active = set(self.active)
            constrained = tuple(
                node
                for node in self.failed
                if any(neighbor in active for neighbor in self.graph.neighbors(node))
            )
            return constrained or failed
        raise ValueError("action_space must be one of: 'failed', 'frontier', 'adjacent'.")


class RecoveryEnv:
    """Budget-constrained recovery environment with post-action cascade waves.

    Step semantics:
      1) Reactivate exactly one failed node (action).
      2) Reward = ANC(after reactivation, before cascade) - ANC(before), so cascade
         side-effects do not affect this step's credit assignment.
      3) Then advance the cascade by one wave (if the pre-step frontier was non-empty).
      4) Budget is per recovery round; rounds repeat until solved or max_rounds is reached.

    Partial observability (``obs_hops=k``): the agent only sees load/capacity for nodes
    within ``k`` hops of **any** currently failed node (partial observability around the
    failure region); other nodes show zero load/capacity in observations while the graph
    structure stays full (for GNN message passing).
    """

    def __init__(
        self,
        graph: nx.Graph,
        alpha: float = 0.2,
        pfail: float = 0.1,
        budget: int = 3,
        max_rounds: int | None = None,
        seed: int | None = None,
        load_fn: LoadFunction | None = None,
        *,
        capacity_noise: float = 0.0,
        failure_bias: str = "uniform",
        action_space: str = "failed",
        obs_hops: int | None = None,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be at least 1.")
        if max_rounds is not None and max_rounds < 1:
            raise ValueError("max_rounds must be at least 1 when provided.")
        if obs_hops is not None and obs_hops < 1:
            raise ValueError("obs_hops must be at least 1 when provided.")

        self.base_graph = graph.copy()
        self.alpha = alpha
        self.pfail = pfail
        self.budget = budget
        self.max_rounds = max_rounds if max_rounds is not None else self.base_graph.number_of_nodes()
        self.load_fn = load_fn
        self.capacity_noise = float(capacity_noise)
        self.failure_bias = str(failure_bias)
        self.action_space = str(action_space)
        self.obs_hops = obs_hops
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
            capacity_noise=self.capacity_noise,
            failure_bias=self.failure_bias,
        )
        self.remaining_budget = self.budget
        self.current_round = 1
        return self.observe()

    def observe(self) -> RecoveryObservation:
        if self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        loads = dict(self.state.loads)
        capacities = dict(self.state.capacities)
        failed_set = frozenset(self.state.failed)
        if self.obs_hops is not None:
            loads, capacities = _apply_obs_hops_mask(
                self.state.graph, loads, capacities, failed_set, self.obs_hops
            )
        return RecoveryObservation(
            graph=self.state.graph,
            loads=loads,
            capacities=capacities,
            active=frozenset(self.state.active),
            failed=failed_set,
            frontier=frozenset(self.state.frontier),
            remaining_budget=self.remaining_budget,
            current_round=self.current_round,
            action_space=self.action_space,
            obs_hops=self.obs_hops,
        )

    def current_anc(self) -> float:
        if self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        return accumulated_normalized_connectivity(self.state.graph, self.state.active)

    def step(self, action: Node) -> tuple[RecoveryObservation, float, bool, dict[str, object]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        if self.remaining_budget <= 0:
            raise RuntimeError("No recovery budget remains.")
        if action not in self.state.failed:
            raise ValueError("Action must be a currently failed node.")

        action_round = self.current_round
        action_index_in_round = self.budget - self.remaining_budget + 1
        previous_anc = accumulated_normalized_connectivity(self.state.graph, self.state.active)
        frontier_before = set(self.state.frontier)
        self.state = reactivate_node(self.state, action)
        self.remaining_budget -= 1

        anc_after_reactivation = accumulated_normalized_connectivity(self.state.graph, self.state.active)
        reward = anc_after_reactivation - previous_anc

        newly_failed: list[Node] = []
        cascade_executed = False
        if frontier_before and self.state.failed:
            cascade_executed = True
            newly_failed = advance_cascade_round(self.state)

        anc_after_cascade = accumulated_normalized_connectivity(self.state.graph, self.state.active)

        round_complete = self.remaining_budget == 0
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
            "anc": anc_after_cascade,
            "anc_after_cascade": anc_after_cascade,
            "anc_after_reactivation": anc_after_reactivation,
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
            "cascade_executed": cascade_executed,
        }
        return self.observe(), reward, done, info
