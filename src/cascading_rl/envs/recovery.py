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
from cascading_rl.metrics.connectivity import normalized_connectivity

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
    budget: int
    current_round: int
    max_rounds: int
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
    """Budget-constrained recovery environment with batch-per-round cascade waves.

    ``step`` exposes intra-round single-node repairs, while ``step_batch`` performs a
    full round decision of up to ``B`` repairs followed by one cascade wave. Both use
    the same reward: net PR/ANC change from the state at the start of the transition to
    after any repairs in that transition and after the cascade wave (if the round ends
    and a wave runs). Intra-round steps omit the wave, so reward equals repair-only gain.
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
        abandonment_nc_threshold: float | None = None,
    ) -> None:
        if budget < 1:
            raise ValueError("budget must be at least 1.")
        if max_rounds is not None and max_rounds < 1:
            raise ValueError("max_rounds must be at least 1 when provided.")
        if obs_hops is not None and obs_hops < 1:
            raise ValueError("obs_hops must be at least 1 when provided.")
        if abandonment_nc_threshold is not None and not (
            0.0 <= abandonment_nc_threshold <= 1.0
        ):
            raise ValueError("abandonment_nc_threshold must lie in [0, 1] when set.")

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
        self.abandonment_nc_threshold = (
            float(abandonment_nc_threshold)
            if abandonment_nc_threshold is not None
            else None
        )
        self._rng = Random(seed)
        self.state: CascadeState | None = None
        self.remaining_budget = budget
        self.current_round = 1

    def reset(self, seed: int | None = None) -> RecoveryObservation:
        # When ``seed`` is given, the environment RNG is fully re-seeded before
        # ``build_initial_state``; the constructor ``seed=`` does not affect
        # failure sampling after such a reset (only ``reset(seed=...)`` matters).
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
            budget=self.budget,
            current_round=self.current_round,
            max_rounds=self.max_rounds,
            action_space=self.action_space,
            obs_hops=self.obs_hops,
        )

    def current_nc(self) -> float:
        if self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        return normalized_connectivity(self.state.graph, self.state.active)

    def _abandon_due_to_low_nc(self, post_cascade_nc: float) -> bool:
        if self.abandonment_nc_threshold is None or self.state is None:
            return False
        return post_cascade_nc < self.abandonment_nc_threshold and bool(self.state.failed)

    def step(self, action: Node) -> tuple[RecoveryObservation, float, bool, dict[str, object]]:
        if self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        if self.remaining_budget <= 0:
            raise RuntimeError("No recovery budget remains.")
        if action not in self.observe().valid_actions:
            raise ValueError("Action must be a currently valid recovery choice.")

        action_round = self.current_round
        action_index_in_round = self.budget - self.remaining_budget + 1
        self.state = reactivate_node(self.state, action)
        self.remaining_budget -= 1

        nc_after_reactivation = normalized_connectivity(self.state.graph, self.state.active)

        newly_failed: list[Node] = []
        cascade_executed = False
        round_complete = self.remaining_budget == 0
        if round_complete and self.state.frontier and self.state.failed:
            cascade_executed = True
            newly_failed = advance_cascade_round(self.state)

        nc_after_cascade = normalized_connectivity(self.state.graph, self.state.active)
        reward = nc_after_cascade if round_complete else 0.0

        exhausted_rounds = action_round >= self.max_rounds
        abandoned = self._abandon_due_to_low_nc(nc_after_cascade)
        if not self.state.failed:
            done = True
            abandoned = False
        elif abandoned:
            done = True
        elif round_complete and exhausted_rounds:
            done = True
        else:
            done = False

        if round_complete and not done:
            self.current_round += 1
            self.remaining_budget = self.budget

        info = {
            "nc": nc_after_cascade,
            "nc_after_cascade": nc_after_cascade,
            "nc_after_reactivation": nc_after_reactivation,
            "failed_nodes": len(self.state.failed),
            "active_nodes": len(self.state.active),
            "frontier_nodes": len(self.state.frontier),
            "action_round": action_round,
            "action_index_in_round": action_index_in_round,
            "next_round": self.current_round,
            "remaining_budget": self.remaining_budget,
            "round_complete": round_complete,
            "max_rounds_reached": (
                round_complete
                and exhausted_rounds
                and bool(self.state.failed)
                and not abandoned
            ),
            "reactivated_node": action,
            "newly_failed_nodes": newly_failed,
            "cascade_executed": cascade_executed,
            "abandoned": abandoned,
        }
        return self.observe(), reward, done, info

    def step_batch(self, actions: list[Node]) -> tuple[RecoveryObservation, float, bool, dict[str, object]]:
        """Reactivate up to the remaining budget at once, then fire one cascade wave."""
        if self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        if len(actions) > self.remaining_budget:
            raise ValueError(f"Cannot reactivate more than {self.remaining_budget} nodes this round.")
        if len(set(actions)) != len(actions):
            raise ValueError("Duplicate actions in batch.")

        valid_actions = set(self.observe().valid_actions)
        invalid_actions = [action for action in actions if action not in valid_actions]
        if invalid_actions:
            raise ValueError(f"Invalid recovery actions: {invalid_actions}")

        action_round = self.current_round

        for action in actions:
            self.state = reactivate_node(self.state, action)
        repaired_nc = normalized_connectivity(self.state.graph, self.state.active)

        newly_failed: list[Node] = []
        cascade_executed = False
        if self.state.frontier and self.state.failed:
            cascade_executed = True
            newly_failed = advance_cascade_round(self.state)

        post_cascade_nc = normalized_connectivity(self.state.graph, self.state.active)
        reward = post_cascade_nc

        exhausted_rounds = self.current_round >= self.max_rounds
        abandoned = self._abandon_due_to_low_nc(post_cascade_nc)
        if not self.state.failed:
            done = True
            abandoned = False
        elif abandoned:
            done = True
        else:
            done = exhausted_rounds

        if not done:
            self.current_round += 1
            self.remaining_budget = self.budget
        else:
            self.remaining_budget = max(0, self.remaining_budget - len(actions))

        info = {
            "nc": post_cascade_nc,
            "nc_after_cascade": post_cascade_nc,
            "nc_after_reactivation": repaired_nc,
            "failed_nodes": len(self.state.failed),
            "active_nodes": len(self.state.active),
            "frontier_nodes": len(self.state.frontier),
            "newly_failed_nodes": newly_failed,
            "action_round": action_round,
            "actions": list(actions),
            "remaining_budget": self.remaining_budget,
            "round_complete": True,
            "max_rounds_reached": (
                exhausted_rounds and bool(self.state.failed) and not abandoned
            ),
            "cascade_executed": cascade_executed,
            "abandoned": abandoned,
        }
        return self.observe(), reward, done, info
