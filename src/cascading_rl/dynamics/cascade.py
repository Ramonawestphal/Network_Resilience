from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from random import Random

import networkx as nx

Node = Hashable
LoadFunction = Callable[[nx.Graph, Node], float]


@dataclass
class CascadeState:
    """Mutable network state used by the environment and heuristic rollouts."""

    graph: nx.Graph
    loads: dict[Node, float]
    capacities: dict[Node, float]
    active: set[Node]
    failed: set[Node]
    frontier: set[Node]

    def copy(self) -> "CascadeState":
        return CascadeState(
            graph=self.graph,
            loads=dict(self.loads),
            capacities=dict(self.capacities),
            active=set(self.active),
            failed=set(self.failed),
            frontier=set(self.frontier),
        )


def degree_load(graph: nx.Graph, node: Node) -> float:
    """Default initial load: node degree."""
    return float(graph.degree(node))


def initialize_loads_and_capacities(
    graph: nx.Graph,
    alpha: float = 0.2,
    load_fn: LoadFunction | None = None,
    *,
    capacity_noise: float = 0.0,
    rng: Random | None = None,
) -> tuple[dict[Node, float], dict[Node, float]]:
    """Set initial loads and capacities using the proposal's baseline rule."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if capacity_noise < 0:
        raise ValueError("capacity_noise must be non-negative.")

    load_fn = load_fn or degree_load
    rng = rng or Random()
    loads: dict[Node, float] = {}
    capacities: dict[Node, float] = {}
    for node in graph.nodes():
        load = float(load_fn(graph, node))
        if load < 0:
            raise ValueError("load_fn must return non-negative values.")
        loads[node] = load
        baseline_capacity = (1.0 + alpha) * load
        if capacity_noise > 0.0 and baseline_capacity > 0.0:
            multiplier = max(0.0, 1.0 + rng.gauss(0.0, capacity_noise))
            capacities[node] = baseline_capacity * multiplier
        else:
            capacities[node] = baseline_capacity
    return loads, capacities


def sample_initial_failures(
    graph: nx.Graph,
    pfail: float = 0.1,
    rng: Random | None = None,
    active: Iterable[Node] | None = None,
    *,
    weights: dict[Node, float] | None = None,
) -> set[Node]:
    """Sample initial failures across active nodes."""
    if not 0.0 <= pfail <= 1.0:
        raise ValueError("pfail must lie in [0, 1].")

    rng = rng or Random()
    candidate_nodes = list(active if active is not None else graph.nodes())
    if not candidate_nodes or pfail == 0.0:
        return set()

    if weights is None:
        return {node for node in candidate_nodes if rng.random() < pfail}

    positive_weights = [max(0.0, float(weights.get(node, 0.0))) for node in candidate_nodes]
    mean_weight = sum(positive_weights) / max(1, len(positive_weights))
    if mean_weight <= 0.0:
        return {node for node in candidate_nodes if rng.random() < pfail}

    failures: set[Node] = set()
    for node, weight in zip(candidate_nodes, positive_weights, strict=False):
        tilted_p = pfail * (weight / mean_weight)
        if rng.random() < min(1.0, max(0.0, tilted_p)):
            failures.add(node)
    return failures


def redistribute_load(
    graph: nx.Graph,
    failed_node: Node,
    loads: dict[Node, float],
    capacities: dict[Node, float],
    active: set[Node],
) -> None:
    """Redistribute a failed node's load to active neighbors by capacity share."""
    surviving_neighbors = [node for node in graph.neighbors(failed_node) if node in active]
    total_capacity = sum(capacities[node] for node in surviving_neighbors)

    if total_capacity > 0.0:
        failed_load = loads[failed_node]
        for node in surviving_neighbors:
            loads[node] += failed_load * (capacities[node] / total_capacity)
    loads[failed_node] = 0.0


def mark_failed_nodes(
    nodes_to_fail: Iterable[Node],
    active: set[Node],
    failed: set[Node],
) -> list[Node]:
    """Mark active nodes as failed without redistributing their load yet."""
    failed_nodes = [node for node in nodes_to_fail if node in active]
    for node in failed_nodes:
        active.remove(node)
        failed.add(node)
    return failed_nodes


def redistribute_frontier(
    graph: nx.Graph,
    frontier: Iterable[Node],
    loads: dict[Node, float],
    capacities: dict[Node, float],
    active: set[Node],
) -> list[Node]:
    """Redistribute the load from the nodes that failed in the current wave."""
    processed = list(frontier)
    for node in processed:
        redistribute_load(graph, node, loads, capacities, active)
    return processed


def identify_overloaded_nodes(
    active: Iterable[Node],
    loads: dict[Node, float],
    capacities: dict[Node, float],
) -> list[Node]:
    """Return the currently active nodes whose load exceeds capacity."""
    return [node for node in active if loads[node] > capacities[node]]


def advance_cascade_round(state: CascadeState) -> list[Node]:
    """Advance the cascade by exactly one redistribution/failure wave."""
    if not state.frontier:
        return []

    redistribute_frontier(
        state.graph,
        state.frontier,
        state.loads,
        state.capacities,
        state.active,
    )
    newly_failed = mark_failed_nodes(
        identify_overloaded_nodes(state.active, state.loads, state.capacities),
        state.active,
        state.failed,
    )
    state.frontier = set(newly_failed)
    return newly_failed


def propagate_cascade(
    graph: nx.Graph,
    loads: dict[Node, float],
    capacities: dict[Node, float],
    active: set[Node],
    failed: set[Node] | None = None,
    frontier: Iterable[Node] | None = None,
) -> list[Node]:
    """Apply overload failures wave by wave until the cascade settles."""
    failed_nodes = set() if failed is None else set(failed)

    state = CascadeState(
        graph=graph,
        loads=loads,
        capacities=capacities,
        active=active,
        failed=failed_nodes,
        frontier=set(frontier or []),
    )
    failed_order: list[Node] = []

    if not state.frontier:
        initial_overloaded = mark_failed_nodes(
            identify_overloaded_nodes(state.active, state.loads, state.capacities),
            state.active,
            state.failed,
        )
        state.frontier = set(initial_overloaded)
        failed_order.extend(initial_overloaded)

    while state.frontier:
        failed_order.extend(advance_cascade_round(state))

    return failed_order


def build_initial_state(
    graph: nx.Graph,
    alpha: float = 0.2,
    pfail: float = 0.1,
    rng: Random | None = None,
    load_fn: LoadFunction | None = None,
    *,
    capacity_noise: float = 0.0,
    failure_bias: str = "uniform",
) -> CascadeState:
    """Create a fresh state after the single exogenous failure event at t=0."""
    working_graph = graph.copy()
    rng = rng or Random()
    loads, capacities = initialize_loads_and_capacities(
        working_graph,
        alpha=alpha,
        load_fn=load_fn,
        capacity_noise=capacity_noise,
        rng=rng,
    )
    active = set(working_graph.nodes())
    failed: set[Node] = set()

    failure_bias = str(failure_bias or "uniform").lower()
    weights: dict[Node, float] | None
    if failure_bias == "uniform":
        weights = None
    elif failure_bias == "degree":
        weights = {node: float(working_graph.degree(node)) for node in working_graph.nodes()}
    elif failure_bias == "load":
        weights = {node: float(loads[node]) for node in working_graph.nodes()}
    else:
        raise ValueError("failure_bias must be one of: 'uniform', 'degree', 'load'.")

    initial_failures = sample_initial_failures(
        working_graph,
        pfail=pfail,
        rng=rng,
        active=active,
        weights=weights,
    )
    frontier = set(mark_failed_nodes(initial_failures, active, failed))
    return CascadeState(
        graph=working_graph,
        loads=loads,
        capacities=capacities,
        active=active,
        failed=failed,
        frontier=frontier,
    )


def reactivate_node(state: CascadeState, node: Node) -> CascadeState:
    """Return the state immediately after reactivating a failed node."""
    if node not in state.failed:
        raise ValueError("Only failed nodes can be reactivated.")

    next_state = state.copy()
    next_state.active.add(node)
    next_state.failed.remove(node)
    next_state.frontier.discard(node)
    next_state.loads[node] = 0.0
    return next_state
