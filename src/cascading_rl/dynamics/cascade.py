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

    def copy(self) -> "CascadeState":
        return CascadeState(
            graph=self.graph,
            loads=dict(self.loads),
            capacities=dict(self.capacities),
            active=set(self.active),
            failed=set(self.failed),
        )


def degree_load(graph: nx.Graph, node: Node) -> float:
    """Default initial load: node degree."""
    return float(graph.degree(node))


def initialize_loads_and_capacities(
    graph: nx.Graph,
    alpha: float = 0.2,
    load_fn: LoadFunction | None = None,
) -> tuple[dict[Node, float], dict[Node, float]]:
    """Set initial loads and capacities using the proposal's baseline rule."""
    if alpha < 0:
        raise ValueError("alpha must be non-negative.")

    load_fn = load_fn or degree_load
    loads: dict[Node, float] = {}
    capacities: dict[Node, float] = {}
    for node in graph.nodes():
        load = float(load_fn(graph, node))
        if load < 0:
            raise ValueError("load_fn must return non-negative values.")
        loads[node] = load
        capacities[node] = (1.0 + alpha) * load
    return loads, capacities


def sample_initial_failures(
    graph: nx.Graph,
    pfail: float = 0.1,
    rng: Random | None = None,
    active: Iterable[Node] | None = None,
) -> set[Node]:
    """Sample initial failures uniformly across active nodes."""
    if not 0.0 <= pfail <= 1.0:
        raise ValueError("pfail must lie in [0, 1].")

    rng = rng or Random()
    candidate_nodes = list(active if active is not None else graph.nodes())
    return {node for node in candidate_nodes if rng.random() < pfail}


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


def fail_nodes(
    graph: nx.Graph,
    nodes_to_fail: Iterable[Node],
    loads: dict[Node, float],
    capacities: dict[Node, float],
    active: set[Node],
) -> list[Node]:
    """Remove nodes from service, then redistribute their load to survivors."""
    failed_nodes = [node for node in nodes_to_fail if node in active]
    for node in failed_nodes:
        active.remove(node)
    for node in failed_nodes:
        redistribute_load(graph, node, loads, capacities, active)
    return failed_nodes


def propagate_cascade(
    graph: nx.Graph,
    loads: dict[Node, float],
    capacities: dict[Node, float],
    active: set[Node],
) -> list[Node]:
    """Apply overload failures round by round until the cascade settles."""
    failed_order: list[Node] = []
    while True:
        overloaded = [node for node in active if loads[node] > capacities[node]]
        if not overloaded:
            return failed_order
        failed_order.extend(fail_nodes(graph, overloaded, loads, capacities, active))


def build_initial_state(
    graph: nx.Graph,
    alpha: float = 0.2,
    pfail: float = 0.1,
    rng: Random | None = None,
    load_fn: LoadFunction | None = None,
) -> CascadeState:
    """Create a fresh cascade state after initial failures and cascade settling."""
    loads, capacities = initialize_loads_and_capacities(graph, alpha=alpha, load_fn=load_fn)
    active = set(graph.nodes())

    initial_failures = sample_initial_failures(graph, pfail=pfail, rng=rng, active=active)
    fail_nodes(graph, initial_failures, loads, capacities, active)
    propagate_cascade(graph, loads, capacities, active)

    return CascadeState(
        graph=graph.copy(),
        loads=loads,
        capacities=capacities,
        active=active,
        failed=set(graph.nodes()) - active,
    )


def reactivate_node(state: CascadeState, node: Node) -> CascadeState:
    """Return the post-action state after reactivating a failed node."""
    if node not in state.failed:
        raise ValueError("Only failed nodes can be reactivated.")

    next_state = state.copy()
    next_state.active.add(node)
    next_state.failed.remove(node)
    next_state.loads[node] = 0.0

    propagate_cascade(
        next_state.graph,
        next_state.loads,
        next_state.capacities,
        next_state.active,
    )
    next_state.failed = set(next_state.graph.nodes()) - next_state.active
    return next_state
