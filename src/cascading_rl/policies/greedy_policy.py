from __future__ import annotations

from collections.abc import Hashable

from cascading_rl.dynamics.cascade import CascadeState, advance_cascade_round, reactivate_node
from cascading_rl.envs.recovery import RecoveryObservation
from cascading_rl.metrics.connectivity import accumulated_normalized_connectivity

Node = Hashable


def choose_greedy_anc_node(observation: RecoveryObservation) -> Node:
    """Choose the failed node with the highest one-step ANC gain."""
    if not observation.failed:
        raise ValueError("No failed nodes remain to reactivate.")

    current_anc = accumulated_normalized_connectivity(observation.graph, observation.active)
    best_node: Node | None = None
    best_gain = float("-inf")

    for node in observation.failed:
        trial_state = CascadeState(
            graph=observation.graph,
            loads=dict(observation.loads),
            capacities=dict(observation.capacities),
            active=set(observation.active),
            failed=set(observation.failed),
            frontier=set(observation.frontier),
        )
        next_state = reactivate_node(trial_state, node)
        if next_state.failed and next_state.frontier:
            advance_cascade_round(next_state)
        next_anc = accumulated_normalized_connectivity(next_state.graph, next_state.active)
        gain = next_anc - current_anc
        if gain > best_gain or (gain == best_gain and str(node) > str(best_node)):
            best_gain = gain
            best_node = node

    assert best_node is not None
    return best_node
