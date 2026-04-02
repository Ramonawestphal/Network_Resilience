from __future__ import annotations

from collections.abc import Hashable

from cascading_rl.envs.recovery import RecoveryObservation

Node = Hashable


def choose_highest_degree_failed_node(observation: RecoveryObservation) -> Node:
    """Prioritize the failed node with the highest original degree."""
    valid_actions = observation.valid_actions
    if not valid_actions:
        raise ValueError("No valid nodes remain to reactivate.")
    return max(
        valid_actions,
        key=lambda node: (observation.graph.degree(node), str(node)),
    )
