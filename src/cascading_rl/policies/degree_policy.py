from __future__ import annotations

from collections.abc import Hashable

from cascading_rl.envs.recovery import RecoveryObservation

Node = Hashable


def choose_highest_degree_failed_node(observation: RecoveryObservation) -> Node:
    """Prioritize the failed node with the highest original degree."""
    if not observation.failed:
        raise ValueError("No failed nodes remain to reactivate.")
    return max(
        observation.failed,
        key=lambda node: (observation.graph.degree(node), str(node)),
    )
