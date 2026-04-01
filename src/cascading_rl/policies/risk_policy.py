from __future__ import annotations

from collections.abc import Hashable

from cascading_rl.envs.recovery import RecoveryObservation

Node = Hashable


def choose_highest_overload_risk_node(observation: RecoveryObservation) -> Node:
    """Choose the failed node whose active neighbors are closest to capacity."""
    valid_actions = observation.valid_actions
    if not valid_actions:
        raise ValueError("No valid nodes remain to reactivate.")

    def overload_risk(node: Node) -> float:
        active_neighbors = [
            neighbor
            for neighbor in observation.graph.neighbors(node)
            if neighbor in observation.active
        ]
        if not active_neighbors:
            return 0.0
        return max(
            observation.loads[neighbor] / observation.capacities[neighbor]
            if observation.capacities[neighbor] > 0.0
            else 0.0
            for neighbor in active_neighbors
        )

    return max(valid_actions, key=lambda node: (overload_risk(node), str(node)))
