from __future__ import annotations

from collections.abc import Hashable

import networkx as nx

from cascading_rl.envs.recovery import RecoveryObservation

Node = Hashable


def choose_highest_betweenness_failed_node(observation: RecoveryObservation) -> Node:
    """Prioritize the failed node with the highest graph betweenness centrality."""
    valid_actions = observation.valid_actions
    if not valid_actions:
        raise ValueError("No valid nodes remain to reactivate.")

    centrality = nx.betweenness_centrality(observation.graph)
    return max(valid_actions, key=lambda node: (centrality[node], str(node)))
