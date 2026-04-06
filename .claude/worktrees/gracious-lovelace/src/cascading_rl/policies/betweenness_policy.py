from __future__ import annotations

from collections.abc import Hashable

import networkx as nx

from cascading_rl.envs.recovery import RecoveryObservation

Node = Hashable


def choose_highest_betweenness_failed_node(observation: RecoveryObservation) -> Node:
    """Prioritize the failed node with the highest graph betweenness centrality."""
    if not observation.failed:
        raise ValueError("No failed nodes remain to reactivate.")

    centrality = nx.betweenness_centrality(observation.graph)
    return max(observation.failed, key=lambda node: (centrality[node], str(node)))
