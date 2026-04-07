from __future__ import annotations

import itertools
from collections.abc import Hashable, Sequence

from cascading_rl.dynamics.cascade import (
    CascadeState,
    advance_cascade_round,
    reactivate_node,
)
from cascading_rl.envs.recovery import RecoveryObservation
from cascading_rl.metrics.connectivity import normalized_connectivity

Node = Hashable


def observation_to_cascade_state(observation: RecoveryObservation) -> CascadeState:
    return CascadeState(
        graph=observation.graph,
        loads=dict(observation.loads),
        capacities=dict(observation.capacities),
        active=set(observation.active),
        failed=set(observation.failed),
        frontier=set(observation.frontier),
    )


def delta_nc_after_round_batch(state: CascadeState, nodes: Sequence[Node]) -> float:
    """ANC change after reactivating ``nodes`` (in sorted order) then one cascade wave, matching ``step_batch``."""
    trial = state.copy()
    previous_anc = normalized_connectivity(trial.graph, trial.active)
    ordered = sorted(nodes, key=str)
    for node in ordered:
        trial = reactivate_node(trial, node)
    if trial.frontier and trial.failed:
        advance_cascade_round(trial)
    post_anc = normalized_connectivity(trial.graph, trial.active)
    return post_anc - previous_anc


def choose_greedy_nc_node(observation: RecoveryObservation) -> list[Node]:
    """Choose up to ``k`` failed nodes maximizing ANC gain after reactivations and one cascade wave.

    ``k = min(remaining_budget, len(valid_actions))``. Returns nodes in ascending ``str(node)`` order
    for deterministic ``step_batch`` application.
    """
    valid = observation.valid_actions
    if not valid:
        raise ValueError("No failed nodes remain to reactivate.")

    k = min(int(observation.remaining_budget), len(valid))
    if k < 1:
        raise ValueError("No recovery budget remains.")

    base = observation_to_cascade_state(observation)
    best_delta = float("-inf")
    best_key: tuple[str, ...] | None = None
    best_nodes: tuple[Node, ...] | None = None

    for combo in itertools.combinations(valid, k):
        delta = delta_nc_after_round_batch(base, combo)
        sorted_combo = tuple(sorted(combo, key=str))
        tie_key = tuple(str(n) for n in sorted_combo)
        if best_nodes is None:
            best_delta = delta
            best_key = tie_key
            best_nodes = sorted_combo
        elif delta > best_delta:
            best_delta = delta
            best_key = tie_key
            best_nodes = sorted_combo
        elif delta == best_delta and tie_key < best_key:
            best_key = tie_key
            best_nodes = sorted_combo

    assert best_nodes is not None
    return list(best_nodes)
