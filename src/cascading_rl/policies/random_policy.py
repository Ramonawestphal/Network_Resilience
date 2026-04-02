from __future__ import annotations

from collections.abc import Hashable
from random import Random

from cascading_rl.envs.recovery import RecoveryObservation

Node = Hashable


def choose_random_failed_node(
    observation: RecoveryObservation, rng: Random | None = None
) -> Node:
    """Sample a failed node uniformly at random."""
    valid_actions = observation.valid_actions
    if not valid_actions:
        raise ValueError("No valid nodes remain to reactivate.")
    rng = rng or Random()
    return rng.choice(valid_actions)
