from __future__ import annotations

from collections.abc import Hashable
from random import Random

from cascading_rl.envs.recovery import RecoveryObservation

Node = Hashable


def choose_random_failed_node(
    observation: RecoveryObservation, rng: Random | None = None
) -> Node:
    """Sample a failed node uniformly at random."""
    if not observation.failed:
        raise ValueError("No failed nodes remain to reactivate.")
    rng = rng or Random()
    return rng.choice(tuple(observation.failed))
