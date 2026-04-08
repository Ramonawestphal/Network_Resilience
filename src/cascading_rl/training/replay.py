from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from random import Random

from cascading_rl.envs.recovery import RecoveryObservation


@dataclass(frozen=True)
class Transition:
    observation: RecoveryObservation
    action: object
    reward: float
    next_observation: RecoveryObservation
    done: bool
    bootstrap_steps: int = 1


class ReplayBuffer:
    """Simple replay buffer for variable-size graph transitions."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")
        self._buffer: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def sample(self, batch_size: int, rng: Random | None = None) -> list[Transition]:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if batch_size > len(self._buffer):
            raise ValueError("Cannot sample more transitions than are stored.")
        rng = rng or Random()
        return rng.sample(list(self._buffer), batch_size)

    def __len__(self) -> int:
        return len(self._buffer)
