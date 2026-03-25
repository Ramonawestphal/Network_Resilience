from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation

Policy = Callable[[RecoveryObservation], object]


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    final_anc: float
    steps: int
    remaining_failed_nodes: int


def rollout_policy(
    env: RecoveryEnv, policy: Policy, seed: int | None = None
) -> EpisodeResult:
    """Run one episode under a policy and collect core comparison metrics."""
    observation = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = policy(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

    return EpisodeResult(
        total_reward=total_reward,
        final_anc=float(info["anc"]),
        steps=steps,
        remaining_failed_nodes=int(info["failed_nodes"]),
    )
