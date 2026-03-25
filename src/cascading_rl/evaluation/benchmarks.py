from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from math import sqrt
from statistics import mean, pstdev

from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation

Policy = Callable[[RecoveryObservation], object]


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    final_anc: float
    steps: int
    remaining_failed_nodes: int
    threshold_step: int | None


@dataclass(frozen=True)
class AggregateMetric:
    mean: float
    stderr: float


@dataclass(frozen=True)
class PolicyEvaluationSummary:
    final_anc: AggregateMetric
    total_reward: AggregateMetric
    steps: AggregateMetric
    solved_fraction: AggregateMetric
    threshold_hit_fraction: AggregateMetric
    threshold_step: AggregateMetric | None


def _aggregate(values: list[float]) -> AggregateMetric:
    if not values:
        raise ValueError("Cannot aggregate an empty metric list.")
    if len(values) == 1:
        return AggregateMetric(mean=values[0], stderr=0.0)
    return AggregateMetric(mean=mean(values), stderr=pstdev(values) / sqrt(len(values)))


def rollout_policy(
    env: RecoveryEnv,
    policy: Policy,
    seed: int | None = None,
    tau: float | None = None,
) -> EpisodeResult:
    """Run one episode under a policy and collect core comparison metrics."""
    observation = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    current_anc = env.current_anc()
    threshold_step = 0 if tau is not None and current_anc >= tau else None

    if not observation.failed or observation.remaining_budget <= 0:
        return EpisodeResult(
            total_reward=total_reward,
            final_anc=current_anc,
            steps=steps,
            remaining_failed_nodes=len(observation.failed),
            threshold_step=threshold_step,
        )

    done = False
    info = {
        "anc": current_anc,
        "failed_nodes": len(observation.failed),
    }

    while not done:
        action = policy(observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if tau is not None and threshold_step is None and float(info["anc"]) >= tau:
            threshold_step = steps

    return EpisodeResult(
        total_reward=total_reward,
        final_anc=float(info["anc"]),
        steps=steps,
        remaining_failed_nodes=int(info["failed_nodes"]),
        threshold_step=threshold_step,
    )


def evaluate_policies(
    policy_map: Mapping[str, Policy],
    env_factory: Callable[[int], RecoveryEnv],
    seeds: Iterable[int],
    tau: float | None = None,
) -> dict[str, PolicyEvaluationSummary]:
    """Evaluate multiple policies with matched seeds and aggregate the outcomes."""
    summaries: dict[str, PolicyEvaluationSummary] = {}

    for policy_name, policy in policy_map.items():
        episode_results = [
            rollout_policy(env_factory(seed), policy, seed=seed, tau=tau)
            for seed in seeds
        ]

        threshold_steps = [
            float(result.threshold_step)
            for result in episode_results
            if result.threshold_step is not None
        ]
        summaries[policy_name] = PolicyEvaluationSummary(
            final_anc=_aggregate([result.final_anc for result in episode_results]),
            total_reward=_aggregate([result.total_reward for result in episode_results]),
            steps=_aggregate([float(result.steps) for result in episode_results]),
            solved_fraction=_aggregate(
                [1.0 if result.remaining_failed_nodes == 0 else 0.0 for result in episode_results]
            ),
            threshold_hit_fraction=_aggregate(
                [1.0 if result.threshold_step is not None else 0.0 for result in episode_results]
            ),
            threshold_step=_aggregate(threshold_steps) if threshold_steps else None,
        )

    return summaries
