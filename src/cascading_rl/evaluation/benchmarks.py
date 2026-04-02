from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from random import Random
from statistics import mean, stdev

import networkx as nx

from cascading_rl.budgeting import compute_scaled_budget
from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.policies import (
    choose_greedy_anc_node,
    choose_highest_betweenness_failed_node,
    choose_highest_degree_failed_node,
    choose_highest_overload_risk_node,
    choose_random_failed_node,
)

Policy = Callable[[RecoveryObservation], object]
PolicyFactory = Callable[[int, int], Policy]


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    final_anc: float
    steps: int
    rounds: int
    remaining_failed_nodes: int
    threshold_step: int | None
    threshold_round: int | None


@dataclass(frozen=True)
class AggregateMetric:
    mean: float
    stderr: float


@dataclass(frozen=True)
class PolicyEvaluationSummary:
    final_anc: AggregateMetric
    total_reward: AggregateMetric
    steps: AggregateMetric
    rounds: AggregateMetric
    solved_fraction: AggregateMetric
    threshold_hit_fraction: AggregateMetric
    threshold_step: AggregateMetric | None
    threshold_round: AggregateMetric | None


def _aggregate(values: list[float]) -> AggregateMetric:
    if not values:
        raise ValueError("Cannot aggregate an empty metric list.")
    if len(values) == 1:
        return AggregateMetric(mean=values[0], stderr=0.0)
    return AggregateMetric(mean=mean(values), stderr=stdev(values) / sqrt(len(values)))


def summarize_episode_results(episode_results: Sequence[EpisodeResult]) -> PolicyEvaluationSummary:
    """Aggregate per-episode results into one policy summary."""
    threshold_steps = [
        float(result.threshold_step)
        for result in episode_results
        if result.threshold_step is not None
    ]
    threshold_rounds = [
        float(result.threshold_round)
        for result in episode_results
        if result.threshold_round is not None
    ]
    return PolicyEvaluationSummary(
        final_anc=_aggregate([result.final_anc for result in episode_results]),
        total_reward=_aggregate([result.total_reward for result in episode_results]),
        steps=_aggregate([float(result.steps) for result in episode_results]),
        rounds=_aggregate([float(result.rounds) for result in episode_results]),
        solved_fraction=_aggregate(
            [1.0 if result.remaining_failed_nodes == 0 else 0.0 for result in episode_results]
        ),
        threshold_hit_fraction=_aggregate(
            [1.0 if result.threshold_step is not None else 0.0 for result in episode_results]
        ),
        threshold_step=_aggregate(threshold_steps) if threshold_steps else None,
        threshold_round=_aggregate(threshold_rounds) if threshold_rounds else None,
    )


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
    threshold_round = 0 if threshold_step == 0 else None

    if not observation.failed:
        return EpisodeResult(
            total_reward=total_reward,
            final_anc=current_anc,
            steps=steps,
            rounds=0,
            remaining_failed_nodes=len(observation.failed),
            threshold_step=threshold_step,
            threshold_round=threshold_round,
        )

    done = False
    info = {
        "anc": current_anc,
        "failed_nodes": len(observation.failed),
    }

    while not done:
        if observation.remaining_budget > 0:
            action = policy(observation)
            observation, reward, done, info = env.step(action)
        else:
            # `step` requires remaining budget per repair; advance dynamics with a no-op round.
            observation, reward, done, info = env.step_batch([])
        total_reward += reward
        steps += 1
        if tau is not None and threshold_step is None and float(info["anc"]) >= tau:
            threshold_step = steps
            threshold_round = int(info["action_round"])

    return EpisodeResult(
        total_reward=total_reward,
        final_anc=float(info["anc"]),
        steps=steps,
        rounds=env.current_round,
        remaining_failed_nodes=int(info["failed_nodes"]),
        threshold_step=threshold_step,
        threshold_round=threshold_round,
    )


def evaluate_policies(
    policy_map: Mapping[str, Policy],
    env_factory: Callable[[int], RecoveryEnv],
    seeds: Iterable[int],
    tau: float | None = None,
) -> dict[str, PolicyEvaluationSummary]:
    """Evaluate multiple policies with matched seeds and aggregate the outcomes."""
    seeds_list = list(seeds)
    summaries: dict[str, PolicyEvaluationSummary] = {}

    for policy_name, policy in policy_map.items():
        episode_results = [
            rollout_policy(env_factory(seed), policy, seed=seed, tau=tau)
            for seed in seeds_list
        ]
        summaries[policy_name] = summarize_episode_results(episode_results)

    return summaries


def build_policy_factories(base_seed: int = 0) -> dict[str, PolicyFactory]:
    """Create baseline policy factories for matched-seed sweeps."""

    def random_factory(graph_index: int, seed: int) -> Policy:
        rng = Random(f"{base_seed}:{graph_index}:{seed}")
        return lambda observation: choose_random_failed_node(observation, rng=rng)

    return {
        "random": random_factory,
        "degree": lambda graph_index, seed: choose_highest_degree_failed_node,
        "risk": lambda graph_index, seed: choose_highest_overload_risk_node,
        "greedy": lambda graph_index, seed: choose_greedy_anc_node,
        "betweenness": lambda graph_index, seed: choose_highest_betweenness_failed_node,
    }


def evaluate_policy_factories_on_graphs(
    graphs: Sequence[nx.Graph],
    policy_factories: Mapping[str, PolicyFactory],
    *,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int | None = None,
    seeds: Iterable[int],
    tau: float,
    env_kwargs: Mapping[str, object] | None = None,
    scale_budget: bool = False,
    reference_n: int = 40,
) -> dict[str, PolicyEvaluationSummary]:
    """Evaluate policy factories across fixed graphs and matched seeds."""
    seeds_list = list(seeds)
    episode_results_by_policy: dict[str, list[EpisodeResult]] = {
        name: [] for name in policy_factories
    }
    env_kwargs = dict(env_kwargs or {})

    for graph_index, graph in enumerate(graphs):
        resolved_budget = compute_scaled_budget(
            budget,
            num_nodes=graph.number_of_nodes(),
            reference_n=reference_n,
            enabled=scale_budget,
        )
        for seed in seeds_list:
            for policy_name, policy_factory in policy_factories.items():
                env = RecoveryEnv(
                    graph,
                    alpha=alpha,
                    pfail=pfail,
                    budget=resolved_budget,
                    max_rounds=max_rounds,
                    seed=seed,
                    **env_kwargs,
                )
                policy = policy_factory(graph_index, seed)
                result = rollout_policy(env, policy, seed=seed, tau=tau)
                episode_results_by_policy[policy_name].append(result)

    return {
        policy_name: summarize_episode_results(episode_results)
        for policy_name, episode_results in episode_results_by_policy.items()
    }
