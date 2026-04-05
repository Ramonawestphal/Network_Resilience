from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt
from random import Random
from statistics import mean, stdev

import networkx as nx

from cascading_rl.budgeting import compute_scaled_budget, compute_scaled_max_rounds
from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.policies import (
    choose_greedy_anc_node,
    choose_highest_betweenness_failed_node,
    choose_highest_degree_failed_node,
    choose_highest_overload_risk_node,
    choose_random_failed_node,
)

PolicyAction = object
Policy = Callable[[RecoveryObservation], PolicyAction]
PolicyFactory = Callable[[int, int], Policy]

DEFAULT_FINAL_ANC_FAILURE_THRESHOLD = 0.3


def final_anc_failure_threshold_for_reporting(
    env_kwargs: Mapping[str, object] | None,
) -> float:
    """Threshold for ``unsolved_low_final_anc`` stats: env abandonment value, else 0.3."""
    if not env_kwargs:
        return DEFAULT_FINAL_ANC_FAILURE_THRESHOLD
    raw = env_kwargs.get("abandonment_anc_threshold")
    if raw is None:
        return DEFAULT_FINAL_ANC_FAILURE_THRESHOLD
    return float(raw)


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    final_anc: float
    steps: int
    rounds: int
    remaining_failed_nodes: int
    anc_by_round: list[float] = field(default_factory=list)
    mean_delta_anc_per_round: float = 0.0


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
    rounds_when_solved: AggregateMetric | None
    fully_restored_count: int
    episode_count: int
    unsolved_low_final_anc_count: int = 0
    unsolved_low_final_anc_fraction: float = 0.0
    final_anc_failure_threshold_used: float | None = None
    mean_anc_on_failed: AggregateMetric | None = None
    anc_by_round: list[AggregateMetric] = field(default_factory=list)
    mean_delta_anc_per_round: AggregateMetric = field(
        default_factory=lambda: AggregateMetric(mean=0.0, stderr=0.0)
    )


def _aggregate(values: list[float]) -> AggregateMetric:
    if not values:
        raise ValueError("Cannot aggregate an empty metric list.")
    if len(values) == 1:
        return AggregateMetric(mean=values[0], stderr=0.0)
    return AggregateMetric(mean=mean(values), stderr=stdev(values) / sqrt(len(values)))


def summarize_episode_results(
    episode_results: Sequence[EpisodeResult],
    *,
    final_anc_failure_threshold: float | None = None,
) -> PolicyEvaluationSummary:
    """Aggregate per-episode results into one policy summary."""
    solved_rounds = [
        float(result.rounds)
        for result in episode_results
        if result.remaining_failed_nodes == 0
    ]
    fully_restored_count = sum(
        1 for result in episode_results if result.remaining_failed_nodes == 0
    )
    episode_count = len(episode_results)
    failed_episode_anc = [
        result.final_anc
        for result in episode_results
        if result.remaining_failed_nodes > 0
    ]
    max_rounds_observed = max((len(result.anc_by_round) for result in episode_results), default=0)
    anc_by_round = [
        _aggregate(
            [
                result.anc_by_round[round_index]
                for result in episode_results
                if len(result.anc_by_round) > round_index
            ]
        )
        for round_index in range(max_rounds_observed)
    ]
    if final_anc_failure_threshold is not None:
        thr = float(final_anc_failure_threshold)
        unsolved_low = sum(
            1
            for result in episode_results
            if result.remaining_failed_nodes > 0 and result.final_anc < thr
        )
        low_frac = unsolved_low / episode_count if episode_count else 0.0
        thr_used: float | None = thr
    else:
        unsolved_low = 0
        low_frac = 0.0
        thr_used = None

    return PolicyEvaluationSummary(
        final_anc=_aggregate([result.final_anc for result in episode_results]),
        total_reward=_aggregate([result.total_reward for result in episode_results]),
        steps=_aggregate([float(result.steps) for result in episode_results]),
        rounds=_aggregate([float(result.rounds) for result in episode_results]),
        solved_fraction=_aggregate(
            [1.0 if result.remaining_failed_nodes == 0 else 0.0 for result in episode_results]
        ),
        rounds_when_solved=_aggregate(solved_rounds) if solved_rounds else None,
        fully_restored_count=fully_restored_count,
        episode_count=episode_count,
        unsolved_low_final_anc_count=unsolved_low,
        unsolved_low_final_anc_fraction=low_frac,
        final_anc_failure_threshold_used=thr_used,
        mean_anc_on_failed=_aggregate(failed_episode_anc) if failed_episode_anc else None,
        anc_by_round=anc_by_round,
        mean_delta_anc_per_round=_aggregate(
            [result.mean_delta_anc_per_round for result in episode_results]
        ),
    )


def rollout_policy(
    env: RecoveryEnv,
    policy: Policy,
    seed: int | None = None,
) -> EpisodeResult:
    """Run one episode under a policy and collect core comparison metrics."""
    observation = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    initial_anc = env.current_anc()
    anc_by_round: list[float] = []

    if not observation.failed:
        return EpisodeResult(
            total_reward=total_reward,
            final_anc=initial_anc,
            steps=steps,
            rounds=0,
            remaining_failed_nodes=len(observation.failed),
            anc_by_round=anc_by_round,
            mean_delta_anc_per_round=0.0,
        )

    done = False
    info = {
        "anc": initial_anc,
        "failed_nodes": len(observation.failed),
    }

    while not done:
        action = policy(observation)
        if isinstance(action, (list, tuple)):
            observation, reward, done, info = env.step_batch(list(action))
        else:
            observation, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if bool(info.get("round_complete")):
            anc_by_round.append(float(info["anc"]))

    final_anc = float(info["anc"])
    rounds = env.current_round
    if rounds > len(anc_by_round):
        anc_by_round.append(final_anc)
    mean_delta_anc_per_round = (final_anc - initial_anc) / rounds if rounds > 0 else 0.0

    return EpisodeResult(
        total_reward=total_reward,
        final_anc=final_anc,
        steps=steps,
        rounds=rounds,
        remaining_failed_nodes=int(info["failed_nodes"]),
        anc_by_round=anc_by_round,
        mean_delta_anc_per_round=mean_delta_anc_per_round,
    )


def evaluate_policies(
    policy_map: Mapping[str, Policy],
    env_factory: Callable[[int], RecoveryEnv],
    seeds: Iterable[int],
) -> dict[str, PolicyEvaluationSummary]:
    """Evaluate multiple policies with matched seeds and aggregate the outcomes."""
    seeds_list = list(seeds)
    summaries: dict[str, PolicyEvaluationSummary] = {}
    thr = final_anc_failure_threshold_for_reporting(None)

    for policy_name, policy in policy_map.items():
        episode_results = [
            rollout_policy(env_factory(seed), policy, seed=seed)
            for seed in seeds_list
        ]
        summaries[policy_name] = summarize_episode_results(
            episode_results,
            final_anc_failure_threshold=thr,
        )

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
    env_kwargs: Mapping[str, object] | None = None,
    scale_budget: bool = False,
    scale_max_rounds: bool = False,
    reference_n: int = 40,
) -> dict[str, PolicyEvaluationSummary]:
    """Evaluate policy factories across fixed graphs and matched seeds."""
    seeds_list = list(seeds)
    episode_results_by_policy: dict[str, list[EpisodeResult]] = {
        name: [] for name in policy_factories
    }
    env_kwargs = dict(env_kwargs or {})
    thr = final_anc_failure_threshold_for_reporting(env_kwargs)

    for graph_index, graph in enumerate(graphs):
        resolved_budget = compute_scaled_budget(
            budget,
            num_nodes=graph.number_of_nodes(),
            reference_n=reference_n,
            enabled=scale_budget,
        )
        resolved_max_rounds = (
            compute_scaled_max_rounds(
                max_rounds,
                num_nodes=graph.number_of_nodes(),
                reference_n=reference_n,
                enabled=scale_max_rounds,
            )
            if max_rounds is not None
            else None
        )
        for seed in seeds_list:
            for policy_name, policy_factory in policy_factories.items():
                env = RecoveryEnv(
                    graph,
                    alpha=alpha,
                    pfail=pfail,
                    budget=resolved_budget,
                    max_rounds=resolved_max_rounds,
                    seed=seed,
                    **env_kwargs,
                )
                policy = policy_factory(graph_index, seed)
                result = rollout_policy(env, policy, seed=seed)
                episode_results_by_policy[policy_name].append(result)

    return {
        policy_name: summarize_episode_results(
            episode_results,
            final_anc_failure_threshold=thr,
        )
        for policy_name, episode_results in episode_results_by_policy.items()
    }
