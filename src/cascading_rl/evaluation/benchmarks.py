from __future__ import annotations


from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt
from random import Random
from statistics import mean, stdev

import networkx as nx

from cascading_rl.budgeting import compute_scaled_budget, compute_scaled_max_rounds
from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.metrics.connectivity import anc_fixed_horizon, anc_adaptive_horizon
from cascading_rl.policies import (
    choose_greedy_nc_node,
    choose_highest_betweenness_failed_node,
    choose_highest_degree_failed_node,
    choose_highest_overload_risk_node,
    choose_random_failed_node,
)
from cascading_rl.policies.greedy_policy import (
    delta_nc_after_round_batch,
    observation_to_cascade_state,
)

PolicyAction = object
Policy = Callable[[RecoveryObservation], PolicyAction]
PolicyFactory = Callable[[int, int], Policy]

DEFAULT_FINAL_NC_FAILURE_THRESHOLD = 0.3


@dataclass(frozen=True)
class StepMetrics:
    round: int
    degree_ratio: float       # degree(chosen) / max(degree(f) for f in failed)
    overload_risk: float      # max load/capacity among active neighbors of chosen node
    nc_gain: float            # NC gain of chosen action (greedy lookahead)
    greedy_nc_gain: float     # NC gain of the greedy-optimal action
    action_rank: int          # rank of chosen action by greedy NC gain (1 = optimal)


def final_nc_failure_threshold_for_reporting(
    env_kwargs: Mapping[str, object] | None,
) -> float:
    """Threshold for ``unsolved_low_final_nc`` stats: env abandonment value, else 0.3."""
    if not env_kwargs:
        return DEFAULT_FINAL_NC_FAILURE_THRESHOLD
    raw = env_kwargs.get("abandonment_nc_threshold")
    if raw is None:
        return DEFAULT_FINAL_NC_FAILURE_THRESHOLD
    return float(raw)


@dataclass(frozen=True)
class EpisodeResult:
    total_reward: float
    final_nc: float
    steps: int
    rounds: int
    remaining_failed_nodes: int
    nc_by_round: list[float] = field(default_factory=list)
    mean_delta_nc_per_round: float = 0.0
    anc_fixed: float = 0.0
    anc_adaptive: float = 0.0
    step_metrics: list[StepMetrics] = field(default_factory=list)


@dataclass(frozen=True)
class AggregateMetric:
    mean: float
    stderr: float


@dataclass(frozen=True)
class PolicyEvaluationSummary:
    final_nc: AggregateMetric
    total_reward: AggregateMetric
    steps: AggregateMetric
    rounds: AggregateMetric
    solved_fraction: AggregateMetric
    rounds_when_solved: AggregateMetric | None
    fully_restored_count: int
    episode_count: int
    unsolved_low_final_nc_count: int = 0
    unsolved_low_final_nc_fraction: float = 0.0
    final_nc_failure_threshold_used: float | None = None
    mean_nc_on_failed: AggregateMetric | None = None
    nc_by_round: list[AggregateMetric] = field(default_factory=list)
    mean_delta_nc_per_round: AggregateMetric = field(
        default_factory=lambda: AggregateMetric(mean=0.0, stderr=0.0)
    )
    anc_fixed: AggregateMetric = field(default_factory=lambda: AggregateMetric(0.0, 0.0))
    anc_adaptive: AggregateMetric = field(default_factory=lambda: AggregateMetric(0.0, 0.0))
    mean_degree_ratio: AggregateMetric = field(default_factory=lambda: AggregateMetric(0.0, 0.0))
    mean_overload_risk: AggregateMetric = field(default_factory=lambda: AggregateMetric(0.0, 0.0))
    mean_nc_gain: AggregateMetric = field(default_factory=lambda: AggregateMetric(0.0, 0.0))
    mean_greedy_nc_gain: AggregateMetric = field(default_factory=lambda: AggregateMetric(0.0, 0.0))
    mean_action_rank: AggregateMetric = field(default_factory=lambda: AggregateMetric(0.0, 0.0))


def _compute_step_metrics(
    observation: RecoveryObservation,
    action: object,
    current_round: int,
) -> StepMetrics:
    """Compute per-step diagnostic metrics for the chosen action."""
    graph = observation.graph
    loads = observation.loads
    capacities = observation.capacities

    # Normalise action to a list of chosen nodes
    if isinstance(action, (list, tuple)):
        chosen_nodes: list = list(action)
    else:
        chosen_nodes = [action]
    chosen = chosen_nodes[0]

    # degree_ratio: degree of chosen node vs max degree among failed nodes
    if observation.failed:
        max_failed_degree = max(graph.degree(f) for f in observation.failed)
        degree_ratio = graph.degree(chosen) / max_failed_degree if max_failed_degree > 0 else 0.0
    else:
        degree_ratio = 0.0

    # overload_risk: max load/capacity ratio among active neighbours of chosen node
    active_neighbours = [n for n in graph.neighbors(chosen) if n in observation.active]
    if active_neighbours:
        overload_risk = max(loads[n] / capacities[n] for n in active_neighbours)
    else:
        overload_risk = 0.0

    # nc_gain: NC gain of the actual chosen action
    base_state = observation_to_cascade_state(observation)
    nc_gain = delta_nc_after_round_batch(base_state, chosen_nodes)

    # Rank each valid action individually (O(|failed|) instead of O(|failed|^B))
    valid = list(observation.valid_actions)
    singleton_deltas: list[tuple[float, object]] = []
    for node in valid:
        delta = delta_nc_after_round_batch(base_state, [node])
        singleton_deltas.append((delta, node))
    singleton_deltas.sort(key=lambda x: (-x[0], str(x[1])))

    greedy_nc_gain = singleton_deltas[0][0] if singleton_deltas else nc_gain

    # Rank of chosen node among all singletons (1 = best)
    action_rank = 1
    for rank, (_, node) in enumerate(singleton_deltas, start=1):
        if node == chosen:
            action_rank = rank
            break

    return StepMetrics(
        round=current_round,
        degree_ratio=degree_ratio,
        overload_risk=overload_risk,
        nc_gain=nc_gain,
        greedy_nc_gain=greedy_nc_gain,
        action_rank=action_rank,
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
    final_nc_failure_threshold: float | None = None,
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
    failed_episode_nc = [
        result.final_nc
        for result in episode_results
        if result.remaining_failed_nodes > 0
    ]
    max_rounds_observed = max((len(result.nc_by_round) for result in episode_results), default=0)
    nc_by_round = [
        _aggregate(
            [
                result.nc_by_round[round_index]
                for result in episode_results
                if len(result.nc_by_round) > round_index
            ]
        )
        for round_index in range(max_rounds_observed)
    ]
    if final_nc_failure_threshold is not None:
        thr = float(final_nc_failure_threshold)
        unsolved_low = sum(
            1
            for result in episode_results
            if result.remaining_failed_nodes > 0 and result.final_nc < thr
        )
        low_frac = unsolved_low / episode_count if episode_count else 0.0
        thr_used: float | None = thr
    else:
        unsolved_low = 0
        low_frac = 0.0
        thr_used = None

    # Flatten all StepMetrics across every episode
    all_step_metrics: list[StepMetrics] = [
        sm for result in episode_results for sm in result.step_metrics
    ]

    def _agg_step_field(attr: str) -> AggregateMetric:
        vals = [float(getattr(sm, attr)) for sm in all_step_metrics]
        return _aggregate(vals) if vals else AggregateMetric(0.0, 0.0)

    return PolicyEvaluationSummary(
        final_nc=_aggregate([result.final_nc for result in episode_results]),
        total_reward=_aggregate([result.total_reward for result in episode_results]),
        steps=_aggregate([float(result.steps) for result in episode_results]),
        rounds=_aggregate([float(result.rounds) for result in episode_results]),
        solved_fraction=_aggregate(
            [1.0 if result.remaining_failed_nodes == 0 else 0.0 for result in episode_results]
        ),
        rounds_when_solved=_aggregate(solved_rounds) if solved_rounds else None,
        fully_restored_count=fully_restored_count,
        episode_count=episode_count,
        unsolved_low_final_nc_count=unsolved_low,
        unsolved_low_final_nc_fraction=low_frac,
        final_nc_failure_threshold_used=thr_used,
        mean_nc_on_failed=_aggregate(failed_episode_nc) if failed_episode_nc else None,
        nc_by_round=nc_by_round,
        mean_delta_nc_per_round=_aggregate(
            [result.mean_delta_nc_per_round for result in episode_results]
        ),
        anc_fixed=_aggregate([result.anc_fixed for result in episode_results]),
        anc_adaptive=_aggregate([result.anc_adaptive for result in episode_results]),
        mean_degree_ratio=_agg_step_field("degree_ratio"),
        mean_overload_risk=_agg_step_field("overload_risk"),
        mean_nc_gain=_agg_step_field("nc_gain"),
        mean_greedy_nc_gain=_agg_step_field("greedy_nc_gain"),
        mean_action_rank=_agg_step_field("action_rank"),
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
    initial_nc = env.current_nc()
    nc_by_round: list[float] = []
    step_metrics_list: list[StepMetrics] = []

    if not observation.failed:
        return EpisodeResult(
            total_reward=total_reward,
            final_nc=initial_nc,
            steps=steps,
            rounds=0,
            remaining_failed_nodes=len(observation.failed),
            nc_by_round=nc_by_round,
            mean_delta_nc_per_round=0.0,
            anc_fixed=anc_fixed_horizon(nc_by_round, env.max_rounds or 1),
            anc_adaptive=anc_adaptive_horizon(nc_by_round),
            step_metrics=step_metrics_list,
        )

    done = False
    info = {
        "nc": initial_nc,
        "failed_nodes": len(observation.failed),
    }

    while not done:
        action = policy(observation)
        step_metrics_list.append(
            _compute_step_metrics(observation, action, env.current_round)
        )
        if isinstance(action, (list, tuple)):
            observation, reward, done, info = env.step_batch(list(action))
        else:
            observation, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1
        if bool(info.get("round_complete")):
            nc_by_round.append(float(info["nc"]))

    final_nc = float(info["nc"])
    rounds = env.current_round
    if rounds > len(nc_by_round):
        nc_by_round.append(final_nc)
    mean_delta_nc_per_round = (final_nc - initial_nc) / rounds if rounds > 0 else 0.0
    max_rounds_for_anc = env.max_rounds

    return EpisodeResult(
        total_reward=total_reward,
        final_nc=final_nc,
        steps=steps,
        rounds=rounds,
        remaining_failed_nodes=int(info["failed_nodes"]),
        nc_by_round=nc_by_round,
        mean_delta_nc_per_round=mean_delta_nc_per_round,
        anc_fixed=anc_fixed_horizon(nc_by_round, max_rounds_for_anc),
        anc_adaptive=anc_adaptive_horizon(nc_by_round),
        step_metrics=step_metrics_list,
    )


def evaluate_policies(
    policy_map: Mapping[str, Policy],
    env_factory: Callable[[int], RecoveryEnv],
    seeds: Iterable[int],
) -> dict[str, PolicyEvaluationSummary]:
    """Evaluate multiple policies with matched seeds and aggregate the outcomes."""
    seeds_list = list(seeds)
    summaries: dict[str, PolicyEvaluationSummary] = {}
    probe_seed = seeds_list[0] if seeds_list else 0
    probe_env = env_factory(probe_seed)
    thr = final_nc_failure_threshold_for_reporting(
        {"abandonment_nc_threshold": probe_env.abandonment_nc_threshold}
    )

    for policy_name, policy in policy_map.items():
        episode_results = [
            rollout_policy(env_factory(seed), policy, seed=seed)
            for seed in seeds_list
        ]
        summaries[policy_name] = summarize_episode_results(
            episode_results,
            final_nc_failure_threshold=thr,
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
        "greedy": lambda graph_index, seed: choose_greedy_nc_node,
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
    thr = final_nc_failure_threshold_for_reporting(env_kwargs)

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
            final_nc_failure_threshold=thr,
        )
        for policy_name, episode_results in episode_results_by_policy.items()
    }
