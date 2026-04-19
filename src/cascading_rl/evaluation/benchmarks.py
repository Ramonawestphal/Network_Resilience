from __future__ import annotations


import itertools
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from math import sqrt
from random import Random
from statistics import mean, stdev

import networkx as nx
from scipy import stats as _scipy_stats

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
class PolicyComparisonResult:
    """Pairwise statistical comparison between two policies on matched episodes.

    Episodes are matched by their position in the evaluation list, which
    corresponds to identical (graph_index, seed) pairs when produced by
    evaluate_policy_factories_on_graphs. The comparison is therefore a
    paired test: each observation is the per-episode difference d_i = A_i - B_i
    on the same environmental realisation.

    wilcoxon_statistic / wilcoxon_p_value: two-sided Wilcoxon signed-rank test
        H0: median(d_i) = 0  vs  H1: median(d_i) ≠ 0
        Non-parametric; does not assume normality of differences. Appropriate
        for ANC scores which are bounded in [0,1] and may be skewed.

    bootstrap_ci_low / bootstrap_ci_high: percentile bootstrap CI for E[d_i].
        Uses n_boot resamples with replacement from {d_i}. Parametric-free;
        valid under mild regularity conditions regardless of ANC distribution.
    """

    policy_a: str
    policy_b: str
    metric: str
    n_pairs: int
    mean_difference: float       # E[A_i - B_i]: positive means A > B
    bootstrap_ci_low: float      # lower bound of (1-alpha)% CI for mean difference
    bootstrap_ci_high: float     # upper bound of (1-alpha)% CI for mean difference
    ci_level: float              # e.g. 0.95
    wilcoxon_statistic: float
    wilcoxon_p_value: float
    alpha_level: float           # significance threshold used
    significant: bool            # wilcoxon_p_value < alpha_level


def _extract_metric(result: "EpisodeResult", metric: str) -> float:
    """Extract a scalar metric from an EpisodeResult by name."""
    if metric == "anc_fixed":
        return result.anc_fixed
    if metric == "anc_adaptive":
        return result.anc_adaptive
    if metric == "final_nc":
        return result.final_nc
    if metric == "total_reward":
        return result.total_reward
    if metric == "rounds":
        return float(result.rounds)
    if metric == "solved":
        return 1.0 if result.remaining_failed_nodes == 0 else 0.0
    raise ValueError(
        f"Unknown metric '{metric}'. Choose from: anc_fixed, anc_adaptive, "
        "final_nc, total_reward, rounds, solved."
    )


def bootstrap_mean_ci(
    differences: list[float],
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    rng: Random | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean paired difference.

    Resamples differences with replacement n_boot times. Returns the
    (alpha/2, 1-alpha/2) empirical quantiles of the bootstrap distribution of
    the sample mean, where alpha = 1 - ci.

    This estimator is consistent under mild regularity conditions and requires
    no parametric assumption about the distribution of d_i = A_i - B_i.

    Parameters
    ----------
    differences : list of d_i = metric(A_i) - metric(B_i) for matched pairs i
    n_boot      : number of bootstrap replicates (default 10 000)
    ci          : nominal coverage level, e.g. 0.95 for 95% CI
    rng         : optional Random instance for reproducibility

    Returns
    -------
    (ci_low, ci_high) : lower and upper percentile bounds
    """
    if not differences:
        raise ValueError("differences must be non-empty.")
    rng = rng or Random(0)
    n = len(differences)
    boot_means: list[float] = []
    for _ in range(n_boot):
        resample = [differences[rng.randint(0, n - 1)] for _ in range(n)]
        boot_means.append(sum(resample) / n)
    boot_means.sort()
    lo_idx = int((1.0 - ci) / 2.0 * n_boot)
    hi_idx = min(int((1.0 + ci) / 2.0 * n_boot), n_boot - 1)
    return boot_means[lo_idx], boot_means[hi_idx]


def compare_policy_pair(
    episodes_a: list["EpisodeResult"],
    episodes_b: list["EpisodeResult"],
    *,
    name_a: str,
    name_b: str,
    metric: str = "anc_fixed",
    n_boot: int = 10_000,
    ci: float = 0.95,
    alpha_level: float = 0.05,
    rng: Random | None = None,
) -> PolicyComparisonResult:
    """Paired statistical comparison between two policy episode lists.

    Episodes must be aligned: episodes_a[i] and episodes_b[i] must correspond
    to the same (graph, seed) pair. evaluate_policy_factories_on_graphs
    guarantees this when both policies are evaluated in the same call.

    Statistical procedure
    ---------------------
    1. Compute paired differences d_i = metric(A_i) - metric(B_i).
    2. Wilcoxon signed-rank test (two-sided):
       H0: median(d_i) = 0  vs  H1: median(d_i) ≠ 0
       Chosen over paired t-test because ANC scores are bounded in [0,1] and
       may not satisfy normality; Wilcoxon is distribution-free.
    3. Percentile bootstrap CI for E[d_i] with n_boot resamples.
    """
    if len(episodes_a) != len(episodes_b):
        raise ValueError(
            f"Episode lists must have equal length for paired comparison. "
            f"Got {len(episodes_a)} for '{name_a}' and {len(episodes_b)} for '{name_b}'."
        )
    n = len(episodes_a)
    if n == 0:
        raise ValueError(
            f"Episode lists for '{name_a}' and '{name_b}' must not be empty "
            "for paired comparison."
        )
    values_a = [_extract_metric(r, metric) for r in episodes_a]
    values_b = [_extract_metric(r, metric) for r in episodes_b]
    differences = [a - b for a, b in zip(values_a, values_b)]
    mean_diff = sum(differences) / n

    ci_low, ci_high = bootstrap_mean_ci(differences, n_boot=n_boot, ci=ci, rng=rng)

    try:
        wstat, wpval = _scipy_stats.wilcoxon(differences, alternative="two-sided")
    except ValueError:
        # All differences are zero: the test is degenerate (p=1.0 by convention).
        wstat, wpval = 0.0, 1.0

    return PolicyComparisonResult(
        policy_a=name_a,
        policy_b=name_b,
        metric=metric,
        n_pairs=n,
        mean_difference=mean_diff,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        ci_level=ci,
        wilcoxon_statistic=float(wstat),
        wilcoxon_p_value=float(wpval),
        alpha_level=alpha_level,
        significant=float(wpval) < alpha_level,
    )


def compare_all_pairs(
    episodes_by_policy: Mapping[str, list["EpisodeResult"]],
    *,
    baseline: str,
    metric: str = "anc_fixed",
    n_boot: int = 10_000,
    ci: float = 0.95,
    alpha_level: float = 0.05,
    rng: Random | None = None,
) -> list[PolicyComparisonResult]:
    """Compare every policy against a designated baseline policy.

    Parameters
    ----------
    episodes_by_policy : dict mapping policy name → list of EpisodeResult,
        where lists are aligned (same index = same graph/seed pair).
    baseline           : name of the reference policy (e.g. 'degree').
        Each non-baseline policy is compared as (policy, baseline).
    metric             : episode-level scalar to compare (default 'anc_fixed').
    n_boot             : bootstrap replicates (default 10 000).
    ci                 : CI coverage, e.g. 0.95.
    alpha_level        : significance threshold for Wilcoxon test.
    rng                : optional Random for reproducibility.

    Returns
    -------
    List of PolicyComparisonResult, one per non-baseline policy, sorted by
    mean_difference descending (best improvement over baseline first).
    """
    if baseline not in episodes_by_policy:
        raise ValueError(
            f"Baseline policy '{baseline}' not found in episodes_by_policy. "
            f"Available policies: {list(episodes_by_policy.keys())}"
        )
    baseline_episodes = episodes_by_policy[baseline]
    results = []
    for name, episodes in episodes_by_policy.items():
        if name == baseline:
            continue
        results.append(
            compare_policy_pair(
                episodes,
                baseline_episodes,
                name_a=name,
                name_b=baseline,
                metric=metric,
                n_boot=n_boot,
                ci=ci,
                alpha_level=alpha_level,
                rng=rng,
            )
        )
    results.sort(key=lambda r: r.mean_difference, reverse=True)
    return results


def collect_matched_episodes(
    graphs: Sequence[nx.Graph],
    policy_factories: Mapping[str, "PolicyFactory"],
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
    collect_step_metrics: bool = False,
    progress_tick: Callable[[], None] | None = None,
) -> dict[str, list["EpisodeResult"]]:
    """Evaluate policy factories across fixed graphs and return per-episode results.

    Returns a dict mapping policy name → list[EpisodeResult] where all lists
    have equal length and are index-aligned: index i corresponds to the same
    (graph_index, seed) pair for every policy. This alignment is required for
    the paired statistical tests in compare_policy_pair and compare_all_pairs.

    Unlike evaluate_policy_factories_on_graphs, this function does not
    aggregate — it preserves the full episode-level resolution needed for
    paired Wilcoxon and bootstrap CI computations.
    """
    from cascading_rl.budgeting import compute_scaled_budget, compute_scaled_max_rounds

    seeds_list = list(seeds)
    episode_results_by_policy: dict[str, list[EpisodeResult]] = {
        name: [] for name in policy_factories
    }
    env_kw = dict(env_kwargs or {})

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
                    **env_kw,
                )
                policy = policy_factory(graph_index, seed)
                result = rollout_policy(env, policy, seed=seed, collect_step_metrics=collect_step_metrics)
                episode_results_by_policy[policy_name].append(result)
                if progress_tick is not None:
                    progress_tick()

    return episode_results_by_policy


@dataclass(frozen=True)
class PolicyEvaluationSummary:
    final_nc: AggregateMetric
    total_reward: AggregateMetric
    steps: AggregateMetric
    rounds: AggregateMetric
    solved_fraction: AggregateMetric
    rounds_when_solved: AggregateMetric | None
    rounds_when_failed: AggregateMetric | None
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

    valid = list(observation.valid_actions)
    if len(chosen_nodes) == 1:
        # Rank each valid singleton action.
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
    else:
        batch_size = len(chosen_nodes)
        chosen_key = tuple(str(node) for node in sorted(chosen_nodes, key=str))
        batch_deltas: list[tuple[float, tuple[object, ...], tuple[str, ...]]] = []
        for candidate_batch in itertools.combinations(valid, batch_size):
            normalized_batch = tuple(sorted(candidate_batch, key=str))
            delta = delta_nc_after_round_batch(base_state, normalized_batch)
            batch_deltas.append(
                (
                    delta,
                    normalized_batch,
                    tuple(str(node) for node in normalized_batch),
                )
            )
        batch_deltas.sort(key=lambda x: (-x[0], x[2]))

        greedy_nc_gain = batch_deltas[0][0] if batch_deltas else nc_gain
        action_rank = 1
        for rank, (_, _batch, batch_key) in enumerate(batch_deltas, start=1):
            if batch_key == chosen_key:
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
    failed_rounds = [
        float(result.rounds)
        for result in episode_results
        if result.remaining_failed_nodes > 0
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
        rounds_when_failed=_aggregate(failed_rounds) if failed_rounds else None,
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
    *,
    collect_step_metrics: bool = False,
) -> EpisodeResult:
    """Run one episode under a policy and collect core comparison metrics.

    ``collect_step_metrics=False`` (the default) skips the per-step greedy
    lookahead that computes ``StepMetrics``.  That lookahead re-evaluates every
    valid action at every step (O(|failed|) cascade simulations per step) and
    dominates runtime during training-time validation.  Pass
    ``collect_step_metrics=True`` only for offline diagnostic analysis where
    ``mean_greedy_nc_gain`` / ``mean_action_rank`` are actually needed.
    """
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
        if collect_step_metrics:
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
    max_rounds_for_anc = env.max_rounds if env.max_rounds is not None else (rounds or len(nc_by_round) or 1)

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


def build_policy_factories(
    base_seed: int = 0,
    *,
    sequential_greedy: bool = False,
) -> dict[str, PolicyFactory]:
    """Create baseline policy factories for matched-seed sweeps.

    ``sequential_greedy`` is accepted for CLI compatibility with evaluation scripts;
    the current greedy implementation does not branch on this flag.
    """

    _ = sequential_greedy

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


def fmt_policy_summary(summary: "PolicyEvaluationSummary") -> dict:
    """Serialize a PolicyEvaluationSummary to a flat JSON-compatible dict.

    Conditional fields (rounds_when_solved, mean_nc_on_failed) are included as
    None when no episodes matched the condition rather than being omitted, so
    downstream consumers can always find the key.
    """
    return {
        # Primary ANC metrics
        "anc_fixed_mean": round(summary.anc_fixed.mean, 4),
        "anc_fixed_stderr": round(summary.anc_fixed.stderr, 4),
        "anc_adaptive_mean": round(summary.anc_adaptive.mean, 4),
        "anc_adaptive_stderr": round(summary.anc_adaptive.stderr, 4),
        # Final connectivity
        "final_nc_mean": round(summary.final_nc.mean, 4),
        "final_nc_stderr": round(summary.final_nc.stderr, 4),
        # Recovery outcomes
        "solved_fraction_mean": round(summary.solved_fraction.mean, 4),
        "fully_restored_count": summary.fully_restored_count,
        "episode_count": summary.episode_count,
        "unsolved_low_final_nc_count": summary.unsolved_low_final_nc_count,
        "unsolved_low_final_nc_fraction": round(summary.unsolved_low_final_nc_fraction, 4),
        # Timing
        "rounds_mean": round(summary.rounds.mean, 2),
        "rounds_stderr": round(summary.rounds.stderr, 2),
        "steps_mean": round(summary.steps.mean, 2),
        "steps_stderr": round(summary.steps.stderr, 2),
        # Conditional: only populated when at least one episode was solved
        "rounds_when_solved_mean": (
            round(summary.rounds_when_solved.mean, 2)
            if summary.rounds_when_solved is not None else None
        ),
        "rounds_when_solved_stderr": (
            round(summary.rounds_when_solved.stderr, 2)
            if summary.rounds_when_solved is not None else None
        ),
        # Conditional: only populated when at least one episode failed
        "mean_nc_on_failed_mean": (
            round(summary.mean_nc_on_failed.mean, 4)
            if summary.mean_nc_on_failed is not None else None
        ),
        "mean_nc_on_failed_stderr": (
            round(summary.mean_nc_on_failed.stderr, 4)
            if summary.mean_nc_on_failed is not None else None
        ),
        # Reward
        "total_reward_mean": round(summary.total_reward.mean, 4),
        "total_reward_stderr": round(summary.total_reward.stderr, 2),
        # Per-round NC trajectory (mean across episodes at each round index)
        "nc_by_round": [round(m.mean, 4) for m in summary.nc_by_round],
        "mean_delta_nc_per_round_mean": round(summary.mean_delta_nc_per_round.mean, 4),
        "mean_delta_nc_per_round_stderr": round(summary.mean_delta_nc_per_round.stderr, 4),
        # Action-quality diagnostics
        "mean_degree_ratio_mean": round(summary.mean_degree_ratio.mean, 4),
        "mean_degree_ratio_stderr": round(summary.mean_degree_ratio.stderr, 4),
        "mean_overload_risk_mean": round(summary.mean_overload_risk.mean, 4),
        "mean_overload_risk_stderr": round(summary.mean_overload_risk.stderr, 4),
        "mean_nc_gain_mean": round(summary.mean_nc_gain.mean, 4),
        "mean_nc_gain_stderr": round(summary.mean_nc_gain.stderr, 4),
        "mean_greedy_nc_gain_mean": round(summary.mean_greedy_nc_gain.mean, 4),
        "mean_greedy_nc_gain_stderr": round(summary.mean_greedy_nc_gain.stderr, 4),
        "mean_action_rank_mean": round(summary.mean_action_rank.mean, 4),
        "mean_action_rank_stderr": round(summary.mean_action_rank.stderr, 4),
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
