from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from random import Random

import networkx as nx

from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.evaluation.benchmarks import (
    PolicyEvaluationSummary,
    rollout_policy,
    summarize_episode_results,
)
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
class RegimeDiagnostics:
    regime_label: str
    interesting_for_rl: bool
    interestingness_score: float
    final_anc_spread: float
    threshold_hit_spread: float
    rounds_spread: float
    mean_final_anc: float
    mean_threshold_hit: float
    budget_sensitivity: float | None
    best_policy: str
    worst_policy: str


@dataclass(frozen=True)
class RegimeCellResult:
    alpha: float
    pfail: float
    budget: int
    diagnostics: RegimeDiagnostics
    policy_summaries: dict[str, PolicyEvaluationSummary]


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
) -> dict[str, PolicyEvaluationSummary]:
    """Evaluate policies across fixed graph instances and matched seeds."""
    episode_results_by_policy: dict[str, list] = {name: [] for name in policy_factories}

    for graph_index, graph in enumerate(graphs):
        for seed in seeds:
            for policy_name, policy_factory in policy_factories.items():
                env = RecoveryEnv(
                    graph,
                    alpha=alpha,
                    pfail=pfail,
                    budget=budget,
                    max_rounds=max_rounds,
                    seed=seed,
                )
                policy = policy_factory(graph_index, seed)
                result = rollout_policy(env, policy, seed=seed, tau=tau)
                episode_results_by_policy[policy_name].append(result)

    return {
        policy_name: summarize_episode_results(episode_results)
        for policy_name, episode_results in episode_results_by_policy.items()
    }


def compute_regime_diagnostics(
    policy_summaries: Mapping[str, PolicyEvaluationSummary],
    *,
    hopeless_threshold: float = 0.25,
    trivial_threshold: float = 0.75,
    spread_threshold: float = 0.05,
    budget_sensitivity: float | None = None,
) -> RegimeDiagnostics:
    """Summarize whether a parameter cell is trivial, hopeless, or interesting."""
    final_anc_by_policy = {
        policy_name: summary.final_anc.mean
        for policy_name, summary in policy_summaries.items()
    }
    threshold_hit_by_policy = {
        policy_name: summary.threshold_hit_fraction.mean
        for policy_name, summary in policy_summaries.items()
    }
    rounds_by_policy = {
        policy_name: summary.rounds.mean
        for policy_name, summary in policy_summaries.items()
    }

    best_policy = max(final_anc_by_policy, key=final_anc_by_policy.get)
    worst_policy = min(final_anc_by_policy, key=final_anc_by_policy.get)

    final_anc_spread = max(final_anc_by_policy.values()) - min(final_anc_by_policy.values())
    threshold_hit_spread = max(threshold_hit_by_policy.values()) - min(
        threshold_hit_by_policy.values()
    )
    rounds_spread = max(rounds_by_policy.values()) - min(rounds_by_policy.values())
    mean_final_anc = sum(final_anc_by_policy.values()) / len(final_anc_by_policy)
    mean_threshold_hit = sum(threshold_hit_by_policy.values()) / len(threshold_hit_by_policy)

    middle_final_anc = max(0.0, 1.0 - 2.0 * abs(mean_final_anc - 0.5))
    middle_threshold_hit = max(0.0, 1.0 - 2.0 * abs(mean_threshold_hit - 0.5))
    interestingness_score = (
        0.35 * final_anc_spread
        + 0.25 * threshold_hit_spread
        + 0.20 * middle_final_anc
        + 0.20 * middle_threshold_hit
    )
    if budget_sensitivity is not None:
        interestingness_score += 0.20 * budget_sensitivity

    best_final_anc = max(final_anc_by_policy.values())
    best_threshold_hit = max(threshold_hit_by_policy.values())
    worst_final_anc = min(final_anc_by_policy.values())
    worst_threshold_hit = min(threshold_hit_by_policy.values())

    if best_final_anc <= hopeless_threshold and best_threshold_hit <= hopeless_threshold:
        regime_label = "hopeless"
    elif worst_final_anc >= trivial_threshold and worst_threshold_hit >= trivial_threshold:
        regime_label = "trivial"
    elif final_anc_spread >= spread_threshold or threshold_hit_spread >= spread_threshold:
        regime_label = "interesting"
    else:
        regime_label = "interesting"

    return RegimeDiagnostics(
        regime_label=regime_label,
        interesting_for_rl=regime_label == "interesting",
        interestingness_score=interestingness_score,
        final_anc_spread=final_anc_spread,
        threshold_hit_spread=threshold_hit_spread,
        rounds_spread=rounds_spread,
        mean_final_anc=mean_final_anc,
        mean_threshold_hit=mean_threshold_hit,
        budget_sensitivity=budget_sensitivity,
        best_policy=best_policy,
        worst_policy=worst_policy,
    )


def build_regime_cells(
    graphs: Sequence[nx.Graph],
    policy_factories: Mapping[str, PolicyFactory],
    *,
    alpha_values: Sequence[float],
    pfail_values: Sequence[float],
    budgets: Sequence[int],
    max_rounds: int | None = None,
    seeds: Iterable[int],
    tau: float,
    hopeless_threshold: float = 0.25,
    trivial_threshold: float = 0.75,
    spread_threshold: float = 0.05,
) -> list[RegimeCellResult]:
    """Evaluate the full parameter grid and attach per-cell diagnostics."""
    cells: list[RegimeCellResult] = []
    grouped_best_anc: dict[tuple[float, float], list[float]] = {}
    grouped_cells: dict[tuple[float, float], list[tuple[int, dict[str, PolicyEvaluationSummary]]]] = {}

    for alpha in alpha_values:
        for pfail in pfail_values:
            for budget in budgets:
                policy_summaries = evaluate_policy_factories_on_graphs(
                    graphs,
                    policy_factories,
                    alpha=alpha,
                    pfail=pfail,
                    budget=budget,
                    max_rounds=max_rounds,
                    seeds=seeds,
                    tau=tau,
                )
                grouped_cells.setdefault((alpha, pfail), []).append((budget, policy_summaries))
                grouped_best_anc.setdefault((alpha, pfail), []).append(
                    max(summary.final_anc.mean for summary in policy_summaries.values())
                )

    for (alpha, pfail), budget_summaries in grouped_cells.items():
        anc_values = grouped_best_anc[(alpha, pfail)]
        budget_sensitivity = max(anc_values) - min(anc_values) if len(anc_values) > 1 else 0.0
        for budget, policy_summaries in budget_summaries:
            diagnostics = compute_regime_diagnostics(
                policy_summaries,
                hopeless_threshold=hopeless_threshold,
                trivial_threshold=trivial_threshold,
                spread_threshold=spread_threshold,
                budget_sensitivity=budget_sensitivity,
            )
            cells.append(
                RegimeCellResult(
                    alpha=alpha,
                    pfail=pfail,
                    budget=budget,
                    diagnostics=diagnostics,
                    policy_summaries=dict(policy_summaries),
                )
            )

    return sorted(cells, key=lambda cell: (cell.alpha, cell.pfail, cell.budget))
