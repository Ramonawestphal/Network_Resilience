from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from random import Random

import networkx as nx

from cascading_rl.budgeting import DEFAULT_REFERENCE_N, compute_scaled_budget, compute_scaled_max_rounds
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
    solved_fraction_spread: float
    rounds_spread: float
    mean_final_anc: float
    mean_solved_fraction: float
    budget_sensitivity: float | None
    best_policy: str
    worst_policy: str
    best_heuristic: str | None
    best_heuristic_final_anc: float | None
    rl_vs_best_heuristic_gap: float | None


@dataclass(frozen=True)
class RegimeCellResult:
    alpha: float
    pfail: float
    budget: int
    diagnostics: RegimeDiagnostics
    policy_summaries: dict[str, PolicyEvaluationSummary]
    scaling: dict[str, Any] | None = None


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
    """Evaluate policies across fixed graph instances and matched seeds."""
    episode_results_by_policy: dict[str, list] = {name: [] for name in policy_factories}
    resolved_env_kwargs = dict(env_kwargs or {})
    seeds_seq = tuple(seeds)

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
        for seed in seeds_seq:
            for policy_name, policy_factory in policy_factories.items():
                env = RecoveryEnv(
                    graph,
                    alpha=alpha,
                    pfail=pfail,
                    budget=resolved_budget,
                    max_rounds=resolved_max_rounds,
                    seed=seed,
                    **resolved_env_kwargs,
                )
                policy = policy_factory(graph_index, seed)
                result = rollout_policy(env, policy, seed=seed)
                episode_results_by_policy[policy_name].append(result)

    return {
        policy_name: summarize_episode_results(episode_results)
        for policy_name, episode_results in episode_results_by_policy.items()
    }


def filter_interesting_graphs(
    graphs: Sequence[nx.Graph],
    policy_factories: Mapping[str, PolicyFactory],
    *,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int | None = None,
    seeds: Iterable[int],
    spread_threshold: float = 0.05,
    env_kwargs: Mapping[str, object] | None = None,
    scale_budget: bool = False,
    scale_max_rounds: bool = False,
    reference_n: int = DEFAULT_REFERENCE_N,
) -> list[nx.Graph]:
    """Keep only graphs whose per-policy final-ANC spread exceeds the threshold."""
    filtered_graphs: list[nx.Graph] = []
    seeds_seq = tuple(seeds)

    for graph in graphs:
        summaries = evaluate_policy_factories_on_graphs(
            [graph],
            policy_factories,
            alpha=alpha,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            seeds=seeds_seq,
            env_kwargs=env_kwargs,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_n=reference_n,
        )
        final_anc_values = [summary.final_anc.mean for summary in summaries.values()]
        if max(final_anc_values) - min(final_anc_values) > spread_threshold:
            filtered_graphs.append(graph)

    return filtered_graphs


def compute_regime_diagnostics(
    policy_summaries: Mapping[str, PolicyEvaluationSummary],
    *,
    hopeless_threshold: float = 0.25,
    trivial_threshold: float = 0.75,
    spread_threshold: float = 0.05,
    budget_sensitivity: float | None = None,
) -> RegimeDiagnostics:
    """Summarize whether a parameter cell is trivial, hopeless, ambiguous, or DS."""
    final_anc_by_policy = {
        policy_name: summary.final_anc.mean
        for policy_name, summary in policy_summaries.items()
    }
    solved_fraction_by_policy = {
        policy_name: summary.solved_fraction.mean
        for policy_name, summary in policy_summaries.items()
    }
    rounds_by_policy = {
        policy_name: summary.rounds.mean
        for policy_name, summary in policy_summaries.items()
    }

    best_policy = max(final_anc_by_policy, key=final_anc_by_policy.get)
    worst_policy = min(final_anc_by_policy, key=final_anc_by_policy.get)
    heuristic_final_anc = {
        policy_name: value
        for policy_name, value in final_anc_by_policy.items()
        if policy_name != "rl"
    }

    final_anc_spread = max(final_anc_by_policy.values()) - min(final_anc_by_policy.values())
    solved_fraction_spread = max(solved_fraction_by_policy.values()) - min(
        solved_fraction_by_policy.values()
    )
    rounds_spread = max(rounds_by_policy.values()) - min(rounds_by_policy.values())
    mean_final_anc = sum(final_anc_by_policy.values()) / len(final_anc_by_policy)
    mean_solved_fraction = sum(solved_fraction_by_policy.values()) / len(
        solved_fraction_by_policy
    )

    middle_final_anc = max(0.0, 1.0 - 2.0 * abs(mean_final_anc - 0.5))
    middle_solved = max(0.0, 1.0 - 2.0 * abs(mean_solved_fraction - 0.5))
    interestingness_score = (
        0.35 * final_anc_spread
        + 0.25 * solved_fraction_spread
        + 0.20 * middle_final_anc
        + 0.20 * middle_solved
    )
    if budget_sensitivity is not None:
        interestingness_score += 0.20 * budget_sensitivity

    best_final_anc = max(final_anc_by_policy.values())
    worst_final_anc = min(final_anc_by_policy.values())
    best_solved_fraction = max(solved_fraction_by_policy.values())
    worst_solved_fraction = min(solved_fraction_by_policy.values())
    if heuristic_final_anc:
        best_heuristic = max(heuristic_final_anc, key=heuristic_final_anc.get)
        best_heuristic_final_anc = heuristic_final_anc[best_heuristic]
    else:
        best_heuristic = None
        best_heuristic_final_anc = None
    rl_final_anc = final_anc_by_policy.get("rl")
    rl_vs_best_heuristic_gap = (
        rl_final_anc - best_heuristic_final_anc
        if rl_final_anc is not None and best_heuristic_final_anc is not None
        else None
    )

    if best_final_anc <= hopeless_threshold and best_solved_fraction <= hopeless_threshold:
        regime_label = "hopeless"
    elif worst_final_anc >= trivial_threshold and worst_solved_fraction >= trivial_threshold:
        regime_label = "trivial"
    elif (
        final_anc_spread > spread_threshold
        or solved_fraction_spread > spread_threshold
    ):
        regime_label = "decision-sensitive"
    else:
        regime_label = "ambiguous"

    return RegimeDiagnostics(
        regime_label=regime_label,
        interesting_for_rl=regime_label == "decision-sensitive",
        interestingness_score=interestingness_score,
        final_anc_spread=final_anc_spread,
        solved_fraction_spread=solved_fraction_spread,
        rounds_spread=rounds_spread,
        mean_final_anc=mean_final_anc,
        mean_solved_fraction=mean_solved_fraction,
        budget_sensitivity=budget_sensitivity,
        best_policy=best_policy,
        worst_policy=worst_policy,
        best_heuristic=best_heuristic,
        best_heuristic_final_anc=best_heuristic_final_anc,
        rl_vs_best_heuristic_gap=rl_vs_best_heuristic_gap,
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
    hopeless_threshold: float = 0.25,
    trivial_threshold: float = 0.75,
    spread_threshold: float = 0.05,
    env_kwargs: Mapping[str, object] | None = None,
    scale_budget: bool = False,
    scale_max_rounds: bool = False,
    reference_n: int = 40,
) -> list[RegimeCellResult]:
    """Evaluate the full parameter grid and attach per-cell diagnostics."""
    cells: list[RegimeCellResult] = []
    grouped_best_anc: dict[tuple[float, float], list[float]] = {}
    grouped_cells: dict[tuple[float, float], list[tuple[int, dict[str, PolicyEvaluationSummary]]]] = {}
    seeds_seq = tuple(seeds)

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
                    seeds=seeds_seq,
                    env_kwargs=env_kwargs,
                    scale_budget=scale_budget,
                    scale_max_rounds=scale_max_rounds,
                    reference_n=reference_n,
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


def serialize_metric(metric: object | None) -> dict[str, float] | None:
    if metric is None:
        return None
    typed_metric = metric
    return {
        "mean": typed_metric.mean,
        "stderr": typed_metric.stderr,
    }


def serialize_policy_summary(summary: PolicyEvaluationSummary) -> dict[str, object]:
    return {
        "final_anc": serialize_metric(summary.final_anc),
        "total_reward": serialize_metric(summary.total_reward),
        "steps": serialize_metric(summary.steps),
        "rounds": serialize_metric(summary.rounds),
        "solved_fraction": serialize_metric(summary.solved_fraction),
        "rounds_when_solved": serialize_metric(summary.rounds_when_solved),
        "fully_restored_count": summary.fully_restored_count,
        "episode_count": summary.episode_count,
    }


def serialize_regime_cell(cell: RegimeCellResult) -> dict[str, object]:
    diagnostics = cell.diagnostics
    payload: dict[str, object] = {
        "alpha": cell.alpha,
        "pfail": cell.pfail,
        "budget": cell.budget,
        "diagnostics": {
            "regime_label": diagnostics.regime_label,
            "interesting_for_rl": diagnostics.interesting_for_rl,
            "interestingness_score": diagnostics.interestingness_score,
            "final_anc_spread": diagnostics.final_anc_spread,
            "solved_fraction_spread": diagnostics.solved_fraction_spread,
            "rounds_spread": diagnostics.rounds_spread,
            "mean_final_anc": diagnostics.mean_final_anc,
            "mean_solved_fraction": diagnostics.mean_solved_fraction,
            "budget_sensitivity": diagnostics.budget_sensitivity,
            "best_policy": diagnostics.best_policy,
            "worst_policy": diagnostics.worst_policy,
            "best_heuristic": diagnostics.best_heuristic,
            "best_heuristic_final_anc": diagnostics.best_heuristic_final_anc,
            "rl_vs_best_heuristic_gap": diagnostics.rl_vs_best_heuristic_gap,
        },
        "policy_summaries": {
            policy_name: serialize_policy_summary(summary)
            for policy_name, summary in cell.policy_summaries.items()
        },
    }
    if cell.scaling is not None:
        payload["scaling"] = cell.scaling
    return payload


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_regime_buckets(
    cells: Sequence[RegimeCellResult],
) -> dict[str, dict[str, object]]:
    bucket_names = ["overall"] + sorted({cell.diagnostics.regime_label for cell in cells})
    summaries: dict[str, dict[str, object]] = {}

    for bucket_name in bucket_names:
        if bucket_name == "overall":
            bucket_cells = list(cells)
        else:
            bucket_cells = [
                cell for cell in cells if cell.diagnostics.regime_label == bucket_name
            ]
        if not bucket_cells:
            continue

        policy_names = sorted(
            {
                policy_name
                for cell in bucket_cells
                for policy_name in cell.policy_summaries
            }
        )
        winner_counts: dict[str, int] = {}
        for cell in bucket_cells:
            winner_counts[cell.diagnostics.best_policy] = (
                winner_counts.get(cell.diagnostics.best_policy, 0) + 1
            )

        policy_means: dict[str, dict[str, float]] = {}
        for policy_name in policy_names:
            matching_summaries = [
                cell.policy_summaries[policy_name]
                for cell in bucket_cells
                if policy_name in cell.policy_summaries
            ]
            rounds_ws = [
                summary.rounds_when_solved.mean
                for summary in matching_summaries
                if summary.rounds_when_solved is not None
            ]
            policy_means[policy_name] = {
                "final_anc_mean": _mean(
                    [summary.final_anc.mean for summary in matching_summaries]
                ),
                "rounds_mean": _mean(
                    [summary.rounds.mean for summary in matching_summaries]
                ),
                "solved_fraction_mean": _mean(
                    [summary.solved_fraction.mean for summary in matching_summaries]
                ),
                "rounds_when_solved_mean": _mean(rounds_ws) if rounds_ws else float("nan"),
            }

        rl_gaps = [
            gap
            for gap in (
                cell.diagnostics.rl_vs_best_heuristic_gap for cell in bucket_cells
            )
            if gap is not None
        ]
        summaries[bucket_name] = {
            "cell_count": len(bucket_cells),
            "mean_interestingness_score": _mean(
                [cell.diagnostics.interestingness_score for cell in bucket_cells]
            ),
            "mean_budget_sensitivity": _mean(
                [
                    cell.diagnostics.budget_sensitivity or 0.0
                    for cell in bucket_cells
                ]
            ),
            "winner_counts": winner_counts,
            "policy_means": policy_means,
            "rl_vs_best_heuristic_gap": (
                {
                    "mean": _mean(rl_gaps),
                    "min": min(rl_gaps),
                    "max": max(rl_gaps),
                    "positive_fraction": sum(gap > 0.0 for gap in rl_gaps) / len(rl_gaps),
                }
                if rl_gaps
                else None
            ),
        }

    return summaries
