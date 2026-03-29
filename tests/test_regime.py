import networkx as nx
import pytest

from cascading_rl.evaluation import (
    AggregateMetric,
    PolicyEvaluationSummary,
    RegimeCellResult,
    build_policy_factories,
)
from cascading_rl.evaluation.regime import (
    build_regime_cells,
    compute_regime_diagnostics,
    evaluate_policy_factories_on_graphs,
    serialize_regime_cell,
    summarize_regime_buckets,
)


def test_evaluate_policy_factories_on_graphs_returns_all_selected_policies():
    graphs = [nx.star_graph(3)]
    policy_factories = build_policy_factories()

    summaries = evaluate_policy_factories_on_graphs(
        graphs,
        {"degree": policy_factories["degree"], "greedy": policy_factories["greedy"]},
        alpha=0.2,
        pfail=0.0,
        budget=2,
        seeds=[0, 1],
        tau=0.5,
    )

    assert set(summaries) == {"degree", "greedy"}
    assert summaries["degree"].rounds.mean >= 0.0


def test_compute_regime_diagnostics_marks_trivial_when_all_policies_succeed():
    graphs = [nx.path_graph(5)]
    policy_factories = build_policy_factories()
    summaries = evaluate_policy_factories_on_graphs(
        graphs,
        {"degree": policy_factories["degree"], "risk": policy_factories["risk"]},
        alpha=1.0,
        pfail=0.0,
        budget=3,
        seeds=[0, 1],
        tau=0.5,
    )

    diagnostics = compute_regime_diagnostics(summaries)

    assert diagnostics.regime_label == "trivial"
    assert diagnostics.interesting_for_rl is False


def test_build_regime_cells_produces_budget_sensitivity_for_same_alpha_pfail():
    graphs = [nx.star_graph(4)]
    policy_factories = build_policy_factories()

    cells = build_regime_cells(
        graphs,
        {"degree": policy_factories["degree"], "greedy": policy_factories["greedy"]},
        alpha_values=[0.4],
        pfail_values=[0.05],
        budgets=[1, 2],
        seeds=[0, 1],
        tau=0.8,
    )

    assert len(cells) == 2
    assert all(cell.diagnostics.budget_sensitivity is not None for cell in cells)


def _summary(
    *,
    final_anc: float,
    threshold_hit: float,
    rounds: float,
    solved_fraction: float,
) -> PolicyEvaluationSummary:
    metric = lambda value: AggregateMetric(mean=value, stderr=0.0)
    return PolicyEvaluationSummary(
        final_anc=metric(final_anc),
        total_reward=metric(final_anc),
        steps=metric(1.0),
        rounds=metric(rounds),
        solved_fraction=metric(solved_fraction),
        threshold_hit_fraction=metric(threshold_hit),
        threshold_step=metric(1.0),
        threshold_round=metric(1.0),
    )


def test_compute_regime_diagnostics_tracks_best_heuristic_gap():
    summaries = {
        "rl": _summary(final_anc=0.68, threshold_hit=0.70, rounds=2.0, solved_fraction=0.60),
        "greedy": _summary(
            final_anc=0.61, threshold_hit=0.58, rounds=2.5, solved_fraction=0.50
        ),
        "degree": _summary(
            final_anc=0.55, threshold_hit=0.52, rounds=2.8, solved_fraction=0.40
        ),
    }

    diagnostics = compute_regime_diagnostics(summaries)

    assert diagnostics.regime_label == "decision-sensitive"
    assert diagnostics.interesting_for_rl is True
    assert diagnostics.best_heuristic == "greedy"
    assert diagnostics.best_heuristic_final_anc == 0.61
    assert diagnostics.rl_vs_best_heuristic_gap == pytest.approx(0.07)


def test_summarize_regime_buckets_reports_rl_gap():
    trivial = RegimeCellResult(
        alpha=0.1,
        pfail=0.05,
        budget=2,
        diagnostics=compute_regime_diagnostics(
            {
                "rl": _summary(
                    final_anc=0.95, threshold_hit=1.0, rounds=1.0, solved_fraction=1.0
                ),
                "greedy": _summary(
                    final_anc=0.92, threshold_hit=1.0, rounds=1.0, solved_fraction=1.0
                ),
            }
        ),
        policy_summaries={
            "rl": _summary(final_anc=0.95, threshold_hit=1.0, rounds=1.0, solved_fraction=1.0),
            "greedy": _summary(
                final_anc=0.92, threshold_hit=1.0, rounds=1.0, solved_fraction=1.0
            ),
        },
    )
    decision_sensitive = RegimeCellResult(
        alpha=0.2,
        pfail=0.1,
        budget=2,
        diagnostics=compute_regime_diagnostics(
            {
                "rl": _summary(
                    final_anc=0.66, threshold_hit=0.64, rounds=2.0, solved_fraction=0.50
                ),
                "greedy": _summary(
                    final_anc=0.58, threshold_hit=0.55, rounds=2.5, solved_fraction=0.45
                ),
            }
        ),
        policy_summaries={
            "rl": _summary(final_anc=0.66, threshold_hit=0.64, rounds=2.0, solved_fraction=0.50),
            "greedy": _summary(
                final_anc=0.58, threshold_hit=0.55, rounds=2.5, solved_fraction=0.45
            ),
        },
    )

    bucket_summary = summarize_regime_buckets([trivial, decision_sensitive])
    serialized = serialize_regime_cell(decision_sensitive)

    assert set(bucket_summary) == {"overall", "decision-sensitive", "trivial"}
    assert bucket_summary["decision-sensitive"]["cell_count"] == 1
    assert bucket_summary["decision-sensitive"]["rl_vs_best_heuristic_gap"]["mean"] > 0.0
    assert serialized["diagnostics"]["best_heuristic"] == "greedy"
