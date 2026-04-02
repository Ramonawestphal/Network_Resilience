import math

import networkx as nx
import pytest

from cascading_rl.evaluation import (
    AggregateMetric,
    PolicyEvaluationSummary,
    build_policy_factories,
    serialize_regime_cell,
    summarize_regime_buckets,
)
from cascading_rl.evaluation.regime import (
    RegimeCellResult,
    RegimeDiagnostics,
    build_regime_cells,
    compute_regime_diagnostics,
    evaluate_policy_factories_on_graphs,
    filter_interesting_graphs,
)


def make_summary(
    *,
    final_anc: float,
    threshold_hit: float,
    rounds: float,
    solved_fraction: float | None = None,
) -> PolicyEvaluationSummary:
    solved = final_anc if solved_fraction is None else solved_fraction
    return PolicyEvaluationSummary(
        final_anc=AggregateMetric(mean=final_anc, stderr=0.0),
        total_reward=AggregateMetric(mean=final_anc, stderr=0.0),
        steps=AggregateMetric(mean=rounds, stderr=0.0),
        rounds=AggregateMetric(mean=rounds, stderr=0.0),
        solved_fraction=AggregateMetric(mean=solved, stderr=0.0),
        threshold_hit_fraction=AggregateMetric(mean=threshold_hit, stderr=0.0),
        threshold_step=AggregateMetric(mean=rounds, stderr=0.0),
        threshold_round=AggregateMetric(mean=rounds, stderr=0.0),
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


def test_compute_regime_diagnostics_reports_best_heuristic_and_rl_gap():
    diagnostics = compute_regime_diagnostics(
        {
            "rl": make_summary(final_anc=0.65, threshold_hit=0.60, rounds=3.0),
            "greedy": make_summary(final_anc=0.55, threshold_hit=0.50, rounds=3.0),
            "random": make_summary(final_anc=0.40, threshold_hit=0.20, rounds=4.0),
        }
    )

    assert diagnostics.regime_label == "decision-sensitive"
    assert diagnostics.interesting_for_rl is True
    assert diagnostics.best_policy == "rl"
    assert diagnostics.best_heuristic == "greedy"
    assert diagnostics.best_heuristic_final_anc == 0.55
    assert math.isclose(diagnostics.rl_vs_best_heuristic_gap or 0.0, 0.10)


def test_compute_regime_diagnostics_respects_spread_threshold():
    diagnostics = compute_regime_diagnostics(
        {
            "rl": make_summary(final_anc=0.61, threshold_hit=0.54, rounds=3.0),
            "greedy": make_summary(final_anc=0.59, threshold_hit=0.52, rounds=3.1),
            "random": make_summary(final_anc=0.58, threshold_hit=0.51, rounds=3.2),
        },
        spread_threshold=0.05,
    )

    assert diagnostics.regime_label == "recoverable"
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


def test_serialize_regime_cell_and_bucket_summary_include_rl_gap():
    decision_sensitive = RegimeCellResult(
        alpha=0.2,
        pfail=0.1,
        budget=2,
        diagnostics=RegimeDiagnostics(
            regime_label="decision-sensitive",
            interesting_for_rl=True,
            interestingness_score=0.42,
            final_anc_spread=0.20,
            threshold_hit_spread=0.30,
            rounds_spread=1.0,
            mean_final_anc=0.53,
            mean_threshold_hit=0.43,
            budget_sensitivity=0.11,
            best_policy="rl",
            worst_policy="random",
            best_heuristic="greedy",
            best_heuristic_final_anc=0.55,
            rl_vs_best_heuristic_gap=0.10,
        ),
        policy_summaries={
            "rl": make_summary(final_anc=0.65, threshold_hit=0.60, rounds=3.0),
            "greedy": make_summary(final_anc=0.55, threshold_hit=0.50, rounds=3.0),
            "random": make_summary(final_anc=0.40, threshold_hit=0.20, rounds=4.0),
        },
    )
    trivial = RegimeCellResult(
        alpha=0.4,
        pfail=0.05,
        budget=2,
        diagnostics=RegimeDiagnostics(
            regime_label="trivial",
            interesting_for_rl=False,
            interestingness_score=0.05,
            final_anc_spread=0.01,
            threshold_hit_spread=0.00,
            rounds_spread=0.25,
            mean_final_anc=0.95,
            mean_threshold_hit=1.0,
            budget_sensitivity=0.0,
            best_policy="greedy",
            worst_policy="random",
            best_heuristic="greedy",
            best_heuristic_final_anc=0.95,
            rl_vs_best_heuristic_gap=None,
        ),
        policy_summaries={
            "greedy": make_summary(final_anc=0.95, threshold_hit=1.0, rounds=1.0),
            "random": make_summary(final_anc=0.94, threshold_hit=1.0, rounds=1.2),
        },
    )

    bucket_summary = summarize_regime_buckets([trivial, decision_sensitive])
    serialized = serialize_regime_cell(decision_sensitive)

    assert set(bucket_summary) == {"overall", "decision-sensitive", "trivial"}
    assert bucket_summary["decision-sensitive"]["cell_count"] == 1
    assert bucket_summary["decision-sensitive"]["rl_vs_best_heuristic_gap"]["mean"] > 0.0
    assert serialized["diagnostics"]["best_heuristic"] == "greedy"


def test_filter_interesting_graphs_keeps_only_graphs_above_spread_threshold(monkeypatch: pytest.MonkeyPatch):
    graphs = [nx.path_graph(3), nx.star_graph(3)]

    def summary(value: float) -> PolicyEvaluationSummary:
        metric = AggregateMetric(mean=value, stderr=0.0)
        return PolicyEvaluationSummary(
            final_anc=metric,
            total_reward=metric,
            steps=metric,
            rounds=metric,
            solved_fraction=metric,
            threshold_hit_fraction=metric,
            threshold_step=metric,
            threshold_round=metric,
        )

    def fake_evaluate_policy_factories_on_graphs(
        graphs,
        policy_factories,
        *,
        alpha,
        pfail,
        budget,
        max_rounds=None,
        seeds,
        tau,
    ):
        assert len(graphs) == 1
        graph = graphs[0]
        if graph.number_of_edges() == 2:
            return {"a": summary(0.60), "b": summary(0.62)}
        return {"a": summary(0.40), "b": summary(0.55)}

    monkeypatch.setattr(
        "cascading_rl.evaluation.regime.evaluate_policy_factories_on_graphs",
        fake_evaluate_policy_factories_on_graphs,
    )

    filtered = filter_interesting_graphs(
        graphs,
        {"a": object(), "b": object()},
        alpha=0.2,
        pfail=0.05,
        budget=2,
        max_rounds=5,
        seeds=[0, 1],
        tau=0.8,
        spread_threshold=0.05,
    )

    assert filtered == [graphs[1]]
