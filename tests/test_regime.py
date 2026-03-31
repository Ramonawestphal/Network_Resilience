import pytest
import networkx as nx

from cascading_rl.evaluation import build_policy_factories
from cascading_rl.evaluation.benchmarks import AggregateMetric, PolicyEvaluationSummary
from cascading_rl.evaluation.regime import (
    build_regime_cells,
    compute_regime_diagnostics,
    evaluate_policy_factories_on_graphs,
    filter_interesting_graphs,
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
