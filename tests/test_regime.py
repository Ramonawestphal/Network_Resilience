import networkx as nx

from cascading_rl.evaluation import build_policy_factories
from cascading_rl.evaluation.regime import (
    build_regime_cells,
    compute_regime_diagnostics,
    evaluate_policy_factories_on_graphs,
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
