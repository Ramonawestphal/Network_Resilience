from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascading_rl.evaluation import build_policy_factories
from cascading_rl.envs.recovery import RecoveryEnv
from scripts import map_regime_comprehensive as comprehensive


def tiny_config() -> comprehensive.MappingConfig:
    return comprehensive.MappingConfig(
        alpha_values=(0.05, 0.10),
        pfail_values=(0.05, 0.10),
        budget_values=(2,),
        n_graphs=3,
        n_seeds=2,
        graph_n_range=(30, 32),
        graph_m=2,
        max_rounds=5,
        reference_n=40,
        master_seed=2026,
        delta_h=0.30,
        delta_t=0.80,
        delta_s=0.15,
        min_ds_frac=0.50,
        delta_h_candidates=(0.20, 0.25, 0.30, 0.35),
        delta_t_candidates=(0.70, 0.75, 0.80, 0.85),
        delta_s_candidates=(0.05, 0.10, 0.15, 0.20),
        min_ds_frac_candidates=(0.30, 0.40, 0.50, 0.60),
        output_dir="experiments/regime_comprehensive_test",
    )


def test_regime_comprehensive_smoke_creates_all_outputs(tmp_path: Path):
    result = comprehensive.run_analysis(tiny_config(), output_dir=tmp_path / "regime")
    output_dir = result["output_dir"]

    expected_files = [
        output_dir / "checkpoint.parquet",
        output_dir / "regime_instances.parquet",
        output_dir / "regime_instances.csv",
        output_dir / "regime_cells.json",
        output_dir / "regime_cells.csv",
        output_dir / "budget_summary.json",
        output_dir / "threshold_sensitivity.json",
        output_dir / "plots" / "spread_histogram_by_alpha.png",
        output_dir / "plots" / "anc_degree_histogram_by_alpha.png",
        output_dir / "plots" / "decision_sensitive_fraction_heatmap.png",
        output_dir / "plots" / "interestingness_heatmap.png",
        output_dir / "plots" / "budget_comparison_barplot.png",
    ]

    for path in expected_files:
        assert path.exists(), f"Missing output artifact: {path}"


def test_same_graph_invariant_holds_in_instances(tmp_path: Path):
    result = comprehensive.run_analysis(tiny_config(), output_dir=tmp_path / "regime")
    instances = result["instances"]

    comprehensive.validate_same_graph_invariant(instances)
    assert bool((instances.groupby("graph_id")["n"].nunique() == 1).all())
    assert bool((instances.groupby("graph_id")["graph_seed"].nunique() == 1).all())


def test_same_seed_invariant_holds_for_degree_and_random():
    config = tiny_config()
    graphs, graph_frame = comprehensive.build_graph_bank(config)
    graph = graphs[0]
    seed = 0
    alpha = config.alpha_values[0]
    pfail = config.pfail_values[0]
    budget_ref = config.budget_values[0]
    scaled_budget = comprehensive.compute_scaled_budget(
        budget_ref,
        num_nodes=graph.number_of_nodes(),
        reference_n=config.reference_n,
        enabled=True,
    )

    env_degree = RecoveryEnv(graph, alpha=alpha, pfail=pfail, budget=scaled_budget, max_rounds=config.max_rounds, seed=seed)
    env_random = RecoveryEnv(graph, alpha=alpha, pfail=pfail, budget=scaled_budget, max_rounds=config.max_rounds, seed=seed)
    degree_observation = env_degree.reset(seed=seed)
    random_observation = env_random.reset(seed=seed)

    assert degree_observation.failed == random_observation.failed
    assert degree_observation.frontier == random_observation.frontier
    assert comprehensive.count_post_cascade_failures(env_degree.state) == comprehensive.count_post_cascade_failures(env_random.state)

    policy_factories = build_policy_factories(base_seed=config.master_seed)
    row = comprehensive.evaluate_instance(
        graph,
        graph_frame.iloc[0].to_dict(),
        alpha=alpha,
        pfail=pfail,
        budget_ref=budget_ref,
        seed=seed,
        config=config,
        policy_factories=policy_factories,
    )
    assert row["n_post_cascade_failures"] >= row["n_initial_failures"]


def test_threshold_sensitivity_matches_main_aggregation(tmp_path: Path):
    config = tiny_config()
    result = comprehensive.run_analysis(config, output_dir=tmp_path / "regime")
    cells = result["cells"]
    sensitivity = result["threshold_sensitivity"]
    default_entry = comprehensive.sensitivity_entry_lookup(
        sensitivity,
        delta_h=config.delta_h,
        delta_t=config.delta_t,
        delta_s=config.delta_s,
        min_ds_frac=config.min_ds_frac,
    )
    label_counts = cells["cell_label"].value_counts()

    assert int(default_entry["n_cells_ds"]) == int(label_counts.get("decision_sensitive", 0))
    assert int(default_entry["n_cells_hopeless"]) == int(label_counts.get("hopeless", 0))
    assert int(default_entry["n_cells_trivial"]) == int(label_counts.get("trivial", 0))
    assert int(default_entry["n_cells_mixed"]) == int(label_counts.get("mixed", 0))
