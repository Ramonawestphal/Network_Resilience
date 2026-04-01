from __future__ import annotations

from argparse import Namespace

from scripts.evaluate_policy import resolve_grid_spec, serialize_legacy_summary


def make_config() -> dict:
    return {
        "graph": {"n_range": [30, 50], "m": 2},
        "evaluation": {"tau": 0.8, "budgets": [1, 2, 3]},
        "training": {
            "seed": 7,
            "benchmark_graphs": 3,
            "benchmark_seeds": [0, 1, 2],
            "regime": {"alpha": 0.2, "pfail": 0.1, "budget": 2, "max_rounds": 5},
            "graph": {"n_range": [30, 50], "m": 2},
        },
        "regime_mapping": {
            "num_graphs": 4,
            "seeds": [3, 4],
            "graph_seed": 123,
            "alpha_values": [0.4, 0.6],
            "pfail_values": [0.05, 0.08],
            "budgets": [1, 3],
            "max_rounds": 6,
            "hopeless_threshold": 0.25,
            "trivial_threshold": 0.75,
            "spread_threshold": 0.05,
        },
    }


def make_args(**overrides) -> Namespace:
    defaults = {
        "grid_source": "training",
        "alpha_values": None,
        "pfail_values": None,
        "budgets": None,
        "num_graphs": None,
        "seeds": None,
        "max_rounds": None,
        "graph_seed": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_resolve_grid_spec_uses_selected_grid_as_primary_for_regime_mapping():
    grid_spec = resolve_grid_spec(make_config(), make_args(grid_source="regime_mapping"))

    assert grid_spec["alpha_values"] == [0.4, 0.6]
    assert grid_spec["pfail_values"] == [0.05, 0.08]
    assert grid_spec["budgets"] == [1, 3]
    assert grid_spec["primary_alpha"] == 0.4
    assert grid_spec["primary_pfail"] == 0.05
    assert grid_spec["primary_budget"] == 1
    assert grid_spec["primary_max_rounds"] == 6


def test_serialize_legacy_summary_matches_previous_evaluation_shape():
    primary_cell = {
        "policy_summaries": {
            "rl": {
                "final_anc": {"mean": 0.8, "stderr": 0.1},
                "threshold_hit_fraction": {"mean": 0.7, "stderr": 0.0},
                "rounds": {"mean": 3.0, "stderr": 0.0},
                "solved_fraction": {"mean": 0.6, "stderr": 0.0},
            },
            "degree": {
                "final_anc": {"mean": 0.9, "stderr": 0.05},
                "threshold_hit_fraction": {"mean": 0.8, "stderr": 0.0},
                "rounds": {"mean": 2.0, "stderr": 0.0},
                "solved_fraction": {"mean": 0.7, "stderr": 0.0},
            },
        }
    }

    legacy = serialize_legacy_summary(primary_cell, {"rl": 3, "degree": 2})

    assert legacy == {
        "rl": {
            "final_anc_mean": 0.8,
            "final_anc_stderr": 0.1,
            "threshold_hit_mean": 0.7,
            "rounds_mean": 3.0,
            "solved_fraction_mean": 0.6,
            "b_star": 3,
        },
        "degree": {
            "final_anc_mean": 0.9,
            "final_anc_stderr": 0.05,
            "threshold_hit_mean": 0.8,
            "rounds_mean": 2.0,
            "solved_fraction_mean": 0.7,
            "b_star": 2,
        },
    }
