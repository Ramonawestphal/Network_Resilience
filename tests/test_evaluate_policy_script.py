from __future__ import annotations

import json
import pickle
import warnings
from argparse import Namespace
from pathlib import Path
import sys

import networkx as nx
import pytest
import yaml

from scripts import evaluate_policy
from scripts.evaluate_policy import resolve_grid_spec, serialize_legacy_summary
from cascading_rl.models import RecoveryQNetwork
from cascading_rl.training import TrainingConfig, TrainingState, save_checkpoint


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


def test_resolve_grid_spec_hard_regime_rejects_non_mapping():
    cfg = make_config()
    cfg["hard_regime"] = "not-a-dict"
    with pytest.raises(ValueError, match=r"hard_regime.*mapping"):
        resolve_grid_spec(cfg, make_args(grid_source="hard_regime"))


def test_resolve_grid_spec_hard_regime_incomplete_falls_back_to_training():
    cfg = make_config()
    cfg["hard_regime"] = {"alpha_values": [0.05]}
    grid_spec = resolve_grid_spec(cfg, make_args(grid_source="hard_regime"))

    assert grid_spec["alpha_values"] == [0.05]
    assert grid_spec["pfail_values"] == [0.1]
    assert grid_spec["budgets"] == [2]
    assert grid_spec["max_rounds"] == 5
    assert grid_spec["n_range"] == (30, 50)
    assert grid_spec["m"] == 2
    assert grid_spec["seeds"] == [0, 1, 2]
    assert grid_spec["num_graphs"] == 3
    assert grid_spec["graph_seed"] == 7 + 2000


def test_resolve_grid_spec_training_primary_follows_cli_overrides():
    grid_spec = resolve_grid_spec(
        make_config(),
        make_args(
            alpha_values=[0.33],
            pfail_values=[0.07],
            budgets=[4],
            max_rounds=9,
        ),
    )
    assert grid_spec["primary_alpha"] == 0.33
    assert grid_spec["primary_pfail"] == 0.07
    assert grid_spec["primary_budget"] == 4
    assert grid_spec["primary_max_rounds"] == 9


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


def test_evaluate_policy_writes_legacy_and_grid_outputs(tmp_path: Path, monkeypatch):
    config = make_config()
    config["training"].update(
        {
            "benchmark_dir": "unused",
            "benchmark_graphs": 1,
            "benchmark_seeds": [0],
        }
    )
    config["training"]["regime"].update(
        {
            "capacity_noise": 0.0,
            "failure_bias": "uniform",
            "action_space": "failed",
            "obs_hops": None,
        }
    )
    config["regime_mapping"].update(
        {
            "alpha_values": [0.2],
            "pfail_values": [0.1],
            "budgets": [2],
            "num_graphs": 1,
            "seeds": [0],
            "max_rounds": 5,
        }
    )

    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    checkpoint_path = save_checkpoint(
        RecoveryQNetwork(),
        TrainingConfig(checkpoint_dir=str(tmp_path), checkpoint_name="stub.pt"),
        TrainingState(),
        tmp_path / "stub.pt",
        episode=0,
    )

    output_dir = tmp_path / "eval_outputs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_policy.py",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--policies",
            "rl",
            "degree",
        ],
    )
    evaluate_policy.main()

    summary_path = output_dir / "evaluation_summary.json"
    grid_path = output_dir / "evaluation_grid_summary.json"
    regime_path = output_dir / "evaluation_regime_summary.json"
    metadata_path = output_dir / "run_metadata.json"

    assert summary_path.exists()
    assert grid_path.exists()
    assert regime_path.exists()
    assert metadata_path.exists()

    with grid_path.open("r", encoding="utf-8") as file:
        grid_payload = json.load(file)
    with regime_path.open("r", encoding="utf-8") as file:
        regime_payload = json.load(file)

    assert grid_payload == regime_payload
    assert grid_payload["policies"] == ["rl", "degree"]


def _minimal_eval_set_config() -> dict:
    return {
        "evaluation": {"tau": 0.8},
        "training": {
            "seed": 1,
            "regime": {
                "capacity_noise": 0.0,
                "failure_bias": "uniform",
                "action_space": "failed",
                "obs_hops": None,
            },
        },
        "regime_mapping": {
            "hopeless_threshold": 0.25,
            "trivial_threshold": 0.75,
            "spread_threshold": 0.05,
        },
    }


def test_run_eval_set_mode_warns_when_no_decision_sensitive_instances(tmp_path: Path):
    checkpoint_path = save_checkpoint(
        RecoveryQNetwork(),
        TrainingConfig(checkpoint_dir=str(tmp_path), checkpoint_name="stub.pt"),
        TrainingState(),
        tmp_path / "stub.pt",
        episode=0,
    )
    inst = {
        "graph": nx.path_graph(8),
        "alpha": 0.2,
        "p_fail": 0.4,
        "budget": 2,
        "max_rounds": 4,
        "failure_seed": 3,
        "regime_label": "trivial",
    }
    pkl = tmp_path / "no_ds.pkl"
    with pkl.open("wb") as file:
        pickle.dump([inst], file, protocol=4)

    args = Namespace(
        eval_set=pkl,
        checkpoint=checkpoint_path,
        policies=["degree"],
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        evaluate_policy.run_eval_set_mode(args, _minimal_eval_set_config())

    assert any(
        "no decision-sensitive" in str(w.message).lower() for w in caught
    ), f"expected UserWarning, got: {[w.message for w in caught]}"


def test_run_eval_set_mode_large_graph_pickle_requires_b_scaled(tmp_path: Path):
    checkpoint_path = save_checkpoint(
        RecoveryQNetwork(),
        TrainingConfig(checkpoint_dir=str(tmp_path), checkpoint_name="stub2.pt"),
        TrainingState(),
        tmp_path / "stub2.pt",
        episode=0,
    )
    inst = {
        "graph": nx.path_graph(20),
        "alpha": 0.15,
        "p_fail": 0.18,
        "budget": 3,
        "max_rounds": 5,
        "failure_seed": 1,
        "regime_label": "decision-sensitive",
    }
    pkl = tmp_path / "large_graph_medium.pkl"
    with pkl.open("wb") as file:
        pickle.dump([inst], file, protocol=4)

    args = Namespace(
        eval_set=pkl,
        checkpoint=checkpoint_path,
        policies=["degree"],
    )
    with pytest.raises(ValueError, match="b_scaled"):
        evaluate_policy.run_eval_set_mode(args, _minimal_eval_set_config())
