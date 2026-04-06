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
            "regime": {"alpha": 0.2, "pfail": 0.1, "budget": 2, "max_rounds": 20},
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


def test_resolve_grid_spec_unknown_grid_source_raises():
    cfg = make_config()
    with pytest.raises(ValueError, match=r"Unknown grid_source"):
        resolve_grid_spec(cfg, make_args(grid_source="not_a_valid_source"))


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
                "rounds": {"mean": 3.0, "stderr": 0.0},
                "solved_fraction": {"mean": 0.6, "stderr": 0.0},
                "fully_restored_count": 6,
                "episode_count": 10,
                "rounds_when_solved": {"mean": 2.5, "stderr": 0.1},
            },
            "degree": {
                "final_anc": {"mean": 0.9, "stderr": 0.05},
                "rounds": {"mean": 2.0, "stderr": 0.0},
                "solved_fraction": {"mean": 0.7, "stderr": 0.0},
                "fully_restored_count": 7,
                "episode_count": 10,
                "rounds_when_solved": {"mean": 2.0, "stderr": 0.0},
            },
        }
    }

    legacy = serialize_legacy_summary(primary_cell, {"rl": 3, "degree": 2})

    assert legacy == {
        "rl": {
            "final_anc_mean": 0.8,
            "final_anc_stderr": 0.1,
            "rounds_mean": 3.0,
            "solved_fraction_mean": 0.6,
            "fully_restored_count": 6,
            "episode_count": 10,
            "rounds_when_solved_mean": 2.5,
            "b_star": 3,
        },
        "degree": {
            "final_anc_mean": 0.9,
            "final_anc_stderr": 0.05,
            "rounds_mean": 2.0,
            "solved_fraction_mean": 0.7,
            "fully_restored_count": 7,
            "episode_count": 10,
            "rounds_when_solved_mean": 2.0,
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
            "max_rounds": 20,
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


def test_run_eval_set_mode_writes_eval_set_log(tmp_path: Path):
    checkpoint_path = save_checkpoint(
        RecoveryQNetwork(),
        TrainingConfig(checkpoint_dir=str(tmp_path), checkpoint_name="stub_log.pt"),
        TrainingState(),
        tmp_path / "stub_log.pt",
        episode=0,
    )
    inst = {
        "graph": nx.path_graph(8),
        "alpha": 0.2,
        "p_fail": 0.4,
        "budget": 2,
        "max_rounds": 4,
        "failure_seed": 3,
        "regime_label": "decision-sensitive",
    }
    pkl = tmp_path / "with_ds.pkl"
    with pkl.open("wb") as file:
        pickle.dump([inst], file, protocol=4)

    log_path = tmp_path / "subdir" / "eval_report.txt"
    args = Namespace(
        eval_set=pkl,
        checkpoint=checkpoint_path,
        policies=["degree"],
        eval_set_log=log_path,
    )
    evaluate_policy.run_eval_set_mode(args, _minimal_eval_set_config())
    assert log_path.is_file()
    text = log_path.read_text(encoding="utf-8")
    assert "=== Saved eval set:" in text
    assert "degree:" in text


def test_run_eval_set_mode_scaled_pickle_requires_b_scaled_on_all_instances(tmp_path: Path):
    checkpoint_path = save_checkpoint(
        RecoveryQNetwork(),
        TrainingConfig(checkpoint_dir=str(tmp_path), checkpoint_name="stub2.pt"),
        TrainingState(),
        tmp_path / "stub2.pt",
        episode=0,
    )
    common = {
        "graph": nx.path_graph(20),
        "alpha": 0.15,
        "p_fail": 0.18,
        "max_rounds": 20,
        "regime_label": "decision-sensitive",
    }
    inst_scaled = {**common, "b_scaled": 3, "failure_seed": 1}
    inst_missing_scaled = {**common, "budget": 3, "failure_seed": 2}
    pkl = tmp_path / "scaled_eval_mixed.pkl"
    with pkl.open("wb") as file:
        pickle.dump([inst_scaled, inst_missing_scaled], file, protocol=4)

    args = Namespace(
        eval_set=pkl,
        checkpoint=checkpoint_path,
        policies=["degree"],
    )
    with pytest.raises(ValueError, match="b_scaled"):
        evaluate_policy.run_eval_set_mode(args, _minimal_eval_set_config())
