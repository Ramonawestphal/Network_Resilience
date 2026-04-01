from __future__ import annotations

from pathlib import Path
import sys

import yaml

from scripts import map_regime


def test_map_regime_writes_expected_outputs(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "regime_map"
    config = {
        "graph": {"n_range": [12, 12], "m": 2},
        "budget_scaling": {"enabled": False, "reference_n": 40},
        "evaluation": {"tau": 0.8, "budgets": [1, 2], "trials": 1, "matched_seeds": True},
        "regime_mapping": {
            "output_dir": str(output_dir),
            "graph_seed": 123,
            "num_graphs": 1,
            "seeds": [0],
            "alpha_values": [0.2],
            "pfail_values": [0.05],
            "budgets": [1, 2],
            "policies": ["random", "degree"],
            "hopeless_threshold": 0.25,
            "trivial_threshold": 0.75,
            "spread_threshold": 0.05,
            "max_rounds": 4,
        },
        "training": {
            "regime": {
                "capacity_noise": 0.0,
                "failure_bias": "uniform",
                "action_space": "failed",
                "obs_hops": None,
            }
        },
    }

    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    monkeypatch.setattr(
        sys,
        "argv",
        ["map_regime.py", "--config", str(config_path)],
    )
    map_regime.main()

    expected_paths = [
        output_dir / "regime_results.json",
        output_dir / "regime_results.csv",
        output_dir / "recommended_regime.md",
        output_dir / "interestingness_heatmap.png",
        output_dir / "budget_curves.png",
        output_dir / "run_metadata.json",
    ]
    for path in expected_paths:
        assert path.exists(), f"Missing expected artifact: {path}"
