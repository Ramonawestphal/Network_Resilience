from __future__ import annotations

import json
from pathlib import Path

from scripts import map_regime_comprehensive as comprehensive


def tiny_config() -> comprehensive.MappingConfig:
    return comprehensive.MappingConfig(
        alpha_values=(0.2,),
        pfail_values=(0.05,),
        budgets=(1, 2),
        num_graphs=1,
        seeds=(0,),
        n_range=(12, 12),
        m=2,
        graph_seed=123,
        max_rounds=4,
        tau=0.8,
        hopeless_threshold=0.25,
        trivial_threshold=0.75,
        spread_threshold=0.05,
        output_dir="experiments/regime_comprehensive_test",
    )


def test_run_analysis_writes_expected_outputs(tmp_path: Path):
    output_dir = tmp_path / "regime_outputs"
    results = comprehensive.run_analysis(tiny_config(), output_dir=output_dir)

    assert len(results["cells"]) == 2
    expected_paths = [
        output_dir / "checkpoint.json",
        output_dir / "regime_cells.json",
        output_dir / "regime_cells.csv",
        output_dir / "budget_summary.json",
        output_dir / "training_recommendation.json",
        output_dir / "run_metadata.json",
        output_dir / "plots" / "interestingness_heatmap.png",
        output_dir / "plots" / "budget_curves.png",
    ]
    for path in expected_paths:
        assert path.exists(), f"Missing expected artifact: {path}"


def test_run_analysis_resumes_from_checkpoint(tmp_path: Path, capsys):
    output_dir = tmp_path / "regime_resume"

    try:
        comprehensive.run_analysis(
            tiny_config(),
            output_dir=output_dir,
            fail_after_cells=1,
        )
    except RuntimeError as exc:
        assert str(exc) == "Intentional interruption"
    else:
        raise AssertionError("Expected fail_after_cells to interrupt the analysis")

    with (output_dir / "checkpoint.json").open("r", encoding="utf-8") as file:
        checkpoint = json.load(file)
    assert checkpoint["completed_cells"] == 1

    results = comprehensive.run_analysis(tiny_config(), output_dir=output_dir)
    resumed_output = capsys.readouterr().out
    assert "Resuming: 1 of 2 cells already complete" in resumed_output
    assert len(results["cells"]) == 2
