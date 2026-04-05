from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
        max_rounds=20,
        reference_n=40,
        master_seed=2026,
        delta_h=0.30,
        delta_t=0.80,
        delta_s=0.15,
        min_ds_frac=0.50,
        sens_delta_h=(0.20, 0.25, 0.30, 0.35),
        sens_delta_t=(0.70, 0.75, 0.80, 0.85),
        sens_delta_s=(0.05, 0.10, 0.15, 0.20, 0.25),
        sens_min_ds=(0.30, 0.40, 0.50, 0.60),
        output_dir="experiments/regime_comprehensive_test",
    )


def read_checkpoint(output_dir: Path) -> pd.DataFrame:
    parquet_path = output_dir / "checkpoint.parquet"
    csv_path = output_dir / "checkpoint.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.read_csv(csv_path)


def test_constants_defined():
    assert isinstance(comprehensive.ALPHA_VALUES, list)
    assert isinstance(comprehensive.PFAIL_VALUES, list)
    assert isinstance(comprehensive.BUDGET_VALUES, list)
    assert isinstance(comprehensive.N_GRAPHS, int)
    assert isinstance(comprehensive.N_SEEDS, int)
    assert comprehensive.N_SEEDS == 5
    assert comprehensive.MAX_ROUNDS == 20
    assert len(comprehensive.POLICY_NAMES) == 5
    assert len(comprehensive.ALPHA_VALUES) == 9
    assert len(comprehensive.PFAIL_VALUES) == 7
    assert len(comprehensive.BUDGET_VALUES) == 6
    total_cells = (
        len(comprehensive.ALPHA_VALUES)
        * len(comprehensive.PFAIL_VALUES)
        * len(comprehensive.BUDGET_VALUES)
    )
    assert total_cells == 378


def test_graph_generation_invariant():
    config = comprehensive.default_config()
    graphs_a, graph_frame_a = comprehensive.build_graph_bank(config)
    graphs_b, graph_frame_b = comprehensive.build_graph_bank(config)

    assert len(graphs_a) == config.n_graphs
    assert graph_frame_a["n"].between(*config.graph_n_range).all()
    assert nx.is_isomorphic(graphs_a[5], graphs_b[5])
    assert list(graphs_a[5].edges()) == list(graphs_b[5].edges())
    assert graph_frame_a.iloc[5].to_dict() == graph_frame_b.iloc[5].to_dict()


def test_same_seed_invariant():
    config = MappingConfigOverride(
        tiny_config(),
        alpha_values=(0.15,),
        pfail_values=(0.10,),
        budget_values=(4,),
    ).value
    graphs, graph_frame = comprehensive.build_graph_bank(config)
    graph = graphs[0]
    graph_meta = graph_frame.iloc[0].to_dict()

    rows = comprehensive.evaluate_instance_rows(
        graph,
        graph_meta,
        alpha=0.15,
        pfail=0.10,
        budget_ref=4,
        seed_index=0,
        config=config,
    )
    policies = {row["policy"] for row in rows}
    labels = {row["instance_label"] for row in rows}
    assert len(rows) == len(comprehensive.POLICY_NAMES)
    assert policies == set(comprehensive.POLICY_NAMES)
    assert len(labels) == 1


@dataclass(frozen=True)
class MappingConfigOverride:
    base: comprehensive.MappingConfig
    alpha_values: tuple[float, ...] | None = None
    pfail_values: tuple[float, ...] | None = None
    budget_values: tuple[int, ...] | None = None

    @property
    def value(self) -> comprehensive.MappingConfig:
        return comprehensive.MappingConfig(
            alpha_values=self.alpha_values or self.base.alpha_values,
            pfail_values=self.pfail_values or self.base.pfail_values,
            budget_values=self.budget_values or self.base.budget_values,
            n_graphs=self.base.n_graphs,
            n_seeds=self.base.n_seeds,
            graph_n_range=self.base.graph_n_range,
            graph_m=self.base.graph_m,
            max_rounds=self.base.max_rounds,
            reference_n=self.base.reference_n,
            master_seed=self.base.master_seed,
            delta_h=self.base.delta_h,
            delta_t=self.base.delta_t,
            delta_s=self.base.delta_s,
            min_ds_frac=self.base.min_ds_frac,
            sens_delta_h=self.base.sens_delta_h,
            sens_delta_t=self.base.sens_delta_t,
            sens_delta_s=self.base.sens_delta_s,
            sens_min_ds=self.base.sens_min_ds,
            output_dir=self.base.output_dir,
        )


def test_pr_metric_used_correctly():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=1, max_rounds=1)
    env.reset(seed=0)
    env.state.active = {0, 1, 2, 3}
    env.state.failed = set()
    env.state.frontier = set()

    assert math.isclose(env.current_anc(), 1.0)


def test_instance_label_from_heuristic_outcomes():
    lbl = comprehensive.instance_label_from_heuristic_outcomes
    pol = comprehensive.POLICY_NAMES
    others = [p for p in pol if p != "random"]

    def make_solved(random_ok: bool, other_ok: list[bool]) -> dict[str, bool]:
        out = dict(zip(others, other_ok, strict=True))
        out["random"] = random_ok
        return out

    assert lbl(make_solved(False, [True, False, False, False])) == "decision_sensitive"
    assert lbl(make_solved(False, [False, False, False, False])) == "all_fail"
    assert lbl(make_solved(True, [False, False, False, False])) == "random_recovers"
    assert lbl(make_solved(True, [True, True, True, True])) == "random_recovers"


def test_checkpoint_resume(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    config = tiny_config()
    output_dir = tmp_path / "regime_resume"

    with pytest.raises(RuntimeError, match="Intentional interruption"):
        comprehensive.run_analysis(config, output_dir=output_dir, fail_after_cells=1)

    n_pol = len(comprehensive.POLICY_NAMES)
    checkpoint_frame = read_checkpoint(output_dir)
    assert len(checkpoint_frame) == 3 * 2 * n_pol

    comprehensive.run_analysis(config, output_dir=output_dir)
    resumed_output = capsys.readouterr().out
    assert "Resuming: 1 of 4 cells already complete" in resumed_output
    final_checkpoint = read_checkpoint(output_dir)
    assert len(final_checkpoint) == 2 * 2 * 1 * 3 * 2 * n_pol


def test_output_files_created(tmp_path: Path):
    config = tiny_config()
    output_dir = tmp_path / "regime_outputs"
    comprehensive.run_analysis(config, output_dir=output_dir)

    expected_paths = [
        output_dir / "regime_cells.json",
        output_dir / "regime_cells.csv",
        output_dir / "regime_instances.csv",
        output_dir / "budget_summary.json",
        output_dir / "training_recommendation.json",
        output_dir / "run_metadata.json",
        output_dir / "graph_variance.json",
    ]
    if comprehensive.PARQUET_AVAILABLE:
        expected_paths.append(output_dir / "regime_instances.parquet")
        expected_paths.append(output_dir / "checkpoint.parquet")
    else:
        expected_paths.append(output_dir / "regime_instances_no_parquet.csv")
        expected_paths.append(output_dir / "checkpoint.csv")

    for path in expected_paths:
        assert path.exists(), f"Missing expected artifact: {path}"

    png_files = list((output_dir / "plots").glob("*.png"))
    assert len(png_files) == len(comprehensive.PNG_FILENAMES)
