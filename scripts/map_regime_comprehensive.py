from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.evaluation import build_policy_factories, build_regime_cells, serialize_regime_cell
from cascading_rl.graph.generation import make_graph_batch
from scripts.map_regime import build_recommendation, write_csv
from scripts.plot_regime import plot_budget_curves, plot_interestingness_heatmaps

POLICY_NAMES = ("random", "degree", "risk", "greedy", "betweenness")


@dataclass(frozen=True)
class MappingConfig:
    alpha_values: tuple[float, ...]
    pfail_values: tuple[float, ...]
    budgets: tuple[int, ...]
    num_graphs: int
    seeds: tuple[int, ...]
    n_range: tuple[int, int]
    m: int
    graph_seed: int
    max_rounds: int
    tau: float
    hopeless_threshold: float
    trivial_threshold: float
    spread_threshold: float
    output_dir: str = "experiments/regime_comprehensive"

    @property
    def total_cells(self) -> int:
        return len(self.alpha_values) * len(self.pfail_values) * len(self.budgets)


def load_config(path: Path) -> MappingConfig:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")

    regime_mapping = data["regime_mapping"]
    graph_cfg = data["graph"]
    evaluation = data["evaluation"]
    return MappingConfig(
        alpha_values=tuple(float(value) for value in regime_mapping["alpha_values"]),
        pfail_values=tuple(float(value) for value in regime_mapping["pfail_values"]),
        budgets=tuple(int(value) for value in regime_mapping["budgets"]),
        num_graphs=int(regime_mapping["num_graphs"]),
        seeds=tuple(int(value) for value in regime_mapping["seeds"]),
        n_range=tuple(graph_cfg["n_range"]),
        m=int(graph_cfg["m"]),
        graph_seed=int(regime_mapping["graph_seed"]),
        max_rounds=int(regime_mapping["max_rounds"]),
        tau=float(evaluation["tau"]),
        hopeless_threshold=float(regime_mapping["hopeless_threshold"]),
        trivial_threshold=float(regime_mapping["trivial_threshold"]),
        spread_threshold=float(regime_mapping["spread_threshold"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable, analysis-focused regime mapping pass."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="YAML config file for the comprehensive regime mapping pass.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def cell_key(alpha: float, pfail: float, budget: int) -> str:
    return f"{alpha:.8f}|{pfail:.8f}|{budget}"


def checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "checkpoint.json"


def load_checkpoint(output_dir: Path) -> list[dict[str, Any]]:
    path = checkpoint_path(output_dir)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        return []
    cells = data.get("cells", [])
    return cells if isinstance(cells, list) else []


def save_checkpoint(output_dir: Path, cells: list[dict[str, Any]], total_cells: int) -> None:
    path = checkpoint_path(output_dir)
    payload = {
        "saved_at": timestamp_utc(),
        "completed_cells": len(cells),
        "total_cells": total_cells,
        "cells": cells,
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def aggregate_budget_summary(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    budgets = sorted({int(cell["budget"]) for cell in cells})
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        budget_cells = [cell for cell in cells if int(cell["budget"]) == budget]
        decision_sensitive = [
            cell for cell in budget_cells if cell["diagnostics"]["regime_label"] == "decision-sensitive"
        ]
        rows.append(
            {
                "budget": budget,
                "cell_count": len(budget_cells),
                "decision_sensitive_fraction": len(decision_sensitive) / len(budget_cells),
                "mean_interestingness_score": sum(
                    cell["diagnostics"]["interestingness_score"] for cell in budget_cells
                )
                / len(budget_cells),
            }
        )
    return rows


def evaluate_all_cells(
    config: MappingConfig,
    *,
    output_dir: Path,
    fail_after_cells: int | None = None,
) -> list[dict[str, Any]]:
    graphs = make_graph_batch(
        num_graphs=config.num_graphs,
        n_range=config.n_range,
        m=config.m,
        seed=config.graph_seed,
    )
    base_factories = build_policy_factories(base_seed=config.graph_seed)
    policy_factories = {
        policy_name: base_factories[policy_name] for policy_name in POLICY_NAMES
    }

    serialized_cells = load_checkpoint(output_dir)
    completed = {
        cell_key(float(cell["alpha"]), float(cell["pfail"]), int(cell["budget"]))
        for cell in serialized_cells
    }
    if completed:
        print(f"Resuming: {len(completed)} of {config.total_cells} cells already complete")

    completed_this_run = 0
    for alpha in config.alpha_values:
        for pfail in config.pfail_values:
            budget_keys = [cell_key(alpha, pfail, budget) for budget in config.budgets]
            if all(key in completed for key in budget_keys):
                continue

            cells_for_pair = build_regime_cells(
                graphs,
                policy_factories,
                alpha_values=[alpha],
                pfail_values=[pfail],
                budgets=list(config.budgets),
                max_rounds=config.max_rounds,
                seeds=config.seeds,
                tau=config.tau,
                hopeless_threshold=config.hopeless_threshold,
                trivial_threshold=config.trivial_threshold,
                spread_threshold=config.spread_threshold,
            )
            for cell in cells_for_pair:
                key = cell_key(cell.alpha, cell.pfail, cell.budget)
                if key in completed:
                    continue

                serialized_cells.append(serialize_regime_cell(cell))
                completed.add(key)
                completed_this_run += 1
                save_checkpoint(output_dir, serialized_cells, config.total_cells)

                if fail_after_cells is not None and completed_this_run >= fail_after_cells:
                    raise RuntimeError("Intentional interruption")

    serialized_cells.sort(key=lambda cell: (cell["alpha"], cell["pfail"], cell["budget"]))
    save_checkpoint(output_dir, serialized_cells, config.total_cells)
    return serialized_cells


def run_analysis(
    config: MappingConfig,
    *,
    output_dir: Path | None = None,
    fail_after_cells: int | None = None,
) -> dict[str, Any]:
    resolved_output_dir = output_dir or (ROOT / config.output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    cells = evaluate_all_cells(
        config,
        output_dir=resolved_output_dir,
        fail_after_cells=fail_after_cells,
    )
    recommendation = build_recommendation(cells)
    budget_summary = aggregate_budget_summary(cells)
    results = {
        "config": asdict(config),
        "policies": list(POLICY_NAMES),
        "cells": cells,
        "recommendation": recommendation,
    }
    run_metadata = {
        "generated_at": timestamp_utc(),
        "output_dir": str(resolved_output_dir),
        "cell_count": len(cells),
        "policy_names": list(POLICY_NAMES),
    }

    with (resolved_output_dir / "regime_cells.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    write_csv(cells, resolved_output_dir / "regime_cells.csv", list(POLICY_NAMES))
    with (resolved_output_dir / "budget_summary.json").open("w", encoding="utf-8") as file:
        json.dump(budget_summary, file, indent=2)
    with (resolved_output_dir / "training_recommendation.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(recommendation, file, indent=2)
    with (resolved_output_dir / "run_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(run_metadata, file, indent=2)

    plot_results = {
        "policies": list(POLICY_NAMES),
        "cells": cells,
        "recommendation": recommendation,
    }
    plots_dir = resolved_output_dir / "plots"
    plot_interestingness_heatmaps(plot_results, plots_dir / "interestingness_heatmap.png")
    plot_budget_curves(plot_results, plots_dir / "budget_curves.png")
    return results


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or (ROOT / config.output_dir)
    results = run_analysis(config, output_dir=output_dir)
    print(f"Saved comprehensive regime analysis under {output_dir}")
    print(f"Completed {len(results['cells'])} cells")


if __name__ == "__main__":
    main()
