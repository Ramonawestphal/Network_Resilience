from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.evaluation import (
    build_policy_factories,
    build_regime_cells,
    serialize_regime_cell,
    summarize_regime_buckets,
)
from cascading_rl.graph.generation import make_graph_batch
from scripts.plot_regime import plot_budget_curves, plot_interestingness_heatmaps
from scripts.reproducibility import write_run_metadata


def resolve_env_kwargs(config: dict[str, Any]) -> dict[str, object]:
    regime = config["training"]["regime"]
    obs_hops = regime.get("obs_hops")
    return {
        "capacity_noise": float(regime.get("capacity_noise", 0.0)),
        "failure_bias": str(regime.get("failure_bias", "uniform")),
        "action_space": str(regime.get("action_space", "failed")),
        "obs_hops": int(obs_hops) if obs_hops is not None else None,
    }


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def write_csv(rows: list[dict[str, Any]], output_path: Path, policies: list[str]) -> None:
    fieldnames = [
        "alpha",
        "pfail",
        "budget",
        "regime_label",
        "interestingness_score",
        "final_anc_spread",
        "threshold_hit_spread",
        "rounds_spread",
        "mean_final_anc",
        "mean_threshold_hit",
        "budget_sensitivity",
        "best_policy",
        "worst_policy",
        "best_heuristic",
        "best_heuristic_final_anc",
        "rl_vs_best_heuristic_gap",
    ]
    for policy_name in policies:
        fieldnames.extend(
            [
                f"{policy_name}_final_anc_mean",
                f"{policy_name}_final_anc_stderr",
                f"{policy_name}_threshold_hit_mean",
                f"{policy_name}_rounds_mean",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            diagnostics = row["diagnostics"]
            csv_row = {
                "alpha": row["alpha"],
                "pfail": row["pfail"],
                "budget": row["budget"],
                "regime_label": diagnostics["regime_label"],
                "interestingness_score": diagnostics["interestingness_score"],
                "final_anc_spread": diagnostics["final_anc_spread"],
                "threshold_hit_spread": diagnostics["threshold_hit_spread"],
                "rounds_spread": diagnostics["rounds_spread"],
                "mean_final_anc": diagnostics["mean_final_anc"],
                "mean_threshold_hit": diagnostics["mean_threshold_hit"],
                "budget_sensitivity": diagnostics["budget_sensitivity"],
                "best_policy": diagnostics["best_policy"],
                "worst_policy": diagnostics["worst_policy"],
                "best_heuristic": diagnostics["best_heuristic"],
                "best_heuristic_final_anc": diagnostics["best_heuristic_final_anc"],
                "rl_vs_best_heuristic_gap": diagnostics["rl_vs_best_heuristic_gap"],
            }
            for policy_name in policies:
                policy_summary = row["policy_summaries"][policy_name]
                csv_row[f"{policy_name}_final_anc_mean"] = policy_summary["final_anc"]["mean"]
                csv_row[f"{policy_name}_final_anc_stderr"] = policy_summary["final_anc"]["stderr"]
                csv_row[f"{policy_name}_threshold_hit_mean"] = policy_summary[
                    "threshold_hit_fraction"
                ]["mean"]
                csv_row[f"{policy_name}_rounds_mean"] = policy_summary["rounds"]["mean"]
            writer.writerow(csv_row)


def build_recommendation(serialized_cells: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not serialized_cells:
        return None

    spread_cells = [
        cell
        for cell in serialized_cells
        if cell["diagnostics"]["final_anc_spread"] > 0.0
        or cell["diagnostics"]["threshold_hit_spread"] > 0.0
    ]
    decision_sensitive = [
        cell for cell in spread_cells if cell["diagnostics"]["regime_label"] == "decision-sensitive"
    ]
    candidate_cells = decision_sensitive or spread_cells or serialized_cells
    best_cell = max(
        candidate_cells,
        key=lambda cell: (
            cell["diagnostics"]["interestingness_score"],
            cell["diagnostics"]["final_anc_spread"],
            cell["diagnostics"]["threshold_hit_spread"],
            cell["diagnostics"]["budget_sensitivity"] or 0.0,
        ),
    )
    diagnostics = best_cell["diagnostics"]
    return {
        "alpha": best_cell["alpha"],
        "pfail": best_cell["pfail"],
        "budget": best_cell["budget"],
        "regime_label": diagnostics["regime_label"],
        "interestingness_score": diagnostics["interestingness_score"],
        "best_policy": diagnostics["best_policy"],
        "worst_policy": diagnostics["worst_policy"],
        "best_heuristic": diagnostics["best_heuristic"],
        "best_heuristic_final_anc": diagnostics["best_heuristic_final_anc"],
        "rl_vs_best_heuristic_gap": diagnostics["rl_vs_best_heuristic_gap"],
        "final_anc_spread": diagnostics["final_anc_spread"],
        "threshold_hit_spread": diagnostics["threshold_hit_spread"],
        "budget_sensitivity": diagnostics["budget_sensitivity"],
        "limited_spread": (
            diagnostics["final_anc_spread"] < 0.05
            and diagnostics["threshold_hit_spread"] < 0.05
        ),
    }


def write_note(
    recommendation: dict[str, Any] | None,
    output_path: Path,
    *,
    tau: float,
    num_graphs: int,
    seeds: list[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("# Recommended RL Regime\n\n")
        file.write(f"- `tau`: {tau}\n")
        file.write(f"- fixed graph instances per cell: {num_graphs}\n")
        file.write(f"- matched seeds per graph: {len(seeds)}\n\n")

        if recommendation is None:
            file.write("No candidate cell was available.\n")
            return

        file.write("## Recommendation\n\n")
        file.write(
            f"Start RL training and baseline comparison around `alpha={recommendation['alpha']}`, "
            f"`pfail={recommendation['pfail']}`, and `budget={recommendation['budget']}`.\n\n"
        )
        file.write("## Why This Cell\n\n")
        file.write(f"- regime label: `{recommendation['regime_label']}`\n")
        file.write(
            f"- interestingness score: `{recommendation['interestingness_score']:.3f}`\n"
        )
        file.write(
            f"- final ANC spread across policies: `{recommendation['final_anc_spread']:.3f}`\n"
        )
        file.write(
            f"- threshold-hit spread across policies: "
            f"`{recommendation['threshold_hit_spread']:.3f}`\n"
        )
        if recommendation["budget_sensitivity"] is not None:
            file.write(
                f"- budget sensitivity at this `(alpha, pfail)`: "
                f"`{recommendation['budget_sensitivity']:.3f}`\n"
            )
        if recommendation["best_heuristic"] is not None:
            file.write(
                f"- best heuristic in this cell: `{recommendation['best_heuristic']}`\n"
            )
        if recommendation["rl_vs_best_heuristic_gap"] is not None:
            file.write(
                f"- RL vs best heuristic final-ANC gap: "
                f"`{recommendation['rl_vs_best_heuristic_gap']:.3f}`\n"
            )
        file.write(f"- best overall policy in this cell: `{recommendation['best_policy']}`\n")
        file.write(f"- weakest overall policy in this cell: `{recommendation['worst_policy']}`\n")
        if recommendation["limited_spread"]:
            file.write(
                "\n## Caveat\n\n"
                "The coarse sweep found only a small spread between policies in this cell. "
                "This is still the best candidate found, but it suggests the regime map should "
                "be refined further before assuming there is large headroom for RL.\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map the parameter region where policy matters.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="YAML config file for regime mapping.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    regime_config = config["regime_mapping"]
    graph_config = config["graph"]
    evaluation_config = config["evaluation"]
    budget_scaling = config.get("budget_scaling", {})
    env_kwargs = resolve_env_kwargs(config)
    scale_budget = bool(budget_scaling.get("enabled", False))
    reference_n = int(budget_scaling.get("reference_n", 40))

    output_dir = ROOT / regime_config["output_dir"]
    tau = float(evaluation_config["tau"])
    seeds = list(regime_config["seeds"])
    graphs = make_graph_batch(
        num_graphs=int(regime_config["num_graphs"]),
        n_range=tuple(graph_config["n_range"]),
        m=int(graph_config["m"]),
        seed=int(regime_config["graph_seed"]),
    )

    policy_factories = build_policy_factories(base_seed=int(regime_config["graph_seed"]))
    selected_policies = list(regime_config["policies"])
    policy_factories = {
        policy_name: policy_factories[policy_name] for policy_name in selected_policies
    }

    cells = build_regime_cells(
        graphs,
        policy_factories,
        alpha_values=regime_config["alpha_values"],
        pfail_values=regime_config["pfail_values"],
        budgets=regime_config["budgets"],
        max_rounds=regime_config.get("max_rounds"),
        seeds=seeds,
        tau=tau,
        hopeless_threshold=float(regime_config["hopeless_threshold"]),
        trivial_threshold=float(regime_config["trivial_threshold"]),
        spread_threshold=float(regime_config["spread_threshold"]),
        env_kwargs=env_kwargs,
        scale_budget=scale_budget,
        reference_n=reference_n,
    )

    serialized_cells = [serialize_regime_cell(cell) for cell in cells]
    recommendation = build_recommendation(serialized_cells)
    bucket_summary = summarize_regime_buckets(cells)
    results = {
        "config_path": str(args.config),
        "policies": selected_policies,
        "tau": tau,
        "seeds": seeds,
        "num_graphs": len(graphs),
        "env": env_kwargs,
        "scaling": {
            "scale_budget": scale_budget,
            "reference_n": reference_n,
        },
        "cells": serialized_cells,
        "bucket_summary": bucket_summary,
        "recommendation": recommendation,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "regime_results.json"
    csv_path = output_dir / "regime_results.csv"
    note_path = output_dir / "recommended_regime.md"
    metadata_path = output_dir / "run_metadata.json"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    write_csv(serialized_cells, csv_path, selected_policies)
    write_note(recommendation, note_path, tau=tau, num_graphs=len(graphs), seeds=seeds)
    plot_interestingness_heatmaps(results, output_dir / "interestingness_heatmap.png")
    plot_budget_curves(results, output_dir / "budget_curves.png")
    write_run_metadata(
        metadata_path,
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        config_path=args.config,
        extra={
            "output_dir": str(output_dir),
            "policy_names": selected_policies,
            "env": env_kwargs,
            "scaling": {
                "scale_budget": scale_budget,
                "reference_n": reference_n,
            },
        },
    )

    print(f"Saved regime map to {json_path}")
    print(f"Saved summary table to {csv_path}")
    print(f"Saved recommendation note to {note_path}")


if __name__ == "__main__":
    main()
