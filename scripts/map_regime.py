from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.evaluation import build_policy_factories, build_regime_cells
from cascading_rl.graph.generation import make_graph_batch
from scripts.plot_regime import plot_budget_curves, plot_interestingness_heatmaps


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def serialize_metric(metric) -> dict | None:
    if metric is None:
        return None
    return {"mean": metric.mean, "stderr": metric.stderr}


def serialize_policy_summary(summary) -> dict:
    return {
        "final_anc": serialize_metric(summary.final_anc),
        "total_reward": serialize_metric(summary.total_reward),
        "steps": serialize_metric(summary.steps),
        "rounds": serialize_metric(summary.rounds),
        "solved_fraction": serialize_metric(summary.solved_fraction),
        "threshold_hit_fraction": serialize_metric(summary.threshold_hit_fraction),
        "threshold_step": serialize_metric(summary.threshold_step),
        "threshold_round": serialize_metric(summary.threshold_round),
    }


def serialize_cell(cell) -> dict:
    return {
        "alpha": cell.alpha,
        "pfail": cell.pfail,
        "budget": cell.budget,
        "diagnostics": {
            "regime_label": cell.diagnostics.regime_label,
            "interesting_for_rl": cell.diagnostics.interesting_for_rl,
            "interestingness_score": cell.diagnostics.interestingness_score,
            "final_anc_spread": cell.diagnostics.final_anc_spread,
            "threshold_hit_spread": cell.diagnostics.threshold_hit_spread,
            "rounds_spread": cell.diagnostics.rounds_spread,
            "mean_final_anc": cell.diagnostics.mean_final_anc,
            "mean_threshold_hit": cell.diagnostics.mean_threshold_hit,
            "budget_sensitivity": cell.diagnostics.budget_sensitivity,
            "best_policy": cell.diagnostics.best_policy,
            "worst_policy": cell.diagnostics.worst_policy,
        },
        "policy_summaries": {
            policy_name: serialize_policy_summary(summary)
            for policy_name, summary in cell.policy_summaries.items()
        },
    }


def write_csv(rows: list[dict], output_path: Path, policies: list[str]) -> None:
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
            csv_row = {
                "alpha": row["alpha"],
                "pfail": row["pfail"],
                "budget": row["budget"],
                "regime_label": row["diagnostics"]["regime_label"],
                "interestingness_score": row["diagnostics"]["interestingness_score"],
                "final_anc_spread": row["diagnostics"]["final_anc_spread"],
                "threshold_hit_spread": row["diagnostics"]["threshold_hit_spread"],
                "rounds_spread": row["diagnostics"]["rounds_spread"],
                "mean_final_anc": row["diagnostics"]["mean_final_anc"],
                "mean_threshold_hit": row["diagnostics"]["mean_threshold_hit"],
                "budget_sensitivity": row["diagnostics"]["budget_sensitivity"],
                "best_policy": row["diagnostics"]["best_policy"],
                "worst_policy": row["diagnostics"]["worst_policy"],
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


def build_recommendation(serialized_cells: list[dict]) -> dict | None:
    spread_cells = [
        cell
        for cell in serialized_cells
        if cell["diagnostics"]["final_anc_spread"] > 0.0
        or cell["diagnostics"]["threshold_hit_spread"] > 0.0
    ]
    interesting_cells = [
        cell
        for cell in spread_cells
        if cell["diagnostics"]["interesting_for_rl"]
    ]
    candidate_cells = interesting_cells or spread_cells or serialized_cells
    if not candidate_cells:
        return None
    best_cell = max(
        candidate_cells,
        key=lambda cell: (
            cell["diagnostics"]["final_anc_spread"],
            cell["diagnostics"]["threshold_hit_spread"],
            cell["diagnostics"]["budget_sensitivity"] or 0.0,
            cell["diagnostics"]["interestingness_score"],
        ),
    )
    return {
        "alpha": best_cell["alpha"],
        "pfail": best_cell["pfail"],
        "budget": best_cell["budget"],
        "regime_label": best_cell["diagnostics"]["regime_label"],
        "interestingness_score": best_cell["diagnostics"]["interestingness_score"],
        "best_policy": best_cell["diagnostics"]["best_policy"],
        "worst_policy": best_cell["diagnostics"]["worst_policy"],
        "final_anc_spread": best_cell["diagnostics"]["final_anc_spread"],
        "threshold_hit_spread": best_cell["diagnostics"]["threshold_hit_spread"],
        "budget_sensitivity": best_cell["diagnostics"]["budget_sensitivity"],
        "limited_spread": (
            best_cell["diagnostics"]["final_anc_spread"] < 0.05
            and best_cell["diagnostics"]["threshold_hit_spread"] < 0.05
        ),
    }


def write_note(
    recommendation: dict | None,
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
            f"- final ANC spread across heuristics: `{recommendation['final_anc_spread']:.3f}`\n"
        )
        file.write(
            f"- threshold-hit spread across heuristics: "
            f"`{recommendation['threshold_hit_spread']:.3f}`\n"
        )
        if recommendation["budget_sensitivity"] is not None:
            file.write(
                f"- budget sensitivity at this `(alpha, pfail)`: "
                f"`{recommendation['budget_sensitivity']:.3f}`\n"
            )
        file.write(
            f"- current best heuristic in this cell: `{recommendation['best_policy']}`\n"
        )
        file.write(
            f"- current weakest heuristic in this cell: `{recommendation['worst_policy']}`\n"
        )
        if recommendation["limited_spread"]:
            file.write(
                "\n## Caveat\n\n"
                "The coarse sweep found only a small spread between heuristic baselines in this "
                "cell. This still marks the strongest policy-sensitive setting observed so far, "
                "but it suggests the regime map should be refined further before assuming there "
                "is large headroom for RL.\n"
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
    selected_policies = regime_config["policies"]
    policy_factories = {
        policy_name: policy_factories[policy_name] for policy_name in selected_policies
    }

    cells = build_regime_cells(
        graphs,
        policy_factories,
        alpha_values=regime_config["alpha_values"],
        pfail_values=regime_config["pfail_values"],
        budgets=regime_config["budgets"],
        seeds=seeds,
        tau=tau,
        hopeless_threshold=float(regime_config["hopeless_threshold"]),
        trivial_threshold=float(regime_config["trivial_threshold"]),
        spread_threshold=float(regime_config["spread_threshold"]),
    )

    serialized_cells = [serialize_cell(cell) for cell in cells]
    recommendation = build_recommendation(serialized_cells)
    results = {
        "config_path": str(args.config),
        "policies": selected_policies,
        "tau": tau,
        "seeds": seeds,
        "num_graphs": len(graphs),
        "cells": serialized_cells,
        "recommendation": recommendation,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "regime_results.json"
    csv_path = output_dir / "regime_results.csv"
    note_path = output_dir / "recommended_regime.md"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    write_csv(serialized_cells, csv_path, selected_policies)
    write_note(recommendation, note_path, tau=tau, num_graphs=len(graphs), seeds=seeds)
    plot_interestingness_heatmaps(results, output_dir / "interestingness_heatmap.png")
    plot_budget_curves(results, output_dir / "budget_curves.png")

    print(f"Saved regime map to {json_path}")
    print(f"Saved summary table to {csv_path}")
    print(f"Saved recommendation note to {note_path}")


if __name__ == "__main__":
    main()
