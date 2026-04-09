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
from cascading_rl.reproducibility import portable_artifact_path
from scripts.plot_regime import (
    plot_budget_curves,
    plot_interestingness_heatmaps,
    plot_policy_metric_heatmaps,
)
from scripts.reproducibility import write_run_metadata


def resolve_env_kwargs(config: dict[str, Any]) -> dict[str, object]:
    regime = config["training"]["regime"]
    obs_hops = regime.get("obs_hops")
    abandon_raw = regime.get("abandonment_nc_threshold")
    return {
        "capacity_noise": float(regime.get("capacity_noise", 0.0)),
        "failure_bias": str(regime.get("failure_bias", "uniform")),
        "action_space": str(regime.get("action_space", "failed")),
        "obs_hops": int(obs_hops) if obs_hops is not None else None,
        "abandonment_nc_threshold": (
            float(abandon_raw) if abandon_raw is not None else None
        ),
    }


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_csv(rows: list[dict], output_path: Path, policies: list[str]) -> None:
    fieldnames = [
        "alpha",
        "pfail",
        "budget",
        "regime_label",
        "interestingness_score",
        "final_anc_spread",
        "solved_fraction_spread",
        "rounds_spread",
        "mean_final_anc",
        "mean_solved_fraction",
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
                f"{policy_name}_final_nc_mean",
                f"{policy_name}_final_nc_stderr",
                f"{policy_name}_solved_fraction_mean",
                f"{policy_name}_fully_restored_count",
                f"{policy_name}_fully_restored_fraction",
                f"{policy_name}_episode_count",
                f"{policy_name}_unsolved_low_nc_count",
                f"{policy_name}_unsolved_low_nc_fraction",
                f"{policy_name}_final_nc_failure_threshold_used",
                f"{policy_name}_rounds_when_solved_mean",
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
                "solved_fraction_spread": row["diagnostics"]["solved_fraction_spread"],
                "rounds_spread": row["diagnostics"]["rounds_spread"],
                "mean_final_anc": row["diagnostics"]["mean_final_anc"],
                "mean_solved_fraction": row["diagnostics"]["mean_solved_fraction"],
                "budget_sensitivity": row["diagnostics"]["budget_sensitivity"],
                "best_policy": row["diagnostics"]["best_policy"],
                "worst_policy": row["diagnostics"]["worst_policy"],
                "best_heuristic": row["diagnostics"]["best_heuristic"],
                "best_heuristic_final_anc": row["diagnostics"]["best_heuristic_final_anc"],
                "rl_vs_best_heuristic_gap": row["diagnostics"]["rl_vs_best_heuristic_gap"],
            }
            for policy_name in policies:
                policy_summary = row["policy_summaries"][policy_name]
                csv_row[f"{policy_name}_final_nc_mean"] = policy_summary["final_nc"]["mean"]
                csv_row[f"{policy_name}_final_nc_stderr"] = policy_summary["final_nc"]["stderr"]
                csv_row[f"{policy_name}_solved_fraction_mean"] = policy_summary[
                    "solved_fraction"
                ]["mean"]
                csv_row[f"{policy_name}_fully_restored_count"] = policy_summary[
                    "fully_restored_count"
                ]
                csv_row[f"{policy_name}_fully_restored_fraction"] = policy_summary.get(
                    "fully_restored_fraction",
                    (
                        policy_summary["fully_restored_count"] / policy_summary["episode_count"]
                        if policy_summary["episode_count"]
                        else 0.0
                    ),
                )
                csv_row[f"{policy_name}_episode_count"] = policy_summary["episode_count"]
                csv_row[f"{policy_name}_unsolved_low_nc_count"] = policy_summary.get(
                    "unsolved_low_final_nc_count", 0
                )
                csv_row[f"{policy_name}_unsolved_low_nc_fraction"] = policy_summary.get(
                    "unsolved_low_final_nc_fraction", 0.0
                )
                thr_used = policy_summary.get("final_nc_failure_threshold_used")
                csv_row[f"{policy_name}_final_nc_failure_threshold_used"] = (
                    thr_used if thr_used is not None else ""
                )
                rws = policy_summary.get("rounds_when_solved")
                csv_row[f"{policy_name}_rounds_when_solved_mean"] = (
                    rws["mean"] if isinstance(rws, dict) else ""
                )
                csv_row[f"{policy_name}_rounds_mean"] = policy_summary["rounds"]["mean"]
            writer.writerow(csv_row)


def build_recommendation(serialized_cells: list[dict]) -> dict | None:
    spread_cells = [
        cell
        for cell in serialized_cells
        if cell["diagnostics"]["final_anc_spread"] > 0.0
        or cell["diagnostics"]["solved_fraction_spread"] > 0.0
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
            cell["diagnostics"]["solved_fraction_spread"],
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
        "solved_fraction_spread": best_cell["diagnostics"]["solved_fraction_spread"],
        "budget_sensitivity": best_cell["diagnostics"]["budget_sensitivity"],
        "limited_spread": (
            best_cell["diagnostics"]["final_anc_spread"] < 0.05
            and best_cell["diagnostics"]["solved_fraction_spread"] < 0.05
        ),
    }


def write_regime_heuristic_summary(
    serialized_cells: list[dict[str, Any]],
    policies: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thr_note = ""
    if serialized_cells and policies:
        ps0 = serialized_cells[0]["policy_summaries"][policies[0]]
        t_raw = ps0.get("final_nc_failure_threshold_used")
        if t_raw is not None:
            thr = float(t_raw)
            thr_note = (
                f'"Unsolved low-NC" counts episodes with remaining failed nodes and final NC '
                f"strictly below **{thr:g}** (from `abandonment_nc_threshold` when set in config, "
                f"else 0.3).\n\n"
            )

    lines = [
        "# Regime heuristic rollup",
        "",
        "Per-policy means over all grid cells (alpha x p_fail x budget).",
        "",
        thr_note,
        "| policy | mean solved fraction | mean unsolved low-NC fraction |",
        "| --- | ---: | ---: |",
    ]
    n_cells = len(serialized_cells)
    for policy in policies:
        if not n_cells:
            lines.append(f"| {policy} | — | — |")
            continue
        solved_acc = 0.0
        low_acc = 0.0
        for cell in serialized_cells:
            ps = cell["policy_summaries"][policy]
            solved_acc += float(ps["solved_fraction"]["mean"])
            low_acc += float(ps.get("unsolved_low_final_nc_fraction", 0.0))
        lines.append(
            f"| {policy} | {solved_acc / n_cells:.4f} | {low_acc / n_cells:.4f} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_note(
    recommendation: dict | None,
    output_path: Path,
    *,
    minimum_budget_solved_target: float,
    env_kwargs: dict[str, object],
    num_graphs: int,
    seeds: list[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        file.write("# Recommended RL Regime\n\n")
        file.write(
            f"- `minimum_budget_solved_target` (for b\\* search in evaluate scripts): "
            f"{minimum_budget_solved_target}\n"
        )
        file.write(
            f"- env stopping: `abandonment_nc_threshold` = "
            f"{env_kwargs.get('abandonment_nc_threshold')!r}\n"
        )
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
            f"- solved-fraction spread across heuristics: "
            f"`{recommendation['solved_fraction_spread']:.3f}`\n"
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
    budget_scaling = config.get("budget_scaling", {})
    env_kwargs = resolve_env_kwargs(config)

    output_dir = ROOT / regime_config["output_dir"]
    target_solved_fraction = float(
        evaluation_config.get("minimum_budget_solved_target", evaluation_config.get("tau", 0.8))
    )
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
        max_rounds=regime_config.get("max_rounds"),
        seeds=seeds,
        hopeless_threshold=float(regime_config["hopeless_threshold"]),
        trivial_threshold=float(regime_config["trivial_threshold"]),
        spread_threshold=float(regime_config["spread_threshold"]),
        env_kwargs=env_kwargs,
        scale_budget=bool(budget_scaling.get("enabled", True)),
        scale_max_rounds=bool(budget_scaling.get("scale_max_rounds", True)),
        reference_n=int(budget_scaling.get("reference_n", 40)),
    )

    serialized_cells = [serialize_regime_cell(cell) for cell in cells]
    recommendation = build_recommendation(serialized_cells)
    bucket_summary = summarize_regime_buckets(cells)
    results = {
        "config_path": str(args.config),
        "policies": selected_policies,
        "minimum_budget_solved_target": target_solved_fraction,
        "seeds": seeds,
        "num_graphs": len(graphs),
        "env": env_kwargs,
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
    write_note(
        recommendation,
        note_path,
        minimum_budget_solved_target=target_solved_fraction,
        env_kwargs=dict(env_kwargs),
        num_graphs=len(graphs),
        seeds=seeds,
    )
    write_run_metadata(
        metadata_path,
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        config_path=args.config,
        extra={
            "output_dir": portable_artifact_path(output_dir),
            "num_graphs": len(graphs),
            "policies": selected_policies,
            "minimum_budget_solved_target": target_solved_fraction,
            "env": env_kwargs,
        },
    )
    plot_interestingness_heatmaps(results, output_dir / "interestingness_heatmap.png")
    plot_budget_curves(results, output_dir / "budget_curves.png")
    plot_policy_metric_heatmaps(
        results,
        output_dir / "solved_fraction_heatmaps.png",
        suptitle="Mean solved fraction (heuristic rollouts)",
        colorbar_label="solved fraction",
        value_fn=lambda cell, pol: float(
            cell["policy_summaries"][pol]["solved_fraction"]["mean"]
        ),
    )
    plot_policy_metric_heatmaps(
        results,
        output_dir / "unsolved_low_anc_heatmaps.png",
        suptitle="Unsolved low-final-ANC fraction (failed & final ANC < threshold)",
        colorbar_label="fraction of episodes",
        value_fn=lambda cell, pol: float(
            cell["policy_summaries"][pol].get("unsolved_low_final_nc_fraction", 0.0)
        ),
    )
    write_regime_heuristic_summary(
        serialized_cells, selected_policies, output_dir / "regime_heuristic_summary.md"
    )

    print(f"Saved regime map to {json_path}")
    print(f"Saved summary table to {csv_path}")
    print(f"Saved recommendation note to {note_path}")
    print(f"Saved run metadata to {metadata_path}")


if __name__ == "__main__":
    main()
