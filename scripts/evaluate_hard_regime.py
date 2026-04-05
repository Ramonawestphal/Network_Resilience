from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.evaluation import (
    build_policy_factories,
    build_regime_cells,
    serialize_regime_cell,
    summarize_regime_buckets,
)
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import build_greedy_policy, load_q_network


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def resolve_env_kwargs(config: dict[str, Any]) -> dict[str, object]:
    regime = config["training"]["regime"]
    obs_hops = regime.get("obs_hops")
    abandon_raw = regime.get("abandonment_anc_threshold")
    return {
        "capacity_noise": float(regime.get("capacity_noise", 0.0)),
        "failure_bias": str(regime.get("failure_bias", "uniform")),
        "action_space": str(regime.get("action_space", "failed")),
        "obs_hops": int(obs_hops) if obs_hops is not None else None,
        "abandonment_anc_threshold": (
            float(abandon_raw) if abandon_raw is not None else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate heuristics and RL in hard cascade regimes.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="YAML config (regime_mapping thresholds, hard_regime grid, evaluation targets).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "experiments" / "learner" / "recovery_q.pt",
        help="Optional RL checkpoint. If missing, only heuristics are evaluated.",
    )
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional alpha override for the hard-regime grid.",
    )
    parser.add_argument(
        "--pfail-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional pfail override for the hard-regime grid.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional matched-seed override.",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=None,
        help="Optional graph-count override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    hard = config["hard_regime"]
    evaluation = config["evaluation"]
    threshold_cfg = config["regime_mapping"]
    budget_scaling = config.get("budget_scaling", {})

    target_solved_fraction = float(
        evaluation.get("minimum_budget_solved_target", evaluation.get("tau", 0.8))
    )
    alpha_values = list(args.alpha_values) if args.alpha_values is not None else list(
        hard.get("alpha_values", [hard["alpha"]])
    )
    pfail_values = list(args.pfail_values) if args.pfail_values is not None else list(
        hard.get("pfail_values", [hard["pfail"]])
    )
    seeds = list(args.seeds) if args.seeds is not None else list(
        hard.get("seeds", range(10))
    )
    num_graphs = int(args.num_graphs) if args.num_graphs is not None else int(
        hard.get("num_graphs", 30)
    )
    n_range = tuple(hard["n_range"])
    m = int(hard["m"])
    budget = int(hard["budget"])
    max_rounds = int(hard["max_rounds"])
    graph_seed = int(hard.get("graph_seed", 4242))
    output_dir = ROOT / hard.get("output_dir", "experiments/hard_regime")
    output_dir.mkdir(parents=True, exist_ok=True)

    graphs = make_graph_batch(
        num_graphs=num_graphs,
        n_range=n_range,
        m=m,
        seed=graph_seed,
    )
    env_kwargs = resolve_env_kwargs(config)
    smr_raw = budget_scaling.get("scale_max_rounds")
    scale_max_rounds = bool(smr_raw) if smr_raw is not None else True
    policy_factories = build_policy_factories(base_seed=graph_seed)
    if args.checkpoint.exists():
        device = torch.device("cpu")
        model, _ = load_q_network(args.checkpoint, map_location=device)
        rl_policy = build_greedy_policy(model, device=device, batch_actions=True)
        policy_factories = {"rl": lambda _gi, _se: rl_policy, **policy_factories}

    cells = build_regime_cells(
        graphs,
        policy_factories,
        alpha_values=alpha_values,
        pfail_values=pfail_values,
        budgets=[budget],
        max_rounds=max_rounds,
        seeds=seeds,
        hopeless_threshold=float(threshold_cfg["hopeless_threshold"]),
        trivial_threshold=float(threshold_cfg["trivial_threshold"]),
        spread_threshold=float(threshold_cfg["spread_threshold"]),
        env_kwargs=env_kwargs,
        scale_budget=bool(budget_scaling.get("enabled", True)),
        scale_max_rounds=scale_max_rounds,
        reference_n=int(budget_scaling.get("reference_n", 40)),
    )
    serialized_cells = [serialize_regime_cell(cell) for cell in cells]
    bucket_summary = summarize_regime_buckets(cells)

    for cell in serialized_cells:
        out_path = output_dir / f"results_{cell['alpha']:.2f}_{cell['pfail']:.2f}.json"
        with out_path.open("w", encoding="utf-8") as file:
            json.dump(cell, file, indent=2)

    summary_path = output_dir / "hard_regime_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": str(args.checkpoint) if args.checkpoint.exists() else None,
                "minimum_budget_solved_target": target_solved_fraction,
                "budget": budget,
                "scale_budget": bool(budget_scaling.get("enabled", True)),
                "budget_reference_n": int(budget_scaling.get("reference_n", 40)),
                "scale_max_rounds": scale_max_rounds,
                "env": env_kwargs,
                "max_rounds": max_rounds,
                "num_graphs": num_graphs,
                "seeds": seeds,
                "alpha_values": alpha_values,
                "pfail_values": pfail_values,
                "cells": serialized_cells,
                "bucket_summary": bucket_summary,
            },
            file,
            indent=2,
        )

    all_policies = sorted(
        {
            policy_name
            for cell in serialized_cells
            for policy_name in cell["policy_summaries"]
        }
    )
    print(
        "Hard-regime evaluation — mean final ANC (average over all episodes; episodes may end "
        "early under abandonment_anc_threshold from config)"
    )
    header = "alpha\tpfail\tlabel\twinner\t" + "\t".join(all_policies) + "\tRL_minus_best_heuristic"
    print(header)
    for cell in serialized_cells:
        means = {
            policy_name: cell["policy_summaries"][policy_name]["final_anc"]["mean"]
            for policy_name in cell["policy_summaries"]
        }
        vals = "\t".join(f"{means.get(policy_name, float('nan')):.3f}" for policy_name in all_policies)
        gap = cell["diagnostics"]["rl_vs_best_heuristic_gap"]
        gap_s = f"{gap:.3f}" if gap is not None else "n/a"
        print(
            f"{cell['alpha']}\t{cell['pfail']}\t{cell['diagnostics']['regime_label']}\t"
            f"{cell['diagnostics']['best_policy']}\t{vals}\t{gap_s}"
        )
    print("\nFully restored count / episodes — mean rounds when restored (n/a if none):")
    for cell in serialized_cells:
        parts: list[str] = []
        for policy_name in all_policies:
            ps = cell["policy_summaries"][policy_name]
            rws = ps.get("rounds_when_solved")
            rws_m = rws["mean"] if isinstance(rws, dict) else None
            rws_s = f"{rws_m:.2f}" if rws_m is not None else "n/a"
            parts.append(
                f"{policy_name}={ps['fully_restored_count']}/{ps['episode_count']}@{rws_s}"
            )
        print(
            f"{cell['alpha']}\t{cell['pfail']}\t{cell['diagnostics']['regime_label']}\t"
            + "\t".join(parts)
        )
    print(f"Saved hard-regime results under {output_dir}")
    print(f"Saved hard-regime summary to {summary_path}")


if __name__ == "__main__":
    main()
