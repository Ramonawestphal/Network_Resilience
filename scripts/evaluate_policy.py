from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from random import Random
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.evaluation import (
    build_policy_factories,
    build_regime_cells,
    estimate_minimum_budget,
    evaluate_policy_factories_on_graphs,
    serialize_regime_cell,
    summarize_regime_buckets,
)
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import build_greedy_policy, load_q_network
from cascading_rl.policies import choose_random_failed_node


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the trained recovery learner.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "experiments" / "learner" / "recovery_q.pt",
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--grid-source",
        choices=("training", "regime_mapping", "hard_regime"),
        default="training",
        help="Which config section should define the regime grid for robust evaluation.",
    )
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional alpha override for the grid evaluation.",
    )
    parser.add_argument(
        "--pfail-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional pfail override for the grid evaluation.",
    )
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=None,
        help="Optional budget override for the grid evaluation.",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=None,
        help="Optional graph-count override for the grid evaluation.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Optional matched-seed override for the grid evaluation.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Optional max-rounds override for the grid evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for evaluation artifacts (default: training.benchmark_dir from config).",
    )
    return parser.parse_args()


def build_eval_policy_factories(checkpoint_path: Path, *, base_seed: int) -> dict[str, Any]:
    model, _ = load_q_network(checkpoint_path)
    rl_policy = build_greedy_policy(model)
    base_factories = build_policy_factories(base_seed=base_seed)
    return {
        "rl": lambda _graph_index, _seed: rl_policy,
        "random": base_factories["random"],
        "degree": base_factories["degree"],
        "risk": base_factories["risk"],
        "greedy": base_factories["greedy"],
        "betweenness": base_factories["betweenness"],
    }


def resolve_env_kwargs(config: dict[str, Any]) -> dict[str, object]:
    regime = config["training"]["regime"]
    obs_hops = regime.get("obs_hops")
    return {
        "capacity_noise": float(regime.get("capacity_noise", 0.0)),
        "failure_bias": str(regime.get("failure_bias", "uniform")),
        "action_space": str(regime.get("action_space", "failed")),
        "obs_hops": int(obs_hops) if obs_hops is not None else None,
    }


def resolve_grid_spec(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    training = config["training"]
    evaluation = config["evaluation"]
    regime_mapping = config["regime_mapping"]

    if args.grid_source == "training":
        regime = training["regime"]
        graph_cfg = training["graph"]
        alpha_values = regime.get("alpha_values") or [regime["alpha"]]
        pfail_values = regime.get("pfail_values") or [regime["pfail"]]
        budgets = evaluation.get("budgets") or [regime["budget"]]
        num_graphs = int(training["benchmark_graphs"])
        seeds = list(training["benchmark_seeds"])
        max_rounds = int(regime["max_rounds"])
        graph_seed = int(training["seed"]) + 1000
        n_range = tuple(graph_cfg["n_range"])
        m = int(graph_cfg["m"])
    elif args.grid_source == "regime_mapping":
        graph_cfg = config["graph"]
        alpha_values = list(regime_mapping["alpha_values"])
        pfail_values = list(regime_mapping["pfail_values"])
        budgets = list(regime_mapping["budgets"])
        num_graphs = int(regime_mapping["num_graphs"])
        seeds = list(regime_mapping["seeds"])
        max_rounds = int(regime_mapping.get("max_rounds"))
        graph_seed = int(regime_mapping["graph_seed"])
        n_range = tuple(graph_cfg["n_range"])
        m = int(graph_cfg["m"])
    else:
        hard = config["hard_regime"]
        alpha_values = list(hard.get("alpha_values", [hard["alpha"]]))
        pfail_values = list(hard.get("pfail_values", [hard["pfail"]]))
        budgets = [int(hard["budget"])]
        num_graphs = int(hard.get("num_graphs", training["benchmark_graphs"]))
        seeds = list(hard.get("seeds", training["benchmark_seeds"]))
        max_rounds = int(hard["max_rounds"])
        graph_seed = int(hard.get("graph_seed", training["seed"]) + 2000)
        n_range = tuple(hard["n_range"])
        m = int(hard["m"])

    return {
        "alpha_values": list(args.alpha_values) if args.alpha_values is not None else list(alpha_values),
        "pfail_values": list(args.pfail_values) if args.pfail_values is not None else list(pfail_values),
        "budgets": list(args.budgets) if args.budgets is not None else list(budgets),
        "num_graphs": int(args.num_graphs) if args.num_graphs is not None else int(num_graphs),
        "seeds": list(args.seeds) if args.seeds is not None else list(seeds),
        "max_rounds": int(args.max_rounds) if args.max_rounds is not None else int(max_rounds),
        "graph_seed": int(graph_seed),
        "n_range": tuple(n_range),
        "m": int(m),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    regime = training["regime"]
    graph_cfg = training["graph"]
    evaluation = config["evaluation"]
    env_kwargs = resolve_env_kwargs(config)

    policy_factories = build_eval_policy_factories(
        args.checkpoint,
        base_seed=int(training["seed"]),
    )
    rl_policy = policy_factories["rl"](0, 0)

    graphs = make_graph_batch(
        num_graphs=int(training["benchmark_graphs"]),
        n_range=tuple(graph_cfg["n_range"]),
        m=int(graph_cfg["m"]),
        seed=int(training["seed"]) + 1000,
    )

    tau = float(evaluation["tau"])
    summaries = evaluate_policy_factories_on_graphs(
        graphs,
        policy_factories,
        alpha=float(regime["alpha"]),
        pfail=float(regime["pfail"]),
        budget=int(regime["budget"]),
        max_rounds=int(regime["max_rounds"]),
        seeds=training["benchmark_seeds"],
        tau=tau,
        env_kwargs=env_kwargs,
    )

    serialized = {
        policy_name: {
            "final_anc_mean": summary.final_anc.mean,
            "final_anc_stderr": summary.final_anc.stderr,
            "threshold_hit_mean": summary.threshold_hit_fraction.mean,
            "rounds_mean": summary.rounds.mean,
            "solved_fraction_mean": summary.solved_fraction.mean,
        }
        for policy_name, summary in summaries.items()
    }

    representative_graph = graphs[0]
    evaluation_budgets = evaluation["budgets"]
    base_factories = build_policy_factories(base_seed=int(training["seed"]))
    b_star = {
        "rl": estimate_minimum_budget(
            representative_graph,
            rl_policy,
            tau=tau,
            budgets=evaluation_budgets,
            trials=len(training["benchmark_seeds"]),
            alpha=float(regime["alpha"]),
            pfail=float(regime["pfail"]),
            max_rounds=int(regime["max_rounds"]),
            env_kwargs=env_kwargs,
        )[0],
        "degree": estimate_minimum_budget(
            representative_graph,
            base_factories["degree"](0, 0),
            tau=tau,
            budgets=evaluation_budgets,
            trials=len(training["benchmark_seeds"]),
            alpha=float(regime["alpha"]),
            pfail=float(regime["pfail"]),
            max_rounds=int(regime["max_rounds"]),
            env_kwargs=env_kwargs,
        )[0],
        "greedy": estimate_minimum_budget(
            representative_graph,
            base_factories["greedy"](0, 0),
            tau=tau,
            budgets=evaluation_budgets,
            trials=len(training["benchmark_seeds"]),
            alpha=float(regime["alpha"]),
            pfail=float(regime["pfail"]),
            max_rounds=int(regime["max_rounds"]),
            env_kwargs=env_kwargs,
        )[0],
        "risk": estimate_minimum_budget(
            representative_graph,
            base_factories["risk"](0, 0),
            tau=tau,
            budgets=evaluation_budgets,
            trials=len(training["benchmark_seeds"]),
            alpha=float(regime["alpha"]),
            pfail=float(regime["pfail"]),
            max_rounds=int(regime["max_rounds"]),
            env_kwargs=env_kwargs,
        )[0],
        "betweenness": estimate_minimum_budget(
            representative_graph,
            base_factories["betweenness"](0, 0),
            tau=tau,
            budgets=evaluation_budgets,
            trials=len(training["benchmark_seeds"]),
            alpha=float(regime["alpha"]),
            pfail=float(regime["pfail"]),
            max_rounds=int(regime["max_rounds"]),
            env_kwargs=env_kwargs,
        )[0],
        "random": estimate_minimum_budget(
            representative_graph,
            lambda observation: choose_random_failed_node(observation, rng=Random(0)),
            tau=tau,
            budgets=evaluation_budgets,
            trials=len(training["benchmark_seeds"]),
            alpha=float(regime["alpha"]),
            pfail=float(regime["pfail"]),
            max_rounds=int(regime["max_rounds"]),
            env_kwargs=env_kwargs,
        )[0],
    }
    for policy_name, value in b_star.items():
        serialized[policy_name]["b_star"] = value

    grid_spec = resolve_grid_spec(config, args)
    grid_graphs = make_graph_batch(
        num_graphs=grid_spec["num_graphs"],
        n_range=grid_spec["n_range"],
        m=grid_spec["m"],
        seed=grid_spec["graph_seed"],
    )
    threshold_cfg = config["regime_mapping"]
    cells = build_regime_cells(
        grid_graphs,
        policy_factories,
        alpha_values=grid_spec["alpha_values"],
        pfail_values=grid_spec["pfail_values"],
        budgets=grid_spec["budgets"],
        max_rounds=grid_spec["max_rounds"],
        seeds=grid_spec["seeds"],
        tau=tau,
        hopeless_threshold=float(threshold_cfg["hopeless_threshold"]),
        trivial_threshold=float(threshold_cfg["trivial_threshold"]),
        spread_threshold=float(threshold_cfg["spread_threshold"]),
        env_kwargs=env_kwargs,
    )
    grid_results = {
        "checkpoint": str(args.checkpoint),
        "grid_source": args.grid_source,
        "env": env_kwargs,
        "grid_spec": {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in grid_spec.items()
        },
        "tau": tau,
        "cells": [serialize_regime_cell(cell) for cell in cells],
        "bucket_summary": summarize_regime_buckets(cells),
    }

    output_dir = args.output_dir if args.output_dir is not None else ROOT / training["benchmark_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "evaluation_summary.json"
    grid_path = output_dir / "evaluation_grid_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(serialized, file, indent=2)
    with grid_path.open("w", encoding="utf-8") as file:
        json.dump(grid_results, file, indent=2)

    print(f"Saved evaluation summary to {summary_path}")
    print(f"Saved grid evaluation summary to {grid_path}")
    for policy_name, metrics in serialized.items():
        print(
            f"{policy_name}: final_anc={metrics['final_anc_mean']:.3f}, "
            f"threshold_hit={metrics['threshold_hit_mean']:.3f}, "
            f"rounds={metrics['rounds_mean']:.3f}, b_star={metrics['b_star']}"
        )
    for bucket_name, bucket in grid_results["bucket_summary"].items():
        rl_gap = bucket["rl_vs_best_heuristic_gap"]
        gap_text = (
            f"{rl_gap['mean']:.3f}" if isinstance(rl_gap, dict) else "n/a"
        )
        print(
            f"[bucket:{bucket_name}] cells={bucket['cell_count']} "
            f"rl_minus_best_heuristic={gap_text}"
        )


if __name__ == "__main__":
    main()
