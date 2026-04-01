from __future__ import annotations

import argparse
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

from cascading_rl.budgeting import DEFAULT_REFERENCE_N
from cascading_rl.evaluation import (
    build_policy_factories,
    build_regime_cells,
    estimate_minimum_budget,
    serialize_regime_cell,
    summarize_regime_buckets,
)
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import build_greedy_policy, load_q_network
from scripts.reproducibility import write_run_metadata

SUPPORTED_POLICIES = ("rl", "random", "degree", "risk", "greedy", "betweenness")


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
        help="Which config section should define the evaluation grid.",
    )
    parser.add_argument("--alpha-values", type=float, nargs="+", default=None)
    parser.add_argument("--pfail-values", type=float, nargs="+", default=None)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--num-graphs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--policies", type=str, nargs="+", default=None)
    parser.add_argument("--graph-seed", type=int, default=None)
    parser.add_argument("--n-range", type=int, nargs=2, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--scale-budget", action="store_true")
    parser.add_argument("--reference-n", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def build_eval_policy_factories(
    checkpoint_path: Path,
    *,
    base_seed: int,
    selected_policies: list[str],
) -> dict[str, Any]:
    invalid = [policy for policy in selected_policies if policy not in SUPPORTED_POLICIES]
    if invalid:
        raise ValueError(
            f"Unsupported policies: {invalid}. Supported values: {list(SUPPORTED_POLICIES)}"
        )

    model, _ = load_q_network(checkpoint_path)
    rl_policy = build_greedy_policy(model)
    base_factories = build_policy_factories(base_seed=base_seed)
    policy_factories: dict[str, Any] = {}
    for policy_name in selected_policies:
        if policy_name == "rl":
            policy_factories["rl"] = lambda _graph_index, _seed: rl_policy
        else:
            policy_factories[policy_name] = base_factories[policy_name]
    return policy_factories


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
    regime = training["regime"]
    graph_cfg = training["graph"]
    regime_mapping = config["regime_mapping"]
    hard_regime = config.get("hard_regime", {})

    if args.grid_source == "training":
        alpha_values = [float(regime["alpha"])]
        pfail_values = [float(regime["pfail"])]
        budgets = [int(regime["budget"])]
        num_graphs = int(training["benchmark_graphs"])
        seeds = list(training["benchmark_seeds"])
        graph_seed = int(training["seed"]) + 1000
        max_rounds = int(regime["max_rounds"])
        n_range = tuple(graph_cfg["n_range"])
        m = int(graph_cfg["m"])
    elif args.grid_source == "regime_mapping":
        alpha_values = [float(value) for value in regime_mapping["alpha_values"]]
        pfail_values = [float(value) for value in regime_mapping["pfail_values"]]
        budgets = [int(value) for value in regime_mapping["budgets"]]
        num_graphs = int(regime_mapping["num_graphs"])
        seeds = list(regime_mapping["seeds"])
        graph_seed = int(regime_mapping["graph_seed"])
        max_rounds = int(regime_mapping.get("max_rounds", regime["max_rounds"]))
        n_range = tuple(config["graph"]["n_range"])
        m = int(config["graph"]["m"])
    else:
        alpha_values = [float(value) for value in hard_regime.get("alpha_values", [hard_regime["alpha"]])]
        pfail_values = [float(value) for value in hard_regime.get("pfail_values", [hard_regime["pfail"]])]
        budgets = [int(hard_regime["budget"])]
        num_graphs = int(hard_regime.get("num_graphs", training["benchmark_graphs"]))
        seeds = list(hard_regime.get("seeds", training["benchmark_seeds"]))
        graph_seed = int(hard_regime.get("graph_seed", training["seed"] + 2000))
        max_rounds = int(hard_regime["max_rounds"])
        n_range = tuple(hard_regime["n_range"])
        m = int(hard_regime["m"])

    if getattr(args, "alpha_values", None):
        alpha_values = list(args.alpha_values)
    if getattr(args, "pfail_values", None):
        pfail_values = list(args.pfail_values)
    if getattr(args, "budgets", None):
        budgets = list(args.budgets)
    if getattr(args, "num_graphs", None) is not None:
        num_graphs = args.num_graphs
    if getattr(args, "seeds", None):
        seeds = list(args.seeds)
    if getattr(args, "graph_seed", None) is not None:
        graph_seed = args.graph_seed
    if getattr(args, "max_rounds", None) is not None:
        max_rounds = args.max_rounds
    if getattr(args, "n_range", None) is not None:
        n_range = tuple(args.n_range)

    if args.grid_source == "training":
        primary_alpha = float(regime["alpha"])
        primary_pfail = float(regime["pfail"])
        primary_budget = int(regime["budget"])
        primary_max_rounds = int(regime["max_rounds"])
    else:
        primary_alpha = float(alpha_values[0])
        primary_pfail = float(pfail_values[0])
        primary_budget = int(budgets[0])
        primary_max_rounds = int(max_rounds)

    return {
        "alpha_values": alpha_values,
        "pfail_values": pfail_values,
        "budgets": budgets,
        "num_graphs": num_graphs,
        "seeds": seeds,
        "graph_seed": graph_seed,
        "max_rounds": max_rounds,
        "n_range": tuple(n_range),
        "m": int(m),
        "primary_alpha": primary_alpha,
        "primary_pfail": primary_pfail,
        "primary_budget": primary_budget,
        "primary_max_rounds": primary_max_rounds,
    }


def select_primary_cell(
    serialized_cells: list[dict[str, Any]],
    *,
    alpha: float,
    pfail: float,
    budget: int,
) -> dict[str, Any] | None:
    for cell in serialized_cells:
        if cell["alpha"] == alpha and cell["pfail"] == pfail and cell["budget"] == budget:
            return cell
    return serialized_cells[0] if serialized_cells else None


def estimate_primary_budgets(
    *,
    graphs: list[Any],
    policy_factories: dict[str, Any],
    budgets: list[int],
    trials: int,
    alpha: float,
    pfail: float,
    max_rounds: int,
    tau: float,
    env_kwargs: dict[str, object],
    scale_budget: bool,
    reference_n: int,
) -> dict[str, int | None]:
    representative_graph = graphs[0]
    results: dict[str, int | None] = {}
    for policy_name, policy_factory in policy_factories.items():
        policy = policy_factory(0, 0)
        minimum_budget, _ = estimate_minimum_budget(
            representative_graph,
            policy,
            tau=tau,
            budgets=budgets,
            trials=trials,
            alpha=alpha,
            pfail=pfail,
            max_rounds=max_rounds,
            env_kwargs=env_kwargs,
            scale_budget=scale_budget,
            reference_n=reference_n,
        )
        results[policy_name] = minimum_budget
    return results


def serialize_legacy_summary(
    primary_cell: dict[str, Any] | None,
    representative_budgets: dict[str, int | None],
) -> dict[str, dict[str, float | int | None]]:
    if primary_cell is None:
        return {}

    legacy_summary: dict[str, dict[str, float | int | None]] = {}
    for policy_name, summary in primary_cell["policy_summaries"].items():
        legacy_summary[policy_name] = {
            "final_anc_mean": summary["final_anc"]["mean"],
            "final_anc_stderr": summary["final_anc"]["stderr"],
            "threshold_hit_mean": summary["threshold_hit_fraction"]["mean"],
            "rounds_mean": summary["rounds"]["mean"],
            "solved_fraction_mean": summary["solved_fraction"]["mean"],
            "b_star": representative_budgets.get(policy_name),
        }
    return legacy_summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    evaluation = config["evaluation"]
    budget_scaling = config.get("budget_scaling", {})
    env_kwargs = resolve_env_kwargs(config)
    grid_spec = resolve_grid_spec(config, args)
    selected_policies = list(dict.fromkeys(args.policies or list(SUPPORTED_POLICIES)))
    tau = float(args.tau if args.tau is not None else evaluation["tau"])
    scale_budget = bool(args.scale_budget or budget_scaling.get("enabled", False))
    reference_n = int(
        args.reference_n
        if args.reference_n is not None
        else budget_scaling.get("reference_n", DEFAULT_REFERENCE_N)
    )

    policy_factories = build_eval_policy_factories(
        args.checkpoint,
        base_seed=int(training["seed"]),
        selected_policies=selected_policies,
    )
    graphs = make_graph_batch(
        num_graphs=grid_spec["num_graphs"],
        n_range=grid_spec["n_range"],
        m=grid_spec["m"],
        seed=grid_spec["graph_seed"],
    )

    cells = build_regime_cells(
        graphs,
        policy_factories,
        alpha_values=grid_spec["alpha_values"],
        pfail_values=grid_spec["pfail_values"],
        budgets=grid_spec["budgets"],
        max_rounds=grid_spec["max_rounds"],
        seeds=grid_spec["seeds"],
        tau=tau,
        hopeless_threshold=float(config["regime_mapping"]["hopeless_threshold"]),
        trivial_threshold=float(config["regime_mapping"]["trivial_threshold"]),
        spread_threshold=float(config["regime_mapping"]["spread_threshold"]),
        env_kwargs=env_kwargs,
        scale_budget=scale_budget,
        reference_n=reference_n,
    )

    serialized_cells = [serialize_regime_cell(cell) for cell in cells]
    bucket_summary = summarize_regime_buckets(cells)
    primary_cell = select_primary_cell(
        serialized_cells,
        alpha=grid_spec["primary_alpha"],
        pfail=grid_spec["primary_pfail"],
        budget=grid_spec["primary_budget"],
    )
    representative_budgets = estimate_primary_budgets(
        graphs=graphs,
        policy_factories=policy_factories,
        budgets=[int(value) for value in evaluation["budgets"]],
        trials=len(grid_spec["seeds"]),
        alpha=grid_spec["primary_alpha"],
        pfail=grid_spec["primary_pfail"],
        max_rounds=grid_spec["primary_max_rounds"],
        tau=tau,
        env_kwargs=env_kwargs,
        scale_budget=scale_budget,
        reference_n=reference_n,
    )

    legacy_summary = serialize_legacy_summary(primary_cell, representative_budgets)
    detailed_output = {
        "config_path": str(args.config),
        "checkpoint_path": str(args.checkpoint),
        "grid_source": args.grid_source,
        "policies": selected_policies,
        "tau": tau,
        "graph_seed": grid_spec["graph_seed"],
        "num_graphs": len(graphs),
        "seeds": grid_spec["seeds"],
        "env": env_kwargs,
        "scaling": {
            "scale_budget": scale_budget,
            "reference_n": reference_n,
        },
        "grid": {
            "alpha_values": grid_spec["alpha_values"],
            "pfail_values": grid_spec["pfail_values"],
            "budgets": grid_spec["budgets"],
            "max_rounds": grid_spec["max_rounds"],
            "n_range": list(grid_spec["n_range"]),
        },
        "cells": serialized_cells,
        "bucket_summary": bucket_summary,
        "primary_cell": primary_cell,
        "representative_b_star": representative_budgets,
    }

    output_dir = args.output_dir or ROOT / training["benchmark_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_summary.json"
    grid_output_path = output_dir / "evaluation_grid_summary.json"
    regime_output_path = output_dir / "evaluation_regime_summary.json"
    metadata_path = output_dir / "run_metadata.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(legacy_summary, file, indent=2)
    with grid_output_path.open("w", encoding="utf-8") as file:
        json.dump(detailed_output, file, indent=2)
    with regime_output_path.open("w", encoding="utf-8") as file:
        json.dump(detailed_output, file, indent=2)
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

    print(f"Saved evaluation summary to {output_path}")
    print(f"Saved grid evaluation summary to {grid_output_path}")
    print(f"Saved regime-aware evaluation summary to {regime_output_path}")
    if primary_cell is not None:
        diagnostics = primary_cell["diagnostics"]
        print(
            "Primary cell: "
            f"alpha={primary_cell['alpha']}, pfail={primary_cell['pfail']}, budget={primary_cell['budget']}, "
            f"label={diagnostics['regime_label']}, best={diagnostics['best_policy']}"
        )
        gap = diagnostics["rl_vs_best_heuristic_gap"]
        if gap is not None:
            print(f"RL vs best heuristic gap: {gap:.3f}")
    for policy_name, metrics in legacy_summary.items():
        print(
            f"{policy_name}: final_anc={metrics['final_anc_mean']:.3f}, "
            f"threshold_hit={metrics['threshold_hit_mean']:.3f}, "
            f"rounds={metrics['rounds_mean']:.3f}, b_star={metrics['b_star']}"
        )


if __name__ == "__main__":
    main()
