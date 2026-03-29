from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from random import Random
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.envs.recovery import RecoveryEnv
from cascading_rl.evaluation import (
    RegimeCellResult,
    build_policy_factories,
    build_regime_cells,
    compute_regime_diagnostics,
    estimate_minimum_budget,
    evaluate_policy_factories_on_graphs,
    rollout_policy,
    serialize_regime_cell,
    summarize_episode_results,
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
        "--scale-budget",
        action="store_true",
        help="Scale the evaluation budget with graph size using beta = budget / reference_n.",
    )
    parser.add_argument(
        "--scale-max-rounds",
        action="store_true",
        help="Scale max_rounds per graph as max(10, ceil(3 * pfail * n / budget)).",
    )
    parser.add_argument(
        "--reference-n",
        type=int,
        default=40,
        help=(
            "Reference graph size used to compute the recovery rate beta = budget / "
            "reference_n when --scale-budget is active. Default: 40 (midpoint of the "
            "default training range [30, 50])."
        ),
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


def compute_scaled_budget(*, budget: int, reference_n: int, num_nodes: int) -> int:
    beta = budget / reference_n
    return max(1, round(beta * num_nodes))


def compute_scaled_max_rounds(*, pfail: float, num_nodes: int, budget: int) -> int:
    return max(10, math.ceil(3 * pfail * num_nodes / budget))


def build_scaling_metadata(
    *,
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_budget: int,
    reference_n: int,
) -> dict[str, Any] | None:
    if not scale_budget and not scale_max_rounds:
        return None
    return {
        "scale_budget": scale_budget,
        "scale_max_rounds": scale_max_rounds,
        "reference_budget": reference_budget,
        "reference_n": reference_n,
        "beta": (reference_budget / reference_n) if scale_budget else None,
    }


def resolve_budget_and_rounds(
    *,
    num_nodes: int,
    pfail: float,
    budget: int,
    max_rounds: int | None,
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_budget: int,
    reference_n: int,
) -> tuple[int, int | None]:
    resolved_budget = (
        compute_scaled_budget(
            budget=reference_budget,
            reference_n=reference_n,
            num_nodes=num_nodes,
        )
        if scale_budget
        else budget
    )
    resolved_max_rounds = (
        compute_scaled_max_rounds(
            pfail=pfail,
            num_nodes=num_nodes,
            budget=resolved_budget,
        )
        if scale_max_rounds
        else max_rounds
    )
    return resolved_budget, resolved_max_rounds


def log_scaling_decisions(
    *,
    graphs: list[Any],
    pfail: float,
    budget: int,
    max_rounds: int | None,
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_budget: int,
    reference_n: int,
) -> None:
    if not scale_budget and not scale_max_rounds:
        return

    if scale_budget:
        beta = reference_budget / reference_n
        print(
            f"Budget scaling active: beta={beta:.3f} "
            f"(budget={reference_budget}, reference_n={reference_n})"
        )
    else:
        print(f"Max-round scaling active: using fixed budget={budget}")

    for num_nodes in sorted({graph.number_of_nodes() for graph in graphs}):
        scaled_budget, scaled_max_rounds = resolve_budget_and_rounds(
            num_nodes=num_nodes,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_budget=reference_budget,
            reference_n=reference_n,
        )
        print(
            f"For n={num_nodes}: scaled_budget={scaled_budget}, "
            f"scaled_max_rounds={scaled_max_rounds}"
        )


def evaluate_policy_factories_with_optional_scaling(
    graphs: list[Any],
    policy_factories: dict[str, Any],
    *,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int | None,
    seeds: list[int],
    tau: float,
    env_kwargs: dict[str, object],
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_budget: int,
    reference_n: int,
) -> dict[str, Any]:
    if not scale_budget and not scale_max_rounds:
        return evaluate_policy_factories_on_graphs(
            graphs,
            policy_factories,
            alpha=alpha,
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            seeds=seeds,
            tau=tau,
            env_kwargs=env_kwargs,
        )

    episode_results_by_policy: dict[str, list[Any]] = {name: [] for name in policy_factories}
    for graph_index, graph in enumerate(graphs):
        resolved_budget, resolved_max_rounds = resolve_budget_and_rounds(
            num_nodes=graph.number_of_nodes(),
            pfail=pfail,
            budget=budget,
            max_rounds=max_rounds,
            scale_budget=scale_budget,
            scale_max_rounds=scale_max_rounds,
            reference_budget=reference_budget,
            reference_n=reference_n,
        )
        for seed in seeds:
            for policy_name, policy_factory in policy_factories.items():
                env = RecoveryEnv(
                    graph,
                    alpha=alpha,
                    pfail=pfail,
                    budget=resolved_budget,
                    max_rounds=resolved_max_rounds,
                    seed=seed,
                    **env_kwargs,
                )
                policy = policy_factory(graph_index, seed)
                result = rollout_policy(env, policy, seed=seed, tau=tau)
                episode_results_by_policy[policy_name].append(result)

    return {
        policy_name: summarize_episode_results(episode_results)
        for policy_name, episode_results in episode_results_by_policy.items()
    }


def build_regime_cells_with_optional_scaling(
    graphs: list[Any],
    policy_factories: dict[str, Any],
    *,
    alpha_values: list[float],
    pfail_values: list[float],
    budgets: list[int],
    max_rounds: int | None,
    seeds: list[int],
    tau: float,
    hopeless_threshold: float,
    trivial_threshold: float,
    spread_threshold: float,
    env_kwargs: dict[str, object],
    scale_budget: bool,
    scale_max_rounds: bool,
    reference_budget: int,
    reference_n: int,
) -> list[RegimeCellResult]:
    if not scale_budget and not scale_max_rounds:
        return build_regime_cells(
            graphs,
            policy_factories,
            alpha_values=alpha_values,
            pfail_values=pfail_values,
            budgets=budgets,
            max_rounds=max_rounds,
            seeds=seeds,
            tau=tau,
            hopeless_threshold=hopeless_threshold,
            trivial_threshold=trivial_threshold,
            spread_threshold=spread_threshold,
            env_kwargs=env_kwargs,
        )

    cells: list[RegimeCellResult] = []
    grouped_best_anc: dict[tuple[float, float], list[float]] = {}
    grouped_cells: dict[tuple[float, float], list[tuple[int, dict[str, Any]]]] = {}

    for alpha in alpha_values:
        for pfail in pfail_values:
            for budget in budgets:
                policy_summaries = evaluate_policy_factories_with_optional_scaling(
                    graphs,
                    policy_factories,
                    alpha=alpha,
                    pfail=pfail,
                    budget=budget,
                    max_rounds=max_rounds,
                    seeds=seeds,
                    tau=tau,
                    env_kwargs=env_kwargs,
                    scale_budget=scale_budget,
                    scale_max_rounds=scale_max_rounds,
                    reference_budget=reference_budget,
                    reference_n=reference_n,
                )
                grouped_cells.setdefault((alpha, pfail), []).append((budget, policy_summaries))
                grouped_best_anc.setdefault((alpha, pfail), []).append(
                    max(summary.final_anc.mean for summary in policy_summaries.values())
                )

    for (alpha, pfail), budget_summaries in grouped_cells.items():
        anc_values = grouped_best_anc[(alpha, pfail)]
        budget_sensitivity = max(anc_values) - min(anc_values) if len(anc_values) > 1 else 0.0
        for budget, policy_summaries in budget_summaries:
            diagnostics = compute_regime_diagnostics(
                policy_summaries,
                hopeless_threshold=hopeless_threshold,
                trivial_threshold=trivial_threshold,
                spread_threshold=spread_threshold,
                budget_sensitivity=budget_sensitivity,
            )
            cells.append(
                RegimeCellResult(
                    alpha=alpha,
                    pfail=pfail,
                    budget=budget,
                    diagnostics=diagnostics,
                    policy_summaries=dict(policy_summaries),
                )
            )

    return sorted(cells, key=lambda cell: (cell.alpha, cell.pfail, cell.budget))


def main() -> None:
    args = parse_args()
    if args.reference_n < 1:
        raise ValueError("--reference-n must be at least 1.")

    config = load_config(args.config)
    training = config["training"]
    regime = training["regime"]
    graph_cfg = training["graph"]
    evaluation = config["evaluation"]
    env_kwargs = resolve_env_kwargs(config)
    benchmark_reference_budget = (
        int(args.budgets[0]) if args.budgets is not None else int(regime["budget"])
    )
    scaling_metadata = build_scaling_metadata(
        scale_budget=args.scale_budget,
        scale_max_rounds=args.scale_max_rounds,
        reference_budget=benchmark_reference_budget,
        reference_n=args.reference_n,
    )

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
    log_scaling_decisions(
        graphs=graphs,
        pfail=float(regime["pfail"]),
        budget=int(regime["budget"]),
        max_rounds=int(regime["max_rounds"]),
        scale_budget=args.scale_budget,
        scale_max_rounds=args.scale_max_rounds,
        reference_budget=benchmark_reference_budget,
        reference_n=args.reference_n,
    )
    summaries = evaluate_policy_factories_with_optional_scaling(
        graphs,
        policy_factories,
        alpha=float(regime["alpha"]),
        pfail=float(regime["pfail"]),
        budget=int(regime["budget"]),
        max_rounds=int(regime["max_rounds"]),
        seeds=list(training["benchmark_seeds"]),
        tau=tau,
        env_kwargs=env_kwargs,
        scale_budget=args.scale_budget,
        scale_max_rounds=args.scale_max_rounds,
        reference_budget=benchmark_reference_budget,
        reference_n=args.reference_n,
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
    serialized["scaling"] = scaling_metadata

    grid_spec = resolve_grid_spec(config, args)
    grid_reference_budget = int(grid_spec["budgets"][0])
    grid_scaling_metadata = build_scaling_metadata(
        scale_budget=args.scale_budget,
        scale_max_rounds=args.scale_max_rounds,
        reference_budget=grid_reference_budget,
        reference_n=args.reference_n,
    )
    grid_graphs = make_graph_batch(
        num_graphs=grid_spec["num_graphs"],
        n_range=grid_spec["n_range"],
        m=grid_spec["m"],
        seed=grid_spec["graph_seed"],
    )
    threshold_cfg = config["regime_mapping"]
    log_scaling_decisions(
        graphs=grid_graphs,
        pfail=float(grid_spec["pfail_values"][0]),
        budget=int(grid_spec["budgets"][0]),
        max_rounds=grid_spec["max_rounds"],
        scale_budget=args.scale_budget,
        scale_max_rounds=args.scale_max_rounds,
        reference_budget=grid_reference_budget,
        reference_n=args.reference_n,
    )
    cells = build_regime_cells_with_optional_scaling(
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
        scale_budget=args.scale_budget,
        scale_max_rounds=args.scale_max_rounds,
        reference_budget=grid_reference_budget,
        reference_n=args.reference_n,
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
        "scaling": grid_scaling_metadata,
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
        if policy_name == "scaling":
            continue
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
