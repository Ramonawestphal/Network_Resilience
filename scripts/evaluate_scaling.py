from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.evaluation import build_policy_factories, evaluate_policy_factories_on_graphs
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import build_greedy_policy, load_q_network

SUPPORTED_POLICIES = ("rl", "random", "degree", "risk", "greedy", "betweenness")


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate checkpoint scaling on larger graphs.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained checkpoint.",
    )
    parser.add_argument(
        "--graph-sizes",
        type=int,
        nargs="+",
        required=True,
        help="Exact graph sizes to evaluate, e.g. 100 300 500 1000.",
    )
    parser.add_argument(
        "--num-graphs",
        type=int,
        default=1,
        help="Number of graphs to sample per graph size.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Matched rollout seeds.",
    )
    parser.add_argument("--alpha", type=float, required=True, help="Cascade alpha.")
    parser.add_argument("--pfail", type=float, required=True, help="Initial failure probability.")
    parser.add_argument("--budget", type=int, required=True, help="Recovery budget.")
    parser.add_argument("--max-rounds", type=int, required=True, help="Maximum recovery rounds.")
    parser.add_argument("--m", type=int, default=2, help="BA attachment parameter.")
    parser.add_argument(
        "--policies",
        type=str,
        nargs="+",
        default=["rl", "degree"],
        help="Subset of policies to evaluate.",
    )
    parser.add_argument(
        "--graph-seed",
        type=int,
        default=1007,
        help="Base seed for graph generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "scaling",
        help="Directory for scaling evaluation artifacts.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="ANC threshold used for threshold_hit_fraction (defaults to config evaluation.tau).",
    )
    return parser.parse_args()


def resolve_env_kwargs(config: dict[str, Any]) -> dict[str, object]:
    regime = config["training"]["regime"]
    obs_hops = regime.get("obs_hops")
    return {
        "capacity_noise": float(regime.get("capacity_noise", 0.0)),
        "failure_bias": str(regime.get("failure_bias", "uniform")),
        "action_space": str(regime.get("action_space", "failed")),
        "obs_hops": int(obs_hops) if obs_hops is not None else None,
    }


def build_selected_policy_factories(
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

    base_factories = build_policy_factories(base_seed=base_seed)
    policy_factories: dict[str, Any] = {}

    if "rl" in selected_policies:
        model, _ = load_q_network(checkpoint_path)
        rl_policy = build_greedy_policy(model)
        policy_factories["rl"] = lambda _graph_index, _seed: rl_policy

    for policy_name in selected_policies:
        if policy_name == "rl":
            continue
        policy_factories[policy_name] = base_factories[policy_name]

    return policy_factories


def serialize_summary(summary: Any) -> dict[str, float]:
    return {
        "final_anc_mean": summary.final_anc.mean,
        "final_anc_stderr": summary.final_anc.stderr,
        "threshold_hit_mean": summary.threshold_hit_fraction.mean,
        "threshold_hit_stderr": summary.threshold_hit_fraction.stderr,
        "rounds_mean": summary.rounds.mean,
        "rounds_stderr": summary.rounds.stderr,
        "solved_fraction_mean": summary.solved_fraction.mean,
        "solved_fraction_stderr": summary.solved_fraction.stderr,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    env_kwargs = resolve_env_kwargs(config)
    tau = float(args.tau) if args.tau is not None else float(config["evaluation"]["tau"])
    selected_policies = list(dict.fromkeys(args.policies))
    policy_factories = build_selected_policy_factories(
        args.checkpoint,
        base_seed=int(config["training"]["seed"]),
        selected_policies=selected_policies,
    )

    started = perf_counter()
    size_results: list[dict[str, Any]] = []
    for graph_size in args.graph_sizes:
        graphs = make_graph_batch(
            num_graphs=args.num_graphs,
            n_range=(graph_size, graph_size),
            m=args.m,
            seed=args.graph_seed + graph_size,
        )
        size_started = perf_counter()
        summaries = evaluate_policy_factories_on_graphs(
            graphs,
            policy_factories,
            alpha=args.alpha,
            pfail=args.pfail,
            budget=args.budget,
            max_rounds=args.max_rounds,
            seeds=args.seeds,
            tau=tau,
            env_kwargs=env_kwargs,
        )
        elapsed_seconds = perf_counter() - size_started
        size_results.append(
            {
                "graph_size": graph_size,
                "num_graphs": args.num_graphs,
                "seeds": list(args.seeds),
                "elapsed_seconds": elapsed_seconds,
                "policy_summaries": {
                    policy_name: serialize_summary(summary)
                    for policy_name, summary in summaries.items()
                },
            }
        )
        policy_text = "  ".join(
            f"{name}: final_anc={serialize_summary(summary)['final_anc_mean']:.3f}"
            for name, summary in summaries.items()
        )
        print(
            f"[scaling] n={graph_size} elapsed={elapsed_seconds:.1f}s {policy_text}",
            flush=True,
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (
        f"{args.checkpoint.parent.name}_{'_'.join(selected_policies)}_"
        f"a{args.alpha:.2f}_p{args.pfail:.2f}_b{args.budget}_mr{args.max_rounds}.json"
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "alpha": args.alpha,
        "pfail": args.pfail,
        "budget": args.budget,
        "max_rounds": args.max_rounds,
        "m": args.m,
        "graph_seed": args.graph_seed,
        "tau": tau,
        "policies": selected_policies,
        "env": env_kwargs,
        "elapsed_seconds_total": perf_counter() - started,
        "results": size_results,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(f"Saved scaling summary to {output_path}")


if __name__ == "__main__":
    main()
