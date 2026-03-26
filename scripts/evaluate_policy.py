from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from random import Random

from cascading_rl.evaluation import (
    build_policy_factories,
    estimate_minimum_budget,
    evaluate_policy_factories_on_graphs,
)
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import build_greedy_policy, load_q_network
from cascading_rl.policies import choose_random_failed_node


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training = config["training"]
    regime = training["regime"]
    graph_cfg = training["graph"]

    model, _ = load_q_network(args.checkpoint)
    rl_policy = build_greedy_policy(model)
    policy_factories = build_policy_factories(base_seed=int(training["seed"]))
    policy_factories = {
        "rl": lambda graph_index, seed: rl_policy,
        "random": policy_factories["random"],
        "degree": policy_factories["degree"],
        "risk": policy_factories["risk"],
        "greedy": policy_factories["greedy"],
        "betweenness": policy_factories["betweenness"],
    }

    graphs = make_graph_batch(
        num_graphs=int(training["benchmark_graphs"]),
        n_range=tuple(graph_cfg["n_range"]),
        m=int(graph_cfg["m"]),
        seed=int(training["seed"]) + 1000,
    )

    summaries = evaluate_policy_factories_on_graphs(
        graphs,
        policy_factories,
        alpha=float(regime["alpha"]),
        pfail=float(regime["pfail"]),
        budget=int(regime["budget"]),
        max_rounds=int(regime["max_rounds"]),
        seeds=training["benchmark_seeds"],
        tau=float(config["evaluation"]["tau"]),
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
    evaluation_budgets = config["evaluation"]["budgets"]
    tau = float(config["evaluation"]["tau"])
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
        )[0],
        "degree": estimate_minimum_budget(
            representative_graph,
            policy_factories["degree"](0, 0),
            tau=tau,
            budgets=evaluation_budgets,
            trials=len(training["benchmark_seeds"]),
            alpha=float(regime["alpha"]),
            pfail=float(regime["pfail"]),
            max_rounds=int(regime["max_rounds"]),
        )[0],
        "greedy": estimate_minimum_budget(
            representative_graph,
            policy_factories["greedy"](0, 0),
            tau=tau,
            budgets=evaluation_budgets,
            trials=len(training["benchmark_seeds"]),
            alpha=float(regime["alpha"]),
            pfail=float(regime["pfail"]),
            max_rounds=int(regime["max_rounds"]),
        )[0],
        "risk": estimate_minimum_budget(
            representative_graph,
            policy_factories["risk"](0, 0),
            tau=tau,
            budgets=evaluation_budgets,
            trials=len(training["benchmark_seeds"]),
            alpha=float(regime["alpha"]),
            pfail=float(regime["pfail"]),
            max_rounds=int(regime["max_rounds"]),
        )[0],
        "betweenness": estimate_minimum_budget(
            representative_graph,
            policy_factories["betweenness"](0, 0),
            tau=tau,
            budgets=evaluation_budgets,
            trials=len(training["benchmark_seeds"]),
            alpha=float(regime["alpha"]),
            pfail=float(regime["pfail"]),
            max_rounds=int(regime["max_rounds"]),
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
        )[0],
    }
    for policy_name, value in b_star.items():
        serialized[policy_name]["b_star"] = value

    output_dir = ROOT / training["benchmark_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "evaluation_summary.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(serialized, file, indent=2)

    print(f"Saved evaluation summary to {output_path}")
    for policy_name, metrics in serialized.items():
        print(
            f"{policy_name}: final_anc={metrics['final_anc_mean']:.3f}, "
            f"threshold_hit={metrics['threshold_hit_mean']:.3f}, "
            f"rounds={metrics['rounds_mean']:.3f}, b_star={metrics['b_star']}"
        )


if __name__ == "__main__":
    main()
