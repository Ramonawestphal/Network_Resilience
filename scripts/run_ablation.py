from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.evaluation import evaluate_policy_factories_on_graphs
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import build_greedy_policy
from cascading_rl.training import (
    FREEZE_GRAPH_SPECS_SEED_OFFSET,
    TrainingConfig,
    generate_episode_graph_specs,
    train_recovery_agent,
)


ABLATION_CONFIGS = [
    {"name": "node_only", "use_global_features": False, "use_virtual_node": False},
    {"name": "node_global", "use_global_features": True, "use_virtual_node": False},
    {"name": "node_virtual", "use_global_features": False, "use_virtual_node": True},
    {"name": "node_global_virtual", "use_global_features": True, "use_virtual_node": True},
]
ABLATION_OUTPUT_DIR = ROOT / "experiments" / "ablation"
ABLATION_OUTPUT_PATH = ABLATION_OUTPUT_DIR / "ablation_comparison.json"
EVAL_GRAPH_SEED_OFFSET = 30_000


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_training_config(config: dict, episodes_override: int | None = None) -> TrainingConfig:
    training = config["training"]
    regime = training["regime"]
    graph = training["graph"]
    return TrainingConfig(
        seed=int(training["seed"]),
        device=str(training["device"]),
        alpha=float(regime["alpha"]),
        pfail=float(regime["pfail"]),
        budget=int(regime["budget"]),
        max_rounds=int(regime["max_rounds"]),
        n_range=tuple(graph["n_range"]),
        m=int(graph["m"]),
        num_episodes=int(episodes_override or training["num_episodes"]),
        replay_capacity=int(training["replay_capacity"]),
        warmup_transitions=int(training["warmup_transitions"]),
        batch_size=int(training["batch_size"]),
        gamma=float(training["gamma"]),
        learning_rate=float(training["learning_rate"]),
        epsilon_start=float(training["epsilon_start"]),
        epsilon_end=float(training["epsilon_end"]),
        epsilon_decay_episodes=int(training["epsilon_decay_episodes"]),
        target_update_interval=int(training["target_update_interval"]),
        hidden_dim=int(training["hidden_dim"]),
        embed_dim=int(training["embed_dim"]),
        num_layers=int(training["num_layers"]),
        validation_graphs=int(training["validation_graphs"]),
        validation_seeds=tuple(training["validation_seeds"]),
        validation_every=int(training["validation_every"]),
        checkpoint_dir=str(ABLATION_OUTPUT_DIR),
        checkpoint_name="placeholder.pt",
        freeze_graphs=bool(training.get("freeze_graphs", False)),
    )


def generate_episode_graph_specs(config: TrainingConfig, *, seed: int) -> tuple[tuple[int, int], ...]:
    rng = Random(seed)
    return tuple(
        (
            rng.randint(config.n_range[0], config.n_range[1]),
            rng.randint(0, 10**9),
        )
        for _ in range(config.num_episodes)
    )


def evaluate_config(model, training_config: TrainingConfig, eval_graphs: list, eval_seeds: list[int], tau: float) -> dict:
    rl_policy = build_greedy_policy(model, device=training_config.device, batch_actions=True)
    summaries = evaluate_policy_factories_on_graphs(
        eval_graphs,
        {"rl": lambda graph_index, seed: rl_policy},
        alpha=training_config.alpha,
        pfail=training_config.pfail,
        budget=training_config.budget,
        max_rounds=training_config.max_rounds,
        seeds=eval_seeds,
        tau=tau,
    )
    summary = summaries["rl"]
    return {
        "final_anc": {"mean": summary.final_anc.mean, "stderr": summary.final_anc.stderr},
        "threshold_hit_fraction": {
            "mean": summary.threshold_hit_fraction.mean,
            "stderr": summary.threshold_hit_fraction.stderr,
        },
        "solved_fraction": {
            "mean": summary.solved_fraction.mean,
            "stderr": summary.solved_fraction.stderr,
        },
    }


def print_comparison_table(results: list[dict]) -> None:
    print("")
    print(
        f"{'config':<22} {'global':<8} {'virtual':<8} "
        f"{'final_anc':<18} {'threshold_hit':<18} {'solved':<18}"
    )
    print("-" * 92)
    for item in results:
        final_anc = item["results"]["final_anc"]
        threshold = item["results"]["threshold_hit_fraction"]
        solved = item["results"]["solved_fraction"]
        print(
            f"{item['name']:<22} "
            f"{str(item['use_global_features']):<8} "
            f"{str(item['use_virtual_node']):<8} "
            f"{final_anc['mean']:.3f}+/-{final_anc['stderr']:.3f}   "
            f"{threshold['mean']:.3f}+/-{threshold['stderr']:.3f}   "
            f"{solved['mean']:.3f}+/-{solved['stderr']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run learner ablations over feature/context settings.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional override for the number of training episodes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    base_training_config = build_training_config(config, episodes_override=args.episodes)
    training = config["training"]
    graph = training["graph"]
    eval_tau = float(config["evaluation"]["tau"])
    eval_seeds = list(training["benchmark_seeds"])
    frozen_episode_graph_specs = generate_episode_graph_specs(
        base_training_config,
        seed=base_training_config.seed + FREEZE_GRAPH_SPECS_SEED_OFFSET,
    )
    eval_graphs = make_graph_batch(
        num_graphs=int(training["benchmark_graphs"]),
        n_range=tuple(graph["n_range"]),
        m=int(graph["m"]),
        seed=int(training["seed"]) + EVAL_GRAPH_SEED_OFFSET,
    )

    ABLATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_results = []

    for ablation_config in ABLATION_CONFIGS:
        training_config = replace(
            base_training_config,
            checkpoint_name=f"{ablation_config['name']}.pt",
            use_global_features=ablation_config["use_global_features"],
            use_virtual_node=ablation_config["use_virtual_node"],
            episode_graph_specs=frozen_episode_graph_specs,
        )
        model, _, checkpoint_path = train_recovery_agent(training_config)
        results = evaluate_config(
            model,
            training_config,
            eval_graphs,
            eval_seeds,
            eval_tau,
        )
        comparison_results.append(
            {
                "name": ablation_config["name"],
                "use_global_features": ablation_config["use_global_features"],
                "use_virtual_node": ablation_config["use_virtual_node"],
                "training_episodes": training_config.num_episodes,
                "checkpoint_path": str(checkpoint_path),
                "results": results,
            }
        )

    payload = {"configs": comparison_results}
    with ABLATION_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print_comparison_table(comparison_results)
    print("")
    print(f"Saved ablation comparison to {ABLATION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
