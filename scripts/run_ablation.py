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
from cascading_rl.models import FEATURE_NAMES, GLOBAL_FEATURE_NAMES, build_greedy_policy
from cascading_rl.reproducibility import portable_artifact_path, portable_repo_relative_path
from cascading_rl.training import (
    FREEZE_GRAPH_SPECS_SEED_OFFSET,
    TrainingConfig,
    generate_episode_graph_specs,
    train_recovery_agent,
)
from cascading_rl.training.trainer import _env_kwargs_from_config


ABLATION_OUTPUT_DIR = ROOT / "experiments" / "ablation"
ABLATION_OUTPUT_PATH = ABLATION_OUTPUT_DIR / "ablation_comparison.json"
EVAL_GRAPH_SEED_OFFSET = 30_000


def build_ablation_runs() -> list[dict[str, object]]:
    runs: list[dict[str, object]] = [
        {
            "name": "node_only",
            "active_node_features": FEATURE_NAMES,
            "active_global_features": (),
            "use_virtual_node": False,
        },
        {
            "name": "node_global",
            "active_node_features": FEATURE_NAMES,
            "active_global_features": GLOBAL_FEATURE_NAMES,
            "use_virtual_node": False,
        },
        {
            "name": "node_virtual",
            "active_node_features": FEATURE_NAMES,
            "active_global_features": (),
            "use_virtual_node": True,
        },
        {
            "name": "node_global_virtual",
            "active_node_features": FEATURE_NAMES,
            "active_global_features": GLOBAL_FEATURE_NAMES,
            "use_virtual_node": True,
        },
    ]
    runs.extend(
        {
            "name": f"drop_global_{feature_name}",
            "active_node_features": FEATURE_NAMES,
            "active_global_features": tuple(
                feature for feature in GLOBAL_FEATURE_NAMES if feature != feature_name
            ),
            "use_virtual_node": False,
        }
        for feature_name in GLOBAL_FEATURE_NAMES
    )
    runs.extend(
        {
            "name": f"drop_node_{feature_name}",
            "active_node_features": tuple(
                feature for feature in FEATURE_NAMES if feature != feature_name
            ),
            "active_global_features": GLOBAL_FEATURE_NAMES,
            "use_virtual_node": False,
        }
        for feature_name in FEATURE_NAMES
    )
    runs.extend(
        {
            "name": f"drop_global_{feature_name}_vn",
            "active_node_features": FEATURE_NAMES,
            "active_global_features": tuple(
                feature for feature in GLOBAL_FEATURE_NAMES if feature != feature_name
            ),
            "use_virtual_node": True,
        }
        for feature_name in GLOBAL_FEATURE_NAMES
    )
    runs.extend(
        {
            "name": f"drop_node_{feature_name}_vn",
            "active_node_features": tuple(
                feature for feature in FEATURE_NAMES if feature != feature_name
            ),
            "active_global_features": GLOBAL_FEATURE_NAMES,
            "use_virtual_node": True,
        }
        for feature_name in FEATURE_NAMES
    )
    return runs


ABLATION_RUNS = build_ablation_runs()


def serialize_policy_summary(summary) -> dict[str, object]:
    rws = summary.rounds_when_solved
    return {
        "final_nc": {"mean": summary.final_nc.mean, "stderr": summary.final_nc.stderr},
        "solved_fraction": {
            "mean": summary.solved_fraction.mean,
            "stderr": summary.solved_fraction.stderr,
        },
        "fully_restored_count": summary.fully_restored_count,
        "fully_restored_fraction": (
            summary.fully_restored_count / summary.episode_count
            if summary.episode_count
            else 0.0
        ),
        "episode_count": summary.episode_count,
        "unsolved_low_final_nc_count": summary.unsolved_low_final_nc_count,
        "unsolved_low_final_nc_fraction": summary.unsolved_low_final_nc_fraction,
        "final_nc_failure_threshold_used": summary.final_nc_failure_threshold_used,
        "rounds_when_solved": (
            {"mean": rws.mean, "stderr": rws.stderr} if rws is not None else None
        ),
        "mean_delta_nc_per_round": {
            "mean": summary.mean_delta_nc_per_round.mean,
            "stderr": summary.mean_delta_nc_per_round.stderr,
        },
        "mean_nc_on_failed": (
            {"mean": summary.mean_nc_on_failed.mean, "stderr": summary.mean_nc_on_failed.stderr}
            if summary.mean_nc_on_failed is not None
            else None
        ),
    }


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_training_config(config: dict, episodes_override: int | None = None) -> TrainingConfig:
    training = config["training"]
    regime = training["regime"]
    graph = training["graph"]
    budget_scaling = config.get("budget_scaling", {})
    defaults = TrainingConfig()
    return TrainingConfig(
        seed=int(training["seed"]),
        device=str(training["device"]),
        alpha=float(regime["alpha"]),
        pfail=float(regime["pfail"]),
        budget=int(regime["budget"]),
        scale_budget=bool(budget_scaling.get("enabled", defaults.scale_budget)),
        budget_reference_n=int(
            budget_scaling.get("reference_n", defaults.budget_reference_n)
        ),
        scale_max_rounds=bool(
            budget_scaling.get("scale_max_rounds", defaults.scale_max_rounds)
        ),
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
        validation_eval_set_path=(
            str(training["validation_eval_set_path"]).strip()
            if training.get("validation_eval_set_path")
            else None
        ),
        checkpoint_dir=str(ABLATION_OUTPUT_DIR),
        checkpoint_name="placeholder.pt",
        freeze_graphs=bool(training.get("freeze_graphs", False)),
    )


def evaluate_config(
    model,
    training_config: TrainingConfig,
    eval_graphs: list,
    eval_seeds: list[int],
) -> dict[str, object]:
    rl_policy = build_greedy_policy(model, device=training_config.device, batch_actions=True)
    summary = evaluate_policy_factories_on_graphs(
        eval_graphs,
        {"rl": lambda _graph_index, _seed: rl_policy},
        alpha=training_config.alpha,
        pfail=training_config.pfail,
        budget=training_config.budget,
        max_rounds=training_config.max_rounds,
        seeds=eval_seeds,
        env_kwargs=_env_kwargs_from_config(training_config),
        scale_budget=training_config.scale_budget,
        scale_max_rounds=training_config.scale_max_rounds,
        reference_n=training_config.budget_reference_n,
    )["rl"]
    return serialize_policy_summary(summary)


def print_comparison_table(results: list[dict[str, object]]) -> None:
    print("")
    print(
        f"{'config':<28} {'node':<6} {'global':<8} {'virtual':<8} "
        f"{'final_nc':<18} {'solved':<18} {'restored':<14} {'rws_mean':<10}"
    )
    print("-" * 108)
    for item in results:
        final_nc = item["results"]["final_nc"]
        solved = item["results"]["solved_fraction"]
        rws = item["results"]["rounds_when_solved"]
        rws_m = rws["mean"] if rws else float("nan")
        restored = item["results"]["fully_restored_count"]
        ep_n = item["results"]["episode_count"]
        print(
            f"{item['name']:<28} "
            f"{len(item['active_node_features']):<6} "
            f"{len(item['active_global_features']):<8} "
            f"{str(item['use_virtual_node']):<8} "
            f"{final_nc['mean']:.3f}+/-{final_nc['stderr']:.3f}   "
            f"{solved['mean']:.3f}+/-{solved['stderr']:.3f}   "
            f"{restored}/{ep_n}  "
            f"{rws_m:.2f}"
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
    if base_training_config.validation_eval_set_path:
        ves = Path(base_training_config.validation_eval_set_path)
        if not ves.is_absolute():
            ves = ROOT / ves
        base_training_config = replace(
            base_training_config,
            validation_eval_set_path=portable_repo_relative_path(ves),
        )
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
    comparison_results: list[dict[str, object]] = []

    for ablation_config in ABLATION_RUNS:
        training_config = replace(
            base_training_config,
            checkpoint_name=f"{ablation_config['name']}.pt",
            use_global_features=bool(ablation_config["active_global_features"]),
            active_node_features=ablation_config["active_node_features"],
            active_global_features=ablation_config["active_global_features"],
            use_virtual_node=ablation_config["use_virtual_node"],
            episode_graph_specs=frozen_episode_graph_specs,
        )
        model, _, checkpoint_path = train_recovery_agent(training_config)
        results = evaluate_config(
            model,
            training_config,
            eval_graphs,
            eval_seeds,
        )
        run_payload = {
            "name": ablation_config["name"],
            "active_node_features": list(ablation_config["active_node_features"]),
            "active_global_features": list(ablation_config["active_global_features"]),
            "use_virtual_node": ablation_config["use_virtual_node"],
            "training_episodes": training_config.num_episodes,
            "checkpoint_path": portable_artifact_path(checkpoint_path),
            "results": results,
        }
        comparison_results.append(run_payload)
        run_output_path = ABLATION_OUTPUT_DIR / f"{ablation_config['name']}.json"
        with run_output_path.open("w", encoding="utf-8") as file:
            json.dump(run_payload, file, indent=2)

    payload = {"configs": comparison_results}
    with ABLATION_OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print_comparison_table(comparison_results)
    print("")
    print(f"Saved ablation comparison to {ABLATION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
