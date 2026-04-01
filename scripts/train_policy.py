from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.training import TrainingConfig, train_recovery_agent
from scripts.reproducibility import write_run_metadata


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def build_training_config(
    config: dict[str, Any],
    *,
    episodes_override: int | None = None,
) -> TrainingConfig:
    defaults = TrainingConfig()
    training = config["training"]
    regime = training["regime"]
    graph = training["graph"]
    budget_scaling = config.get("budget_scaling", {})
    alpha_values_raw = regime.get("alpha_values")
    pfail_values_raw = regime.get("pfail_values")
    obs_hops_raw = regime.get("obs_hops", defaults.obs_hops)
    num_episodes = (
        int(episodes_override)
        if episodes_override is not None
        else int(training["num_episodes"])
    )

    return TrainingConfig(
        seed=int(training["seed"]),
        device=str(training["device"]),
        alpha=float(regime["alpha"]),
        pfail=float(regime["pfail"]),
        alpha_values=(
            tuple(float(value) for value in alpha_values_raw)
            if alpha_values_raw is not None
            else defaults.alpha_values
        ),
        pfail_values=(
            tuple(float(value) for value in pfail_values_raw)
            if pfail_values_raw is not None
            else defaults.pfail_values
        ),
        budget=int(regime["budget"]),
        scale_budget=bool(budget_scaling.get("enabled", defaults.scale_budget)),
        budget_reference_n=int(
            budget_scaling.get("reference_n", defaults.budget_reference_n)
        ),
        max_rounds=int(regime["max_rounds"]),
        capacity_noise=float(regime.get("capacity_noise", defaults.capacity_noise)),
        failure_bias=str(regime.get("failure_bias", defaults.failure_bias)),
        action_space=str(regime.get("action_space", defaults.action_space)),
        obs_hops=int(obs_hops_raw) if obs_hops_raw is not None else None,
        n_range=tuple(graph["n_range"]),
        m=int(graph["m"]),
        num_episodes=num_episodes,
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
        use_global_features=bool(
            training.get("use_global_features", defaults.use_global_features)
        ),
        use_virtual_node=bool(
            training.get("use_virtual_node", defaults.use_virtual_node)
        ),
        validation_graphs=int(training["validation_graphs"]),
        validation_seeds=tuple(training["validation_seeds"]),
        validation_seed=int(training.get("validation_seed", defaults.validation_seed)),
        validation_every=int(training["validation_every"]),
        checkpoint_dir=str(training["checkpoint_dir"]),
        checkpoint_name=str(training["checkpoint_name"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the recovery Q-network.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Override single training alpha.")
    parser.add_argument("--pfail", type=float, default=None, help="Override single training pfail.")
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=None,
        help="Override the per-episode alpha grid.",
    )
    parser.add_argument(
        "--pfail-values",
        type=float,
        nargs="+",
        default=None,
        help="Override the per-episode pfail grid.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional override for the number of training episodes.",
    )
    parser.add_argument(
        "--hard-regime",
        action="store_true",
        help="Swap to the hard-regime training grid from the plan/config.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Optional override for the checkpoint directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_config = build_training_config(config, episodes_override=args.episodes)

    if args.alpha is not None:
        training_config = replace(training_config, alpha=args.alpha)
    if args.pfail is not None:
        training_config = replace(training_config, pfail=args.pfail)
    if args.alpha_values is not None:
        training_config = replace(training_config, alpha_values=tuple(args.alpha_values))
    if args.pfail_values is not None:
        training_config = replace(training_config, pfail_values=tuple(args.pfail_values))
    if args.hard_regime:
        hard = config["hard_regime"]
        training_config = replace(
            training_config,
            alpha=float(hard["alpha"]),
            pfail=float(hard["pfail"]),
            alpha_values=tuple(float(value) for value in hard.get("alpha_values", [hard["alpha"]])),
            pfail_values=tuple(float(value) for value in hard.get("pfail_values", [hard["pfail"]])),
            budget=int(hard["budget"]),
            max_rounds=int(hard["max_rounds"]),
        )
    if args.checkpoint_dir is not None:
        training_config = replace(training_config, checkpoint_dir=args.checkpoint_dir)

    _, training_state, checkpoint_path = train_recovery_agent(training_config)

    summary_path = checkpoint_path.with_suffix(".summary.json")
    recent_rewards = training_state.episode_rewards[-10:]
    recent_anc = training_state.episode_final_anc[-10:]
    recent_losses = training_state.losses[-10:]
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "training_config": asdict(training_config),
        "num_episodes": training_config.num_episodes,
        "alpha_values": list(training_config.alpha_values or (training_config.alpha,)),
        "pfail_values": list(training_config.pfail_values or (training_config.pfail,)),
        "env": {
            "capacity_noise": training_config.capacity_noise,
            "failure_bias": training_config.failure_bias,
            "action_space": training_config.action_space,
            "obs_hops": training_config.obs_hops,
            "scale_budget": training_config.scale_budget,
            "budget_reference_n": training_config.budget_reference_n,
        },
        "final_reward_mean_last_10": sum(recent_rewards) / max(1, len(recent_rewards)),
        "final_anc_mean_last_10": sum(recent_anc) / max(1, len(recent_anc)),
        "final_loss_mean_last_10": sum(recent_losses) / max(1, len(recent_losses)),
        "num_updates": len(training_state.losses),
        "total_steps": training_state.total_steps,
        "validation_history": training_state.validation_history,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    write_run_metadata(
        checkpoint_path.parent / "run_metadata.json",
        script_path=Path(__file__).resolve(),
        argv=sys.argv,
        config_path=args.config,
        extra={
            "checkpoint_path": str(checkpoint_path),
            "summary_path": str(summary_path),
        },
    )

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
