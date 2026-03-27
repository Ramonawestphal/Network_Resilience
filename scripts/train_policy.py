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

from cascading_rl.training import TrainingConfig, train_recovery_agent


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def build_training_config(config: dict, episodes_override: int | None = None) -> TrainingConfig:
    training = config["training"]
    regime = training["regime"]
    graph = training["graph"]
    alpha_values = regime.get("alpha_values")
    pfail_values = regime.get("pfail_values")
    return TrainingConfig(
        seed=int(training["seed"]),
        device=str(training["device"]),
        alpha=float(regime["alpha"]),
        pfail=float(regime["pfail"]),
        alpha_values=tuple(float(value) for value in alpha_values) if alpha_values is not None else None,
        pfail_values=tuple(float(value) for value in pfail_values) if pfail_values is not None else None,
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
    training_config = build_training_config(config, episodes_override=args.episodes)
    _, training_state, checkpoint_path = train_recovery_agent(training_config)

    summary_path = checkpoint_path.with_suffix(".summary.json")
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "num_episodes": training_config.num_episodes,
        "alpha_values": training_config.alpha_values,
        "pfail_values": training_config.pfail_values,
        "final_reward_mean_last_10": (
            sum(training_state.episode_rewards[-10:]) / max(1, len(training_state.episode_rewards[-10:]))
        ),
        "final_anc_mean_last_10": (
            sum(training_state.episode_final_anc[-10:])
            / max(1, len(training_state.episode_final_anc[-10:]))
        ),
        "num_updates": len(training_state.losses),
        "validation_history": training_state.validation_history,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
