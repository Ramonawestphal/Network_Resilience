from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cascading_rl.training import (
    FREEZE_GRAPH_SPECS_SEED_OFFSET,
    TrainingConfig,
    train_recovery_agent,
)


def training_config_for_json(config: TrainingConfig) -> dict[str, Any]:
    """JSON-serializable training config; avoid huge episode_graph_specs in summaries."""
    data: dict[str, Any] = asdict(config)
    if config.episode_graph_specs is not None:
        data["episode_graph_specs"] = {
            "frozen": config.freeze_graphs,
            "count": len(config.episode_graph_specs),
            "spec_seed": config.seed + FREEZE_GRAPH_SPECS_SEED_OFFSET,
        }
    elif config.freeze_graphs:
        data["episode_graph_specs"] = {
            "frozen": True,
            "count": config.num_episodes,
            "spec_seed": config.seed + FREEZE_GRAPH_SPECS_SEED_OFFSET,
        }
    return data


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return data


def build_training_config(config: dict[str, Any], *, episodes_override: int | None = None) -> TrainingConfig:
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
        use_monte_carlo_returns=bool(
            training.get("use_monte_carlo_returns", defaults.use_monte_carlo_returns)
        ),
        learning_rate=float(training["learning_rate"]),
        epsilon_start=float(training["epsilon_start"]),
        epsilon_end=float(training["epsilon_end"]),
        epsilon_decay_episodes=int(
            training.get("epsilon_decay_episodes", defaults.epsilon_decay_episodes)
        ),
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
        use_imitation_warmstart=bool(
            training.get("use_imitation_warmstart", defaults.use_imitation_warmstart)
        ),
        imitation_graphs=int(training.get("imitation_graphs", defaults.imitation_graphs)),
        imitation_seeds=int(training.get("imitation_seeds", defaults.imitation_seeds)),
        imitation_epochs=int(training.get("imitation_epochs", defaults.imitation_epochs)),
        validation_graphs=int(training["validation_graphs"]),
        validation_seeds=tuple(training["validation_seeds"]),
        validation_seed=int(training.get("validation_seed", defaults.validation_seed)),
        validation_every=int(training["validation_every"]),
        validation_tau=float(training.get("validation_tau", defaults.validation_tau)),
        checkpoint_dir=str(training["checkpoint_dir"]),
        checkpoint_name=str(training["checkpoint_name"]),
        freeze_graphs=bool(training.get("freeze_graphs", defaults.freeze_graphs)),
        log_episode_spread=bool(training.get("log_episode_spread", defaults.log_episode_spread)),
        log_grad_norm=bool(training.get("log_grad_norm", defaults.log_grad_norm)),
        validation_eval_set_path=(
            str(training["validation_eval_set_path"]).strip()
            if training.get("validation_eval_set_path")
            else defaults.validation_eval_set_path
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the recovery Q-network.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config" / "default.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Override single regime alpha (reference).")
    parser.add_argument("--pfail", type=float, default=None, help="Override single regime pfail (reference).")
    parser.add_argument(
        "--alpha-values",
        type=float,
        nargs="+",
        default=None,
        help="Override per-episode alpha grid.",
    )
    parser.add_argument(
        "--pfail-values",
        type=float,
        nargs="+",
        default=None,
        help="Override per-episode pfail grid.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of training episodes (default: from config).",
    )
    parser.add_argument(
        "--hard-regime",
        action="store_true",
        help="Use hard-regime grids and 8000 episodes (overridden by --episodes if set).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for checkpoints (default: training.checkpoint_dir from config).",
    )
    parser.add_argument(
        "--log-episode-spread",
        action="store_true",
        help="Log PR(degree)-PR(random) per episode and summary stats (diagnostic).",
    )
    parser.add_argument(
        "--log-grad-norm",
        action="store_true",
        help="After each optimizer step, print sum of per-parameter grad L2 norms (diagnostic).",
    )
    parser.add_argument(
        "--validation-eval-set",
        type=Path,
        default=None,
        help="If set, run periodic validation on this pickle (e.g. eval_sets/ds_validation.pkl).",
    )
    parser.add_argument(
        "--validation-every",
        type=int,
        default=None,
        help="Override validation interval (episodes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    training_config = build_training_config(config, episodes_override=None)

    if args.alpha is not None:
        training_config = replace(training_config, alpha=args.alpha)
    if args.pfail is not None:
        training_config = replace(training_config, pfail=args.pfail)
    if args.alpha_values is not None:
        training_config = replace(training_config, alpha_values=tuple(args.alpha_values))
    if args.pfail_values is not None:
        training_config = replace(training_config, pfail_values=tuple(args.pfail_values))
    if args.hard_regime:
        training_config = replace(
            training_config,
            alpha=0.10,
            pfail=0.15,
            alpha_values=(0.10, 0.15, 0.20),
            pfail_values=(0.10, 0.15, 0.20),
            num_episodes=8000,
        )
    if args.episodes is not None:
        training_config = replace(training_config, num_episodes=args.episodes)
    if args.log_episode_spread:
        training_config = replace(training_config, log_episode_spread=True)
    if args.log_grad_norm:
        training_config = replace(training_config, log_grad_norm=True)
    if args.validation_eval_set is not None:
        ves = args.validation_eval_set
        if not ves.is_absolute():
            ves = ROOT / ves
        training_config = replace(
            training_config, validation_eval_set_path=str(ves.resolve())
        )
    if args.validation_every is not None:
        training_config = replace(
            training_config, validation_every=int(args.validation_every)
        )
    if args.checkpoint_dir is not None:
        training_config = replace(training_config, checkpoint_dir=args.checkpoint_dir)

    _, training_state, checkpoint_path = train_recovery_agent(training_config)

    summary_path = checkpoint_path.with_suffix(".summary.json")
    summary = {
        "checkpoint_path": str(checkpoint_path),
        "training_config": training_config_for_json(training_config),
        "num_episodes": training_config.num_episodes,
        "alpha_values": list(training_config.alpha_values),
        "pfail_values": list(training_config.pfail_values),
        "env": {
            "capacity_noise": training_config.capacity_noise,
            "failure_bias": training_config.failure_bias,
            "action_space": training_config.action_space,
            "obs_hops": training_config.obs_hops,
        },
        "final_reward_mean_last_10": (
            sum(training_state.episode_rewards[-10:])
            / max(1, len(training_state.episode_rewards[-10:]))
        ),
        "final_anc_mean_last_10": (
            sum(training_state.episode_final_anc[-10:])
            / max(1, len(training_state.episode_final_anc[-10:]))
        ),
        "final_loss_mean_last_10": (
            sum(training_state.losses[-10:]) / max(1, len(training_state.losses[-10:]))
        ),
        "num_updates": len(training_state.losses),
        "total_steps": training_state.total_steps,
        "validation_history": training_state.validation_history,
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved training summary to {summary_path}")


if __name__ == "__main__":
    main()
