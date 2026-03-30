from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Any
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from cascading_rl.budgeting import compute_scaled_budget
from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.evaluation import evaluate_policy_factories_on_graphs
from cascading_rl.graph.generation import make_ba_graph, make_graph_batch
from cascading_rl.models import (
    QNetworkConfig,
    RecoveryQNetwork,
    build_greedy_policy,
    observation_to_graph_tensor,
    select_action,
)
from cascading_rl.policies.degree_policy import choose_highest_degree_failed_node
from cascading_rl.training.replay import ReplayBuffer, Transition

Node = Hashable


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    device: str = "cpu"
    alpha: float = 0.10
    pfail: float = 0.15
    alpha_values: tuple[float, ...] = (0.10, 0.15, 0.20)
    pfail_values: tuple[float, ...] = (0.10, 0.15, 0.20)
    budget: int = 2
    scale_budget: bool = True
    budget_reference_n: int = 40
    max_rounds: int = 5
    capacity_noise: float = 0.0
    failure_bias: str = "uniform"
    action_space: str = "failed"
    obs_hops: int | None = None
    n_range: tuple[int, int] = (30, 50)
    m: int = 2
    num_episodes: int = 8000
    replay_capacity: int = 10000
    warmup_transitions: int = 500
    batch_size: int = 64
    gamma: float = 0.99
    use_monte_carlo_returns: bool = True
    learning_rate: float = 3e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 6000
    target_update_interval: int = 200
    hidden_dim: int = 128
    embed_dim: int = 128
    num_layers: int = 2
    use_imitation_warmstart: bool = False
    imitation_graphs: int = 500
    imitation_seeds: int = 5
    imitation_epochs: int = 10
    validation_graphs: int = 2
    validation_seeds: tuple[int, ...] = (0, 1, 2)
    validation_seed: int = 42
    validation_every: int = 200
    checkpoint_dir: str = "experiments/learner"
    checkpoint_name: str = "recovery_q.pt"


@dataclass
class TrainingState:
    episode_rewards: list[float] = field(default_factory=list)
    episode_final_anc: list[float] = field(default_factory=list)
    episode_alpha: list[float] = field(default_factory=list)
    episode_pfail: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    validation_history: list[dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0


@dataclass(frozen=True)
class ImitationSample:
    observation: RecoveryObservation
    action: Node


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def epsilon_for_episode(config: TrainingConfig, episode: int) -> float:
    if config.epsilon_decay_episodes <= 1:
        return config.epsilon_end
    progress = min(1.0, episode / max(1, config.epsilon_decay_episodes - 1))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def build_model_config(config: TrainingConfig) -> QNetworkConfig:
    return QNetworkConfig(
        hidden_dim=config.hidden_dim,
        embed_dim=config.embed_dim,
        num_layers=config.num_layers,
    )


def _mean_recent(values: list[float], window: int = 10) -> float:
    if not values:
        return 0.0
    recent = values[-window:]
    return sum(recent) / len(recent)


def _reset_with_non_empty_failures(
    env: RecoveryEnv,
    base_seed: int,
    rng: Random,
    *,
    max_attempts: int = 1024,
) -> RecoveryObservation:
    """Reset until at least one node failed; Bernoulli pfail can yield an empty episode otherwise."""
    seed = base_seed
    for _ in range(max_attempts):
        observation = env.reset(seed=seed)
        if observation.failed:
            return observation
        seed = rng.randint(0, 10**9)
    raise RuntimeError(
        f"After {max_attempts} reset attempts, no episode started with failed nodes "
        f"(pfail={env.pfail}, n_nodes={env.base_graph.number_of_nodes()}). "
        "Training requires stochastic failures or a positive pfail."
    )


def _render_progress_line(
    episode: int,
    total_episodes: int,
    *,
    epsilon: float,
    training_state: TrainingState,
    bar_width: int = 28,
) -> str:
    completed = episode + 1
    progress = completed / max(1, total_episodes)
    filled = int(bar_width * progress)
    bar = "#" * filled + "-" * (bar_width - filled)
    recent_reward = _mean_recent(training_state.episode_rewards)
    recent_anc = _mean_recent(training_state.episode_final_anc)
    recent_loss = _mean_recent(training_state.losses)
    return (
        f"\r[{bar}] {completed:>4}/{total_episodes} "
        f"eps={epsilon:.3f} "
        f"reward10={recent_reward:.3f} "
        f"anc10={recent_anc:.3f} "
        f"loss10={recent_loss:.4f}"
    )


def _print_validation_update(validation: dict[str, Any]) -> None:
    reference = validation["reference"]
    per_alpha = validation["per_alpha_anc"]
    per_alpha_text = "  ".join(
        f"{alpha:.2f}->{mean:.3f}" for alpha, mean in sorted(per_alpha.items())
    )
    print(
        "\n"
        f"[validation] ep={validation['episode']} "
        f"final_anc={reference['final_anc_mean']:.3f}±{reference['final_anc_stderr']:.3f} "
        f"threshold_hit={reference['threshold_hit_mean']:.3f} "
        f"rounds={reference['rounds_mean']:.3f}\n"
        f"per-alpha: {per_alpha_text}",
        flush=True,
    )


def compute_dqn_loss(
    model: RecoveryQNetwork,
    target_model: RecoveryQNetwork,
    transitions: list[Transition],
    *,
    gamma: float,
    use_monte_carlo_returns: bool,
    device: torch.device,
) -> torch.Tensor:
    """Per-graph forwards; stack predictions and targets; single smooth_l1 over the batch."""
    q_selected: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for transition in transitions:
        graph_tensor = observation_to_graph_tensor(transition.observation, device=device)
        q_values = model(graph_tensor)
        action_index = graph_tensor.node_to_index[transition.action]
        q_selected.append(q_values[action_index])

        with torch.no_grad():
            reward_t = torch.tensor(float(transition.reward), device=device, dtype=torch.float32)
            if use_monte_carlo_returns:
                targets.append(reward_t)
            elif not transition.done and transition.next_observation.failed:
                next_tensor = observation_to_graph_tensor(transition.next_observation, device=device)
                next_q_values = target_model(next_tensor)
                targets.append(reward_t + gamma * torch.max(next_q_values))
            else:
                targets.append(reward_t)

    return F.smooth_l1_loss(torch.stack(q_selected), torch.stack(targets))


def _env_kwargs_from_config(config: TrainingConfig) -> dict[str, Any]:
    return {
        "capacity_noise": config.capacity_noise,
        "failure_bias": config.failure_bias,
        "action_space": config.action_space,
        "obs_hops": config.obs_hops,
    }


def _resolve_budget_for_graph(config: TrainingConfig, graph: Any) -> int:
    return compute_scaled_budget(
        config.budget,
        num_nodes=graph.number_of_nodes(),
        reference_n=config.budget_reference_n,
        enabled=config.scale_budget,
    )


def _compute_discounted_returns(
    transitions: Sequence[Transition],
    *,
    gamma: float,
) -> list[float]:
    returns: list[float] = []
    discounted_return = 0.0
    for transition in reversed(transitions):
        discounted_return = float(transition.reward) + gamma * discounted_return
        returns.append(discounted_return)
    returns.reverse()
    return returns


def generate_imitation_data(
    graphs: Sequence[Any],
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int,
    num_seeds: int,
    policy: Callable[[RecoveryObservation], Node],
    *,
    env_kwargs: dict[str, Any] | None = None,
    base_seed: int = 0,
    scale_budget: bool = False,
    budget_reference_n: int = 40,
) -> list[ImitationSample]:
    if num_seeds < 1:
        raise ValueError("num_seeds must be at least 1.")

    env_kwargs = env_kwargs or {}
    rng = Random(base_seed)
    samples: list[ImitationSample] = []
    for graph_index, graph in enumerate(graphs):
        resolved_budget = compute_scaled_budget(
            budget,
            num_nodes=graph.number_of_nodes(),
            reference_n=budget_reference_n,
            enabled=scale_budget,
        )
        for seed_offset in range(num_seeds):
            rollout_seed = base_seed + graph_index * 10_000 + seed_offset
            env = RecoveryEnv(
                graph,
                alpha=alpha,
                pfail=pfail,
                budget=resolved_budget,
                max_rounds=max_rounds,
                seed=rollout_seed,
                **env_kwargs,
            )
            observation = _reset_with_non_empty_failures(env, rollout_seed, rng)
            done = False
            while not done and observation.failed:
                action = policy(observation)
                samples.append(ImitationSample(observation=observation, action=action))
                observation, _reward, done, _info = env.step(action)
    return samples


def pretrain_by_imitation(
    model: RecoveryQNetwork,
    samples: Sequence[ImitationSample],
    *,
    lr: float = 1e-3,
    epochs: int = 10,
    batch_size: int = 64,
) -> tuple[RecoveryQNetwork, list[float]]:
    if not samples:
        raise ValueError("samples must be non-empty.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    device = next(model.parameters()).device
    optimizer = Adam(model.parameters(), lr=lr)
    epoch_losses: list[float] = []

    for _ in range(epochs):
        permutation = torch.randperm(len(samples)).tolist()
        weighted_loss_sum = 0.0
        sample_count = 0
        model.train()
        for batch_start in range(0, len(permutation), batch_size):
            batch_indices = permutation[batch_start : batch_start + batch_size]
            batch_losses: list[torch.Tensor] = []
            for sample_index in batch_indices:
                sample = samples[sample_index]
                graph_tensor = observation_to_graph_tensor(sample.observation, device=device)
                logits = model(graph_tensor)
                action_index = graph_tensor.node_to_index[sample.action]
                target = torch.tensor([action_index], device=device, dtype=torch.long)
                batch_losses.append(F.cross_entropy(logits.unsqueeze(0), target))
            loss = torch.stack(batch_losses).mean()
            optimizer.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weighted_loss_sum += float(loss.item()) * len(batch_indices)
            sample_count += len(batch_indices)
        epoch_losses.append(weighted_loss_sum / max(1, sample_count))

    model.eval()
    return model, epoch_losses


def _imitation_agreement_rate(
    model: RecoveryQNetwork,
    samples: Sequence[ImitationSample],
    *,
    device: torch.device,
) -> float:
    if not samples:
        return 0.0

    agreements = 0
    for sample in samples:
        predicted_action = select_action(
            model,
            sample.observation,
            epsilon=0.0,
            rng=Random(0),
            device=device,
        )
        if predicted_action == sample.action:
            agreements += 1
    return agreements / len(samples)


def validate_policy(
    model: RecoveryQNetwork,
    config: TrainingConfig,
    *,
    device: torch.device,
    validation_graphs: Sequence[Any],
) -> dict[str, Any]:
    policy = build_greedy_policy(model, device=device)
    env_kwargs = _env_kwargs_from_config(config)
    reference_summaries = evaluate_policy_factories_on_graphs(
        validation_graphs,
        {"rl": lambda graph_index, seed: policy},
        alpha=config.alpha,
        pfail=config.pfail,
        budget=config.budget,
        max_rounds=config.max_rounds,
        seeds=config.validation_seeds,
        tau=0.8,
        env_kwargs=env_kwargs,
        scale_budget=config.scale_budget,
        reference_n=config.budget_reference_n,
    )
    reference_summary = reference_summaries["rl"]
    grid_cells: list[dict[str, float]] = []
    for alpha in config.alpha_values:
        for pfail in config.pfail_values:
            grid_summary = evaluate_policy_factories_on_graphs(
                validation_graphs,
                {"rl": lambda graph_index, seed: policy},
                alpha=alpha,
                pfail=pfail,
                budget=config.budget,
                max_rounds=config.max_rounds,
                seeds=config.validation_seeds,
                tau=0.8,
                env_kwargs=env_kwargs,
                scale_budget=config.scale_budget,
                reference_n=config.budget_reference_n,
            )["rl"]
            grid_cells.append(
                {
                    "alpha": alpha,
                    "pfail": pfail,
                    "final_anc_mean": grid_summary.final_anc.mean,
                    "threshold_hit_mean": grid_summary.threshold_hit_fraction.mean,
                    "rounds_mean": grid_summary.rounds.mean,
                }
            )
    grid_final_anc_mean = sum(cell["final_anc_mean"] for cell in grid_cells) / len(grid_cells)
    grid_threshold_hit_mean = sum(cell["threshold_hit_mean"] for cell in grid_cells) / len(grid_cells)
    grid_rounds_mean = sum(cell["rounds_mean"] for cell in grid_cells) / len(grid_cells)
    per_alpha_anc: dict[float, float] = {}
    for alpha in config.alpha_values:
        per_alpha_summary = evaluate_policy_factories_on_graphs(
            validation_graphs,
            {"rl": lambda graph_index, seed: policy},
            alpha=alpha,
            pfail=config.pfail,
            budget=config.budget,
            max_rounds=config.max_rounds,
            seeds=config.validation_seeds,
            tau=0.8,
            env_kwargs=env_kwargs,
            scale_budget=config.scale_budget,
            reference_n=config.budget_reference_n,
        )["rl"]
        per_alpha_anc[float(alpha)] = per_alpha_summary.final_anc.mean
    return {
        "final_anc_mean": reference_summary.final_anc.mean,
        "final_anc_stderr": reference_summary.final_anc.stderr,
        "threshold_hit_mean": reference_summary.threshold_hit_fraction.mean,
        "rounds_mean": reference_summary.rounds.mean,
        "reference": {
            "alpha": config.alpha,
            "pfail": config.pfail,
            "budget": config.budget,
            "scale_budget": config.scale_budget,
            "budget_reference_n": config.budget_reference_n,
            "max_rounds": config.max_rounds,
            "final_anc_mean": reference_summary.final_anc.mean,
            "final_anc_stderr": reference_summary.final_anc.stderr,
            "threshold_hit_mean": reference_summary.threshold_hit_fraction.mean,
            "rounds_mean": reference_summary.rounds.mean,
        },
        "grid": {
            "alpha_values": list(config.alpha_values),
            "pfail_values": list(config.pfail_values),
            "cell_count": len(grid_cells),
            "final_anc_mean": grid_final_anc_mean,
            "threshold_hit_mean": grid_threshold_hit_mean,
            "rounds_mean": grid_rounds_mean,
            "cells": grid_cells,
        },
        "per_alpha_anc": per_alpha_anc,
        "env": env_kwargs,
    }


def save_checkpoint(
    model: RecoveryQNetwork,
    config: TrainingConfig,
    training_state: TrainingState,
    output_path: str | Path,
    *,
    episode: int,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "model_state": model.state_dict(),
            "model_config": asdict(model.config),
            "training_config": asdict(config),
            "training_state": {
                "episode_rewards": training_state.episode_rewards,
                "episode_final_anc": training_state.episode_final_anc,
                "episode_alpha": training_state.episode_alpha,
                "episode_pfail": training_state.episode_pfail,
                "losses": training_state.losses,
                "validation_history": training_state.validation_history,
                "total_steps": training_state.total_steps,
            },
        },
        output_path,
    )
    return output_path


def train_recovery_agent(config: TrainingConfig) -> tuple[RecoveryQNetwork, TrainingState, Path]:
    assert epsilon_for_episode(config, 100) > 0.5, "epsilon decays too fast"
    assert epsilon_for_episode(config, 6000) < 0.1, "epsilon decays too slow"

    device = resolve_device(config.device)
    rng = Random(config.seed)
    torch.manual_seed(config.seed)

    model = RecoveryQNetwork(build_model_config(config)).to(device)
    target_model = deepcopy(model).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_capacity)
    training_state = TrainingState()

    alpha_values = tuple(config.alpha_values) if config.alpha_values else (config.alpha,)
    pfail_values = tuple(config.pfail_values) if config.pfail_values else (config.pfail,)
    if not alpha_values or not pfail_values:
        raise ValueError("alpha_values and pfail_values must be non-empty.")
    validation_graphs = make_graph_batch(
        num_graphs=config.validation_graphs,
        n_range=config.n_range,
        m=config.m,
        seed=config.validation_seed,
    )
    regime_combinations = [
        (float(alpha), float(pfail)) for alpha in alpha_values for pfail in pfail_values
    ]

    if config.use_imitation_warmstart:
        imitation_graphs = make_graph_batch(
            num_graphs=config.imitation_graphs,
            n_range=config.n_range,
            m=config.m,
            seed=config.seed + 10_000,
        )
        imitation_samples = generate_imitation_data(
            imitation_graphs,
            alpha=config.alpha,
            pfail=config.pfail,
            budget=config.budget,
            max_rounds=config.max_rounds,
            num_seeds=config.imitation_seeds,
            policy=choose_highest_degree_failed_node,
            env_kwargs=_env_kwargs_from_config(config),
            base_seed=config.seed + 20_000,
            scale_budget=config.scale_budget,
            budget_reference_n=config.budget_reference_n,
        )
        model, imitation_losses = pretrain_by_imitation(
            model,
            imitation_samples,
            lr=1e-3,
            epochs=config.imitation_epochs,
            batch_size=config.batch_size,
        )
        target_model.load_state_dict(model.state_dict())
        for epoch_index, loss in enumerate(imitation_losses, start=1):
            print(
                f"[imitation] epoch={epoch_index}/{config.imitation_epochs} loss={loss:.4f}",
                flush=True,
            )
        print(
            "Imitation pre-training complete. "
            f"Final epoch loss: {imitation_losses[-1]:.4f}",
            flush=True,
        )

    checkpoint_path = Path(config.checkpoint_dir) / config.checkpoint_name

    for episode in range(config.num_episodes):
        epsilon = epsilon_for_episode(config, episode)
        cycle_index = episode % len(regime_combinations)
        if cycle_index == 0 and episode > 0:
            rng.shuffle(regime_combinations)
        alpha, pfail = regime_combinations[cycle_index]
        graph_size = rng.randint(config.n_range[0], config.n_range[1])
        graph_seed = rng.randint(0, 10**9)
        graph = make_ba_graph(n=graph_size, m=config.m, seed=graph_seed)
        resolved_budget = _resolve_budget_for_graph(config, graph)
        env = RecoveryEnv(
            graph,
            alpha=alpha,
            pfail=pfail,
            budget=resolved_budget,
            max_rounds=config.max_rounds,
            seed=graph_seed,
            capacity_noise=config.capacity_noise,
            failure_bias=config.failure_bias,
            action_space=config.action_space,
            obs_hops=config.obs_hops,
        )

        observation = _reset_with_non_empty_failures(env, graph_seed, rng)
        done = False
        total_reward = 0.0
        episode_transitions: list[Transition] = []
        episode_step_totals: list[int] = []

        while not done and observation.failed:
            action = select_action(
                model,
                observation,
                epsilon=epsilon,
                rng=rng,
                device=device,
            )
            next_observation, reward, done, _ = env.step(action)
            transition = Transition(
                observation=observation,
                action=action,
                reward=reward,
                next_observation=next_observation,
                done=done,
            )
            episode_transitions.append(transition)
            observation = next_observation
            total_reward += reward
            training_state.total_steps += 1
            episode_step_totals.append(training_state.total_steps)

        if config.use_monte_carlo_returns:
            transition_rewards = _compute_discounted_returns(
                episode_transitions,
                gamma=config.gamma,
            )
        else:
            transition_rewards = [float(transition.reward) for transition in episode_transitions]

        for transition, stored_reward, step_total in zip(
            episode_transitions,
            transition_rewards,
            episode_step_totals,
        ):
            replay_buffer.push(
                Transition(
                    observation=transition.observation,
                    action=transition.action,
                    reward=stored_reward,
                    next_observation=transition.next_observation,
                    done=transition.done,
                )
            )

            if len(replay_buffer) >= max(config.batch_size, config.warmup_transitions):
                batch = replay_buffer.sample(config.batch_size, rng=rng)
                loss = compute_dqn_loss(
                    model,
                    target_model,
                    batch,
                    gamma=config.gamma,
                    use_monte_carlo_returns=config.use_monte_carlo_returns,
                    device=device,
                )
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_state.losses.append(float(loss.item()))

                if step_total % config.target_update_interval == 0:
                    target_model.load_state_dict(model.state_dict())

        training_state.episode_rewards.append(total_reward)
        training_state.episode_final_anc.append(env.current_anc())
        training_state.episode_alpha.append(alpha)
        training_state.episode_pfail.append(pfail)
        progress_line = _render_progress_line(
            episode,
            config.num_episodes,
            epsilon=epsilon,
            training_state=training_state,
        )
        print(progress_line, end="", flush=True)

        if (episode + 1) % config.validation_every == 0:
            validation = validate_policy(
                model,
                config,
                device=device,
                validation_graphs=validation_graphs,
            )
            validation["episode"] = episode + 1
            training_state.validation_history.append(validation)
            _print_validation_update(validation)
            save_checkpoint(
                model,
                config,
                training_state,
                checkpoint_path,
                episode=episode + 1,
            )
            print(
                _render_progress_line(
                    episode,
                    config.num_episodes,
                    epsilon=epsilon,
                    training_state=training_state,
                ),
                end="",
                flush=True,
            )

    print("", file=sys.stdout, flush=True)
    saved_path = save_checkpoint(
        model,
        config,
        training_state,
        checkpoint_path,
        episode=config.num_episodes,
    )
    return model, training_state, saved_path
