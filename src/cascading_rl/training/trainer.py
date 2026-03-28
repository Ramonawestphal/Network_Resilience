from __future__ import annotations

from collections.abc import Iterable
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
from cascading_rl.training.replay import ReplayBuffer, Transition


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    device: str = "cpu"
    alpha: float = 0.2
    pfail: float = 0.1
    alpha_values: tuple[float, ...] = (0.05, 0.1)
    pfail_values: tuple[float, ...] = (0.10, 0.15, 0.20)
    budget: int = 2
    max_rounds: int = 5
    n_range: tuple[int, int] = (30, 50)
    m: int = 2
    num_episodes: int = 8000
    replay_capacity: int = 20000
    warmup_transitions: int = 500
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 6000
    target_update_interval: int = 200
    hidden_dim: int = 64
    embed_dim: int = 64
    num_layers: int = 2
    validation_graphs: int = 20
    validation_seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
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
    print(
        "\n"
        f"[validation] ep={validation['episode']} "
        f"final_anc={validation['final_anc_mean']:.3f}±{validation['final_anc_stderr']:.3f} "
        f"threshold_hit={validation['threshold_hit_mean']:.3f} "
        f"rounds={validation['rounds_mean']:.3f}",
        flush=True,
    )


def compute_dqn_loss(
    model: RecoveryQNetwork,
    target_model: RecoveryQNetwork,
    transitions: list[Transition],
    *,
    gamma: float,
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
            if not transition.done and transition.next_observation.failed:
                next_tensor = observation_to_graph_tensor(transition.next_observation, device=device)
                next_q_values = target_model(next_tensor)
                targets.append(reward_t + gamma * torch.max(next_q_values))
            else:
                targets.append(reward_t)

    return F.smooth_l1_loss(torch.stack(q_selected), torch.stack(targets))


def validate_policy(
    model: RecoveryQNetwork,
    config: TrainingConfig,
    *,
    device: torch.device,
    graph_seed_offset: int,
) -> dict[str, Any]:
    validation_graphs = make_graph_batch(
        num_graphs=config.validation_graphs,
        n_range=config.n_range,
        m=config.m,
        seed=config.seed + graph_seed_offset,
    )
    policy = build_greedy_policy(model, device=device)
    summaries = evaluate_policy_factories_on_graphs(
        validation_graphs,
        {"rl": lambda graph_index, seed: policy},
        alpha=config.alpha,
        pfail=config.pfail,
        budget=config.budget,
        max_rounds=config.max_rounds,
        seeds=config.validation_seeds,
        tau=0.8,
    )
    summary = summaries["rl"]
    return {
        "final_anc_mean": summary.final_anc.mean,
        "final_anc_stderr": summary.final_anc.stderr,
        "threshold_hit_mean": summary.threshold_hit_fraction.mean,
        "rounds_mean": summary.rounds.mean,
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

    checkpoint_path = Path(config.checkpoint_dir) / config.checkpoint_name

    for episode in range(config.num_episodes):
        epsilon = epsilon_for_episode(config, episode)
        alpha = float(rng.choice(alpha_values))
        pfail = float(rng.choice(pfail_values))
        graph_size = rng.randint(config.n_range[0], config.n_range[1])
        graph_seed = rng.randint(0, 10**9)
        graph = make_ba_graph(n=graph_size, m=config.m, seed=graph_seed)
        env = RecoveryEnv(
            graph,
            alpha=alpha,
            pfail=pfail,
            budget=config.budget,
            max_rounds=config.max_rounds,
            seed=graph_seed,
        )

        observation = _reset_with_non_empty_failures(env, graph_seed, rng)
        done = False
        total_reward = 0.0

        while not done and observation.failed:
            action = select_action(
                model,
                observation,
                epsilon=epsilon,
                rng=rng,
                device=device,
            )
            next_observation, reward, done, _ = env.step(action)
            replay_buffer.push(
                Transition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=done,
                )
            )
            observation = next_observation
            total_reward += reward
            training_state.total_steps += 1

            if len(replay_buffer) >= max(config.batch_size, config.warmup_transitions):
                batch = replay_buffer.sample(config.batch_size, rng=rng)
                loss = compute_dqn_loss(
                    model,
                    target_model,
                    batch,
                    gamma=config.gamma,
                    device=device,
                )
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_state.losses.append(float(loss.item()))

                if training_state.total_steps % config.target_update_interval == 0:
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
                graph_seed_offset=episode + 1,
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
