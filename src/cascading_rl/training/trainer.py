from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from cascading_rl.envs.recovery import RecoveryEnv
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
    budget: int = 2
    max_rounds: int = 5
    n_range: tuple[int, int] = (30, 50)
    m: int = 2
    num_episodes: int = 120
    replay_capacity: int = 5000
    warmup_transitions: int = 64
    batch_size: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-3
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 100
    target_update_interval: int = 50
    hidden_dim: int = 64
    embed_dim: int = 64
    num_layers: int = 2
    validation_graphs: int = 2
    validation_seeds: tuple[int, ...] = (0, 1, 2)
    validation_every: int = 20
    checkpoint_dir: str = "experiments/learner"
    checkpoint_name: str = "recovery_q.pt"


@dataclass
class TrainingState:
    episode_rewards: list[float] = field(default_factory=list)
    episode_final_anc: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    validation_history: list[dict] = field(default_factory=list)
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


def compute_dqn_loss(
    model: RecoveryQNetwork,
    target_model: RecoveryQNetwork,
    transitions: list[Transition],
    *,
    gamma: float,
    device: torch.device,
) -> torch.Tensor:
    losses = []
    for transition in transitions:
        graph_tensor = observation_to_graph_tensor(transition.observation, device=device)
        q_values = model(graph_tensor)
        action_index = graph_tensor.node_to_index[transition.action]
        q_sa = q_values[action_index]

        with torch.no_grad():
            target_value = torch.tensor(float(transition.reward), device=device)
            if not transition.done and transition.next_observation.failed:
                next_tensor = observation_to_graph_tensor(transition.next_observation, device=device)
                next_q_values = target_model(next_tensor)
                target_value = target_value + gamma * torch.max(next_q_values)

        losses.append(F.smooth_l1_loss(q_sa, target_value))

    return torch.stack(losses).mean()


def validate_policy(
    model: RecoveryQNetwork,
    config: TrainingConfig,
    *,
    device: torch.device,
    graph_seed_offset: int,
) -> dict:
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
                "losses": training_state.losses,
                "validation_history": training_state.validation_history,
                "total_steps": training_state.total_steps,
            },
        },
        output_path,
    )
    return output_path


def train_recovery_agent(config: TrainingConfig) -> tuple[RecoveryQNetwork, TrainingState, Path]:
    device = resolve_device(config.device)
    rng = Random(config.seed)
    torch.manual_seed(config.seed)

    model = RecoveryQNetwork(build_model_config(config)).to(device)
    target_model = deepcopy(model).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_capacity)
    training_state = TrainingState()

    for episode in range(config.num_episodes):
        epsilon = epsilon_for_episode(config, episode)
        graph_size = rng.randint(config.n_range[0], config.n_range[1])
        graph_seed = rng.randint(0, 10**9)
        graph = make_ba_graph(n=graph_size, m=config.m, seed=graph_seed)
        env = RecoveryEnv(
            graph,
            alpha=config.alpha,
            pfail=config.pfail,
            budget=config.budget,
            max_rounds=config.max_rounds,
            seed=graph_seed,
        )

        observation = env.reset(seed=graph_seed)
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
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_state.losses.append(float(loss.item()))

                if training_state.total_steps % config.target_update_interval == 0:
                    target_model.load_state_dict(model.state_dict())

        training_state.episode_rewards.append(total_reward)
        training_state.episode_final_anc.append(env.current_anc())

        if (episode + 1) % config.validation_every == 0:
            validation = validate_policy(
                model,
                config,
                device=device,
                graph_seed_offset=episode + 1,
            )
            validation["episode"] = episode + 1
            training_state.validation_history.append(validation)

    checkpoint_path = Path(config.checkpoint_dir) / config.checkpoint_name
    saved_path = save_checkpoint(
        model,
        config,
        training_state,
        checkpoint_path,
        episode=config.num_episodes,
    )
    return model, training_state, saved_path
