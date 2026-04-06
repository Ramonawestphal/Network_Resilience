from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
import sys

import torch
from torch.nn import functional as F
from torch.optim import Adam

from cascading_rl.envs.recovery import RecoveryEnv
from cascading_rl.evaluation import evaluate_policy_factories_on_graphs
from cascading_rl.graph.generation import make_ba_graph, make_graph_batch
from cascading_rl.models import (
    FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    QNetworkConfig,
    RecoveryQNetwork,
    build_greedy_policy,
    select_top_b,
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
    use_global_features: bool = True
    active_node_features: tuple[str, ...] = FEATURE_NAMES
    active_global_features: tuple[str, ...] = GLOBAL_FEATURE_NAMES
    use_virtual_node: bool = False
    validation_graphs: int = 2
    validation_seeds: tuple[int, ...] = (0, 1, 2)
    validation_every: int = 20
    checkpoint_dir: str = "experiments/learner"
    checkpoint_name: str = "recovery_q.pt"
    episode_graph_specs: tuple[tuple[int, int], ...] | None = None


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
        use_global_features=config.use_global_features,
        active_node_features=config.active_node_features,
        active_global_features=config.active_global_features,
        use_virtual_node=config.use_virtual_node,
    )


def _mean_recent(values: list[float], window: int = 10) -> float:
    if not values:
        return 0.0
    recent = values[-window:]
    return sum(recent) / len(recent)


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


def _print_validation_update(validation: dict) -> None:
    print(
        "\n"
        f"[validation] ep={validation['episode']} "
        f"final_anc={validation['final_anc_mean']:.3f}+/-{validation['final_anc_stderr']:.3f} "
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
    budget: int,
) -> torch.Tensor:
    losses = []
    for transition in transitions:
        graph_tensor, q_values = model.score_observation(
            transition.observation,
            device=device,
        )

        # mean Q over selected actions — each node in the batch contributed equally
        action_indices = [
            graph_tensor.node_to_index[a] for a in transition.action
        ]
        q_selected = torch.stack([q_values[i] for i in action_indices]).mean()


        with torch.no_grad():
            target_value = torch.tensor(float(transition.reward), device=device)
            if not transition.done and transition.next_observation.failed:
                next_tensor, next_q = target_model.score_observation(
                    transition.next_observation,
                    device=device,
                )
                # target: mean of top-B Q-values in next state
                valid_next = [
                    next_q[next_tensor.node_to_index[n]].item()
                    for n in transition.next_observation.failed
                ]
                top_b_next = sorted(valid_next, reverse=True)[:budget]
                target_value = target_value + gamma * torch.tensor(
                    sum(top_b_next) / len(top_b_next), device=device
                )

        losses.append(F.smooth_l1_loss(q_selected, target_value))

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
    if config.episode_graph_specs is not None and len(config.episode_graph_specs) != config.num_episodes:
        raise ValueError("episode_graph_specs length must match num_episodes.")

    model = RecoveryQNetwork(build_model_config(config)).to(device)
    target_model = deepcopy(model).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    replay_buffer = ReplayBuffer(config.replay_capacity)
    training_state = TrainingState()

    for episode in range(config.num_episodes):
        epsilon = epsilon_for_episode(config, episode)
        if config.episode_graph_specs is not None:
            graph_size, graph_seed = config.episode_graph_specs[episode]
        else:
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
            epsilon = epsilon_for_episode(config, episode)

            actions = select_top_b(
                model,
                observation,
                budget=config.budget,
                epsilon=epsilon,
                rng=rng,
                device=device,
            )
            next_observation, reward, done, info = env.step_batch(actions)
            
            # store as single transition (one per round, not per reactivation)
            replay_buffer.push(
                Transition(
                    observation=observation,
                    action=actions,          # list of nodes now
                    reward=reward,
                    next_observation=next_observation,
                    done=done,
                )
            )

            observation = next_observation
            total_reward += reward
            training_state.total_steps += 1

            # learning step unchanged
            if len(replay_buffer) >= max(config.batch_size, config.warmup_transitions):
                batch = replay_buffer.sample(config.batch_size, rng=rng)
                loss = compute_dqn_loss(model, target_model, batch, gamma=config.gamma, device=device, budget=config.budget,)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                training_state.losses.append(float(loss.item()))

                if training_state.total_steps % config.target_update_interval == 0:
                    target_model.load_state_dict(model.state_dict())

        training_state.episode_rewards.append(total_reward)
        training_state.episode_final_anc.append(env.current_anc())
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

    checkpoint_path = Path(config.checkpoint_dir) / config.checkpoint_name
    print("", file=sys.stdout, flush=True)
    saved_path = save_checkpoint(
        model,
        config,
        training_state,
        checkpoint_path,
        episode=config.num_episodes,
    )
    return model, training_state, saved_path
