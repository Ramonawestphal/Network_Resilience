from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from typing import Any
import sys
import warnings

import torch
from torch.nn import functional as F
from torch.optim import Adam

from cascading_rl.budgeting import compute_scaled_budget, compute_scaled_max_rounds
from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.evaluation import evaluate_policy_factories_on_graphs
from cascading_rl.graph.generation import make_ba_graph, make_graph_batch
from cascading_rl.models import (
    FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    QNetworkConfig,
    RecoveryQNetwork,
    build_greedy_policy,
    observation_to_global_features,
    observation_to_graph_tensor,
    select_action,
    select_top_b,
)
from cascading_rl.training.replay import ReplayBuffer, Transition

Node = Hashable

GRAPH_BUFFER_MAXLEN = 20
# Large offset ensures graph-generation seeds are statistically independent from
# episode/failure seeds (which start near 0), avoiding accidental seed collisions.
FREEZE_GRAPH_SPECS_SEED_OFFSET = 20_000


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7
    device: str = "cpu"
    alpha: float = 0.15
    pfail: float = 0.18
    alpha_values: tuple[float, ...] = (0.15,)
    pfail_values: tuple[float, ...] = (0.18,)
    budget: int = 3
    scale_budget: bool = True
    budget_reference_n: int = 40
    max_rounds: int = 20
    scale_max_rounds: bool = True
    capacity_noise: float = 0.0
    failure_bias: str = "uniform"
    action_space: str = "failed"
    obs_hops: int | None = None
    abandonment_nc_threshold: float | None = None
    n_range: tuple[int, int] = (30, 50)
    m: int = 2
    num_episodes: int = 10000
    replay_capacity: int = 10000
    warmup_transitions: int = 500
    batch_size: int = 64
    gamma: float = 0.99
    use_monte_carlo_returns: bool = False
    learning_rate: float = 3e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 10000
    target_update_interval: int = 200
    hidden_dim: int = 128
    embed_dim: int = 128
    num_layers: int = 2
    use_global_features: bool = False
    active_node_features: tuple[str, ...] = FEATURE_NAMES
    active_global_features: tuple[str, ...] = GLOBAL_FEATURE_NAMES
    use_virtual_node: bool = False
    use_imitation_warmstart: bool = False
    imitation_graphs: int = 500
    imitation_seeds: int = 5
    imitation_epochs: int = 10
    validation_graphs: int = 2
    validation_seeds: tuple[int, ...] = (0, 1, 2)
    validation_seed: int = 42
    validation_every: int = 200
    validation_tau: float = 0.8
    checkpoint_dir: str = "experiments/learner"
    checkpoint_name: str = "recovery_q.pt"
    freeze_graphs: bool = False
    episode_graph_specs: tuple[tuple[int, int], ...] | None = None
    # Diagnostics: log PR(degree)-PR(random) per episode; optional JSON/YAML eval set path.
    log_episode_spread: bool = False
    log_grad_norm: bool = False
    validation_eval_set_path: str | None = None


@dataclass
class TrainingState:
    episode_rewards: list[float] = field(default_factory=list)
    episode_final_nc: list[float] = field(default_factory=list)
    episode_alpha: list[float] = field(default_factory=list)
    episode_pfail: list[float] = field(default_factory=list)
    episode_spreads: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    validation_history: list[dict[str, Any]] = field(default_factory=list)
    total_steps: int = 0


@dataclass(frozen=True)
class ImitationSample:
    observation: RecoveryObservation
    action: object


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def generate_episode_graph_specs(config: TrainingConfig, *, seed: int) -> tuple[tuple[int, int], ...]:
    """Sample (n, graph_seed) pairs for Phase-1 frozen-graph training (independent RNG stream)."""
    rng = Random(seed)
    return tuple(
        (
            rng.randint(config.n_range[0], config.n_range[1]),
            rng.randint(0, 10**9),
        )
        for _ in range(config.num_episodes)
    )


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


def _normalize_action_batch(action: object) -> tuple[Node, ...]:
    if isinstance(action, tuple):
        return action
    if isinstance(action, list):
        return tuple(action)
    return (action,)


def _graph_tensor_for_model(
    model: RecoveryQNetwork,
    observation: RecoveryObservation,
    *,
    device: torch.device,
):
    return observation_to_graph_tensor(
        observation,
        use_virtual_node=model.config.use_virtual_node,
        feature_names=model.feature_names,
        device=device,
    )


def _global_features_for_model(
    model: RecoveryQNetwork,
    observation: RecoveryObservation,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    if not model.config.use_global_features:
        return None
    return observation_to_global_features(
        observation,
        global_feature_names=model.global_feature_names,
    ).to(device)


def _choose_degree_batch(observation: RecoveryObservation) -> tuple[Node, ...]:
    ranked = sorted(
        observation.valid_actions,
        key=lambda node: (observation.graph.degree(node), str(node)),
        reverse=True,
    )
    return tuple(ranked[: observation.remaining_budget])


def _reset_with_non_empty_failures(
    env: RecoveryEnv,
    base_seed: int,
    rng: Random,
    *,
    max_attempts: int = 1024,
) -> RecoveryObservation:
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
    recent_anc = _mean_recent(training_state.episode_final_nc)
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
    tag = (
        "[validation:eval_set]"
        if validation.get("validation_source") == "eval_set_file"
        else "[validation]"
    )
    print(
        "\n"
        f"{tag} ep={validation['episode']} "
        f"final_anc={reference['final_anc_mean']:.3f}±{reference['final_anc_stderr']:.3f} "
        f"solved_fraction={reference['solved_fraction_mean']:.3f} "
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
    device: torch.device,
) -> torch.Tensor:
    # Guard: budget_coverage is the only node feature that distinguishes
    # intra-round states (b < B, reward=0, no cascade) from last-step states
    # (b = B, cascade-inclusive reward). Without it the Q-network receives
    # conflicting Bellman targets for observationally identical inputs, and
    # the loss converges to a biased mixture of the two reward regimes.
    if "budget_coverage" not in model.feature_names:
        raise ValueError(
            "budget_coverage must be present in active node features when "
            "training with the single-step (env.step) interface. It is the "
            "only input that lets the Q-network distinguish intra-round steps "
            "(reward=0) from last-round steps (cascade-inclusive reward). "
            "Remove it only from step_batch training where remaining_budget "
            "is constant within each call. See docs/architecture.md §0."
        )
    losses: list[torch.Tensor] = []
    for transition in transitions:
        graph_tensor = _graph_tensor_for_model(model, transition.observation, device=device)
        global_features = _global_features_for_model(model, transition.observation, device=device)
        q_values = model(graph_tensor, global_features)

        action_indices = [graph_tensor.node_to_index[action] for action in _normalize_action_batch(transition.action)]
        q_selected = torch.stack([q_values[index] for index in action_indices]).mean()

        with torch.no_grad():
            target_value = torch.tensor(float(transition.reward), device=device, dtype=torch.float32)
            if not transition.done and transition.next_observation.failed:
                next_tensor = _graph_tensor_for_model(target_model, transition.next_observation, device=device)
                next_global = _global_features_for_model(
                    target_model,
                    transition.next_observation,
                    device=device,
                )
                next_q_values = target_model(next_tensor, next_global)
                valid_next_indices = [
                    next_tensor.node_to_index[node] for node in transition.next_observation.valid_actions
                ]
                if valid_next_indices:
                    valid_next_q = next_q_values[valid_next_indices]
                    top_next_q = valid_next_q.max()
                    target_value = target_value + gamma * top_next_q

        losses.append(F.smooth_l1_loss(q_selected, target_value))

    return torch.stack(losses).mean()


def _env_kwargs_from_config(config: TrainingConfig) -> dict[str, Any]:
    return {
        "capacity_noise": config.capacity_noise,
        "failure_bias": config.failure_bias,
        "action_space": config.action_space,
        "obs_hops": config.obs_hops,
        "abandonment_nc_threshold": config.abandonment_nc_threshold,
    }


def _resolve_budget_for_graph(config: TrainingConfig, graph: Any) -> int:
    return compute_scaled_budget(
        config.budget,
        num_nodes=graph.number_of_nodes(),
        reference_n=config.budget_reference_n,
        enabled=config.scale_budget,
    )


def _resolve_max_rounds_for_graph(config: TrainingConfig, graph: Any) -> int:
    return compute_scaled_max_rounds(
        config.max_rounds,
        num_nodes=graph.number_of_nodes(),
        reference_n=config.budget_reference_n,
        enabled=config.scale_max_rounds,
    )


def generate_imitation_data(
    graphs: Sequence[Any],
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int,
    num_seeds: int,
    policy: Callable[[RecoveryObservation], object],
    *,
    env_kwargs: dict[str, Any] | None = None,
    base_seed: int = 0,
    scale_budget: bool = False,
    scale_max_rounds: bool = False,
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
        resolved_max_rounds = compute_scaled_max_rounds(
            max_rounds,
            num_nodes=graph.number_of_nodes(),
            reference_n=budget_reference_n,
            enabled=scale_max_rounds,
        )
        for seed_offset in range(num_seeds):
            rollout_seed = base_seed + graph_index * 10_000 + seed_offset
            env = RecoveryEnv(
                graph,
                alpha=alpha,
                pfail=pfail,
                budget=resolved_budget,
                max_rounds=resolved_max_rounds,
                seed=rollout_seed,
                **env_kwargs,
            )
            observation = _reset_with_non_empty_failures(env, rollout_seed, rng)
            done = False
            while not done and observation.failed:
                # Take one action at a time (env.step) rather than the full
                # batch (env.step_batch). This is required so that the imitation
                # dataset covers budget_coverage values in {1/n, 2/n, ..., B/n},
                # matching the RL training distribution P_RL(beta) = Uniform.
                # With step_batch, every observation has remaining_budget=B so
                # budget_coverage = B/n always — a point mass that leaves the
                # network unable to generalise to intra-round budget states seen
                # during RL fine-tuning (KL divergence = log B, infinite in the
                # strict sense for values b < B).
                action_batch = _normalize_action_batch(policy(observation))
                single_action = action_batch[0]
                samples.append(ImitationSample(observation=observation, action=(single_action,)))
                observation, _reward, done, _info = env.step(single_action)
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
                graph_tensor = _graph_tensor_for_model(model, sample.observation, device=device)
                global_features = _global_features_for_model(model, sample.observation, device=device)
                logits = model(graph_tensor, global_features)
                action_indices = [
                    graph_tensor.node_to_index[action] for action in _normalize_action_batch(sample.action)
                ]
                log_probs = F.log_softmax(logits, dim=0)
                batch_losses.append(-log_probs[action_indices].mean())
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

    overlap_sum = 0.0
    for sample in samples:
        predicted_actions = set(
            select_top_b(
                model,
                sample.observation,
                budget=sample.observation.remaining_budget,
                epsilon=0.0,
                rng=Random(0),
                device=device,
            )
        )
        target_actions = set(_normalize_action_batch(sample.action))
        overlap_sum += len(predicted_actions & target_actions) / max(1, len(target_actions))
    return overlap_sum / len(samples)


def _degree_minus_random_spread(
    graph: Any,
    *,
    alpha: float,
    pfail: float,
    budget: int,
    max_rounds: int,
    failure_seed: int,
    env_kwargs: dict[str, object],
    episode_index: int,
    factory_base_seed: int,
) -> float:
    """Heuristic spread for the same graph template and failure seed as the training episode."""
    from cascading_rl.evaluation.regime import build_policy_factories
    from cascading_rl.evaluation.saved_eval_sets import rollout_final_nc_on_instance

    factories = build_policy_factories(base_seed=factory_base_seed)
    pol_degree = factories["degree"](episode_index, failure_seed)
    pol_random = factories["random"](episode_index, failure_seed)
    pr_degree = rollout_final_nc_on_instance(
        graph,
        alpha=alpha,
        p_fail=pfail,
        budget=budget,
        max_rounds=max_rounds,
        failure_seed=failure_seed,
        env_kwargs=env_kwargs,
        policy=pol_degree,
    )
    pr_random = rollout_final_nc_on_instance(
        graph,
        alpha=alpha,
        p_fail=pfail,
        budget=budget,
        max_rounds=max_rounds,
        failure_seed=failure_seed,
        env_kwargs=env_kwargs,
        policy=pol_random,
    )
    return pr_degree - pr_random


def validate_policy_on_eval_set(
    model: RecoveryQNetwork,
    config: TrainingConfig,
    *,
    device: torch.device,
    instances: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run greedy RL on a saved eval set file (same protocol as ``--eval-set``)."""
    from cascading_rl.evaluation.saved_eval_sets import evaluate_policies_on_saved_instances

    env_kwargs = _env_kwargs_from_config(config)
    policy = build_greedy_policy(model, device=device, batch_actions=True)
    factories: dict[str, Callable[[int, int], Any]] = {
        "rl": lambda _gi, _se: policy,
    }
    overall, _ = evaluate_policies_on_saved_instances(
        instances,
        factories,
        env_kwargs=env_kwargs,
        policy_names=["rl"],
    )
    summary = overall["rl"]
    reference = {
        "alpha": config.alpha,
        "pfail": config.pfail,
        "budget": config.budget,
        "scale_budget": config.scale_budget,
        "scale_max_rounds": config.scale_max_rounds,
        "budget_reference_n": config.budget_reference_n,
        "max_rounds": config.max_rounds,
        "final_anc_mean": summary.final_nc.mean,
        "final_anc_stderr": summary.final_nc.stderr,
        "solved_fraction_mean": summary.solved_fraction.mean,
        "rounds_mean": summary.rounds.mean,
    }
    mean_anc = summary.final_nc.mean
    per_alpha_anc = {float(a): mean_anc for a in config.alpha_values}
    return {
        "final_anc_mean": summary.final_nc.mean,
        "final_anc_stderr": summary.final_nc.stderr,
        "solved_fraction_mean": summary.solved_fraction.mean,
        "rounds_mean": summary.rounds.mean,
        "reference": reference,
        "per_alpha_anc": per_alpha_anc,
        "grid": {
            "alpha_values": list(config.alpha_values),
            "pfail_values": list(config.pfail_values),
            "cell_count": len(config.alpha_values) * len(config.pfail_values),
            "final_anc_mean": mean_anc,
            "solved_fraction_mean": summary.solved_fraction.mean,
            "rounds_mean": summary.rounds.mean,
            "cells": [],
        },
        "env": env_kwargs,
        "validation_source": "eval_set_file",
    }


def validate_policy(
    model: RecoveryQNetwork,
    config: TrainingConfig,
    *,
    device: torch.device,
    validation_graphs: Sequence[Any],
) -> dict[str, Any]:
    policy = build_greedy_policy(model, device=device, batch_actions=True)
    env_kwargs = _env_kwargs_from_config(config)
    reference_summaries = evaluate_policy_factories_on_graphs(
        validation_graphs,
        {"rl": lambda graph_index, seed: policy},
        alpha=config.alpha,
        pfail=config.pfail,
        budget=config.budget,
        max_rounds=config.max_rounds,
        seeds=config.validation_seeds,
        env_kwargs=env_kwargs,
        scale_budget=config.scale_budget,
        scale_max_rounds=config.scale_max_rounds,
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
                env_kwargs=env_kwargs,
                scale_budget=config.scale_budget,
                scale_max_rounds=config.scale_max_rounds,
                reference_n=config.budget_reference_n,
            )["rl"]
            grid_cells.append(
                {
                    "alpha": alpha,
                    "pfail": pfail,
                    "final_anc_mean": grid_summary.final_nc.mean,
                    "solved_fraction_mean": grid_summary.solved_fraction.mean,
                    "rounds_mean": grid_summary.rounds.mean,
                }
            )

    grid_final_anc_mean = sum(cell["final_anc_mean"] for cell in grid_cells) / len(grid_cells)
    grid_solved_fraction_mean = sum(cell["solved_fraction_mean"] for cell in grid_cells) / len(
        grid_cells
    )
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
            env_kwargs=env_kwargs,
            scale_budget=config.scale_budget,
            scale_max_rounds=config.scale_max_rounds,
            reference_n=config.budget_reference_n,
        )["rl"]
        per_alpha_anc[float(alpha)] = per_alpha_summary.final_nc.mean

    return {
        "final_anc_mean": reference_summary.final_nc.mean,
        "final_anc_stderr": reference_summary.final_nc.stderr,
        "solved_fraction_mean": reference_summary.solved_fraction.mean,
        "rounds_mean": reference_summary.rounds.mean,
        "reference": {
            "alpha": config.alpha,
            "pfail": config.pfail,
            "budget": config.budget,
            "scale_budget": config.scale_budget,
            "scale_max_rounds": config.scale_max_rounds,
            "budget_reference_n": config.budget_reference_n,
            "max_rounds": config.max_rounds,
            "final_anc_mean": reference_summary.final_nc.mean,
            "final_anc_stderr": reference_summary.final_nc.stderr,
            "solved_fraction_mean": reference_summary.solved_fraction.mean,
            "rounds_mean": reference_summary.rounds.mean,
        },
        "grid": {
            "alpha_values": list(config.alpha_values),
            "pfail_values": list(config.pfail_values),
            "cell_count": len(grid_cells),
            "final_anc_mean": grid_final_anc_mean,
            "solved_fraction_mean": grid_solved_fraction_mean,
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
                "episode_final_nc": training_state.episode_final_nc,
                "episode_alpha": training_state.episode_alpha,
                "episode_pfail": training_state.episode_pfail,
                "episode_spreads": training_state.episode_spreads,
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
    eval_set_instances: list[Mapping[str, Any]] | None = None
    if config.validation_eval_set_path:
        from cascading_rl.evaluation.saved_eval_sets import load_eval_instances
        from cascading_rl.reproducibility import REPO_ROOT

        eval_path = Path(config.validation_eval_set_path)
        if not eval_path.is_absolute():
            eval_path = REPO_ROOT / eval_path
        eval_path = eval_path.resolve()
        if not eval_path.is_file():
            raise FileNotFoundError(
                f"validation_eval_set_path is not a file: {eval_path}"
            )
        eval_set_instances = load_eval_instances(eval_path)

    if (
        eval_set_instances is None
        and config.num_episodes >= config.validation_every >= 1
    ):
        warnings.warn(
            "Validating on synthetic graphs. Results will be noisy and not "
            "comparable across runs. Use --validation-eval-set with a fixed eval set (e.g. "
            "eval_sets/ds_validation.json) instead.",
            UserWarning,
            stacklevel=1,
        )

    regime_combinations = [(float(alpha), float(pfail)) for alpha in alpha_values for pfail in pfail_values]

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
            policy=_choose_degree_batch,
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
    env_kwargs = _env_kwargs_from_config(config)

    resolved_specs: tuple[tuple[int, int], ...] | None
    if config.episode_graph_specs is not None:
        resolved_specs = config.episode_graph_specs
    elif config.freeze_graphs:
        resolved_specs = generate_episode_graph_specs(
            config, seed=config.seed + FREEZE_GRAPH_SPECS_SEED_OFFSET
        )
    else:
        resolved_specs = None

    graph_buffer: deque = deque(maxlen=GRAPH_BUFFER_MAXLEN)

    for episode in range(config.num_episodes):
        epsilon = epsilon_for_episode(config, episode)
        cycle_index = episode % len(regime_combinations)
        if cycle_index == 0 and episode > 0:
            rng.shuffle(regime_combinations)
        alpha, pfail = regime_combinations[cycle_index]

        if resolved_specs is not None:
            n, graph_seed = resolved_specs[episode % len(resolved_specs)]
            graph = make_ba_graph(n=n, m=config.m, seed=graph_seed)
            resolved_budget = _resolve_budget_for_graph(config, graph)
            resolved_max_rounds = _resolve_max_rounds_for_graph(config, graph)
            env = RecoveryEnv(
                graph,
                alpha=alpha,
                pfail=pfail,
                budget=resolved_budget,
                max_rounds=resolved_max_rounds,
                seed=0,
                **env_kwargs,
            )
            observation = _reset_with_non_empty_failures(env, graph_seed, rng)
            episode_failure_seed = graph_seed
        else:
            if not graph_buffer or rng.random() < 0.3:
                graph_size = rng.randint(config.n_range[0], config.n_range[1])
                graph_struct_seed = rng.randint(0, 10**9)
                graph = make_ba_graph(n=graph_size, m=config.m, seed=graph_struct_seed)
                graph_buffer.append(graph)
            else:
                graph = rng.choice(list(graph_buffer))
            failure_seed = rng.randint(0, 10**9)
            resolved_budget = _resolve_budget_for_graph(config, graph)
            resolved_max_rounds = _resolve_max_rounds_for_graph(config, graph)
            env = RecoveryEnv(
                graph,
                alpha=alpha,
                pfail=pfail,
                budget=resolved_budget,
                max_rounds=resolved_max_rounds,
                seed=0,
                **env_kwargs,
            )
            observation = _reset_with_non_empty_failures(env, failure_seed, rng)
            episode_failure_seed = failure_seed
        if config.log_episode_spread:
            spread = _degree_minus_random_spread(
                graph,
                alpha=alpha,
                pfail=pfail,
                budget=resolved_budget,
                max_rounds=resolved_max_rounds,
                failure_seed=episode_failure_seed,
                env_kwargs=env_kwargs,
                episode_index=episode,
                factory_base_seed=config.seed + 400_000,
            )
            training_state.episode_spreads.append(spread)
            print(
                f"\n[diag] ep={episode + 1}/{config.num_episodes} alpha={alpha:.3f} pfail={pfail:.3f} "
                f"deg_minus_random_spread={spread:.4f}",
                flush=True,
            )
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
            next_observation, reward, done, _info = env.step(action)
            replay_buffer.push(
                Transition(
                    observation=observation,
                    action=(action,),
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
                if config.log_grad_norm:
                    grad_norm = sum(
                        p.grad.norm().item()
                        for p in model.parameters()
                        if p.grad is not None
                    )
                    print(f"[diag] grad_norm={grad_norm:.4f}", flush=True)
                training_state.losses.append(float(loss.item()))

                if training_state.total_steps % config.target_update_interval == 0:
                    target_model.load_state_dict(model.state_dict())

        training_state.episode_rewards.append(total_reward)
        training_state.episode_final_nc.append(env.current_nc())
        training_state.episode_alpha.append(alpha)
        training_state.episode_pfail.append(pfail)
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

        if (episode + 1) % config.validation_every == 0:
            if eval_set_instances is not None:
                validation = validate_policy_on_eval_set(
                    model,
                    config,
                    device=device,
                    instances=eval_set_instances,
                )
            else:
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

    if config.log_episode_spread:
        from cascading_rl.evaluation.saved_eval_sets import EVAL_SPREAD_FILTER_DEGREE_RANDOM

        spreads = training_state.episode_spreads
        if spreads:
            above = sum(1 for s in spreads if s > EVAL_SPREAD_FILTER_DEGREE_RANDOM)
            frac = above / len(spreads)
            print(
                f"[diag] spread summary: {above}/{len(spreads)} episodes "
                f"({frac:.1%}) with deg_minus_random > {EVAL_SPREAD_FILTER_DEGREE_RANDOM} "
                f"(each line uses that episode's alpha/pfail, not a fixed validation cell).",
                flush=True,
            )
        first_n = min(20, len(training_state.episode_rewards))
        if first_n:
            r20 = training_state.episode_rewards[:first_n]
            print(
                f"[diag] first {first_n} episode total rewards: "
                f"mean={sum(r20) / first_n:.4f} per-ep={['%.3f' % x for x in r20]}",
                flush=True,
            )
        if training_state.validation_history:
            vals = [
                float(entry["reference"]["final_anc_mean"])
                for entry in training_state.validation_history
            ]
            print(
                f"[diag] validation (eval set) final_anc_mean trajectory: "
                f"episodes={[entry['episode'] for entry in training_state.validation_history]} "
                f"values={[f'{v:.3f}' for v in vals]}",
                flush=True,
            )

    saved_path = save_checkpoint(
        model,
        config,
        training_state,
        checkpoint_path,
        episode=config.num_episodes,
    )
    return model, training_state, saved_path
