from pathlib import Path
from collections import Counter

import networkx as nx
import torch

from cascading_rl.envs.recovery import RecoveryEnv
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import RecoveryQNetwork, observation_to_graph_tensor
from cascading_rl.training import TrainingConfig, train_recovery_agent
from cascading_rl.training.trainer import validate_policy


def test_observation_to_graph_tensor_builds_features_and_mask():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, max_rounds=3, seed=0)

    observation = env.reset(seed=0)
    env.state.active = {0, 1}
    env.state.failed = {2, 3}
    env.state.frontier = {2}
    env.state.loads = {0: 1.0, 1: 2.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.5, 2: 1.5, 3: 1.5}
    observation = env.observe()

    graph_tensor = observation_to_graph_tensor(observation)

    assert graph_tensor.node_features.shape == (5, 9)
    assert graph_tensor.adjacency.shape == (5, 5)
    assert graph_tensor.valid_mask.tolist() == [False, False, True, True, False]
    assert torch.allclose(
        graph_tensor.node_features[4], graph_tensor.node_features[:4].mean(dim=0)
    )


def test_q_network_masks_invalid_actions():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, max_rounds=3, seed=0)

    observation = env.reset(seed=0)
    env.state.active = {0, 1}
    env.state.failed = {2, 3}
    env.state.frontier = {2}
    env.state.loads = {0: 1.0, 1: 2.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.5, 2: 1.5, 3: 1.5}
    observation = env.observe()

    model = RecoveryQNetwork()
    graph_tensor = observation_to_graph_tensor(observation)
    q_values = model(graph_tensor)

    assert q_values.shape[0] == 4
    assert q_values[0].item() < -1e8
    assert q_values[1].item() < -1e8


def test_train_recovery_agent_five_episodes_losses_and_anc_bounds(tmp_path: Path):
    checkpoint_dir = tmp_path / "learner_short"
    config = TrainingConfig(
        num_episodes=5,
        warmup_transitions=4,
        batch_size=4,
        replay_capacity=256,
        alpha_values=(0.2,),
        pfail_values=(0.1,),
        validation_graphs=1,
        validation_seeds=(0,),
        validation_every=100_000,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name="short_run.pt",
        n_range=(10, 12),
        budget=2,
        max_rounds=3,
        device="cpu",
    )

    _model, training_state, checkpoint_path = train_recovery_agent(config)

    assert checkpoint_path.exists()
    assert training_state.losses
    assert all(0.0 <= value <= 1.0 for value in training_state.episode_final_anc)


def test_train_recovery_agent_smoke_runs_and_saves_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "learner"
    config = TrainingConfig(
        alpha=0.10,
        pfail=0.10,
        alpha_values=(0.10,),
        pfail_values=(0.10,),
        num_episodes=6,
        replay_capacity=64,
        warmup_transitions=4,
        batch_size=4,
        capacity_noise=0.05,
        failure_bias="degree",
        obs_hops=1,
        validation_graphs=1,
        validation_seeds=(0,),
        validation_every=3,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name="smoke.pt",
        n_range=(10, 12),
        budget=2,
        max_rounds=3,
        device="cpu",
    )

    _, training_state, checkpoint_path = train_recovery_agent(config)

    assert checkpoint_path.exists()
    assert len(training_state.episode_rewards) == config.num_episodes
    assert len(training_state.validation_history) >= 1
    validation = training_state.validation_history[0]
    assert validation["env"]["capacity_noise"] == config.capacity_noise
    assert validation["env"]["failure_bias"] == config.failure_bias
    assert validation["env"]["action_space"] == config.action_space
    assert validation["env"]["obs_hops"] == config.obs_hops
    assert validation["grid"]["cell_count"] == len(config.alpha_values) * len(config.pfail_values)
    assert set(validation["per_alpha_anc"]) == set(config.alpha_values)


def test_validate_policy_is_deterministic_with_fixed_graphs():
    config = TrainingConfig(
        alpha=0.10,
        pfail=0.10,
        alpha_values=(0.10, 0.15, 0.20),
        pfail_values=(0.10, 0.15, 0.20),
        validation_graphs=1,
        validation_seeds=(0,),
        validation_seed=42,
        n_range=(10, 12),
        budget=2,
        max_rounds=3,
        device="cpu",
    )
    validation_graphs = make_graph_batch(
        num_graphs=config.validation_graphs,
        n_range=config.n_range,
        m=config.m,
        seed=config.validation_seed,
    )
    model = RecoveryQNetwork()

    first = validate_policy(model, config, device=torch.device("cpu"), validation_graphs=validation_graphs)
    second = validate_policy(
        model, config, device=torch.device("cpu"), validation_graphs=validation_graphs
    )

    assert first["final_anc_mean"] == second["final_anc_mean"]
    assert first["per_alpha_anc"] == second["per_alpha_anc"]


def test_train_recovery_agent_cycles_all_regime_combinations_once(tmp_path: Path):
    checkpoint_dir = tmp_path / "learner_regimes"
    config = TrainingConfig(
        num_episodes=9,
        replay_capacity=64,
        warmup_transitions=4,
        batch_size=4,
        alpha_values=(0.10, 0.15, 0.20),
        pfail_values=(0.10, 0.15, 0.20),
        validation_graphs=1,
        validation_seeds=(0,),
        validation_every=100_000,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name="regimes.pt",
        n_range=(10, 12),
        budget=2,
        max_rounds=3,
        device="cpu",
    )

    _, training_state, _checkpoint_path = train_recovery_agent(config)

    seen = Counter(zip(training_state.episode_alpha, training_state.episode_pfail))
    expected = {
        (alpha, pfail) for alpha in config.alpha_values for pfail in config.pfail_values
    }

    assert set(seen) == expected
    assert all(count == 1 for count in seen.values())
