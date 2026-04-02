from pathlib import Path

import networkx as nx
import pytest

from cascading_rl.envs.recovery import RecoveryEnv
from cascading_rl.models import (
    FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    QNetworkConfig,
    RecoveryQNetwork,
    observation_to_global_features,
    observation_to_graph_tensor,
)
from cascading_rl.training import TrainingConfig, train_recovery_agent


def _make_test_observation():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, max_rounds=3, seed=0)

    env.reset(seed=0)
    env.state.active = {0, 1}
    env.state.failed = {2, 3}
    env.state.frontier = {2}
    env.state.loads = {0: 1.0, 1: 2.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.5, 2: 1.5, 3: 1.5}
    return env.observe()


def test_observation_to_graph_tensor_builds_features_and_mask():
    observation = _make_test_observation()

    graph_tensor = observation_to_graph_tensor(observation)

    assert graph_tensor.node_features.shape == (4, 9)
    assert graph_tensor.adjacency.shape == (4, 4)
    assert graph_tensor.valid_mask.tolist() == [False, False, True, True]


def test_q_network_masks_invalid_actions():
    observation = _make_test_observation()

    model = RecoveryQNetwork()
    graph_tensor = observation_to_graph_tensor(observation)
    global_features = observation_to_global_features(observation)
    q_values = model(graph_tensor, global_features)

    assert q_values.shape[0] == 4
    assert q_values[0].item() < -1e8
    assert q_values[1].item() < -1e8


def test_observation_to_graph_tensor_supports_virtual_node():
    observation = _make_test_observation()

    graph_tensor = observation_to_graph_tensor(observation, use_virtual_node=True)

    assert graph_tensor.node_features.shape == (5, 9)
    assert graph_tensor.adjacency.shape == (5, 5)
    assert graph_tensor.valid_mask.tolist() == [False, False, True, True, False]


def test_observation_to_graph_tensor_supports_feature_subsets_in_canonical_order():
    observation = _make_test_observation()

    requested_features = ("degree_norm", "load_norm")
    graph_tensor = observation_to_graph_tensor(
        observation,
        active_node_features=requested_features,
    )

    assert graph_tensor.node_features.shape == (4, 2)
    assert graph_tensor.node_features[0].tolist() == pytest.approx([0.4, 0.5])


def test_observation_to_global_features_supports_feature_subsets_in_canonical_order():
    observation = _make_test_observation()

    requested_features = (
        "max_load_capacity_ratio",
        "failed_fraction",
    )
    global_features = observation_to_global_features(
        observation,
        active_global_features=requested_features,
    )

    assert tuple(global_features.shape) == (2,)
    assert global_features.tolist() == pytest.approx([0.5, 0.8])


def test_q_network_supports_ablation_flags():
    observation = _make_test_observation()

    model = RecoveryQNetwork(
        QNetworkConfig(use_global_features=False, use_virtual_node=True)
    )
    graph_tensor = observation_to_graph_tensor(observation, use_virtual_node=True)
    q_values = model(graph_tensor)

    assert q_values.shape[0] == 4
    assert q_values[0].item() < -1e8
    assert q_values[1].item() < -1e8


def test_q_network_config_derives_dimensions_from_active_features():
    config = QNetworkConfig(
        active_node_features=("degree_norm", "load_norm"),
        active_global_features=("max_load_capacity_ratio", "failed_fraction"),
        use_virtual_node=True,
    )

    model = RecoveryQNetwork(config)

    assert config.active_node_features == ("load_norm", "degree_norm")
    assert config.active_global_features == ("failed_fraction", "max_load_capacity_ratio")
    assert config.input_dim == 2
    assert config.global_feat_dim == 2
    assert model.encoder.layers[0].self_linear.in_features == 2
    assert model.global_readout.proj.in_features == 2 * config.embed_dim + 2


def test_training_config_defaults_match_feature_constants():
    config = TrainingConfig()

    assert config.active_node_features == FEATURE_NAMES
    assert config.active_global_features == GLOBAL_FEATURE_NAMES


def test_train_recovery_agent_smoke_runs_and_saves_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "learner"
    config = TrainingConfig(
        num_episodes=6,
        replay_capacity=64,
        warmup_transitions=4,
        batch_size=4,
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
