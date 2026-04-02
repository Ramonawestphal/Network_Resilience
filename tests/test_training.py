from pathlib import Path

import networkx as nx

from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.models import (
    QNetworkConfig,
    RecoveryQNetwork,
    observation_to_global_features,
    observation_to_graph_tensor,
)
from cascading_rl.models.gnn import VIRTUAL_NODE
from cascading_rl.training import TrainingConfig, train_recovery_agent


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

    assert graph_tensor.node_features.shape == (4, 8)
    assert graph_tensor.adjacency.shape == (4, 4)
    assert graph_tensor.valid_mask.tolist() == [False, False, True, True]


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
    global_features = observation_to_global_features(observation)
    q_values = model(graph_tensor, global_features)

    assert q_values.shape[0] == 4
    assert q_values[0].item() < -1e8
    assert q_values[1].item() < -1e8


def test_virtual_node_sentinel_distinct_from_string_graph_node():
    graph = nx.Graph()
    graph.add_edge("__virtual__", 0)
    graph.add_edge(0, 1)
    observation = RecoveryObservation(
        graph=graph,
        loads={"__virtual__": 1.0, 0: 1.0, 1: 1.0},
        capacities={"__virtual__": 1.0, 0: 1.0, 1: 1.0},
        active=frozenset({0, 1}),
        failed=frozenset({"__virtual__"}),
        frontier=frozenset({"__virtual__"}),
        remaining_budget=2,
        budget=2,
        current_round=1,
        max_rounds=5,
    )
    graph_tensor = observation_to_graph_tensor(observation, use_virtual_node=True)
    n_real = graph.number_of_nodes()
    assert graph_tensor.node_to_index[VIRTUAL_NODE] == n_real
    assert graph_tensor.node_to_index["__virtual__"] in range(n_real)
    assert graph_tensor.node_to_index["__virtual__"] != graph_tensor.node_to_index[VIRTUAL_NODE]


def test_observation_to_graph_tensor_supports_virtual_node():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, max_rounds=3, seed=0)

    observation = env.reset(seed=0)
    env.state.active = {0, 1}
    env.state.failed = {2, 3}
    env.state.frontier = {2}
    env.state.loads = {0: 1.0, 1: 2.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.5, 2: 1.5, 3: 1.5}
    observation = env.observe()

    graph_tensor = observation_to_graph_tensor(observation, use_virtual_node=True)

    assert graph_tensor.node_features.shape == (5, 8)
    assert graph_tensor.adjacency.shape == (5, 5)
    assert graph_tensor.valid_mask.tolist() == [False, False, True, True, False]


def test_q_network_supports_ablation_flags():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, max_rounds=3, seed=0)

    observation = env.reset(seed=0)
    env.state.active = {0, 1}
    env.state.failed = {2, 3}
    env.state.frontier = {2}
    env.state.loads = {0: 1.0, 1: 2.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.5, 2: 1.5, 3: 1.5}
    observation = env.observe()

    model = RecoveryQNetwork(
        QNetworkConfig(use_global_features=False, use_virtual_node=True)
    )
    graph_tensor = observation_to_graph_tensor(observation, use_virtual_node=True)
    q_values = model(graph_tensor)

    assert q_values.shape[0] == 4
    assert q_values[0].item() < -1e8
    assert q_values[1].item() < -1e8


def test_q_network_supports_legacy_checkpoint_feature_width():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, max_rounds=3, seed=0)

    observation = env.reset(seed=0)
    env.state.active = {0, 1}
    env.state.failed = {2, 3}
    env.state.frontier = {2}
    env.state.loads = {0: 1.0, 1: 2.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.5, 2: 1.5, 3: 1.5}
    observation = env.observe()

    model = RecoveryQNetwork(QNetworkConfig(input_dim=9))
    graph_tensor, q_values = model.score_observation(observation)

    assert graph_tensor.node_features.shape == (4, 9)
    assert q_values.shape[0] == 4


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
