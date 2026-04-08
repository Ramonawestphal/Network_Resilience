from pathlib import Path
from collections import Counter
from types import SimpleNamespace

import networkx as nx
import pytest
import torch
from torch import nn

from cascading_rl.budgeting import compute_scaled_budget
from cascading_rl.envs.recovery import RecoveryEnv
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.models import (
    FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    QNetworkConfig,
    RecoveryQNetwork,
    observation_to_graph_tensor,
)
from cascading_rl.training import TrainingConfig, train_recovery_agent
from cascading_rl.training.replay import Transition
from cascading_rl.training.trainer import (
    _imitation_agreement_rate,
    compute_dqn_loss,
    generate_imitation_data,
    pretrain_by_imitation,
    rewrite_round,
    validate_policy,
)


class DummyQModel(nn.Module):
    def __init__(self, q_values_by_node: dict[int, float]) -> None:
        super().__init__()
        self.config = SimpleNamespace(use_virtual_node=False, use_global_features=False)
        self.feature_names = FEATURE_NAMES
        self.global_feature_names: tuple[str, ...] = ()
        self.q_values_by_node = q_values_by_node

    def forward(self, graph_tensor, global_features=None) -> torch.Tensor:  # type: ignore[override]
        return torch.tensor(
            [self.q_values_by_node[node] for node in graph_tensor.node_ids],
            dtype=torch.float32,
            device=graph_tensor.node_features.device,
        )


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

    graph_tensor = observation_to_graph_tensor(observation, use_virtual_node=True)

    assert graph_tensor.node_features.shape == (5, 8)
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


def test_q_network_supports_global_features_and_virtual_node():
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
        config=QNetworkConfig(
            use_global_features=True,
            use_virtual_node=True,
        )
    )
    graph_tensor, q_values = model.score_observation(observation)

    assert graph_tensor.node_features.shape == (5, 8)
    assert q_values.shape[0] == 4
    assert q_values[0].item() < -1e8


def test_q_network_uses_explicit_global_layout_for_node_feature_subsets():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, max_rounds=3, seed=0)

    observation = env.reset(seed=0)
    env.state.active = {0, 1}
    env.state.failed = {2, 3}
    env.state.frontier = {2}
    env.state.loads = {0: 1.0, 1: 2.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.5, 2: 1.5, 3: 1.5}
    observation = env.observe()

    active_node_features = FEATURE_NAMES[:-1]
    model = RecoveryQNetwork(
        config=QNetworkConfig(
            use_global_features=True,
            active_node_features=active_node_features,
        )
    )

    assert model.feature_names == active_node_features
    assert model.global_feature_names == GLOBAL_FEATURE_NAMES

    graph_tensor, q_values = model.score_observation(observation)

    assert graph_tensor.node_features.shape == (4, len(active_node_features))
    assert q_values.shape[0] == 4


def test_train_recovery_agent_five_episodes_losses_and_anc_bounds(tmp_path: Path):
    checkpoint_dir = tmp_path / "learner_short"
    config = TrainingConfig(
        num_episodes=5,
        warmup_transitions=4,
        batch_size=4,
        replay_capacity=256,
        alpha_values=(0.2,),
        pfail_values=(0.1,),
        scale_budget=False,
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
    assert len(training_state.episode_recovered) == config.num_episodes
    assert len(training_state.episode_mean_anc_unconditional) == config.num_episodes
    assert all(0.0 <= v <= 1.0 for v in training_state.episode_mean_anc_unconditional)


def test_train_recovery_agent_smoke_runs_and_saves_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path / "learner"
    config = TrainingConfig(
        alpha=0.10,
        pfail=0.10,
        alpha_values=(0.10,),
        pfail_values=(0.10,),
        scale_budget=False,
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

    with pytest.warns(UserWarning, match="Validating on synthetic"):
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
        scale_budget=False,
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
        scale_budget=False,
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


def test_imitation_pretraining_matches_degree_policy_on_heldout_graphs():
    train_graphs = make_graph_batch(num_graphs=50, n_range=(10, 12), m=2, seed=123)
    expert_policy = lambda observation: tuple(
        sorted(
            observation.valid_actions,
            key=lambda node: (observation.graph.degree(node), str(node)),
            reverse=True,
        )[: observation.remaining_budget]
    )
    samples = generate_imitation_data(
        train_graphs,
        alpha=0.20,
        pfail=0.10,
        budget=2,
        max_rounds=3,
        num_seeds=3,
        policy=expert_policy,
        base_seed=321,
    )
    model = RecoveryQNetwork()
    model, losses = pretrain_by_imitation(
        model,
        samples,
        lr=1e-3,
        epochs=10,
        batch_size=64,
    )
    heldout_graphs = make_graph_batch(num_graphs=10, n_range=(10, 12), m=2, seed=456)
    heldout_samples = generate_imitation_data(
        heldout_graphs,
        alpha=0.20,
        pfail=0.10,
        budget=2,
        max_rounds=3,
        num_seeds=3,
        policy=expert_policy,
        base_seed=654,
    )

    agreement = _imitation_agreement_rate(model, heldout_samples, device=torch.device("cpu"))

    assert losses[-1] <= losses[0]
    assert agreement > 0.60


def test_freeze_graphs_identical_graph_spec_sequence_across_two_runs(tmp_path: Path, monkeypatch):
    """Same training.seed + freeze_graphs must yield the same (n, graph_seed) stream (fair arch search)."""
    from cascading_rl.graph.generation import make_ba_graph as real_make_ba
    from cascading_rl.training import trainer as trainer_mod

    recorded: list[list[tuple[int, int]]] = []

    for run_idx in range(2):
        calls: list[tuple[int, int]] = []

        def recorder(n: int, m: int, seed: int | None = None):
            calls.append((n, int(seed or 0)))
            return real_make_ba(n=n, m=m, seed=seed)

        monkeypatch.setattr(trainer_mod, "make_ba_graph", recorder)
        run_dir = tmp_path / f"freeze_run_{run_idx}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config = TrainingConfig(
            num_episodes=10,
            seed=42,
            freeze_graphs=True,
            alpha_values=(0.2,),
            pfail_values=(0.1,),
            scale_budget=False,
            replay_capacity=128,
            warmup_transitions=8,
            batch_size=4,
            validation_graphs=1,
            validation_seeds=(0,),
            validation_every=1_000_000,
            checkpoint_dir=str(run_dir),
            checkpoint_name="freeze.pt",
            n_range=(18, 40),
            budget=2,
            max_rounds=3,
            device="cpu",
        )
        train_recovery_agent(config)
        recorded.append(list(calls))

    assert recorded[0] == recorded[1]
    assert len(recorded[0]) == 10


def test_train_recovery_agent_uses_episode_graph_specs_when_set(tmp_path: Path, monkeypatch):
    from cascading_rl.graph.generation import make_ba_graph as real_make_ba_graph
    from cascading_rl.training import trainer as trainer_mod

    calls: list[tuple[int, int, int]] = []

    def recorder(n: int, m: int, seed: int | None = None):
        calls.append((n, m, int(seed or 0)))
        return real_make_ba_graph(n=n, m=m, seed=seed)

    monkeypatch.setattr(trainer_mod, "make_ba_graph", recorder)

    checkpoint_dir = tmp_path / "learner_specs"
    config = TrainingConfig(
        num_episodes=4,
        replay_capacity=64,
        warmup_transitions=4,
        batch_size=4,
        alpha_values=(0.2,),
        pfail_values=(0.1,),
        scale_budget=False,
        validation_graphs=1,
        validation_seeds=(0,),
        validation_every=100_000,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name="specs.pt",
        n_range=(30, 50),
        budget=2,
        max_rounds=3,
        device="cpu",
        episode_graph_specs=((10, 999_001), (11, 999_002)),
    )

    train_recovery_agent(config)

    assert calls[0] == (10, config.m, 999_001)
    assert calls[1] == (11, config.m, 999_002)
    assert calls[2] == (10, config.m, 999_001)
    assert calls[3] == (11, config.m, 999_002)


def test_budget_scaling_helper_matches_canonical_reference_rule():
    assert compute_scaled_budget(2, num_nodes=40, reference_n=40, enabled=True) == 2
    assert compute_scaled_budget(2, num_nodes=30, reference_n=40, enabled=True) == 2
    assert compute_scaled_budget(2, num_nodes=50, reference_n=40, enabled=True) == 2
    assert compute_scaled_budget(2, num_nodes=100, reference_n=40, enabled=True) == 5


def test_use_monte_carlo_returns_trains_and_uses_discounted_returns(tmp_path: Path):
    """MC mode must complete training and push transitions with done=True (no bootstrap)."""
    checkpoint_dir = tmp_path / "learner_mc"
    config = TrainingConfig(
        num_episodes=4,
        warmup_transitions=4,
        batch_size=4,
        replay_capacity=256,
        alpha_values=(0.2,),
        pfail_values=(0.1,),
        scale_budget=False,
        validation_graphs=1,
        validation_seeds=(0,),
        validation_every=100_000,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_name="mc_run.pt",
        n_range=(10, 12),
        budget=2,
        max_rounds=3,
        device="cpu",
        use_monte_carlo_returns=True,
    )

    _model, training_state, checkpoint_path = train_recovery_agent(config)

    # Training completes and checkpoints are saved.
    assert checkpoint_path.exists()
    assert len(training_state.episode_rewards) == config.num_episodes
    # ANC values are always valid probabilities.
    assert all(0.0 <= v <= 1.0 for v in training_state.episode_final_anc)
    assert len(training_state.episode_recovered) == config.num_episodes
    assert all(r in (True, False) for r in training_state.episode_recovered)
    assert all(0.0 <= v <= 1.0 for v in training_state.episode_mean_anc_unconditional)


# ---------------------------------------------------------------------------
# rewrite_round unit tests
# ---------------------------------------------------------------------------


def test_rewrite_round_correct_gamma_exponents():
    """rewrite_round produces suffix-discounted rewards with correct γ exponents.

    Three-step round: rewards [0.1, 0.2, 1.0], gamma=0.5
      step 0: 0.1 + 0.5*(0.2 + 0.5*1.0) = 0.1 + 0.35 = 0.45
      step 1: 0.2 + 0.5*1.0              = 0.7
      step 2: 1.0                         = 1.0  (cascade step, γ^0)
    """
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, max_rounds=3, seed=0)
    # Use the same observation object for all transitions — rewrite_round only
    # reads .reward and .done, so the observation content doesn't matter here.
    obs = env.reset(seed=0)
    s_post = env.observe()

    transitions = [
        Transition(observation=obs, action=0, reward=0.1, next_observation=obs, done=False),
        Transition(observation=obs, action=1, reward=0.2, next_observation=obs, done=False),
        Transition(observation=obs, action=2, reward=1.0, next_observation=obs, done=True),
    ]

    rewritten = rewrite_round(transitions, s_post_cascade=s_post, gamma=0.5)

    assert len(rewritten) == 3
    assert rewritten[0].reward == pytest.approx(0.45)
    assert rewritten[1].reward == pytest.approx(0.7)
    assert rewritten[2].reward == pytest.approx(1.0)
    assert [t.bootstrap_steps for t in rewritten] == [3, 2, 1]
    # Every step bootstraps from the post-cascade state.
    assert all(t.next_observation is s_post for t in rewritten)
    # done propagated from the last (cascade) transition.
    assert all(t.done is True for t in rewritten)
    # Observations and actions are preserved unchanged.
    assert all(t.observation is obs for t in rewritten)
    assert [t.action for t in rewritten] == [0, 1, 2]


def test_rewrite_round_single_step():
    """Single-step round (budget=1): reward unchanged, next_obs replaced."""
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=1, max_rounds=3, seed=0)
    obs = env.reset(seed=0)
    s_post = env.observe()

    transitions = [
        Transition(observation=obs, action=0, reward=0.75, next_observation=obs, done=False),
    ]

    rewritten = rewrite_round(transitions, s_post_cascade=s_post, gamma=0.99)

    assert len(rewritten) == 1
    assert rewritten[0].reward == pytest.approx(0.75)
    assert rewritten[0].next_observation is s_post
    assert rewritten[0].done is False
    assert rewritten[0].bootstrap_steps == 1


def test_rewrite_round_empty():
    """Empty list returns empty list without error."""
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=1, max_rounds=3, seed=0)
    obs = env.reset(seed=0)

    assert rewrite_round([], s_post_cascade=obs, gamma=0.99) == []


def test_compute_dqn_loss_uses_bootstrap_steps_exponent():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, max_rounds=3, seed=0)

    env.reset(seed=0)
    env.state.active = {0, 1}
    env.state.failed = {2, 3}
    env.state.frontier = {2}
    env.state.loads = {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.0, 2: 1.0, 3: 1.0}
    observation = env.observe()

    env.state.active = {0, 1, 2}
    env.state.failed = {3}
    env.state.frontier = {3}
    next_observation = env.observe()

    model = DummyQModel({0: -1e9, 1: -1e9, 2: 1.5, 3: 0.0})
    target_model = DummyQModel({0: -1e9, 1: -1e9, 2: -1e9, 3: 4.0})
    transition = Transition(
        observation=observation,
        action=2,
        reward=1.0,
        next_observation=next_observation,
        done=False,
        bootstrap_steps=3,
    )

    loss = compute_dqn_loss(
        model,
        target_model,
        [transition],
        gamma=0.5,
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(0.0)


def test_compute_dqn_loss_defaults_to_standard_one_step_td():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, max_rounds=3, seed=0)

    env.reset(seed=0)
    env.state.active = {0, 1}
    env.state.failed = {2, 3}
    env.state.frontier = {2}
    env.state.loads = {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.0, 2: 1.0, 3: 1.0}
    observation = env.observe()

    env.state.active = {0, 1, 2}
    env.state.failed = {3}
    env.state.frontier = {3}
    next_observation = env.observe()

    model = DummyQModel({0: -1e9, 1: -1e9, 2: 3.0, 3: 0.0})
    target_model = DummyQModel({0: -1e9, 1: -1e9, 2: -1e9, 3: 4.0})
    transition = Transition(
        observation=observation,
        action=2,
        reward=1.0,
        next_observation=next_observation,
        done=False,
    )

    loss = compute_dqn_loss(
        model,
        target_model,
        [transition],
        gamma=0.5,
        device=torch.device("cpu"),
    )

    assert loss.item() == pytest.approx(0.0)
