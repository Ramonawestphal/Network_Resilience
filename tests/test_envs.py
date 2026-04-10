import networkx as nx
import pytest

from cascading_rl.envs.recovery import RecoveryEnv


def test_abandonment_when_post_cascade_anc_below_threshold_step_and_batch():
    """Episode ends with info['abandoned'] when ANC stays below threshold and failures remain."""
    # Three disconnected pairs (0-1), (2-3), (4-5): pairwise connectivity among {0,1,2,3}
    # is 4/30 ≈ 0.133 (two components of size 2 each → 4 connected pairs / 30 total pairs).
    # After repairing 4, active {0,1,2,3,4} has components 2+2+1 → still 4/30 ≈ 0.133 < 0.30.
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (2, 3), (4, 5)])
    env = RecoveryEnv(
        graph,
        alpha=1.0,
        pfail=0.0,
        budget=1,
        max_rounds=10,
        abandonment_nc_threshold=0.30,
    )
    env.reset()
    env.state.active = {0, 1, 2, 3}
    env.state.failed = {4, 5}
    env.state.frontier = {4}
    for node in graph.nodes():
        env.state.loads[node] = 1.0
        env.state.capacities[node] = 3.0
    env.remaining_budget = 1
    env.current_round = 1

    _, _, done, info = env.step(4)
    assert done is True
    assert info["abandoned"] is True
    assert info["nc_after_cascade"] < 0.30
    assert env.state.failed

    env2 = RecoveryEnv(
        graph,
        alpha=1.0,
        pfail=0.0,
        budget=1,
        max_rounds=10,
        abandonment_nc_threshold=0.30,
    )
    env2.reset()
    env2.state.active = {0, 1, 2, 3}
    env2.state.failed = {4, 5}
    env2.state.frontier = {4}
    for node in graph.nodes():
        env2.state.loads[node] = 1.0
        env2.state.capacities[node] = 3.0
    env2.remaining_budget = 1
    env2.current_round = 1

    _, _, done_b, info_b = env2.step_batch([4])
    assert done_b is True
    assert info_b["abandoned"] is True


def test_environment_step_rewards_connectivity_gain():
    graph = nx.star_graph(3)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=2)

    observation = env.reset()
    env.state.active = {1, 2}
    env.state.failed = {0, 3}
    env.state.frontier = {3}
    env.state.loads = {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}
    env.state.capacities = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}

    observation, reward, done, info = env.step(0)

    # Intra-round steps (b < B) now return reward=0 under the homogenised
    # reward scheme: the full round-level NC delta is concentrated at the
    # last step of the round (when the cascade fires) to avoid mixed Bellman
    # targets in the replay buffer. The repair still happened (0 is active).
    assert reward == 0.0
    assert 0 in observation.active
    # |V|=4, active {0,1,2} connected: 3*2 / (4*3) = 0.5
    assert info["anc"] == pytest.approx(0.5)
    assert info["cascade_executed"] is False
    assert done is False


def test_environment_waits_until_round_end_before_cascade():
    graph = nx.star_graph(4)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=2, max_rounds=3)

    env.reset()
    env.state.active = {0, 2}
    env.state.failed = {1, 3, 4}
    env.state.frontier = {1}
    env.state.loads = {0: 0.0, 1: 3.0, 2: 0.0, 3: 0.0, 4: 0.0}
    env.state.capacities = {0: 2.0, 1: 3.0, 2: 2.0, 3: 2.0, 4: 2.0}
    env.remaining_budget = 2
    env.current_round = 1

    obs_after_first, _, done_first, info_first = env.step(3)

    assert done_first is False
    assert info_first["cascade_executed"] is False
    assert info_first["round_complete"] is False
    assert info_first["newly_failed_nodes"] == []
    assert obs_after_first.current_round == 1
    assert obs_after_first.remaining_budget == 1
    assert obs_after_first.frontier == frozenset({1})

    obs_after_second, _, done_second, info_second = env.step(4)

    assert done_second is False
    assert info_second["cascade_executed"] is True
    assert info_second["round_complete"] is True
    assert info_second["newly_failed_nodes"] == [0]
    assert obs_after_second.current_round == 2
    assert obs_after_second.remaining_budget == 2
    assert 0 in obs_after_second.failed


def test_environment_starts_new_round_when_budget_is_exhausted():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=1, max_rounds=3)

    env.reset()
    env.state.active = {0}
    env.state.failed = {1, 2, 3}
    env.state.frontier = set()
    env.state.loads = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}
    env.remaining_budget = 1
    env.current_round = 1

    observation, _, done, info = env.step(1)

    assert done is False
    assert info["round_complete"] is True
    assert info["cascade_executed"] is False
    assert info["action_round"] == 1
    assert observation.current_round == 2
    assert observation.remaining_budget == 1


def test_obs_hops_masks_loads_beyond_one_hop_and_step_preserves_valid_actions():
    graph = nx.path_graph(7)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=2, max_rounds=20, obs_hops=1)
    env.reset(seed=0)
    env.state.active = {0, 2, 3, 4, 6}
    env.state.failed = {1, 5}
    env.state.frontier = set()
    for node in graph.nodes():
        env.state.loads[node] = 1.0
        env.state.capacities[node] = 2.0

    obs_before = env.observe()
    assert obs_before.valid_actions
    for node in graph.nodes():
        dist = min(nx.shortest_path_length(graph, node, f) for f in obs_before.failed)
        if dist > 1:
            assert obs_before.loads[node] == 0.0
            assert obs_before.capacities[node] == 0.0

    obs_after, _, _, _ = env.step(1)
    assert obs_after.valid_actions


def test_environment_stops_when_max_rounds_are_reached():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=1, max_rounds=1)

    env.reset()
    env.state.active = {0}
    env.state.failed = {1, 2, 3}
    env.state.frontier = set()
    env.state.loads = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}
    env.remaining_budget = 1
    env.current_round = 1

    _, _, done, info = env.step(1)

    assert done is True
    assert info["max_rounds_reached"] is True


def test_environment_step_batch_repairs_full_round_before_cascade():
    graph = nx.star_graph(4)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=2, max_rounds=3)

    env.reset()
    env.state.active = {0, 2}
    env.state.failed = {1, 3, 4}
    env.state.frontier = {1}
    env.state.loads = {0: 0.0, 1: 3.0, 2: 0.0, 3: 0.0, 4: 0.0}
    env.state.capacities = {0: 2.0, 1: 3.0, 2: 2.0, 3: 2.0, 4: 2.0}
    env.remaining_budget = 2
    env.current_round = 1

    prev_anc = env.current_nc()
    observation, reward, done, info = env.step_batch([3, 4])

    assert reward == info["nc_after_cascade"] - prev_anc
    assert done is False
    assert info["cascade_executed"] is True
    assert info["actions"] == [3, 4]
    assert observation.current_round == 2
    assert observation.remaining_budget == 2


def test_step_batch_rejects_partial_batch_when_failed_nodes_remain():
    graph = nx.path_graph(5)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=2, max_rounds=3)

    env.reset()
    env.state.active = {0}
    env.state.failed = {1, 2, 3, 4}
    env.state.frontier = set()
    env.state.loads = {node: 0.0 for node in graph.nodes()}
    env.state.capacities = {node: 2.0 for node in graph.nodes()}
    env.remaining_budget = 2
    env.current_round = 1

    with pytest.raises(ValueError, match="Partial step_batch is only valid"):
        env.step_batch([1])


def test_step_batch_updates_round_start_baseline_for_following_steps():
    graph = nx.path_graph(6)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=2, max_rounds=4)

    env.reset(seed=0)
    env.state.active = {0}
    env.state.failed = {1, 2, 3, 4, 5}
    env.state.frontier = set()
    env.state.loads = {node: 0.0 for node in graph.nodes()}
    env.state.capacities = {node: 2.0 for node in graph.nodes()}
    env.remaining_budget = 2
    env.current_round = 1
    env._round_start_nc = env.current_nc()

    obs_after_batch, _, done_batch, info_batch = env.step_batch([1, 2])

    assert done_batch is False
    assert env._round_start_nc == info_batch["nc_after_cascade"]

    round_two_baseline = obs_after_batch.graph.subgraph(obs_after_batch.active).number_of_nodes()
    assert round_two_baseline == 3

    _, reward_first, done_first, _ = env.step(3)
    assert done_first is False
    assert reward_first == 0.0

    _, reward_second, done_second, info_second = env.step(4)

    assert done_second is False
    assert info_second["round_complete"] is True
    assert reward_second == pytest.approx(
        info_second["nc_after_cascade"] - info_batch["nc_after_cascade"]
    )


def test_recovery_env_reset_reseeds_rng_independent_of_constructor_seed():
    """``reset(seed=...)`` fully controls failure sampling; constructor seed must not leak."""
    graph = nx.barabasi_albert_graph(28, 2, seed=0)
    env_low = RecoveryEnv(graph, alpha=0.2, pfail=0.35, budget=3, max_rounds=6, seed=0)
    env_high = RecoveryEnv(graph, alpha=0.2, pfail=0.35, budget=3, max_rounds=6, seed=9_999_999)

    failure_seed = 50_001
    f_low = frozenset(env_low.reset(seed=failure_seed).failed)
    f_high = frozenset(env_high.reset(seed=failure_seed).failed)
    assert f_low == f_high

    same_again = frozenset(env_low.reset(seed=failure_seed).failed)
    assert same_again == f_low

    diff_found = False
    for s in range(50_002, 50_400):
        if frozenset(env_low.reset(seed=s).failed) != f_low:
            diff_found = True
            break
    assert diff_found, "expected different failure_seed to change the initial failure set"
