import networkx as nx

from cascading_rl.envs.recovery import RecoveryEnv


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

    assert reward > 0.0
    assert 0 in observation.active
    assert info["anc"] == 9 / 16
    assert done is False


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
    assert info["action_round"] == 1
    assert observation.current_round == 2
    assert observation.remaining_budget == 1


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


def test_obs_hops_masks_loads_beyond_one_hop_and_preserves_valid_actions():
    graph = nx.path_graph(7)
    env = RecoveryEnv(
        graph,
        alpha=1.0,
        pfail=0.0,
        budget=2,
        max_rounds=5,
        obs_hops=1,
        action_space="frontier",
    )
    env.reset(seed=0)
    env.state.active = {0, 2, 3, 4, 6}
    env.state.failed = {1, 5}
    env.state.frontier = {1}
    for node in graph.nodes():
        env.state.loads[node] = 1.0
        env.state.capacities[node] = 2.0

    observation = env.observe()

    assert observation.valid_actions == (1,)
    for node in graph.nodes():
        dist = min(nx.shortest_path_length(graph, node, failed) for failed in observation.failed)
        if dist > 1:
            assert observation.loads[node] == 0.0
            assert observation.capacities[node] == 0.0


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
