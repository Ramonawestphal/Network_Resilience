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
