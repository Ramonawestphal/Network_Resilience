import networkx as nx

from cascading_rl.envs.recovery import RecoveryEnv


def test_environment_step_rewards_connectivity_gain():
    graph = nx.star_graph(3)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=2)

    observation = env.reset()
    env.state.active = {1, 2}
    env.state.failed = {0, 3}
    env.state.loads = {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}
    env.state.capacities = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}

    observation, reward, done, info = env.step(0)

    assert reward > 0.0
    assert 0 in observation.active
    assert info["anc"] == 9 / 16
    assert done is False
