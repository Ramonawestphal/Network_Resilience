import networkx as nx

from cascading_rl.envs.recovery import RecoveryEnv
from cascading_rl.evaluation import estimate_minimum_budget, evaluate_policies
from cascading_rl.graph.generation import make_graph_batch
from cascading_rl.policies import (
    choose_greedy_anc_node,
    choose_highest_betweenness_failed_node,
    choose_highest_degree_failed_node,
    choose_highest_overload_risk_node,
)


def test_foundations_pipeline_runs_end_to_end_on_generated_graphs():
    graphs = make_graph_batch(num_graphs=2, n_range=(30, 32), m=2, seed=7)
    graph = graphs[0]

    def env_factory(seed: int) -> RecoveryEnv:
        return RecoveryEnv(graph, alpha=0.2, pfail=0.1, budget=3, seed=seed)

    summaries = evaluate_policies(
        {
            "degree": choose_highest_degree_failed_node,
            "risk": choose_highest_overload_risk_node,
            "greedy": choose_greedy_anc_node,
            "betweenness": choose_highest_betweenness_failed_node,
        },
        env_factory=env_factory,
        seeds=range(5),
        tau=0.8,
    )

    assert set(summaries) == {"degree", "risk", "greedy", "betweenness"}
    for summary in summaries.values():
        assert 0.0 <= summary.final_anc.mean <= 1.0
        assert summary.steps.mean >= 0.0
        assert 0.0 <= summary.solved_fraction.mean <= 1.0
        assert 0.0 <= summary.threshold_hit_fraction.mean <= 1.0


def test_budget_search_runs_on_generated_graph():
    graph = make_graph_batch(num_graphs=1, n_range=(30, 30), m=2, seed=11)[0]

    minimum_budget, results = estimate_minimum_budget(
        graph,
        choose_highest_degree_failed_node,
        tau=0.5,
        budgets=range(1, 4),
        trials=5,
        alpha=0.2,
        pfail=0.1,
    )

    assert minimum_budget is None or minimum_budget in {1, 2, 3}
    assert set(results) == {1, 2, 3}
    for mean_anc, stderr in results.values():
        assert 0.0 <= mean_anc <= 1.0
        assert stderr >= 0.0


def test_environment_reward_matches_anc_gain_on_manual_state():
    graph = nx.star_graph(3)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=2)

    observation = env.reset()
    env.state.active = {1, 2}
    env.state.failed = {0, 3}
    env.state.loads = {0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0}
    env.state.capacities = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}

    _, reward, _, info = env.step(0)

    assert reward == info["anc"] - (2 / 16)
