import networkx as nx

from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.evaluation.benchmarks import evaluate_policies, rollout_policy
from cascading_rl.policies.degree_policy import choose_highest_degree_failed_node
from cascading_rl.policies.random_policy import choose_random_failed_node


def test_rollout_policy_handles_zero_failure_episode():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, seed=0)

    result = rollout_policy(env, choose_highest_degree_failed_node, seed=0, tau=0.5)

    assert result.steps == 0
    assert result.remaining_failed_nodes == 0
    assert result.final_anc == 1.0
    assert result.threshold_step == 0


def test_rollout_policy_counts_rounds_after_budget_reset():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=1, max_rounds=3, seed=0)

    env.reset(seed=0)
    env.state.active = {0}
    env.state.failed = {1, 2, 3}
    env.state.frontier = set()
    env.state.loads = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0}
    env.remaining_budget = 1
    env.current_round = 1
    env.reset = lambda seed=None: env.observe()

    result = rollout_policy(env, choose_highest_degree_failed_node, tau=0.9)

    assert result.steps == 3
    assert result.rounds == 3
    assert result.remaining_failed_nodes == 0
    assert result.threshold_round == 3


def test_evaluate_policies_uses_matched_seed_rollouts():
    graph = nx.star_graph(3)

    def env_factory(seed: int) -> RecoveryEnv:
        return RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, seed=seed)

    def deterministic_policy(observation: RecoveryObservation) -> object:
        return choose_highest_degree_failed_node(observation)

    summaries = evaluate_policies(
        {
            "degree": deterministic_policy,
            "random": lambda observation: choose_random_failed_node(
                observation,
            ),
        },
        env_factory,
        seeds=[0, 1, 2],
        tau=0.5,
    )

    assert set(summaries) == {"degree", "random"}
    assert summaries["degree"].final_anc.mean >= 0.0
    assert summaries["degree"].threshold_hit_fraction.mean >= 0.0


def test_rollout_policy_supports_batch_actions():
    graph = nx.path_graph(5)
    env = RecoveryEnv(graph, alpha=1.0, pfail=0.0, budget=2, max_rounds=3, seed=0)

    env.reset(seed=0)
    env.state.active = {0}
    env.state.failed = {1, 2, 3, 4}
    env.state.frontier = {1}
    env.state.loads = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    env.state.capacities = {0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0}
    env.reset = lambda seed=None: env.observe()

    def batch_policy(observation: RecoveryObservation) -> list[int]:
        return list(observation.valid_actions[: observation.remaining_budget])

    result = rollout_policy(env, batch_policy, tau=0.9)

    assert result.steps >= 1
    assert result.rounds >= 1
