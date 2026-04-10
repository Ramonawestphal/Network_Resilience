import networkx as nx
import pytest

from cascading_rl.envs.recovery import RecoveryEnv, RecoveryObservation
from cascading_rl.evaluation.benchmarks import (
    EpisodeResult,
    _compute_step_metrics,
    compare_policy_pair,
    evaluate_policies,
    final_nc_failure_threshold_for_reporting,
    rollout_policy,
    summarize_episode_results,
)
from cascading_rl.policies.degree_policy import choose_highest_degree_failed_node
from cascading_rl.policies.random_policy import choose_random_failed_node


def test_rollout_policy_handles_zero_failure_episode():
    graph = nx.path_graph(4)
    env = RecoveryEnv(graph, alpha=0.2, pfail=0.0, budget=2, seed=0)

    result = rollout_policy(env, choose_highest_degree_failed_node, seed=0)

    assert result.steps == 0
    assert result.remaining_failed_nodes == 0
    assert result.final_nc == 1.0


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

    result = rollout_policy(env, choose_highest_degree_failed_node)

    assert result.steps == 3
    assert result.rounds == 3
    assert result.remaining_failed_nodes == 0


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
    )

    assert set(summaries) == {"degree", "random"}
    assert summaries["degree"].final_nc.mean >= 0.0
    assert 0.0 <= summaries["degree"].solved_fraction.mean <= 1.0


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

    result = rollout_policy(env, batch_policy)

    assert result.steps >= 1
    assert result.rounds >= 1


def test_compute_step_metrics_ranks_batches_against_same_size_candidates(monkeypatch):
    graph = nx.path_graph(4)
    observation = RecoveryObservation(
        graph=graph,
        loads={0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
        capacities={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0},
        active=frozenset({0}),
        failed=frozenset({1, 2, 3}),
        frontier=frozenset({1}),
        remaining_budget=2,
        budget=2,
        current_round=1,
        max_rounds=3,
    )

    delta_by_batch = {
        (1, 2): 0.6,
        (1, 3): 0.9,
        (2, 3): 0.4,
    }

    def fake_delta_nc_after_round_batch(_base_state, candidate_batch):
        return delta_by_batch[tuple(sorted(candidate_batch))]

    monkeypatch.setattr(
        "cascading_rl.evaluation.benchmarks.delta_nc_after_round_batch",
        fake_delta_nc_after_round_batch,
    )

    metrics = _compute_step_metrics(observation, [1, 2], current_round=1)

    assert metrics.nc_gain == pytest.approx(0.6)
    assert metrics.greedy_nc_gain == pytest.approx(0.9)
    assert metrics.action_rank == 2


def test_compare_policy_pair_rejects_empty_episode_lists():
    with pytest.raises(
        ValueError,
        match="Episode lists for 'policy_a' and 'policy_b' must not be empty",
    ):
        compare_policy_pair([], [], name_a="policy_a", name_b="policy_b")


def test_final_nc_failure_threshold_for_reporting_respects_env_and_default():
    assert final_nc_failure_threshold_for_reporting(None) == pytest.approx(0.3)
    assert final_nc_failure_threshold_for_reporting({}) == pytest.approx(0.3)
    assert final_nc_failure_threshold_for_reporting(
        {"abandonment_nc_threshold": 0.25}
    ) == pytest.approx(0.25)


def test_summarize_episode_results_counts_unsolved_low_final_nc():
    episodes = [
        EpisodeResult(
            total_reward=0.0,
            final_nc=0.5,
            steps=1,
            rounds=1,
            remaining_failed_nodes=0,
        ),
        EpisodeResult(
            total_reward=0.0,
            final_nc=0.2,
            steps=2,
            rounds=2,
            remaining_failed_nodes=2,
        ),
        EpisodeResult(
            total_reward=0.0,
            final_nc=0.35,
            steps=2,
            rounds=2,
            remaining_failed_nodes=1,
        ),
    ]
    summary = summarize_episode_results(episodes, final_nc_failure_threshold=0.3)
    assert summary.unsolved_low_final_nc_count == 1
    assert summary.unsolved_low_final_nc_fraction == pytest.approx(1.0 / 3.0)
    assert summary.final_nc_failure_threshold_used == pytest.approx(0.3)

    boundary = summarize_episode_results(
        [
            EpisodeResult(
                total_reward=0.0,
                final_nc=0.3,
                steps=1,
                rounds=1,
                remaining_failed_nodes=1,
            )
        ],
        final_nc_failure_threshold=0.3,
    )
    assert boundary.unsolved_low_final_nc_count == 0
