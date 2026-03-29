from random import Random

import networkx as nx

from cascading_rl.envs.recovery import RecoveryObservation
from cascading_rl.policies.betweenness_policy import choose_highest_betweenness_failed_node
from cascading_rl.policies.degree_policy import choose_highest_degree_failed_node
from cascading_rl.policies.greedy_policy import choose_greedy_anc_node
from cascading_rl.policies.random_policy import choose_random_failed_node
from cascading_rl.policies.risk_policy import choose_highest_overload_risk_node


def make_observation() -> RecoveryObservation:
    graph = nx.star_graph(3)
    return RecoveryObservation(
        graph=graph,
        loads={0: 0.0, 1: 1.8, 2: 1.0, 3: 1.0},
        capacities={0: 2.0, 1: 2.0, 2: 2.0, 3: 2.0},
        active=frozenset({1, 2}),
        failed=frozenset({0, 3}),
        frontier=frozenset({3}),
        remaining_budget=2,
        budget = 2,
        current_round=1,
        max_rounds=5
    )


def test_random_policy_returns_failed_node():
    observation = make_observation()

    action = choose_random_failed_node(observation, rng=Random(0))

    assert action in observation.failed


def test_degree_policy_prefers_high_degree_center():
    observation = make_observation()

    action = choose_highest_degree_failed_node(observation)

    assert action == 0


def test_risk_policy_prefers_node_adjacent_to_stressed_neighbors():
    observation = make_observation()

    action = choose_highest_overload_risk_node(observation)

    assert action == 0


def test_greedy_policy_prefers_largest_connectivity_gain():
    observation = make_observation()

    action = choose_greedy_anc_node(observation)

    assert action == 0


def test_betweenness_policy_prefers_bridge_like_node():
    observation = make_observation()

    action = choose_highest_betweenness_failed_node(observation)

    assert action == 0
