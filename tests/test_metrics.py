import pytest
import networkx as nx

from cascading_rl.metrics.connectivity import (
    accumulated_normalized_connectivity,
    connected_component_sizes,
    largest_component_ratio,
)
from cascading_rl.evaluation.metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    compute_aggregate_metrics,
    compute_episode_metrics,
)


def test_connected_component_sizes_reflect_active_subgraph():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (2, 3)])

    sizes = connected_component_sizes(graph, {0, 1, 3})

    assert sorted(sizes) == [1, 2]


def test_accumulated_normalized_connectivity_matches_manual_value():
    graph = nx.path_graph(4)

    anc = accumulated_normalized_connectivity(graph, {0, 1, 3})

    assert anc == 5 / 16


def test_accumulated_normalized_connectivity_is_one_for_fully_connected_active_graph():
    graph = nx.path_graph(4)

    anc = accumulated_normalized_connectivity(graph, {0, 1, 2, 3})

    assert anc == 1.0


def test_accumulated_normalized_connectivity_handles_two_equal_components():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (2, 3)])

    anc = accumulated_normalized_connectivity(graph, {0, 1, 2, 3})

    assert anc == 0.5


def test_accumulated_normalized_connectivity_handles_single_component_subset():
    graph = nx.path_graph(4)

    anc = accumulated_normalized_connectivity(graph, {0, 1})

    assert anc == 0.25


def test_accumulated_normalized_connectivity_handles_single_active_node():
    graph = nx.path_graph(4)

    anc = accumulated_normalized_connectivity(graph, {0})

    assert anc == 0.0625


def test_largest_component_ratio_uses_total_graph_size():
    graph = nx.path_graph(5)

    ratio = largest_component_ratio(graph, {0, 1, 2})

    assert ratio == 3 / 5


# ---------------------------------------------------------------------------
# EpisodeMetrics tests
# ---------------------------------------------------------------------------


def test_compute_episode_metrics_recovered():
    em = compute_episode_metrics([0.3, 0.7, 1.0], recovered=True)

    assert em.recovered is True
    assert em.rounds_to_recovery == 3
    assert em.rounds_to_termination == 3
    assert em.anc_per_round == [0.3, 0.7, 1.0]
    assert em.mean_anc_conditional == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert em.mean_anc_unconditional == pytest.approx(2.0 / 3.0, rel=1e-6)


def test_compute_episode_metrics_failed():
    em = compute_episode_metrics([0.2, 0.3, 0.25], recovered=False)

    assert em.recovered is False
    assert em.rounds_to_recovery is None
    assert em.rounds_to_termination == 3
    assert em.anc_per_round == [0.2, 0.3, 0.25]
    assert em.mean_anc_conditional is None
    assert em.mean_anc_unconditional == pytest.approx(0.25, rel=1e-6)


def test_compute_aggregate_metrics_mixed():
    episodes = [
        compute_episode_metrics([0.4, 0.8, 1.0], recovered=True),   # recovered in 3 rounds
        compute_episode_metrics([0.3, 0.6, 1.0], recovered=True),   # recovered in 3 rounds
        compute_episode_metrics([0.2, 0.3, 0.25], recovered=False),  # failed, 3 rounds
    ]

    am = compute_aggregate_metrics(episodes)

    assert am.n_episodes == 3
    assert am.recovered_fraction == pytest.approx(2 / 3, rel=1e-6)

    # rounds_to_recovery: both recovered episodes have 3 rounds → mean=3, std=0
    assert am.mean_rounds_to_recovery == pytest.approx(3.0)
    assert am.std_rounds_to_recovery == pytest.approx(0.0)

    # rounds_to_termination_failed: 1 failed episode, 3 rounds → mean=3, std=0
    assert am.mean_rounds_to_termination_failed == pytest.approx(3.0)
    assert am.std_rounds_to_termination_failed == pytest.approx(0.0)

    # conditional ANC: mean of [mean([0.4,0.8,1.0]), mean([0.3,0.6,1.0])]
    #                = mean([0.7333..., 0.6333...]) ≈ 0.6833
    expected_cond = ((0.4 + 0.8 + 1.0) / 3 + (0.3 + 0.6 + 1.0) / 3) / 2
    assert am.mean_anc_conditional == pytest.approx(expected_cond, rel=1e-6)
    assert am.stderr_anc_conditional is not None

    # unconditional ANC: mean of [mean of each episode's trajectory]
    uncond_vals = [
        (0.4 + 0.8 + 1.0) / 3,
        (0.3 + 0.6 + 1.0) / 3,
        (0.2 + 0.3 + 0.25) / 3,
    ]
    expected_uncond = sum(uncond_vals) / 3
    assert am.mean_anc_unconditional == pytest.approx(expected_uncond, rel=1e-6)
    assert am.stderr_anc_unconditional >= 0.0

    # all episodes have length 3
    assert len(am.mean_anc_per_round) == 3
    assert len(am.n_per_round) == 3
    assert all(n == 3 for n in am.n_per_round)


def test_per_round_alignment_unequal_lengths():
    episodes = [
        compute_episode_metrics([0.1, 0.2, 0.3], recovered=False),          # length 3
        compute_episode_metrics([0.2, 0.4, 0.5, 0.7, 0.9], recovered=True), # length 5
        compute_episode_metrics(
            [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 1.0], recovered=True
        ),                                                                    # length 7
    ]

    am = compute_aggregate_metrics(episodes)

    assert len(am.mean_anc_per_round) == 7
    assert len(am.n_per_round) == 7

    # Round index 0: all 3 episodes → n=3
    assert am.n_per_round[0] == 3
    # Round index 2: all 3 episodes → n=3
    assert am.n_per_round[2] == 3
    # Round index 3: only length-5 and length-7 episodes → n=2
    assert am.n_per_round[3] == 2
    # Round index 4: only length-5 and length-7 episodes → n=2
    assert am.n_per_round[4] == 2
    # Round index 5: only length-7 episode → n=1
    assert am.n_per_round[5] == 1
    # Round index 6: only length-7 episode → n=1
    assert am.n_per_round[6] == 1
    # Value at index 6 should equal the single length-7 episode's value at that index
    assert am.mean_anc_per_round[6] == pytest.approx(1.0, rel=1e-6)
