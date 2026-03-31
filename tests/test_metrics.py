import networkx as nx

from cascading_rl.metrics.connectivity import (
    accumulated_normalized_connectivity,
    connected_component_sizes,
    largest_component_ratio,
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
