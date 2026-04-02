import networkx as nx

from cascading_rl.data.loaders import prepare_graph


def test_prepare_graph_makes_graph_simple_and_uses_largest_component():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 0), (1, 1), (2, 3)])
    graph.add_node(4)

    prepared = prepare_graph(graph)

    assert isinstance(prepared, nx.Graph)
    assert set(prepared.nodes()) == {0, 1}
    assert list(nx.selfloop_edges(prepared)) == []


def test_prepare_graph_directed_uses_weakly_connected_components():
    graph = nx.DiGraph()
    graph.add_edges_from([(0, 1), (1, 2)])
    graph.add_edges_from([(10, 11)])  # smaller weak component
    graph.add_node(99)  # isolate, dropped

    prepared = prepare_graph(graph, undirected=False, largest_component_only=True)

    assert isinstance(prepared, nx.DiGraph)
    assert set(prepared.nodes()) == {0, 1, 2}
    assert set(prepared.edges()) == {(0, 1), (1, 2)}
