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
