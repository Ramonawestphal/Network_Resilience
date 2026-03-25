import networkx as nx

from cascading_rl.dynamics.cascade import (
    CascadeState,
    initialize_loads_and_capacities,
    propagate_cascade,
    reactivate_node,
)


def test_initialize_loads_and_capacities_uses_degree_rule():
    graph = nx.path_graph(3)

    loads, capacities = initialize_loads_and_capacities(graph, alpha=0.5)

    assert loads == {0: 1.0, 1: 2.0, 2: 1.0}
    assert capacities == {0: 1.5, 1: 3.0, 2: 1.5}


def test_propagate_cascade_fails_overloaded_neighbors_round_by_round():
    graph = nx.path_graph(3)
    loads = {0: 0.0, 1: 3.0, 2: 0.0}
    capacities = {0: 1.0, 1: 2.0, 2: 1.0}
    active = {0, 1, 2}

    failed_order = propagate_cascade(graph, loads, capacities, active)

    assert failed_order == [1, 0, 2]
    assert active == set()
    assert loads == {0: 0.0, 1: 0.0, 2: 0.0}


def test_reactivate_node_restores_failed_node_with_zero_load():
    graph = nx.path_graph(3)
    state = CascadeState(
        graph=graph,
        loads={0: 1.0, 1: 2.0, 2: 1.0},
        capacities={0: 2.0, 1: 2.5, 2: 2.0},
        active={0, 2},
        failed={1},
        frontier={1},
    )

    next_state = reactivate_node(state, 1)

    assert next_state.active == {0, 1, 2}
    assert next_state.failed == set()
    assert next_state.loads[1] == 0.0
