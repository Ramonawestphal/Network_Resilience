import unittest

import networkx as nx

from graph_functions import CascadingFailureProcess


class TestCascadingFailureProcess(unittest.TestCase):
    def _process_with_graph(self, graph: nx.Graph, alpha: float = 0.2) -> CascadingFailureProcess:
        process = CascadingFailureProcess(
            n=max(2, graph.number_of_nodes()),
            m=1,
            alpha=alpha,
            seed=123,
            redistribution_mode="capacity_weighted",
        )
        process.graph = graph
        process.n = graph.number_of_nodes()
        return process

    def test_initial_load_and_capacity_match_spec(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (1, 2)])
        process = self._process_with_graph(graph, alpha=0.25)
        process.reset(initial_failures=[])

        for node in graph.nodes():
            self.assertEqual(process.initial_load[node], float(graph.degree(node)))
            self.assertEqual(process.current_load[node], float(graph.degree(node)))
            self.assertEqual(process.capacity[node], 1.25 * float(graph.degree(node)))

    def test_capacity_weighted_redistribution_formula(self) -> None:
        # Graph: 0 connected to 1 and 2; node 1 additionally connected to 3.
        # Degrees: k0=2, k1=2, k2=1, k3=1
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (0, 2), (1, 3)])
        process = self._process_with_graph(graph, alpha=0.2)
        process.reset(initial_failures=[0])

        result = process.step()

        # Outgoing load from failed node 0 is 2.0.
        # Active neighbors capacities: C1=2.4, C2=1.2, total=3.6
        # Delta1=2*(2.4/3.6)=4/3 ; Delta2=2*(1.2/3.6)=2/3
        self.assertAlmostEqual(process.current_load[1], 2.0 + (4.0 / 3.0), places=10)
        self.assertAlmostEqual(process.current_load[2], 1.0 + (2.0 / 3.0), places=10)
        self.assertAlmostEqual(process.current_load[0], 0.0, places=10)
        self.assertAlmostEqual((4.0 / 3.0) + (2.0 / 3.0), 2.0, places=10)
        self.assertGreaterEqual(result.info["active_count"], 0)

    def test_failed_nodes_do_not_receive_redistributed_load(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (0, 2), (1, 3)])
        process = self._process_with_graph(graph, alpha=0.2)
        process.reset(initial_failures=[0, 2])

        process.step()

        # Node 2 is failed before redistribution and must not receive load from 0.
        # Failed nodes in the current frontier still shed their own outgoing load to 0.
        self.assertAlmostEqual(process.current_load[2], 0.0, places=10)

    def test_no_active_neighbors_is_safe(self) -> None:
        graph = nx.Graph()
        graph.add_edge(0, 1)
        process = self._process_with_graph(graph, alpha=0.2)
        process.reset(initial_failures=[0, 1])

        result = process.step()
        self.assertAlmostEqual(process.current_load[0], 0.0, places=10)
        self.assertAlmostEqual(process.current_load[1], 0.0, places=10)
        self.assertEqual(result.info["new_failures"], [])

    def test_single_active_neighbor_gets_all_load(self) -> None:
        graph = nx.path_graph(3)  # 0-1-2
        process = self._process_with_graph(graph, alpha=0.2)
        process.reset(initial_failures=[0, 2])  # node 1 is the only active neighbor of each
        pre = process.current_load[1]

        process.step()
        # node 0 sends 1.0 and node 2 sends 1.0 -> node 1 gets all 2.0
        self.assertAlmostEqual(process.current_load[1], pre + 2.0, places=10)

    def test_multiple_simultaneous_overload_failures(self) -> None:
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (0, 2)])
        process = self._process_with_graph(graph, alpha=0.1)
        process.reset(initial_failures=[0])

        result = process.step()
        self.assertCountEqual(result.info["new_failures"], [1, 2])
        self.assertFalse(process.active[1])
        self.assertFalse(process.active[2])

    def test_disconnected_component_and_isolated_node_behavior(self) -> None:
        graph = nx.Graph()
        graph.add_nodes_from([0, 1, 2])
        graph.add_edge(0, 1)  # node 2 is isolated
        process = self._process_with_graph(graph, alpha=0.2)
        process.reset(initial_failures=[2])

        result = process.step()
        self.assertEqual(result.info["new_failures"], [])
        self.assertAlmostEqual(process.current_load[2], 0.0, places=10)

    def test_initial_hub_failure_can_drive_cascade(self) -> None:
        graph = nx.star_graph(4)  # center=0, leaves 1..4
        process = self._process_with_graph(graph, alpha=0.1)
        process.reset(initial_failures=[0])

        result = process.step()
        self.assertCountEqual(result.info["new_failures"], [1, 2, 3, 4])

    def test_full_collapse_and_cascade_stops(self) -> None:
        graph = nx.star_graph(3)  # center=0, leaves 1..3
        process = self._process_with_graph(graph, alpha=0.1)
        process.reset(initial_failures=[0])

        process.step()
        final_step = process.step()
        self.assertTrue(final_step.done)
        self.assertEqual(sum(1 for v in process.active.values() if v), 0)

    def test_no_initial_failures_finishes_immediately(self) -> None:
        graph = nx.path_graph(4)
        process = self._process_with_graph(graph, alpha=0.2)
        process.reset(initial_failures=[])

        self.assertTrue(process.done)
        step_result = process.step()
        self.assertTrue(step_result.done)
        self.assertEqual(step_result.info["new_failures"], [])

    def test_reactivate_node_resets_load_and_status(self) -> None:
        graph = nx.path_graph(3)
        process = self._process_with_graph(graph, alpha=0.2)
        process.reset(initial_failures=[1])
        process.current_load[1] = 9.0
        process.capacity[1] = -1.0

        changed = process.reactivate_node(1)
        self.assertTrue(changed)
        self.assertTrue(process.active[1])
        self.assertAlmostEqual(process.current_load[1], 0.0, places=10)
        self.assertAlmostEqual(process.capacity[1], (1.0 + process.alpha) * process.initial_load[1], places=10)

    def test_reactivated_node_becomes_future_active_recipient(self) -> None:
        graph = nx.path_graph(3)  # 0-1-2
        process = self._process_with_graph(graph, alpha=0.2)
        process.reset(initial_failures=[0, 1])  # only node 2 active
        process.reactivate_node(1)
        pre = process.current_load[1]

        process.step()
        self.assertGreater(process.current_load[1], pre)


if __name__ == "__main__":
    unittest.main()
