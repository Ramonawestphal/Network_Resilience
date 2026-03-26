from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import torch
from torch import nn

from cascading_rl.envs.recovery import RecoveryObservation

Node = Hashable
FEATURE_NAMES = (
    "load_norm",
    "capacity_norm",
    "load_capacity_ratio",
    "failed_flag",
    "active_flag",
    "frontier_flag",
    "remaining_budget_norm",
    "current_round_norm",
    "degree_norm",
)


@dataclass(frozen=True)
class GraphTensor:
    node_features: torch.Tensor
    adjacency: torch.Tensor
    valid_mask: torch.Tensor
    node_ids: tuple[Node, ...]
    node_to_index: dict[Node, int]

    def to(self, device: torch.device | str) -> "GraphTensor":
        return GraphTensor(
            node_features=self.node_features.to(device),
            adjacency=self.adjacency.to(device),
            valid_mask=self.valid_mask.to(device),
            node_ids=self.node_ids,
            node_to_index=self.node_to_index,
        )


def observation_to_graph_tensor(
    observation: RecoveryObservation,
    device: torch.device | str | None = None,
) -> GraphTensor:
    """Convert a RecoveryObservation into tensors for the learner."""
    node_ids = tuple(observation.graph.nodes())
    node_to_index = {node: index for index, node in enumerate(node_ids)}
    num_nodes = len(node_ids)

    max_load = max((float(value) for value in observation.loads.values()), default=1.0)
    max_capacity = max((float(value) for value in observation.capacities.values()), default=1.0)
    max_degree = max((observation.graph.degree(node) for node in node_ids), default=1)
    scale = max(max_load, max_capacity, 1.0)

    node_features = torch.zeros((num_nodes, len(FEATURE_NAMES)), dtype=torch.float32)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    adjacency = torch.eye(num_nodes, dtype=torch.float32)

    for node in node_ids:
        index = node_to_index[node]
        load = float(observation.loads[node])
        capacity = float(observation.capacities[node])
        degree = float(observation.graph.degree(node))
        node_features[index] = torch.tensor(
            [
                load / scale,
                capacity / scale,
                (load / capacity) if capacity > 0.0 else 0.0,
                1.0 if node in observation.failed else 0.0,
                1.0 if node in observation.active else 0.0,
                1.0 if node in observation.frontier else 0.0,
                float(observation.remaining_budget) / max(1.0, float(num_nodes)),
                float(observation.current_round) / max(1.0, float(num_nodes)),
                degree / max(1.0, float(max_degree)),
            ],
            dtype=torch.float32,
        )
        valid_mask[index] = node in observation.failed

    for left_node, right_node in observation.graph.edges():
        left_index = node_to_index[left_node]
        right_index = node_to_index[right_node]
        adjacency[left_index, right_index] = 1.0
        adjacency[right_index, left_index] = 1.0

    degrees = adjacency.sum(dim=1)
    inv_sqrt_degree = torch.pow(degrees, -0.5)
    inv_sqrt_degree[torch.isinf(inv_sqrt_degree)] = 0.0
    normalized_adjacency = (
        inv_sqrt_degree.unsqueeze(1) * adjacency * inv_sqrt_degree.unsqueeze(0)
    )

    graph_tensor = GraphTensor(
        node_features=node_features,
        adjacency=normalized_adjacency,
        valid_mask=valid_mask,
        node_ids=node_ids,
        node_to_index=node_to_index,
    )
    return graph_tensor if device is None else graph_tensor.to(device)


class GraphMessagePassingLayer(nn.Module):
    """Simple adjacency-aware message passing layer."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(in_dim, out_dim)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        self_term = self.self_linear(node_features)
        neighbor_term = self.neighbor_linear(adjacency @ node_features)
        return torch.relu(self.layer_norm(self_term + neighbor_term))


class GraphStateEncoder(nn.Module):
    """Compact message-passing encoder over the current recovery graph."""

    def __init__(
        self,
        input_dim: int = len(FEATURE_NAMES),
        hidden_dim: int = 64,
        embed_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        layers: list[nn.Module] = []
        layer_in_dim = input_dim
        for _ in range(max(1, num_layers - 1)):
            layers.append(GraphMessagePassingLayer(layer_in_dim, hidden_dim))
            layer_in_dim = hidden_dim
        layers.append(GraphMessagePassingLayer(layer_in_dim, embed_dim))
        self.layers = nn.ModuleList(layers)
        self.output_dim = embed_dim

    def forward(self, graph_tensor: GraphTensor) -> torch.Tensor:
        hidden = graph_tensor.node_features
        for layer in self.layers:
            hidden = layer(hidden, graph_tensor.adjacency)
        return hidden
