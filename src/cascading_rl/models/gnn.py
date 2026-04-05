from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import torch
from torch import nn

from cascading_rl.envs.recovery import RecoveryObservation

Node = Hashable

VIRTUAL_NODE: Hashable = object()

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

GLOBAL_FEATURE_NAMES = (
    "failed_fraction",
    "mean_load_capacity_ratio",
    "max_load_capacity_ratio",
)


def observation_to_global_features(
    observation: RecoveryObservation,
    *,
    global_feature_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    global_feature_names = global_feature_names or GLOBAL_FEATURE_NAMES
    num_nodes = observation.graph.number_of_nodes()
    active = observation.active
    failed = observation.failed

    ratios = [
        observation.loads[node] / observation.capacities[node]
        if observation.capacities[node] > 0
        else 0.0
        for node in active
    ] or [0.0]

    feature_values = {
        "failed_fraction": len(failed) / max(num_nodes, 1),
        "mean_load_capacity_ratio": sum(ratios) / len(ratios),
        "max_load_capacity_ratio": max(ratios),
        "current_round_norm": float(observation.current_round) / max(
            1.0, float(observation.max_rounds)
        ),
    }
    return torch.tensor(
        [feature_values[name] for name in global_feature_names],
        dtype=torch.float32,
    )


def resolve_feature_names(input_dim: int) -> tuple[str, ...]:
    if input_dim == len(FEATURE_NAMES):
        return FEATURE_NAMES
    raise ValueError(f"Unsupported node-feature width: {input_dim}")


def resolve_global_feature_names(input_dim: int) -> tuple[str, ...]:
    if input_dim == len(FEATURE_NAMES):
        return GLOBAL_FEATURE_NAMES
    raise ValueError(f"Unsupported node-feature width: {input_dim}")


class GlobalReadout(nn.Module):
    def __init__(self, embed_dim: int, global_feat_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(2 * embed_dim + global_feat_dim, out_dim)

    def forward(self, node_embeddings: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [
                node_embeddings.mean(dim=0),
                node_embeddings.max(dim=0).values,
                global_features,
            ]
        )
        return torch.relu(self.proj(pooled))


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
    *,
    use_virtual_node: bool = False,
    feature_names: tuple[str, ...] | None = None,
    device: torch.device | str | None = None,
) -> GraphTensor:
    """Convert a recovery observation into graph tensors."""
    feature_names = feature_names or FEATURE_NAMES
    node_ids = tuple(observation.graph.nodes())
    node_to_index = {node: index for index, node in enumerate(node_ids)}
    num_real_nodes = len(node_ids)
    num_nodes = num_real_nodes + int(use_virtual_node)
    if use_virtual_node:
        node_to_index[VIRTUAL_NODE] = num_real_nodes

    max_load = max((float(value) for value in observation.loads.values()), default=1.0)
    max_capacity = max((float(value) for value in observation.capacities.values()), default=1.0)
    max_degree = max((observation.graph.degree(node) for node in node_ids), default=1)
    scale = max(max_load, max_capacity, 1.0)

    node_features = torch.zeros((num_nodes, len(feature_names)), dtype=torch.float32)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    adjacency = torch.eye(num_nodes, dtype=torch.float32)

    for node in node_ids:
        index = node_to_index[node]
        load = float(observation.loads[node])
        capacity = float(observation.capacities[node])
        degree = float(observation.graph.degree(node))
        canonical_budget = float(observation.remaining_budget) / max(1.0, float(num_real_nodes))
        legacy_budget = float(observation.remaining_budget) / max(1.0, float(num_real_nodes))
        legacy_round = float(observation.current_round) / max(1.0, float(num_real_nodes))
        feature_values = {
            "load_norm": load / scale,
            "capacity_norm": capacity / scale,
            "load_capacity_ratio": (load / capacity) if capacity > 0.0 else 0.0,
            "failed_flag": 1.0 if node in observation.failed else 0.0,
            "active_flag": 1.0 if node in observation.active else 0.0,
            "frontier_flag": 1.0 if node in observation.frontier else 0.0,
            "budget_coverage": canonical_budget,
            "remaining_budget_norm": legacy_budget,
            "current_round_norm": legacy_round,
            "degree_norm": degree / max(1.0, float(max_degree)),
        }
        node_features[index] = torch.tensor(
            [feature_values[name] for name in feature_names],
            dtype=torch.float32,
        )
        valid_mask[index] = node in observation.failed

    if use_virtual_node:
        node_features[num_real_nodes] = node_features[:num_real_nodes].mean(dim=0)

    for left_node, right_node in observation.graph.edges():
        left_index = node_to_index[left_node]
        right_index = node_to_index[right_node]
        adjacency[left_index, right_index] = 1.0
        adjacency[right_index, left_index] = 1.0

    if use_virtual_node:
        for index in range(num_real_nodes):
            adjacency[index, num_real_nodes] = 1.0
            adjacency[num_real_nodes, index] = 1.0

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
        for _ in range(max(0, num_layers - 1)):
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
