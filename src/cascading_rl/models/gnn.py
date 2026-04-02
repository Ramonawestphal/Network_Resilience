from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import torch
from torch import nn

from cascading_rl.envs.recovery import RecoveryObservation

Node = Hashable
VIRTUAL_NODE_ID = "__virtual__"

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
    "frontier_fraction",
    "mean_load_capacity_ratio",
    "max_load_capacity_ratio",
)


def _canonicalize_feature_subset(
    available_features: tuple[str, ...],
    requested_features: tuple[str, ...] | None,
    *,
    feature_group: str,
) -> tuple[str, ...]:
    if requested_features is None:
        return available_features

    requested_set = set(requested_features)
    if len(requested_set) != len(requested_features):
        raise ValueError(f"{feature_group} feature subsets must not contain duplicates.")

    unknown_features = tuple(
        feature_name
        for feature_name in requested_features
        if feature_name not in available_features
    )
    if unknown_features:
        raise ValueError(
            f"Unknown {feature_group} feature(s): {unknown_features}. "
            f"Expected a subset of {available_features}."
        )

    return tuple(
        feature_name
        for feature_name in available_features
        if feature_name in requested_set
    )


def observation_to_global_features(
    observation: RecoveryObservation,
    active_global_features: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Compute explicit global scalars for an observation.

    Parameters
    ----------
    observation:
        The environment observation to featurize.
    active_global_features:
        Optional subset of `GLOBAL_FEATURE_NAMES` to include. When provided,
        features are returned in the canonical order defined by
        `GLOBAL_FEATURE_NAMES`, not in caller order.
    """
    num_nodes = observation.graph.number_of_nodes()
    active = observation.active
    failed = observation.failed

    ratios = [
        observation.loads[v] / observation.capacities[v]
        if observation.capacities[v] > 0 else 0.0
        for v in active
    ] or [0.0]

    global_features = torch.tensor(
        [
            len(failed) / max(num_nodes, 1),
            len(observation.frontier) / max(num_nodes, 1),
            sum(ratios) / len(ratios),
            max(ratios),
        ],
        dtype=torch.float32,
    )
    selected_global_features = _canonicalize_feature_subset(
        GLOBAL_FEATURE_NAMES,
        active_global_features,
        feature_group="global",
    )
    selected_global_feature_set = set(selected_global_features)
    feature_mask = torch.tensor(
        [feature_name in selected_global_feature_set for feature_name in GLOBAL_FEATURE_NAMES],
        dtype=torch.bool,
    )
    return global_features[feature_mask]


class GlobalReadout(nn.Module):
    def __init__(self, embed_dim: int, global_feat_dim: int, out_dim: int) -> None:
        super().__init__()
        # mean + max pool → 2*embed_dim, plus explicit global features
        self.proj = nn.Linear(2 * embed_dim + global_feat_dim, out_dim)

    def forward(
        self,
        node_embeddings: torch.Tensor,   # (N, embed_dim)
        global_features: torch.Tensor,   # (global_feat_dim,)
    ) -> torch.Tensor:
        pooled = torch.cat([
            node_embeddings.mean(dim=0),
            node_embeddings.max(dim=0).values,
            global_features,
        ])  # (2*embed_dim + global_feat_dim,)
        return torch.relu(self.proj(pooled))  # (out_dim,)


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
    active_node_features: tuple[str, ...] | None = None,
    use_virtual_node: bool = False,
    device: torch.device | str | None = None,
) -> GraphTensor:
    """Convert a `RecoveryObservation` into graph tensors for the learner.

    Parameters
    ----------
    observation:
        The environment observation to featurize.
    active_node_features:
        Optional subset of `FEATURE_NAMES` to include. When provided, features
        are returned in the canonical order defined by `FEATURE_NAMES`, not in
        caller order.
    use_virtual_node:
        Whether to append the virtual node and its incident edges.
    device:
        Optional device to move the returned tensors to.
    """
    node_ids = tuple(observation.graph.nodes())
    node_to_index = {node: index for index, node in enumerate(node_ids)}
    num_real_nodes = len(node_ids)
    num_nodes = num_real_nodes + int(use_virtual_node)
    if use_virtual_node:
        node_to_index[VIRTUAL_NODE_ID] = num_real_nodes

    selected_node_features = _canonicalize_feature_subset(
        FEATURE_NAMES,
        active_node_features,
        feature_group="node",
    )
    selected_node_feature_set = set(selected_node_features)
    feature_mask = torch.tensor(
        [feature_name in selected_node_feature_set for feature_name in FEATURE_NAMES],
        dtype=torch.bool,
    )

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

    if use_virtual_node and num_real_nodes > 0:
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
        node_features=node_features[:, feature_mask],
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
