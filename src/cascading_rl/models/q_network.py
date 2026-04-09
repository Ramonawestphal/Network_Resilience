from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
from torch import nn

from cascading_rl.envs.recovery import RecoveryObservation
from cascading_rl.reproducibility import REPO_ROOT
from cascading_rl.models.gnn import (
    FEATURE_NAMES,
    GLOBAL_FEATURE_NAMES,
    LEGACY_FEATURE_NAMES,
    LEGACY_GLOBAL_FEATURE_NAMES,
    GlobalReadout,
    GraphStateEncoder,
    GraphTensor,
    observation_to_global_features,
    observation_to_graph_tensor,
    resolve_feature_names,
    resolve_global_feature_names,
)

_KNOWN_NODE_FEATURES = frozenset(FEATURE_NAMES) | frozenset(LEGACY_FEATURE_NAMES)
_KNOWN_GLOBAL_FEATURES = frozenset(GLOBAL_FEATURE_NAMES) | frozenset(
    LEGACY_GLOBAL_FEATURE_NAMES
)

Node = Hashable


@dataclass(frozen=True)
class QNetworkConfig:
    input_dim: int = len(FEATURE_NAMES)
    hidden_dim: int = 64
    embed_dim: int = 64
    num_layers: int = 2
    use_global_features: bool = False
    active_node_features: tuple[str, ...] | None = None
    active_global_features: tuple[str, ...] | None = None
    use_virtual_node: bool = False

    def __post_init__(self) -> None:
        if self.active_node_features is not None:
            unknown = tuple(
                feature_name
                for feature_name in self.active_node_features
                if feature_name not in _KNOWN_NODE_FEATURES
            )
            if unknown:
                raise ValueError(f"Unknown node feature(s): {unknown}")
            object.__setattr__(self, "input_dim", len(self.active_node_features))
        if self.active_global_features is not None:
            unknown = tuple(
                feature_name
                for feature_name in self.active_global_features
                if feature_name not in _KNOWN_GLOBAL_FEATURES
            )
            if unknown:
                raise ValueError(f"Unknown global feature(s): {unknown}")
            if not self.active_global_features:
                object.__setattr__(self, "use_global_features", False)

    @property
    def active_node_feature_names(self) -> tuple[str, ...]:
        if self.active_node_features is not None:
            return self.active_node_features
        return resolve_feature_names(self.input_dim)

    @property
    def active_global_feature_names(self) -> tuple[str, ...]:
        if not self.use_global_features:
            return ()
        if self.active_global_features is not None:
            return self.active_global_features
        if self.active_node_features is not None:
            return GLOBAL_FEATURE_NAMES
        return resolve_global_feature_names(self.input_dim)

    @property
    def num_active_global_features(self) -> int:
        return len(self.active_global_feature_names)

    @classmethod
    def from_dict(cls, values: dict) -> "QNetworkConfig":
        config_values = dict(values)
        return cls(**config_values)


class RecoveryQNetwork(nn.Module):
    """FINDER-style node scoring network for failed-node reactivation."""

    def __init__(self, config: QNetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or QNetworkConfig()
        self.feature_names = self.config.active_node_feature_names
        self.global_feature_names = self.config.active_global_feature_names
        self.encoder = GraphStateEncoder(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
        )

        global_out_dim = self.config.embed_dim // 2 if self.config.use_global_features else 0
        self.global_readout = None
        if self.config.use_global_features:
            self.global_readout = GlobalReadout(
                embed_dim=self.config.embed_dim,
                global_feat_dim=self.config.num_active_global_features,
                out_dim=global_out_dim,
            )
        self.q_head = nn.Sequential(
            nn.Linear(self.config.embed_dim + global_out_dim, self.config.embed_dim),
            nn.ReLU(),
            nn.Linear(self.config.embed_dim, 1),
        )

    def forward(
        self,
        graph_tensor: GraphTensor,
        global_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_embeddings = self.encoder(graph_tensor)
        num_real_nodes = len(graph_tensor.node_ids)
        readout_embeddings = node_embeddings[:num_real_nodes] if self.config.use_virtual_node else node_embeddings

        if self.config.use_global_features:
            if global_features is None:
                raise ValueError("global_features are required when use_global_features=True.")
            assert self.global_readout is not None
            global_vec = self.global_readout(readout_embeddings, global_features)
            global_expanded = global_vec.unsqueeze(0).expand(readout_embeddings.size(0), -1)
            node_input = torch.cat([readout_embeddings, global_expanded], dim=1)
        else:
            node_input = readout_embeddings

        q_values = self.q_head(node_input).squeeze(-1)
        valid_mask = (
            graph_tensor.valid_mask[:num_real_nodes]
            if self.config.use_virtual_node
            else graph_tensor.valid_mask
        )
        return q_values.masked_fill(~valid_mask, -1e9)

    def score_observation(
        self,
        observation: RecoveryObservation,
        device: torch.device | str | None = None,
    ) -> tuple[GraphTensor, torch.Tensor]:
        graph_tensor = observation_to_graph_tensor(
            observation,
            use_virtual_node=self.config.use_virtual_node,
            feature_names=self.feature_names,
            device=device,
        )
        global_features = None
        if self.config.use_global_features:
            global_features = observation_to_global_features(
                observation,
                global_feature_names=self.global_feature_names,
            )
            if device is not None:
                global_features = global_features.to(device)
        return graph_tensor, self(graph_tensor, global_features)


def select_action(
    model: RecoveryQNetwork,
    observation: RecoveryObservation,
    *,
    epsilon: float = 0.0,
    rng: Random | None = None,
    device: torch.device | str | None = None,
) -> Node:
    """Choose a failed node using epsilon-greedy action selection."""
    valid_actions = observation.valid_actions
    if not valid_actions:
        raise ValueError("No valid failed-node actions are available.")

    rng = rng or Random()
    if rng.random() < epsilon:
        return rng.choice(valid_actions)

    model.eval()
    with torch.no_grad():
        graph_tensor, q_values = model.score_observation(observation, device=device)
    best_index = int(torch.argmax(q_values).item())
    return graph_tensor.node_ids[best_index]


def select_top_b(
    model: RecoveryQNetwork,
    observation: RecoveryObservation,
    budget: int,
    *,
    epsilon: float = 0.0,
    rng: Random | None = None,
    device: torch.device | str | None = None,
) -> list[Node]:
    """Select up to B failed nodes in one forward pass, ranked by Q-value."""
    valid_actions = observation.valid_actions
    if not valid_actions:
        raise ValueError("No valid failed-node actions available.")

    rng = rng or Random()
    b = min(budget, len(valid_actions))
    if rng.random() < epsilon:
        return rng.sample(list(valid_actions), b)

    model.eval()
    with torch.no_grad():
        graph_tensor, q_values = model.score_observation(observation, device=device)

    valid_indices = [graph_tensor.node_to_index[node] for node in valid_actions]
    valid_q = [(q_values[index].item(), graph_tensor.node_ids[index]) for index in valid_indices]
    valid_q.sort(key=lambda item: item[0], reverse=True)
    return [node for _, node in valid_q[:b]]


def build_greedy_policy(
    model: RecoveryQNetwork,
    *,
    device: torch.device | str | None = None,
    batch_actions: bool = False,
) -> Callable[[RecoveryObservation], Node | list[Node]]:
    """Wrap the Q-network as a greedy policy compatible with evaluation helpers."""

    def policy(observation: RecoveryObservation) -> Node | list[Node]:
        if batch_actions:
            return select_top_b(
                model,
                observation,
                budget=observation.remaining_budget,
                epsilon=0.0,
                device=device,
            )
        return select_action(model, observation, epsilon=0.0, device=device)

    return policy


def load_q_network(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[RecoveryQNetwork, dict]:
    """Load a saved learner checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    checkpoint = torch.load(path, map_location=map_location)
    model_config = QNetworkConfig.from_dict(checkpoint["model_config"])
    model = RecoveryQNetwork(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(map_location)
    model.eval()
    return model, checkpoint
