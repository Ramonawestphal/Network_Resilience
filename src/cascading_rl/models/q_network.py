from __future__ import annotations

from collections.abc import Callable, Hashable
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
    GlobalReadout,
    GraphStateEncoder,
    GraphTensor,
    _canonicalize_feature_subset,
    observation_to_global_features,
    observation_to_graph_tensor,
)

Node = Hashable


@dataclass(frozen=True)
class QNetworkConfig:
    hidden_dim: int = 64
    embed_dim: int = 64
    num_layers: int = 2
    use_global_features: bool = True
    active_node_features: tuple[str, ...] = FEATURE_NAMES
    active_global_features: tuple[str, ...] = GLOBAL_FEATURE_NAMES
    use_virtual_node: bool = False

    def __post_init__(self) -> None:
        canonical_node_features = _canonicalize_feature_subset(
            FEATURE_NAMES,
            self.active_node_features,
            feature_group="node",
        )
        canonical_global_features = _canonicalize_feature_subset(
            GLOBAL_FEATURE_NAMES,
            self.active_global_features,
            feature_group="global",
        )
        object.__setattr__(self, "active_node_features", canonical_node_features)
        object.__setattr__(self, "active_global_features", canonical_global_features)
        object.__setattr__(
            self,
            "use_global_features",
            self.use_global_features and bool(canonical_global_features),
        )

    @property
    def input_dim(self) -> int:
        return len(self.active_node_features)

    @property
    def global_feat_dim(self) -> int:
        return len(self.active_global_features)

    @classmethod
    def from_dict(cls, values: dict) -> "QNetworkConfig":
        config_values = dict(values)
        config_values.pop("input_dim", None)
        config_values.pop("global_feat_dim", None)
        return cls(**config_values)


class RecoveryQNetwork(nn.Module):
    """FINDER-style node scoring network for failed-node reactivation."""

    def __init__(self, config: QNetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or QNetworkConfig()
        # Always featurize the full node vector, then mask to active columns (see `observation_to_graph_tensor`).
        self.feature_names = FEATURE_NAMES
        self.global_feature_names = GLOBAL_FEATURE_NAMES
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
                global_feat_dim=self.config.global_feat_dim,
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
        if self.config.use_virtual_node:
            node_embeddings = node_embeddings[:num_real_nodes]

        if self.config.use_global_features:
            if global_features is None:
                raise ValueError("global_features are required when use_global_features=True.")
            global_vec = self.global_readout(node_embeddings, global_features)
            global_expanded = global_vec.unsqueeze(0).expand(node_embeddings.size(0), -1)
            node_input = torch.cat([node_embeddings, global_expanded], dim=1)
        else:
            node_input = node_embeddings

        q_values = self.q_head(node_input).squeeze(-1)
        valid_mask = graph_tensor.valid_mask[:num_real_nodes] if self.config.use_virtual_node else graph_tensor.valid_mask
        return q_values.masked_fill(~valid_mask, -1e9)

    def score_observation(
        self,
        observation: RecoveryObservation,
        device: torch.device | str | None = None,
    ) -> tuple[GraphTensor, torch.Tensor]:
        graph_tensor = observation_to_graph_tensor(
            observation,
            active_node_features=self.config.active_node_features,
            use_virtual_node=self.config.use_virtual_node,
            feature_names=self.feature_names,
            device=device,
        )
        global_features = None
        if self.config.use_global_features:
            global_features = observation_to_global_features(
                observation,
                active_global_features=self.config.active_global_features,
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


def build_greedy_policy(
    model: RecoveryQNetwork,
    *,
    device: torch.device | str | None = None,
) -> Callable[[RecoveryObservation], Node]:
    """Wrap the Q-network as a greedy policy compatible with evaluation helpers."""

    def policy(observation: RecoveryObservation) -> Node:
        return select_action(model, observation, epsilon=0.0, device=device)

    return policy


def load_q_network(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[RecoveryQNetwork, dict]:
    """Load a saved learner checkpoint.

    Relative paths are resolved against the repository root (parent of ``src/``),
    matching portable ``checkpoint_path`` values stored in artifact JSON.
    """
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
    valid_q = [(q_values[i].item(), graph_tensor.node_ids[i]) for i in valid_indices]
    valid_q.sort(key=lambda x: x[0], reverse=True)

    return [node for _, node in valid_q[:b]]
