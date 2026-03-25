from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from pathlib import Path
from random import Random

import torch
from torch import nn

from cascading_rl.envs.recovery import RecoveryObservation
from cascading_rl.models.gnn import GraphStateEncoder, GraphTensor, observation_to_graph_tensor

Node = Hashable


@dataclass(frozen=True)
class QNetworkConfig:
    input_dim: int = 9
    hidden_dim: int = 64
    embed_dim: int = 64
    num_layers: int = 2


class RecoveryQNetwork(nn.Module):
    """FINDER-style node scoring network for failed-node reactivation."""

    def __init__(self, config: QNetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or QNetworkConfig()
        self.encoder = GraphStateEncoder(
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            embed_dim=self.config.embed_dim,
            num_layers=self.config.num_layers,
        )
        self.q_head = nn.Sequential(
            nn.Linear(self.config.embed_dim, self.config.embed_dim),
            nn.ReLU(),
            nn.Linear(self.config.embed_dim, 1),
        )

    def forward(self, graph_tensor: GraphTensor) -> torch.Tensor:
        node_embeddings = self.encoder(graph_tensor)
        q_values = self.q_head(node_embeddings).squeeze(-1)
        return q_values.masked_fill(~graph_tensor.valid_mask, -1e9)

    def score_observation(
        self,
        observation: RecoveryObservation,
        device: torch.device | str | None = None,
    ) -> tuple[GraphTensor, torch.Tensor]:
        graph_tensor = observation_to_graph_tensor(observation, device=device)
        return graph_tensor, self(graph_tensor)


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
) -> callable:
    """Wrap the Q-network as a greedy policy compatible with evaluation helpers."""

    def policy(observation: RecoveryObservation) -> Node:
        return select_action(model, observation, epsilon=0.0, device=device)

    return policy


def load_q_network(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[RecoveryQNetwork, dict]:
    """Load a saved learner checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model_config = QNetworkConfig(**checkpoint["model_config"])
    model = RecoveryQNetwork(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(map_location)
    model.eval()
    return model, checkpoint
