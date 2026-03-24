# GNN encoder + DQN policy (FINDER-style)

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GNNEncoder(nn.Module):
    def __init__(self, node_feat_dim=4, hidden=64, out=32):
        super().__init__()
        self.conv1 = SAGEConv(node_feat_dim, hidden)
        self.conv2 = SAGEConv(hidden, out)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

class RecoveryDQN(nn.Module):
    def __init__(self, node_feat_dim=4, hidden=64):
        super().__init__()
        self.encoder = GNNEncoder(node_feat_dim, hidden)
        self.q_head  = nn.Linear(32, 1)  # score per node

    def forward(self, x, edge_index, failed_mask):
        emb = self.encoder(x, edge_index)
        scores = self.q_head(emb).squeeze(-1)
        scores[~failed_mask] = -1e9  # mask valid actions to failed nodes only
        return scores
