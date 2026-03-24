# train.py  —  n-step Q-learning, faithful to FINDER (Fan et al., 2020)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import networkx as nx
import random
from collections import deque
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

from graph_utils import make_graph_batch
from cascade_env import CascadeRecoveryEnv, compute_anc

# ─────────────────────────────────────────────
#  Hyperparameters  (from FINDER paper)
# ─────────────────────────────────────────────
CONFIG = {
    # Graph generation
    "n_range":          (30, 50),
    "m":                2,
    "alpha":            0.2,
    "pfail":            0.1,
    "budget":           3,

    # Training  (FINDER defaults)
    "num_episodes":     10_000,       # ε annealed over full run
    "n_step":           5,            # n-step return horizon
    "batch_size":       64,
    "replay_capacity":  50_000,       # M=50,000 as in paper
    "gamma":            0.99,
    "lr":               1e-3,

    # Epsilon: LINEAR decay 1.0 → 0.05 over num_episodes
    "eps_start":        1.0,
    "eps_end":          0.05,

    # GNN architecture
    "node_feat_dim":    4,            # [load_norm, cap_norm, deg_norm, is_failed]
    "embed_dim":        64,           # d in paper
    "mlp_hidden":       32,

    # Checkpointing
    "save_every":       1000,
    "checkpoint_path":  "decade_finder.pt",
}

# ─────────────────────────────────────────────
#  Encoder  (GraphSAGE + virtual node)
# ─────────────────────────────────────────────
class FINDEREncoder(nn.Module):
    """
    Two-layer GraphSAGE encoder.
    Produces:
      - node_emb  : (N, embed_dim)  — per-node (action) embedding
      - graph_emb : (1, embed_dim)  — virtual-node (state) embedding
    """
    def __init__(self, in_dim, embed_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim,     embed_dim)
        self.conv2 = SAGEConv(embed_dim,  embed_dim)
        # Virtual node aggregation: mean-pool nodes → MLP
        self.vn_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        h = torch.relu(self.conv2(h, edge_index))       # (N, embed_dim)
        graph_h = self.vn_mlp(h.mean(dim=0, keepdim=True))  # (1, embed_dim)
        return h, graph_h                               # node, graph embeddings


# ─────────────────────────────────────────────
#  Decoder  (outer product → MLP → Q scalar)
# ─────────────────────────────────────────────
class FINDERDecoder(nn.Module):
    """
    Q(s, a) = MLP( node_emb_a  ⊗  graph_emb_s )
    Outer product: (embed_dim, embed_dim) → flatten → MLP → scalar
    """
    def __init__(self, embed_dim, mlp_hidden):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * embed_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, node_emb, graph_emb, failed_mask):
        """
        node_emb  : (N, embed_dim)
        graph_emb : (1, embed_dim)
        Returns Q : (N,)  — masked to -inf for active nodes
        """
        N, D = node_emb.shape
        # Outer product for each node: (N, D, D) → (N, D*D)
        outer = torch.bmm(
            node_emb.unsqueeze(2),                      # (N, D, 1)
            graph_emb.expand(N, -1).unsqueeze(1)        # (N, 1, D)
        ).view(N, D * D)
        q = self.mlp(outer).squeeze(-1)                 # (N,)
        q = q.masked_fill(~failed_mask, float('-inf'))  # mask valid actions
        return q


# ─────────────────────────────────────────────
#  Full FINDER agent
# ─────────────────────────────────────────────
class FINDERAgent(nn.Module):
    def __init__(self, node_feat_dim, embed_dim, mlp_hidden):
        super().__init__()
        self.encoder = FINDEREncoder(node_feat_dim, embed_dim)
        self.decoder = FINDERDecoder(embed_dim, mlp_hidden)

    def forward(self, x, edge_index, failed_mask):
        node_emb, graph_emb = self.encoder(x, edge_index)
        return self.decoder(node_emb, graph_emb, failed_mask)


# ─────────────────────────────────────────────
#  Graph reconstruction loss  (auxiliary)
# ─────────────────────────────────────────────
def graph_reconstruction_loss(node_emb, adj_matrix):
    """
    Encourage embeddings to reconstruct adjacency: σ(emb @ emb.T) ≈ A
    (structural deep network embedding, Wang et al. 2016)
    """
    pred = torch.sigmoid(node_emb @ node_emb.t())
    return nn.functional.binary_cross_entropy(pred, adj_matrix)


# ─────────────────────────────────────────────
#  State → PyG conversion
# ─────────────────────────────────────────────
def state_to_pyg(state, G, device):
    nodes    = list(G.nodes())
    node2idx = {v: i for i, v in enumerate(nodes)}
    n        = len(nodes)

    max_load = max(state["loads"].values()) + 1e-8
    max_cap  = max(state["caps"].values())  + 1e-8
    max_deg  = max(d for _, d in G.degree()) + 1e-8

    x = torch.zeros((n, 4), dtype=torch.float32)
    for i, node in enumerate(nodes):
        x[i, 0] = state["loads"][node] / max_load
        x[i, 1] = state["caps"][node]  / max_cap
        x[i, 2] = G.degree(node)       / max_deg
        x[i, 3] = 1.0 if node in state["failed"] else 0.0

    edges = [(node2idx[u], node2idx[v]) for u, v in G.edges()]
    edges += [(v, u) for u, v in edges]
    edge_index = (torch.tensor(edges, dtype=torch.long).t().contiguous()
                  if edges else torch.zeros((2, 0), dtype=torch.long))

    failed_mask = torch.tensor(
        [node in state["failed"] for node in nodes], dtype=torch.bool)

    # Adjacency matrix for reconstruction loss
    adj = torch.zeros((n, n), dtype=torch.float32)
    for u, v in G.edges():
        adj[node2idx[u], node2idx[v]] = 1.0
        adj[node2idx[v], node2idx[u]] = 1.0

    data = Data(x=x.to(device), edge_index=edge_index.to(device))
    return data, failed_mask.to(device), node2idx, nodes, adj.to(device)


# ─────────────────────────────────────────────
#  n-step Replay Buffer
# ─────────────────────────────────────────────
class NStepReplayBuffer:
    """
    Stores (S_i, A_i, R_{i→i+n}, S_{i+n}, done) transitions.
    n-step return: R_{i,i+n} = Σ_{k=i}^{i+n-1} γ^(k-i) * r_k
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, n_step_return, next_state, done):
        self.buffer.append((state, action_idx, n_step_return, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def compute_nstep_return(rewards, gamma, n):
    """R_{i, i+n} = Σ_{k=0}^{n-1} γ^k * r_{i+k}"""
    G = 0.0
    for k, r in enumerate(rewards[:n]):
        G += (gamma ** k) * r
    return G


# ─────────────────────────────────────────────
#  Action selection  (ε-greedy)
# ─────────────────────────────────────────────
def select_action(agent, state, G, epsilon, device):
    failed_nodes = list(state["failed"])
    if not failed_nodes:
        return None, None

    if random.random() < epsilon:
        chosen = random.choice(failed_nodes)
        data, failed_mask, node2idx, nodes, _ = state_to_pyg(state, G, device)
        return chosen, node2idx[chosen]

    data, failed_mask, node2idx, nodes, _ = state_to_pyg(state, G, device)
    agent.eval()
    with torch.no_grad():
        q = agent(data.x, data.edge_index, failed_mask)
    agent.train()
    best_idx = q.argmax().item()
    return nodes[best_idx], best_idx


# ─────────────────────────────────────────────
#  n-step Q-learning loss
# ─────────────────────────────────────────────
def compute_loss(batch, agent, gamma, n_step, device):
    q_loss_list = []
    rc_loss_list = []

    for (state, action_idx, n_ret, next_state, done) in batch:
        G = state["G"]

        # Current state
        data, fmask, _, _, adj = state_to_pyg(state, G, device)
        node_emb, graph_emb = agent.encoder(data.x, data.edge_index)
        q_values = agent.decoder(node_emb, graph_emb, fmask)
        q_sa = q_values[action_idx].unsqueeze(0)

        # n-step target: R_{i,i+n} + γ^n * max_a' Q(S_{i+n}, a')
        if done or not next_state["failed"]:
            target = torch.tensor([n_ret], dtype=torch.float32, device=device)
        else:
            nd, nfm, _, _, _ = state_to_pyg(next_state, G, device)
            with torch.no_grad():
                q_next = agent(nd.x, nd.edge_index, nfm)
                max_q_next = q_next.max()
            target = torch.tensor(
                [n_ret + (gamma ** n_step) * max_q_next.item()],
                dtype=torch.float32, device=device)

        q_loss_list.append(nn.functional.smooth_l1_loss(q_sa, target))

        # Graph reconstruction loss (auxiliary, per FINDER Eq. S27)
        rc_loss_list.append(graph_reconstruction_loss(node_emb, adj))

    # Combined loss: Q-learning + reconstruction (equal weight, tune λ if needed)
    q_loss  = torch.stack(q_loss_list).mean()
    rc_loss = torch.stack(rc_loss_list).mean()
    return q_loss + rc_loss, q_loss.item(), rc_loss.item()


# ─────────────────────────────────────────────
#  Training loop
# ─────────────────────────────────────────────
def train(config=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    agent = FINDERAgent(
        node_feat_dim=config["node_feat_dim"],
        embed_dim=config["embed_dim"],
        mlp_hidden=config["mlp_hidden"],
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=config["lr"])
    buffer    = NStepReplayBuffer(config["replay_capacity"])

    # Linear epsilon schedule: 1.0 → 0.05 over num_episodes
    eps_schedule = np.linspace(
        config["eps_start"], config["eps_end"], config["num_episodes"])

    episode_rewards, episode_ancs = [], []

    for episode in range(config["num_episodes"]):
        epsilon = eps_schedule[episode]

        # Fresh random BA graph every episode → forces generalization
        graphs = make_graph_batch(
            num_graphs=1, n_range=config["n_range"],
            m=config["m"], alpha=config["alpha"], seed_offset=episode)
        G = graphs[0]

        env   = CascadeRecoveryEnv(G, alpha=config["alpha"],
                                   pfail=config["pfail"], budget=config["budget"])
        state = env.reset()
        state["G"] = G

        ep_reward    = 0.0
        done         = False
        step_buffer  = []   # temporary n-step window: list of (state, action_idx, reward)

        while not done:
            action, action_idx = select_action(agent, state, G, epsilon, device)
            if action is None:
                break

            next_state, reward, done, _ = env.step(action)
            next_state["G"] = G
            ep_reward += reward

            step_buffer.append((state, action_idx, reward))

            # Once we have n steps, push n-step transition to replay buffer
            if len(step_buffer) >= config["n_step"]:
                s0, a0, _ = step_buffer[0]
                rewards_window = [t[2] for t in step_buffer]
                n_ret = compute_nstep_return(
                    rewards_window, config["gamma"], config["n_step"])
                buffer.push(s0, a0, n_ret, next_state, done)
                step_buffer.pop(0)

            state = next_state

            # Flush remaining steps at episode end
            if done:
                for j in range(len(step_buffer)):
                    s_j, a_j, _ = step_buffer[j]
                    remaining = [t[2] for t in step_buffer[j:]]
                    n_ret = compute_nstep_return(
                        remaining, config["gamma"], len(remaining))
                    buffer.push(s_j, a_j, n_ret, next_state, True)

            # Learn from replay
            if len(buffer) >= config["batch_size"]:
                batch = buffer.sample(config["batch_size"])
                loss, q_l, rc_l = compute_loss(
                    batch, agent, config["gamma"], config["n_step"], device)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

        final_anc = compute_anc(G, env.active)
        episode_rewards.append(ep_reward)
        episode_ancs.append(final_anc)

        if (episode + 1) % 100 == 0:
            print(f"Ep {episode+1:>5} | ε={epsilon:.3f} | "
                  f"Reward(100): {np.mean(episode_rewards[-100:]):.4f} | "
                  f"ANC(100): {np.mean(episode_ancs[-100:]):.4f} | "
                  f"Buffer: {len(buffer)}")

        if (episode + 1) % config["save_every"] == 0:
            torch.save({
                "episode":  episode + 1,
                "agent":    agent.state_dict(),
                "optimizer":optimizer.state_dict(),
            }, config["checkpoint_path"])
            print(f"  → Saved checkpoint at episode {episode + 1}")

    print("Training complete.")
    return agent, episode_rewards, episode_ancs


if __name__ == "__main__":
    agent, rewards, ancs = train()
