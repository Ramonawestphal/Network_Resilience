# Architecture Notes

## 1. Global Readout — `src/cascading_rl/models/gnn.py`

### What was the problem?

The original GNN encoder used message passing to compute a local embedding for
each node — meaning each node's representation only reflected its own features
and its immediate neighborhood. The agent had no way to know things like:

- How much of the network is currently failed?
- How stressed are the active nodes on average?
- How far into the episode are we?

These are exactly the quantities a rational recovery agent should condition on.
Without them, the agent cannot distinguish between "3 nodes failed out of 40,
situation under control" and "30 nodes failed out of 40, system collapsing" if
the local neighborhood happens to look the same.

### What was added?
 
**`GLOBAL_FEATURE_NAMES`** — a tuple naming four explicit global scalars:
 
```python
GLOBAL_FEATURE_NAMES = (
    "failed_fraction",
    "mean_load_capacity_ratio",
    "max_load_capacity_ratio",
    "current_round_norm",
)
```

**`observation_to_global_features(observation)`** — a function that computes
these four scalars from the current observation:
 
| Feature | Formula | What it encodes |
|---|---|---|
| `failed_fraction` | \|failed\| / \|V\| | How severe is the damage right now |
| `mean_load_capacity_ratio` | mean(L_i / C_i) over active nodes | Average stress — high values mean the cascade is likely to continue |
| `max_load_capacity_ratio` | max(L_i / C_i) over active nodes | Worst-case stress — is any single node about to trigger another wave |
| `current_round_norm` | current_round / max_rounds | How far into the episode we are — agent should behave differently near the horizon |
 
Note: `active_fraction` was excluded because it is always
`1 - failed_fraction` and therefore fully redundant. Including it would add
collinearity with no new information.

**`GlobalReadout` (new nn.Module)** — takes the node embeddings produced by
the GNN, mean-pools and max-pools them into a graph-level summary, concatenates
the four explicit global scalars, and projects everything down to a
`global_dim`-dimensional vector:

```
node embeddings (N × embed_dim)
        ↓
mean-pool → (embed_dim,)
max-pool  → (embed_dim,)
        ↓
concat with 4 global scalars → (2*embed_dim + 4,)
        ↓
linear + ReLU → (global_dim,)
```

This global vector is then broadcast to every node and concatenated with its
local embedding before the Q-head scores it. Every node's Q-score is therefore
conditioned on both its local structural context and the global system state.

### Why mean-pool AND max-pool?
 
- Mean-pool captures the average state of the network (e.g. average embedding
  across all nodes).
- Max-pool captures the most extreme state (e.g. the most critical node's
  embedding).
 
Together they give a richer summary than either alone.

### Per-node feature changes — `FEATURE_NAMES` reduced from 9 to 8
 
The original per-node features included `remaining_budget_norm` and
`current_round_norm`, both normalized by dividing by the number of nodes.
These were treated differently:

**`remaining_budget / num_nodes` — kept, renamed to `budget_coverage`.**
Dividing budget by graph size is semantically meaningful. Reactivating 5 nodes
out of a 30-node graph is a much more impactful round than 5 out of a 100-node
graph — the agent recovers a larger fraction of the system in the same number
of actions. This is a genuine signal about how much the agent can move the
needle in a single round, relative to the scale of the problem. The
normalization by `num_nodes` is therefore intentional and correct here.

**`current_round / num_nodes` — removed.** Round number has nothing to do
with graph size. Round 2 of 5 on a 40-node graph is the exact same temporal
situation as round 2 of 5 on a 50-node graph — the agent is 40% through its
episode horizon either way. Dividing by `num_nodes` instead of `max_rounds`
gives different feature values for identical situations just because the graph
is a different size. This is noise, not signal, and directly hurts
generalization across graph sizes.

`current_round_norm` now lives exclusively in the global features, computed
as `current_round / max_rounds` — the correct denominator — once per step,
not once per node

The updated per-node feature set:
 
```python
FEATURE_NAMES = (
    "load_norm",
    "capacity_norm",
    "load_capacity_ratio",
    "failed_flag",
    "active_flag",
    "frontier_flag",
    "budget_coverage",      # remaining_budget / num_nodes
    "degree_norm",
)
```
---
 
## 2. Q-Network with Global Context — `src/cascading_rl/models/q_network.py`
 
### What changed?
 
**`QNetworkConfig`** — `input_dim` updated from 9 to 8 to reflect the reduced
per-node feature set.
 
**`RecoveryQNetwork.__init__`** now creates the global readout layer and
adjusts the Q-head input size accordingly.
```
 
**`forward(graph_tensor, global_features)`** now takes both inputs:
 
1. Runs message passing to get node embeddings
2. Computes the global vector via `GlobalReadout`
3. Broadcasts the global vector to every node and concatenates
4. Scores each node through the Q-head
5. Masks invalid (non-failed) nodes to -1e9
 
**`score_observation`** now computes global features internally so callers
do not need to handle this themselves.
 
**`select_top_b` (new function)** — replaces `select_action` for the batch
design. Instead of selecting one node at a time, it runs the Q-network once
and returns the top-B failed nodes ranked by Q-value:
 
```python
def select_top_b(model, observation, budget, *, epsilon, rng, device):
    # one forward pass
    # rank all failed nodes by Q-value
    # return top-B as a list
```
 
With epsilon-greedy exploration, a random subset of B nodes is returned instead
of the top-B. This preserves exploration during training.
 
---
 
## 3. Environment Updates — `src/cascading_rl/envs/recovery.py`
 
### `RecoveryObservation` — two new fields
 
```python
@dataclass(frozen=True)
class RecoveryObservation:
    ...
    budget: int       # total reactivations allowed per round (fixed, from env)
    max_rounds: int   # episode horizon (fixed, from env)
```
 
These were added so that `observation_to_global_features` and the per-node
feature normalization can access them directly from the observation without
needing them passed as separate arguments through every function call. This
keeps the code self-contained and avoids threading extra parameters everywhere.
 
`env.observe()` was updated to populate both fields from `self.budget` and
`self.max_rounds`.
 
### `step_batch` (new method)
 
The original `step()` method handled one reactivation at a time and fired the
cascade after the budget was exhausted. The new `step_batch()` accepts a list
of nodes to reactivate all at once, then fires one cascade wave:
 
```python
def step_batch(self, actions: list[Node]):
    # reactivate all B nodes
    for action in actions:
        self.state = reactivate_node(self.state, action)
    # fire cascade once
    newly_failed = advance_cascade_round(self.state)
    # compute reward as ΔANC
    reward = next_anc - previous_anc
    ...
```
 
**Why this is correct:** load does not redistribute between individual
reactivations within a round — the cascade only fires at round end. So the
state seen by the Q-network before pick 1 and before pick 2 within the same
round is structurally different (one more node is active) but dynamically
identical (loads unchanged). Sequential Q-value recomputation was therefore
giving the agent no new information at extra compute cost.
 
The original `step()` is kept unchanged for backward compatibility with
existing tests.
 
---
 
## 4. Training Loop — `src/cascading_rl/training/trainer.py`
 
### Episode loop simplified
 
The episode loop was simplified from B sequential steps per round to one
batch step per round:
 
**Before:**
```
while not done:
    action = select_action(...)           # one node, one forward pass
    obs, reward, done, info = env.step(action)
    replay_buffer.push(single transition)
    # repeated B times per round
```
 
**After:**
```
while not done:
    actions = select_top_b(...)           # B nodes, one forward pass
    obs, reward, done, info = env.step_batch(actions)
    replay_buffer.push(single transition with list of actions)
    # once per round
```
 
Each transition in the replay buffer now stores a **list of actions** (the
full batch selected that round) rather than a single action. This means one
transition per round instead of B transitions per round — a cleaner
correspondence between the unit of decision-making and the unit of experience.
 
### `compute_dqn_loss` update
 
The loss function was updated to handle batch actions. For each transition,
the Q-values of all selected nodes are averaged, and the target uses the
mean of the top-B Q-values in the next state:
 
```python
# Q-value of the batch = mean Q over selected nodes
q_selected = torch.stack([q_values[i] for i in action_indices]).mean()
 
# target = reward + gamma * mean of top-B Q-values in next state
top_b_next = sorted(valid_next, reverse=True)[:budget]
target_value = reward + gamma * mean(top_b_next)
```
 
`compute_dqn_loss` also now calls `observation_to_global_features` for both
the current and next observation so the model receives global context during
training, matching exactly what it receives at inference time.
 
---
 
## Summary of files changed
 
| File | What changed |
|---|---|
| `src/cascading_rl/models/gnn.py` | Added `GLOBAL_FEATURE_NAMES`, `observation_to_global_features`, `GlobalReadout`. Removed `current_round_norm` from per-node features, kept `budget_coverage` (remaining_budget / num_nodes). `FEATURE_NAMES` reduced from 9 to 8 entries. |
| `src/cascading_rl/models/q_network.py` | `QNetworkConfig.input_dim` changed from 9 to 8. Updated `RecoveryQNetwork` to use global readout. Added `select_top_b`. Updated `score_observation` and `compute_dqn_loss` for global features and batch actions. |
| `src/cascading_rl/envs/recovery.py` | Added `budget` and `max_rounds` to `RecoveryObservation`. Added `step_batch` method. Updated `observe()` to populate new fields. |
| `src/cascading_rl/training/trainer.py` | Simplified training loop to use `select_top_b` and `step_batch`. Updated `compute_dqn_loss` for batch actions and global features. |
| `src/cascading_rl/models/__init__.py` | Exported `GlobalReadout`, `GLOBAL_FEATURE_NAMES`, `observation_to_global_features`, `select_top_b`. |
| `docs/architecture.md` | Updated with global readout design and motivation. |
 