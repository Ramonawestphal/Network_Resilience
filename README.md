# Cascading-RL

This project studies reinforcement learning for budget-constrained recovery from
cascading failures on graphs. The agent is given a per-round repair budget `B`
and must choose which failed nodes to reactivate to maximise pairwise
connectivity over a fixed horizon. The full motivation, methodology, and
evaluation are described in `paper.tex`.

> **Reference of truth.** When the README and the paper disagree, the
> canonical reference is the code in `src/cascading_rl/` together with
> `config/default.yaml`. The paper is the second reference. This README is a
> short pointer, not the spec.

## Environment Semantics

The environment lives in `src/cascading_rl/envs/recovery.py`.

Per-round step order:

1. At `t = 0`, sample one exogenous failure event: each node fails
   independently with probability `pfail`.
2. Within a round, the agent reactivates up to `B` failed nodes one at a
   time. Intra-round repairs receive zero reward.
3. Once the round's `B` repairs are committed, exactly one cascade wave is
   advanced from the current frontier.
4. The round-closing step receives the cascade-inclusive pairwise
   connectivity gain
   `PC(G, F_post) - PC(G, F_round_start)`
   as its reward (the *homogenised round reward*; cf. Eq. (5) in the paper).
5. The episode ends when there are no failed nodes left, when `max_rounds`
   is reached, or (optionally) when post-cascade pairwise connectivity falls
   strictly below `training.regime.abandonment_nc_threshold` while failures
   remain (`info["abandoned"]`).

Note that the codebase historically uses `nc` / `anc` ("normalised
connectivity" / "accumulated normalised connectivity") for what the paper
calls `PC` / `APC`. The two names refer to the same quantity defined in
Eq. (3) and Eq. (4) of the paper.

## Canonical Budget Scaling

Budget scaling is canonical across training and evaluation through
`config/default.yaml`.

- The configured `budget` is interpreted as a *reference budget* at
  `reference_n = 40` nodes.
- The actual per-graph budget is
  `B_eff = max(1, round(budget * n / reference_n))`.
- When `budget_scaling.scale_max_rounds: true`, the same linear rule is
  applied to `max_rounds`.

The implementation lives in `src/cascading_rl/budgeting.py`.

## Default Training Setup

The default training entry point is `scripts/train_policy.py`, backed by
`src/cascading_rl/training/trainer.py`. The values below reflect
`config/default.yaml` and should be treated as authoritative.

- training regime: `alpha = 0.25`, `pfail = 0.20`, `budget = 2`,
  `max_rounds = 20`
- training graph distribution: Barabási–Albert, `n` sampled from `[30, 50]`,
  attachment parameter `m = 2`
- `num_episodes = 20000`, `warmup_transitions = 1000`, `batch_size = 64`
- `gamma = 0.99`, `learning_rate = 2e-4`
- linear ε-greedy schedule from `1.0` to `0.05` over `18000` episodes
- target network synced every `200` gradient steps
- replay capacity: `20000`
- GNN encoder: 2 message-passing layers, hidden / embedding dimension `128`,
  `use_global_features: true`, `use_virtual_node: false`
- validation: `30` held-out graphs × `7` seeds (`validation_seeds = [0..6]`)
  every `validation_every = 1000` episodes
- training-time graph buffering: a 50-graph rolling buffer, with each
  episode reusing a buffered graph with probability `0.30` and otherwise
  drawing a fresh one
- round-bounded n-step returns (paper Eq. 8 / 9) are used by default; the
  Monte-Carlo return mode (`use_monte_carlo_returns`) is **off** by
  default and is exposed only as an experimental switch in
  `TrainingConfig`.

## Main Entry Points

- `python scripts/map_regime.py` — heuristic-only regime map.
- `python scripts/train_policy.py` — train a fresh GDQ-N checkpoint.
- `python scripts/evaluate_policy.py` — canonical checkpoint evaluation.
- `python scripts/run_full_evaluation.py` — orchestrates the full
  parameter-generalisation, topology-ablation, and OOD evaluation suites.
- `python -m pytest tests/ -x -q` — full verification suite.

## Notes

`README_RESEARCH.md` is the running experiment log.
