# Cascading-RL

This project studies reinforcement learning for recovery from cascading failures on Barabasi-Albert graphs. The current codebase has been reset to the intended environment semantics: each recovery round allows up to `B` single-node repairs, and the cascade advances by exactly one wave only after the round ends.

## Current Baseline

All experiment artifacts from the old interleaved semantics were deleted. Only post-reset results should be treated as valid.

The fresh heuristic-only regime analysis under the corrected semantics currently shows:

- regime map grid: `alpha in {0.10, 0.15, 0.20, 0.25}`, `pfail in {0.05, 0.08, 0.10, 0.15}`, `budget in {1, 2, 3, 4}`
- bucket counts: `41` `decision-sensitive`, `22` `trivial`, `1` `hopeless`
- recommended training cell: `alpha=0.10`, `pfail=0.15`, `budget=4`
- strongest baseline heuristic: usually `degree`, with `betweenness` winning some harder cells

Fresh artifacts:

- `experiments/regime_map/regime_results.json`
- `experiments/regime_map/regime_results.csv`
- `experiments/regime_map/recommended_regime.md`
- `experiments/reference_regime/evaluation_summary.json`
- `experiments/reference_regime/evaluation_grid_summary.json`

## Environment Semantics

The environment lives in `src/cascading_rl/envs/recovery.py`.

Current step order:

1. Sample one exogenous failure event at `t=0`.
2. Within a round, choose up to `B` failed nodes sequentially.
3. After the round budget is exhausted, run exactly one cascade wave from the current failed frontier.
4. Repeat until there are no failed nodes left, `max_rounds` is reached, or (optionally) post-cascade ANC falls strictly below `training.regime.abandonment_anc_threshold` while failures remain (`info["abandoned"]`).

Reward is always the ANC gain immediately after the chosen repair and before any round-end cascade wave. That keeps credit assignment local to the repair decision while still letting the next state reflect cascade consequences.

## Canonical Budget Rule

Budget scaling is now canonical everywhere through `config/default.yaml`.

- `budget` is interpreted as a reference budget
- `reference_n=40`
- actual per-graph budget is `round(budget * n / reference_n)`, clipped to at least `1`

That same rule is used in training, validation, regime mapping, and evaluation scripts.

## Current Default Training Setup

The default training entry point is `scripts/train_policy.py`, backed by `src/cascading_rl/training/trainer.py`.

Current defaults:

- reference regime: `alpha=0.10`, `pfail=0.15`
- mixed training grid: `alpha_values=(0.10, 0.15, 0.20)`, `pfail_values=(0.10, 0.15, 0.20)`
- reference budget: `4`
- `max_rounds=20` (with `budget_scaling.scale_max_rounds`, scaled linearly with graph size like the budget)
- `num_episodes=8000`
- `warmup_transitions=500`
- `batch_size=64`
- `use_monte_carlo_returns=True`
- fixed validation graphs via `validation_seed`
- stratified cycling over `(alpha, pfail)` combinations

The model uses the GNN in `src/cascading_rl/models/gnn.py` with a virtual global node for one-step access to global context.

## Main Entry Points

- `python scripts/map_regime.py`
  - heuristic-only regime map under the corrected semantics
- `python scripts/train_policy.py`
  - train a fresh RL checkpoint under the corrected environment
- `python scripts/evaluate_policy.py`
  - canonical checkpoint evaluation path
- `python -m pytest tests/ -x -q`
  - full verification suite

## Notes

`README_RESEARCH.md` is the running experiment log. It now documents only the post-reset baseline and the next clean training steps from this corrected starting point.
