# Cascading-RL Research Log

This log now starts from the batch-repair reset on 2026-03-30. All earlier RL checkpoints and conclusions were produced under the wrong environment semantics and were intentionally deleted.

## Reset Summary

The corrected baseline now uses:

- batch-per-round repair in `src/cascading_rl/envs/recovery.py`
- one cascade wave only after the round budget is exhausted
- pre-cascade ANC gain as the repair reward
- graph-size-scaled budget everywhere via `config/default.yaml`
- fresh post-reset heuristic artifacts only in `experiments/`

Reset verification:

- `pytest tests/ -x -q` -> `37 passed`
- targeted env behavior is now covered by tests:
  - no cascade on repairs `1..B-1`
  - exactly one cascade at round end
  - canonical budget scaling helper used by training and evaluation

## Regime Taxonomy

Keep using the same three labels:

- `trivial`: all policies are near ceiling
- `hopeless`: all policies fail badly
- `decision-sensitive`: policy quality still matters

Thresholds remain driven by `config/default.yaml`:

- `hopeless_threshold = 0.25`
- `trivial_threshold = 0.75`
- `spread_threshold = 0.05`

## Post-Reset Baseline Artifacts

### Regime Map

- Command: `python scripts/map_regime.py`
- Artifacts:
  - `experiments/regime_map/regime_results.json`
  - `experiments/regime_map/regime_results.csv`
  - `experiments/regime_map/recommended_regime.md`
- Grid:
  - `alpha in {0.10, 0.15, 0.20, 0.25}`
  - `pfail in {0.05, 0.08, 0.10, 0.15}`
  - `budget in {1, 2, 3, 4}`
  - `max_rounds = 5`
- Main result:
  - `64` total cells
  - `41` `decision-sensitive`
  - `22` `trivial`
  - `1` `hopeless`
- Recommendation:
  - start fresh RL training around `alpha=0.10`, `pfail=0.15`, `budget=4`
  - this cell has interestingness score `0.616`
  - heuristic spread is large enough to make RL comparisons meaningful
- Interpretation:
  - the corrected batch-repair environment is no longer dominated by trivially easy cells
  - after making repair rounds stronger, the meaningful regimes shifted toward lower `alpha`, higher `pfail`, and higher reference budget

### Hard-Regime Heuristic Sweep

- Command: `python scripts/evaluate_hard_regime.py`
- Artifact:
  - `experiments/hard_regime/hard_regime_summary.json`
- Grid:
  - `alpha in {0.10, 0.15, 0.20}`
  - `pfail in {0.10, 0.15, 0.20}`
  - `budget = 4`
  - `max_rounds = 5`
- Main result:
  - `alpha=0.15, pfail=0.10` and `alpha=0.20, pfail=0.10` are now `trivial`
  - the rest of the grid remains `decision-sensitive`
  - no cell in this focused hard grid became `hopeless`
- Heuristic ranking:
  - `degree` is usually best
  - `betweenness` wins several of the harder cells and stays extremely close to `degree`
  - `random` and `greedy` lag far behind in the harder settings
- Example harder cells:
  - `alpha=0.10, pfail=0.15`: `betweenness=0.971`, `degree=0.970`, `risk=0.692`, `greedy=0.605`, `random=0.434`
  - `alpha=0.10, pfail=0.20`: `degree=0.915`, `betweenness=0.913`, `risk=0.565`, `greedy=0.494`, `random=0.248`

### Canonical Evaluation Path Check

- Command: `python scripts/evaluate_policy.py --grid-source hard_regime --policies random degree risk greedy betweenness --output-dir experiments/reference_regime`
- Artifacts:
  - `experiments/reference_regime/evaluation_summary.json`
  - `experiments/reference_regime/evaluation_grid_summary.json`
- Main result:
  - reference benchmark at the new default regime still shows a clear structural ordering:
    - `degree=0.990`
    - `betweenness=0.990`
    - `greedy=0.796`
    - `risk=0.742`
    - `random=0.677`
  - the grid summary over the `hard_regime` cells reports `7` `decision-sensitive` and `2` `trivial` cells
- Interpretation:
  - the maintained evaluation entry point is now aligned with the standalone hard-regime script
  - fresh post-reset artifact generation is consistent across the main heuristic analysis paths

## Current Default Config After Reset

`config/default.yaml` now encodes the corrected baseline:

- training reference regime: `alpha=0.10`, `pfail=0.15`
- mixed training grid: `alpha_values=(0.10, 0.15, 0.20)`, `pfail_values=(0.10, 0.15, 0.20)`
- reference budget: `4`
- budget scaling: enabled with `reference_n=40`
- `max_rounds=5`

This is the clean starting point for the next RL run. No retained RL checkpoint currently exists under the corrected semantics.

## Immediate Next Steps

1. Train a fresh RL checkpoint from scratch with the corrected environment and the new default config.
2. Evaluate that checkpoint with `scripts/evaluate_policy.py` against `degree`, `betweenness`, `risk`, `greedy`, and `random`.
3. Compare RL primarily inside the `decision-sensitive` cells from the new regime map.
4. Only after a fresh corrected-semantics RL baseline exists, revisit imitation warmstart or other training ablations.
