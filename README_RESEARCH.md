# Cascading-RL Research Log

This log now reflects only the retained regime-mapping outputs under the corrected batch-repair environment semantics. Older heuristic benchmark directories were intentionally removed so the artifact set is unambiguous.

## Current Scope

The retained experiment outputs are:

- `experiments/regime_comprehensive_smoke`
- `experiments/regime_comprehensive`

The smoke run is a reduced-sample pipeline check. The comprehensive run is the full retained evaluation and should be treated as the main empirical baseline for regime selection.

## Environment And Metric

The current baseline uses:

- batch-per-round repair in `src/cascading_rl/envs/recovery.py`
- one cascade wave only after the round budget is exhausted
- pre-cascade pairwise reachability gain as the reward
- graph-size-scaled budgets via `config/default.yaml`

The connectivity metric is the project's historical `accumulated_normalized_connectivity()` function, interpreted as pairwise reachability:

- `PR(G, A) = sum_k (|C_k| / |V|)^2`

## Policy Set

The comprehensive regime mapping evaluates exactly three heuristic policies:

- `degree`
- `random`
- `betweenness`

No RL checkpoint is included in this artifact set, and other heuristics such as `greedy` or `risk` are not part of this retained evaluation.

## Smoke Verification Run

`experiments/regime_comprehensive_smoke` is a reduced-sample run used only to verify that the rewritten pipeline works end to end before spending the full compute budget.

Smoke configuration:

- `10` graphs
- `3` seeds
- `alpha in {0.05, 0.10, 0.15}`
- `pfail in {0.05, 0.10, 0.15}`
- `budget in {1, 2, 3}`

Smoke outcome:

- pipeline completed successfully
- checkpoint/resume worked
- all summary files were produced
- all expected plots were generated

Smoke recommendation:

- no valid single best cell under the final eligibility rule
- best mixed-training budget in the smoke run: `budget_ref = 3`
- proposed thresholds already fell inside the identified stable `delta_s` zone `[0.05, 0.25]`

Interpretation:

- use this run only as an engineering validation of the rewritten script
- do not treat the smoke outputs as the scientific conclusion

## Full Comprehensive Evaluation

`experiments/regime_comprehensive` is the retained full extensive evaluation.

Full grid:

- `alpha in {0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30}`
- `pfail in {0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20}`
- `budget in {1, 2, 3, 4, 5, 6}`
- `100` BA graphs
- `10` seeds per graph
- `378` cells total
- `1,134,000` policy-instance rows total

Main retained artifacts:

- `experiments/regime_comprehensive/regime_instances.csv`
- `experiments/regime_comprehensive/regime_cells.json`
- `experiments/regime_comprehensive/regime_cells.csv`
- `experiments/regime_comprehensive/budget_summary.json`
- `experiments/regime_comprehensive/threshold_sensitivity.json`
- `experiments/regime_comprehensive/training_recommendation.json`
- `experiments/regime_comprehensive/graph_variance.json`
- `experiments/regime_comprehensive/run_metadata.json`

These files mean:

- `regime_instances`: policy-level rows for every `(graph_id, alpha, pfail, budget_ref, seed_index, policy)` instance
- `regime_cells`: aggregated per-cell statistics and labels
- `budget_summary`: how decision-sensitivity and feasibility change with reference budget
- `threshold_sensitivity`: relabeling results across the threshold grid without rerunning simulations
- `training_recommendation`: best single cell, best mixed budget, and stable-threshold recommendation
- `graph_variance`: decomposition of structural vs stochastic variance

### Full-Run Results

Cell-label counts:

- `60` `decision_sensitive`
- `233` `trivial`
- `56` `hopeless`
- `29` `mixed`

Top interesting cells by `interestingness_degree`:

- `alpha=0.05`, `pfail=0.20`, `budget=4` -> `0.651`
- `alpha=0.10`, `pfail=0.20`, `budget=4` -> `0.647`
- `alpha=0.08`, `pfail=0.20`, `budget=4` -> `0.646`

Training recommendation from `training_recommendation.json`:

- best single training cell:
  - `alpha=0.05`
  - `pfail=0.20`
  - `budget_ref=5`
  - `f_ds=0.536`
  - `interestingness=0.464`
  - `feasibility_ratio_mean=0.948`
- best mixed-training budget:
  - `budget_ref=3`
- recommended mixed regime coverage at `budget_ref=3`:
  - all retained `alpha` values
  - `pfail in {0.12, 0.15, 0.18, 0.20}`
  - `27` decision-sensitive cells covered

Budget-level interpretation from `budget_summary.json`:

- `budget_ref=1` is mostly too hard:
  - `44` hopeless cells
  - mean feasibility ratio `2.955`
- `budget_ref=2` still contains many infeasible cells:
  - `15` decision-sensitive cells
  - mean feasibility ratio `1.477`
- `budget_ref=3` is the main mixed-training sweet spot:
  - `27` decision-sensitive cells
  - mean feasibility ratio `0.979`
- `budget_ref=4` still has strong individual interesting cells, but the grid as a whole shifts toward triviality:
  - `15` decision-sensitive cells
  - `48` trivial cells
- `budget_ref in {5, 6}` becomes increasingly easy overall:
  - `60` trivial cells at `5`
  - `63` trivial cells at `6`

Threshold-sensitivity interpretation:

- proposed thresholds remain in the stable zone:
  - `delta_s in [0.05, 0.25]`
- at the proposed thresholds, the full run yields:
  - `60` decision-sensitive cells
- most permissive threshold combination in the stored sensitivity grid:
  - `delta_h=0.20`
  - `delta_t=0.85`
  - `delta_s=0.05`
  - `min_ds_frac=0.30`
  - `88` decision-sensitive cells

## Interpretation Caveat

Two exported diagnostics need careful interpretation:

- `n_failed_at_start`
- `pr_post_cascade`

In the current environment implementation, `env.reset()` does not expose a first observation after an executed cascade wave. To keep the analysis read-only with respect to `src/`, these two fields are derived from a one-wave cascade preview built from a cloned reset state.

That means:

- the main regime map is valid for the implemented analysis pipeline
- these two diagnostics should be read as preview-based severity measures, not as a directly emitted environment observation

## Current Reading Of The Full Run

The full extensive evaluation supports the following conclusions:

- the retained regime grid is not degenerate; it contains meaningful `decision_sensitive`, `trivial`, and `hopeless` regions
- the most useful mixed-training regime is centered on `budget_ref=3`
- the hardest still-feasible single-cell recommendation shifts toward very high `pfail` and relatively low `alpha`
- larger budgets quickly make the grid trivial, even though a few `budget_ref=4` cells remain individually interesting
- the chosen threshold family is reasonably stable, so the qualitative regime split is not an artifact of a single fragile `delta_s` setting

## Next Step

Use `experiments/regime_comprehensive/training_recommendation.json` as the reference for the next RL training design, and compare RL primarily inside the `decision_sensitive` cells from the retained full evaluation.
