# Architecture Notes

This document covers the codebase as it exists after the ablation, regime-map,
interestingness filtering, and evaluation-metric changes. 

The high-level message is:

- The core learner architecture did not change.
- The input interface to that learner became configurable.
- The evaluation stack became richer and more selective.
- The ablation pipeline became a first-class experiment driver rather than a
  few hardcoded toggles.

## 1. What Stayed the Same

The following pieces are intentionally unchanged in structure:

- `GraphTensor`
- `GraphMessagePassingLayer`
- `GraphStateEncoder`
- `GlobalReadout`
- the Q-head shape and scoring pattern
- the environment dynamics in `src/cascading_rl/envs/recovery.py`
- the batch-training loop design based on `step_batch()`

Most later changes are about:

- choosing which features are exposed to the model
- standardizing fair comparisons across model variants
- enriching evaluation outputs
- selecting more informative benchmark graphs

## 2. Feature-Configurable Model Inputs

The biggest architectural change after the original GNN/global-readout work is
that the learner no longer assumes one fixed feature set. Instead, the model
configuration now decides which node features, which global features, and
whether a virtual node are used.

### 2.1 Canonical feature definitions

The live feature definitions are in `src/cascading_rl/models/gnn.py`:

- `FEATURE_NAMES`
- `GLOBAL_FEATURE_NAMES`

At the time of writing, the code exposes:

- 9 node features
- 3 global scalar features

### 2.2 `observation_to_graph_tensor(...)`

File:

- `src/cascading_rl/models/gnn.py`

This function now accepts:

- `active_node_features: tuple[str, ...] | None = None`
- `use_virtual_node: bool = False`

Behavior:

- If `active_node_features is None`, all node features are used.
- If a subset is provided, it is reordered into the canonical order from
  `FEATURE_NAMES`.
- Unknown or duplicate names raise an error.
- The virtual node can be included or excluded without changing any downstream
  encoder code.

Why this matters:

- all feature-selection logic is centralized at the observation-to-tensor
  boundary
- the rest of the model still sees a regular dense node-feature matrix
- ablation runs can safely vary feature subsets without hand-editing tensor
  dimensions

Main place to tweak:

- add/remove raw node features in `FEATURE_NAMES` and in the per-node feature
  construction inside `observation_to_graph_tensor`

### 2.3 `observation_to_global_features(...)`

File:

- `src/cascading_rl/models/gnn.py`

This function now accepts:

- `active_global_features: tuple[str, ...] | None = None`

Behavior:

- it computes the full global feature vector first
- it then selects only the requested subset
- selection is always returned in canonical `GLOBAL_FEATURE_NAMES` order

This means:

- all global-feature ablations are also handled at the observation boundary
- callers do not need custom logic for different global feature subsets

### 2.4 `QNetworkConfig`

File:

- `src/cascading_rl/models/q_network.py`

New config fields:

- `active_node_features`
- `active_global_features`
- `use_virtual_node`

Important design choice:

- `input_dim` is no longer something the caller should manually reason about
- `global_feat_dim` is derived from the chosen global feature subset

The config canonicalizes the feature tuples at construction time, so a model is
self-consistent as soon as it exists.

This is the main entry point if you want to:

- define a new feature ablation
- disable all global features
- remove the virtual node
- introduce a new default feature subset

### 2.5 Where the feature config is consumed

Files:

- `src/cascading_rl/models/q_network.py`
- `src/cascading_rl/training/trainer.py`

`RecoveryQNetwork.score_observation(...)`, `select_top_b(...)`, and the DQN
loss path all read feature settings from `model.config`, not from external
callers.

This keeps the rest of training/evaluation agnostic to the selected feature
subset.

`TrainingConfig` mirrors the same feature fields in
`src/cascading_rl/training/trainer.py`, and `build_model_config(...)` passes
them into `QNetworkConfig`.

If you want a new training-time default for all learner runs, this is the path:

1. change `TrainingConfig`
2. keep `build_model_config(...)` in sync
3. avoid threading feature flags manually anywhere else

## 3. Ablation Pipeline After the Refactor

The ablation code now lives as a reusable experiment driver rather than four
hardcoded boolean switches.

File:

- `scripts/run_ablation.py`

### 3.1 Run definitions

`build_ablation_runs()` constructs the full ablation list.

It includes:

- `node_only`
- `node_global`
- `node_virtual`
- `node_global_virtual`
- one run dropping each global feature
- one run dropping each node feature

Each run is represented as a dict with:

- `name`
- `active_node_features`
- `active_global_features`
- `use_virtual_node`

This is the first place to edit if you want to:

- add another ablation family
- compare a hand-picked sparse feature subset
- test multiple virtual-node policies
- rename output files

### 3.2 Fairness across ablation runs

The ablation script intentionally makes the comparison fair across model
variants.

Shared across all runs:

- one frozen training graph-spec sequence
- one shared evaluation graph batch
- one shared benchmark seed list

In practice:

- `generate_episode_graph_specs(...)` is called once
- the resulting `frozen_episode_graph_specs` is reused for every run
- `eval_graphs` is generated once before the loop and reused for every run

This means:

- all models train on the same training graph sequence
- all models are evaluated on the same held-out evaluation graph set
- differences in ablation output come from model configuration, not graph draw

### 3.3 Outputs

The script writes:

- one JSON file per run in `experiments/ablation/`
- one aggregate file: `experiments/ablation/ablation_comparison.json`

Each run payload stores:

- the run name
- the active node/global feature lists
- the virtual-node flag
- checkpoint path
- the RL evaluation summary

## 4. Regime Mapping Changes

The regime map now does more than sweep parameters and print baseline scores.
It explicitly scores how policy-sensitive a cell is and produces a
recommendation artifact.

Main files:

- `src/cascading_rl/evaluation/regime.py`
- `scripts/map_regime.py`
- `config/default.yaml` under `regime_mapping`

### 4.1 Diagnostics per parameter cell

`compute_regime_diagnostics(...)` in `src/cascading_rl/evaluation/regime.py`
computes:

- `final_anc_spread`
- `threshold_hit_spread`
- `rounds_spread`
- `mean_final_anc`
- `mean_threshold_hit`
- `budget_sensitivity`
- `interestingness_score`
- best/worst policy labels
- a regime label such as `trivial`, `hopeless`, or `interesting`

`build_regime_cells(...)` evaluates the full `(alpha, pfail, budget)` grid and
attaches these diagnostics to each cell.

### 4.2 Configurable thresholds

The regime map is controlled from `config/default.yaml` via `regime_mapping`,
including:

- `num_graphs`
- `seeds`
- `alpha_values`
- `pfail_values`
- `budgets`
- `hopeless_threshold`
- `trivial_threshold`
- `spread_threshold`

If you want the regime map to be stricter or looser about what counts as
interesting, `spread_threshold` is the first thing to adjust.

### 4.3 Outputs from `scripts/map_regime.py`

The regime script now produces:

- `regime_results.json`
- `regime_results.csv`
- `recommended_regime.md`
- `interestingness_heatmap.png`
- `budget_curves.png`

`recommended_regime.md` is generated from the serialized cells and is meant to
answer: "Where should I train RL next?"

If you want to change the recommendation heuristic, look at:

- `build_recommendation(...)`
- `write_note(...)`

inside `scripts/map_regime.py`

## 5. Filtering Benchmark Graphs by Interestingness

Later evaluation now filters out graphs where policies are too similar before
benchmarking the learner.

Main files:

- `src/cascading_rl/evaluation/regime.py`
- `scripts/evaluate_policy.py`

### 5.1 `filter_interesting_graphs(...)`

File:

- `src/cascading_rl/evaluation/regime.py`

This helper:

1. takes a list of candidate graphs
2. evaluates each graph individually using
   `evaluate_policy_factories_on_graphs([graph], ...)`
3. computes the spread in `final_anc` across policies
4. keeps only graphs whose spread exceeds `spread_threshold`

This is intentionally lightweight:

- no environment changes
- no new benchmark format
- no special-case RL logic

It reuses the same policy-factory evaluation path as the rest of the regime
tools.

### 5.2 Where filtering is applied

File:

- `scripts/evaluate_policy.py`

The evaluation script now:

1. generates a benchmark graph batch
2. filters it with `filter_interesting_graphs(...)`
3. evaluates all policies on the filtered list
4. computes `b_star` from the filtered set as well

The script prints how many graphs were kept and how many were filtered out.

If you want to disable filtering, lower the threshold, or move filtering to a
different stage, `scripts/evaluate_policy.py` is the place to start.

## 6. New Evaluation Metrics

The evaluation layer was extended so both the main benchmark script and the
ablation script expose more than just final ANC and threshold hit rate.

Main files:

- `src/cascading_rl/evaluation/benchmarks.py`
- `scripts/evaluate_policy.py`
- `scripts/run_ablation.py`

### 6.1 Episode-level metrics

In `src/cascading_rl/evaluation/benchmarks.py`, `EpisodeResult` now includes:

- `anc_by_round: list[float]`
- `mean_delta_anc_per_round: float`

#### `anc_by_round`

This stores ANC at the end of each recovery round:

- after repair
- after cascade
- excluding round 0

The values are collected inside `rollout_policy(...)`.

One subtle implementation detail:

- evaluation still uses `env.step()`, not `step_batch()`
- ANC is appended whenever `info["round_complete"]` is true
- if an episode ends early in the middle of a round because all failed nodes
  were repaired, the final ANC is still appended once so that the curve length
  matches the reported round count

That detail matters if you later compare curves across solved and unsolved
episodes.

#### `mean_delta_anc_per_round`

This is computed as:

`(final_anc - initial_anc) / rounds`

where `initial_anc` is measured right after `env.reset()`.

This gives a compact "average recovery efficiency per round" statistic.

### 6.2 Policy-level aggregated metrics

`PolicyEvaluationSummary` now also includes:

- `mean_anc_on_failed: AggregateMetric | None`
- `anc_by_round: list[AggregateMetric]`
- `mean_delta_anc_per_round: AggregateMetric`

#### `mean_anc_on_failed`

This averages `final_anc` only over unsolved episodes, i.e. episodes with
remaining failed nodes at termination.

Interpretation:

- useful when two policies have similar solved fractions
- distinguishes "fails gracefully" from "fails catastrophically"

#### aggregated `anc_by_round`

This aligns episodes by round index and aggregates only the episodes that
actually lasted that long.

So if some episodes end after 2 rounds and others after 5:

- round 1 aggregates all episodes
- round 5 aggregates only the longer episodes

This is an important survivorship-style detail to remember when interpreting
the tail of the curve.

### 6.3 Where the new metrics are serialized

#### Main benchmark output

File:

- `scripts/evaluate_policy.py`

Each policy in `evaluation_summary.json` now includes:

- `mean_delta_anc_per_round`
- `mean_delta_anc_per_round_stderr`
- `mean_anc_on_failed`
- `anc_by_round`

The script also prints these values to the terminal summary line.

#### Ablation output

File:

- `scripts/run_ablation.py`

The same fields are now included in:

- each per-run ablation JSON file
- `ablation_comparison.json`

The serialization logic is kept in `serialize_policy_summary(...)`, so that is
the single best place to tweak ablation JSON structure.

### 6.4 What did not change

The new metrics do not change:

- rollout dynamics
- action selection during evaluation
- the existing summary keys
- the existing JSON nesting

They are appended as additional fields.

## 7. Main Files to Edit for Common Future Tweaks

If you want to make a small change later, here is where to start:

- Add/remove a raw node feature:
  `src/cascading_rl/models/gnn.py`
- Add/remove a raw global feature:
  `src/cascading_rl/models/gnn.py`
- Change default model feature subsets:
  `src/cascading_rl/models/q_network.py`
- Change training-time defaults for those subsets:
  `src/cascading_rl/training/trainer.py`
- Add a new ablation family:
  `scripts/run_ablation.py`
- Change what is written into ablation JSON:
  `scripts/run_ablation.py`
- Change benchmark graph filtering:
  `src/cascading_rl/evaluation/regime.py` and `scripts/evaluate_policy.py`
- Change regime-map thresholds or sweep size:
  `config/default.yaml` and `scripts/map_regime.py`
- Change recommendation wording or selection logic:
  `scripts/map_regime.py`
- Change how evaluation metrics are aggregated:
  `src/cascading_rl/evaluation/benchmarks.py`

## 8. Summary of Post-Ablation Changes

From the ablation stage onward, the codebase changed in four connected ways:

1. The model input interface became configurable by feature subset and
   virtual-node usage.
2. Ablation experiments became systematic and fair across runs, including
   single-feature leave-one-out experiments.
3. Regime mapping became diagnostic-driven and now produces recommendation
   artifacts.
4. Evaluation became both more selective, via interesting-graph filtering, and
   richer, via ANC-over-rounds and failure-conditioned metrics.

That combination is what turns the project from "train one learner and compare
it to heuristics" into a more complete experimental workflow:

- find promising regimes
- train controlled model variants
- benchmark only on informative graphs
- inspect not just who wins, but how recovery unfolds over rounds
