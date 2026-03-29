# Cascading-RL Research Log

This file is the working research log for the project. Keep it updated after every experiment batch so the methodology, artifacts, and conclusions stay reproducible.

## Research Question

How can a FINDER-style RL agent be trained to heal failed nodes in cascading-failure environments while remaining robust across regime shifts in:

- cascade severity (`alpha`, `pfail`)
- recovery budget and episode horizon (`budget`, `max_rounds`)
- observability and action constraints (`obs_hops`, action-space variants)
- structural heterogeneity (graph size, capacity noise, failure bias)

The main comparison target is not only raw reward, but whether RL outperforms strong heuristics in regimes where the decision actually matters.

## Regime Taxonomy

Use the same three labels across all experiments:

- `trivial`: all policies are already near ceiling.
- `hopeless`: all policies fail badly.
- `decision-sensitive`: neither trivial nor hopeless; there is meaningful room for policy quality to matter.

Current thresholds come from `config/default.yaml`:

- `hopeless_threshold = 0.25`
- `trivial_threshold = 0.75`
- `spread_threshold = 0.05`

These thresholds drive the diagnostics in `src/cascading_rl/evaluation/regime.py`.

## Standard Reporting Rules

Every benchmark run should report:

- `final_anc`
- `threshold_hit_fraction`
- `rounds`
- `solved_fraction`
- regime bucket counts
- RL gap to the best heuristic inside each bucket

Do not claim RL is better based only on a global average. Always separate:

- RL wins in `decision-sensitive` regimes
- RL ties in `trivial` regimes
- everyone fails in `hopeless` regimes

## Current Evaluation Entry Points

- `python scripts/map_regime.py`
  - heuristic-only regime map
  - writes `regime_results.json`, `regime_results.csv`, recommendation note, plots
  - now also includes bucket summaries in the JSON output

- `python scripts/evaluate_policy.py`
  - standard checkpoint-vs-heuristics benchmark at the training reference regime
  - also writes `evaluation_grid_summary.json` for regime-grid robustness analysis
  - supports `--grid-source training|regime_mapping|hard_regime`

- `python scripts/evaluate_hard_regime.py`
  - hard-regime convenience sweep
  - reads the hard-regime grid from `config/default.yaml`
  - writes one summary JSON plus per-cell JSON files

## Experiment Matrix

Run ablations in stages. Only move to the next stage after the current one has a stable benchmark story.

### Stage A: Difficulty Controls

- single-regime vs mixed-regime training
- narrow vs broad `alpha_values`
- narrow vs broad `pfail_values`
- budget sweep
- max-round sweep
- graph-size sweep

### Stage B: Information and Action Constraints

- full observability vs `obs_hops`
- action space = `failed`
- action space = `frontier`
- action space = `adjacent`

### Stage C: Structural Heterogeneity

- homogeneous capacities vs `capacity_noise`
- uniform initial failures vs `failure_bias = degree`
- uniform initial failures vs `failure_bias = load`

### Stage D: Model and Training Choices

- no virtual node vs virtual global node
- hidden/embed dimension sweep
- replay / warmup stability checks
- narrow-regime specialization vs robust mixed-regime policy

## Experiment Template

Copy this block for each run:

```md
### Experiment ID
- Date:
- Question:
- Training config:
- Evaluation command:
- Artifacts:
- Main result:
- Regime-bucket summary:
- RL minus best heuristic:
- Interpretation:
- Next action:
```

## Progress Log

### 2026-03-28

- Added shared regime-reporting utilities in `src/cascading_rl/evaluation/regime.py`.
- Standardized the taxonomy on `trivial`, `hopeless`, and `decision-sensitive`.
- Added RL-vs-best-heuristic gap tracking to cell diagnostics.
- Added bucket-level summaries for regime-aware reporting.
- Updated `scripts/evaluate_policy.py` so a checkpoint can be evaluated on a configurable regime grid and saved to `evaluation_grid_summary.json`.
- Updated `scripts/evaluate_hard_regime.py` to read hard-regime grid settings from `config/default.yaml` instead of hardcoded constants.
- Updated `scripts/map_regime.py` to reuse shared serializers and include bucket summaries.
- Added hard-regime grid settings to `config/default.yaml`.
- Added evaluation tests covering regime labels, heuristic-gap tracking, and bucket summaries.
- Verified the updated workflow with:
  - `python -m pytest -q`
  - `python scripts/evaluate_policy.py --num-graphs 1 --seeds 0 --alpha-values 0.15 --pfail-values 0.05 --budgets 2`
  - `python scripts/evaluate_hard_regime.py --num-graphs 1 --seeds 0 --alpha-values 0.10 --pfail-values 0.10`
  - `python scripts/map_regime.py`

### Experiment E001: Current checkpoint on full regime grid

- Date: 2026-03-28
- Question: Does the current RL checkpoint already outperform heuristics on the regime-mapping grid, especially in `decision-sensitive` cells?
- Training config: Existing checkpoint at `experiments/learner/recovery_q.pt`
- Evaluation command: `python scripts/evaluate_policy.py --grid-source regime_mapping`
- Artifacts:
  - `experiments/learner_benchmark/evaluation_summary.json`
  - `experiments/learner_benchmark/evaluation_grid_summary.json`
- Main result:
  - reference-regime benchmark: RL `final_anc=0.638`, greedy heuristic `final_anc=0.776`
  - full grid: `64` cells total
  - bucket split: `16` `decision-sensitive`, `48` `trivial`, `0` `hopeless`
- Regime-bucket summary:
  - overall RL minus best heuristic: `-0.042`
  - `decision-sensitive` RL minus best heuristic: `-0.139`
  - `trivial` RL minus best heuristic: `-0.010`
- RL minus best heuristic:
  - RL is currently behind in the cells that matter most
  - the gap is small in trivial cells and much larger in decision-sensitive ones
- Interpretation:
  - the current checkpoint is not yet a strong robustness baseline
  - it is not enough to improve global averages; Stage A should target better performance specifically in the `decision-sensitive` slice
- Next action:
  - train fresh narrow-regime and mixed-regime baselines under the current code, then re-evaluate both on the same regime grid

### Experiment E002: Stage A training comparison launched

- Date: 2026-03-28
- Question: Is a narrow training distribution around the recommended cell better than mixed-regime training on the regime grid?
- Training configs:
  - mixed: `python scripts/train_policy.py --episodes 500 --checkpoint-dir experiments/stage_a_mixed --alpha-values 0.15 0.20 0.25 --pfail-values 0.05 0.08 0.10`
  - narrow: `python scripts/train_policy.py --episodes 500 --checkpoint-dir experiments/stage_a_narrow --alpha 0.20 --pfail 0.08 --alpha-values 0.20 --pfail-values 0.08`
- Evaluation command:
  - `python scripts/evaluate_policy.py --checkpoint experiments/stage_a_narrow/recovery_q.pt --grid-source regime_mapping`
  - `python scripts/evaluate_policy.py --checkpoint experiments/stage_a_mixed/recovery_q.pt --grid-source regime_mapping`
- Artifacts:
  - `experiments/stage_a_mixed/recovery_q.pt`
  - `experiments/stage_a_mixed/recovery_q.summary.json`
  - `experiments/stage_a_mixed/evaluation_summary.json`
  - `experiments/stage_a_mixed/evaluation_grid_summary.json`
  - `experiments/stage_a_narrow/recovery_q.pt`
  - `experiments/stage_a_narrow/recovery_q.summary.json`
  - `experiments/stage_a_narrow/evaluation_summary.json`
  - `experiments/stage_a_narrow/evaluation_grid_summary.json`
- Main result:
  - mixed checkpoint reference-regime RL `final_anc=0.373`
  - narrow checkpoint reference-regime RL `final_anc=0.311`
  - mixed outperforms narrow, but both underperform the original checkpoint from `E001`
- Regime-bucket summary:
  - mixed:
    - overall RL minus best heuristic: `-0.072`
    - `decision-sensitive` RL minus best heuristic: `-0.225`
    - `trivial` RL minus best heuristic: `-0.021`
  - narrow:
    - overall RL minus best heuristic: `-0.094`
    - `decision-sensitive` RL minus best heuristic: `-0.337`
    - `trivial` RL minus best heuristic: `-0.013`
- RL minus best heuristic:
  - mixed is consistently closer to the best heuristic than narrow
  - both are still weaker than the original checkpoint from `E001` (`-0.042` overall, `-0.139` in `decision-sensitive`)
- Interpretation:
  - a 500-episode retrain is not enough to produce a competitive robust policy here
  - training on a slightly broader distribution seems better than narrow specialization around the recommended cell
  - the stronger result of the original checkpoint suggests that either more training, a better training regime, or both are needed before Stage B / C ablations
- Next action:
  - promote mixed-regime training as the better Stage A direction for now
  - run a longer mixed-regime training job before expanding the ablation space
  - compare training duration explicitly (for example 500 vs 2000+ episodes) before changing observability or heterogeneity settings

### Experiment E003: Post-fix rollout checkpoint at episode 800

- Date: 2026-03-28
- Question: After aligning the greedy baseline with pre-cascade reward, promoting the robust defaults, and adding env-knob-aware validation/reporting, does the first post-fix checkpoint improve regime-grid robustness?
- Training config:
  - checkpoint: `experiments/rl_fixes_rollout/recovery_q.pt`
  - intended default run: `python scripts/train_policy.py --checkpoint-dir experiments/rl_fixes_rollout`
  - effective saved artifact: checkpoint written at `episode=800` from the new default config (`alpha_values=[0.10, 0.15, 0.20]`, `pfail_values=[0.10, 0.15, 0.20]`, `warmup_transitions=500`, `batch_size=64`, `num_episodes=8000`)
  - note: the long run was interrupted after the `ep=800` checkpoint was already saved, so this is an early post-fix measurement rather than a finished 8000-episode result
- Evaluation command:
  - `python scripts/evaluate_policy.py --checkpoint experiments/rl_fixes_rollout/recovery_q.pt --grid-source regime_mapping --output-dir experiments/rl_fixes_rollout/benchmark`
- Artifacts:
  - `experiments/rl_fixes_rollout/recovery_q.pt`
  - `experiments/rl_fixes_rollout/recovery_q.summary.json`
  - `experiments/rl_fixes_rollout/benchmark/evaluation_summary.json`
  - `experiments/rl_fixes_rollout/benchmark/evaluation_grid_summary.json`
- Main result:
  - reference-regime benchmark: RL `final_anc=0.149`
  - strongest heuristic at the reference regime was `degree` with `final_anc=0.227`
  - RL validation history on the training grid improved from `0.111` mean final ANC at `ep=200` to `0.337` at `ep=400`, held at `0.334` at `ep=600`, then regressed to `0.225` at `ep=800`
- Regime-bucket summary:
  - overall RL minus best heuristic: `-0.081`
  - `decision-sensitive` RL minus best heuristic: `-0.243`
  - `trivial` RL minus best heuristic: `-0.027`
- RL minus best heuristic:
  - RL still loses in every bucket on this checkpoint
  - the gap is again much larger in `decision-sensitive` cells than in `trivial` cells
- Interpretation:
  - the code-level fixes were implemented successfully, but they were not sufficient on their own to produce a competitive checkpoint by `ep=800`
  - the new in-loop validation is useful: it shows the checkpoint looked materially better around `ep=400-600` than it did by `ep=800`, suggesting instability or overtraining rather than simple monotonic improvement
  - the pre-cascade greedy baseline is now a fairer comparator, but RL is still behind strong structural heuristics (`degree`, `betweenness`) on the regime grid that matters
- Next action:
  - rerun the robust default training without interruption and keep the best checkpoint by validation-grid mean, not only the latest checkpoint
  - compare the `ep=400`, `ep=600`, and final checkpoints explicitly on the same regime grid
  - only start Stage B/C ablations after the longer Stage A run has a stable checkpoint-selection story

### Implementation note: trainer stability diagnostics

- Date: 2026-03-28
- Change summary:
  - `src/cascading_rl/training/trainer.py` now builds validation graphs once from a fixed `validation_seed`, so repeated validation calls use the same holdout set
  - training now cycles through all `(alpha, pfail)` combinations exactly once per cycle before reshuffling the cycle order
  - validation history now records `per_alpha_anc`, and the console output prints the per-alpha breakdown alongside the aggregate validation metrics
- Verification:
  - deterministic validation check added: two consecutive `validate_policy()` calls with the same model and fixed validation graphs produce identical `final_anc_mean`
  - regime coverage check added: after 9 episodes with `alpha_values=[0.10, 0.15, 0.20]` and `pfail_values=[0.10, 0.15, 0.20]`, each of the 9 combinations appears exactly once in `episode_alpha` / `episode_pfail`
  - test command: `python -m pytest tests/ -x -q`
  - result: `33 passed`

## Immediate Next Runs

1. Rerun `python scripts/train_policy.py --checkpoint-dir experiments/rl_fixes_rollout_full` and let the default 8000-episode job finish.
2. Evaluate intermediate and final checkpoints from the robust-default run on the same regime grid.
3. Add best-checkpoint selection based on validation-grid metrics before the next long run if instability persists.
4. Only after that, start Stage B and Stage C ablations.
