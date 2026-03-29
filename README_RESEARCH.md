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
  - the fixed in-loop reference validation slice looked better around `ep=400-600` (`0.337` at `ep=400`, `0.334` at `ep=600`) than at `ep=800` (`0.225`); treat this as internal monitoring context rather than a grid-level result
- Regime-bucket summary:
  - overall RL minus best heuristic: `-0.081`
  - `decision-sensitive` RL minus best heuristic: `-0.243`
  - `trivial` RL minus best heuristic: `-0.027`
- RL minus best heuristic:
  - RL still loses in every bucket on this checkpoint
  - the gap is again much larger in `decision-sensitive` cells than in `trivial` cells
- Interpretation:
  - the code-level fixes were implemented successfully, but they were not sufficient on their own to produce a competitive checkpoint by `ep=800`
  - the fixed in-loop reference slice suggested the checkpoint looked materially better around `ep=400-600` than it did by `ep=800`, but that should be treated as a provisional monitoring signal rather than proof of grid-level instability or overtraining
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
- Caveat:
  - the printed in-loop validation line still reports the `reference` slice from `config.alpha` / `config.pfail`, while the grid-wide values are stored separately under `validation_history[*]["grid"]`; use checkpoint artifacts or explicit evaluation runs for authoritative regime-grid conclusions
- Verification:
  - deterministic validation check added: two consecutive `validate_policy()` calls with the same model and fixed validation graphs produce identical `final_anc_mean`
  - regime coverage check added: after 9 episodes with `alpha_values=[0.10, 0.15, 0.20]` and `pfail_values=[0.10, 0.15, 0.20]`, each of the 9 combinations appears exactly once in `episode_alpha` / `episode_pfail`
  - test command: `python -m pytest tests/ -x -q`
  - result: `33 passed`

### Experiment E004: Monte Carlo checkpoint sanity check on a broader holdout

- Date: 2026-03-28
- Question: Does the apparently improved `experiments/mc_returns/recovery_q.pt` checkpoint really generalize on a broader `alpha=0.20`, `pfail=0.10`, `budget=2` holdout, and is it simply copying the degree heuristic?
- Training config:
  - checkpoint under inspection: `experiments/mc_returns/recovery_q.pt`
  - provisional in-loop reference-slice snapshots looked mildly encouraging:
    - `ep=600`: `final_anc=0.190`, `threshold_hit=0.333`, `per-alpha: 0.10->0.190  0.15->0.356  0.20->0.509`
    - `ep=1800`: `final_anc=0.195`, `threshold_hit=0.333`, `per-alpha: 0.10->0.195  0.15->0.191  0.20->0.514`
- Evaluation command:
  - `python scripts/evaluate_policy.py --checkpoint experiments/mc_returns/recovery_q.pt --num-graphs 30 --seeds 0 1 2 3 4 --alpha-values 0.20 --pfail-values 0.10 --budgets 2`
  - `python scripts/action_comparison.py --checkpoint experiments/mc_returns/recovery_q.pt --num-graphs 10 --seed 42 --alpha 0.20 --pfail 0.10 --budget 2 --max-rounds 10 --decision-steps 5`
- Artifacts:
  - `experiments/learner_benchmark/evaluation_grid_summary.json`
  - `scripts/action_comparison.py`
- Main result:
  - broader holdout cell (`30` graphs x `5` seeds): RL `final_anc=0.379`, `threshold_hit=0.513`, `rounds=3.793`
  - strongest heuristics in the same cell were `degree` (`final_anc=0.546`) and `betweenness` (`final_anc=0.535`)
  - RL minus best heuristic in the cell: `-0.167`
  - RL-vs-degree action agreement on shared rollouts was only `37.2%` across `43` compared decisions
- Regime-bucket summary:
  - single-cell evaluation only, labeled `decision-sensitive`
  - RL remains behind the best heuristic even when the regime is clearly non-trivial
- RL minus best heuristic:
  - the current MC checkpoint is not just a noisy copy of degree; it is making materially different decisions
  - those different decisions are still weaker than strong structural heuristics on the broader holdout
- Interpretation:
  - the broader holdout evaluation is the authoritative result here; the tiny fixed in-loop validation slice was too optimistic about the strength of the `alpha=0.20` behavior
  - the MC-return change may have improved learning signal somewhat, but the current checkpoint still does not generalize well enough to beat `degree` or `betweenness` in this regime
  - the apparent drop in `alpha=0.15` between the two terminal snapshots is best treated as a hypothesis about unstable or uneven learning, not as a confirmed conclusion from a proper grid evaluation
- Next action:
  - add a stronger initialization to test whether RL can start from a sensible structural prior instead of discovering it from scratch
  - keep treating the printed training validation as a cheap internal monitor, not as the final benchmark story

### Experiment E005: Curriculum run finished and evaluated on its actual training regime

- Date: 2026-03-29
- Question: After training directly on the curriculum `alpha=0.20`, `pfail in {0.08, 0.10}` for `3000` episodes, does the resulting checkpoint perform competitively on a broader holdout drawn from that same regime family?
- Training config:
  - run command: `python scripts/train_policy.py --checkpoint-dir experiments/curriculum_easy --alpha-values 0.20 --pfail-values 0.08 0.10 --episodes 3000`
  - saved training config kept the default reference slice (`alpha=0.10`, `pfail=0.15`) for the printed validation line, but the checkpoint summary also stored the curriculum-grid metrics under `validation_history[*]["grid"]`
- Evaluation command:
  - `python scripts/evaluate_policy.py --checkpoint experiments/curriculum_easy/recovery_q.pt --num-graphs 30 --seeds 0 1 2 3 4 --alpha-values 0.20 --pfail-values 0.08 0.10 --budgets 2 --output-dir experiments/curriculum_easy/benchmark_curriculum`
- Artifacts:
  - `experiments/curriculum_easy/recovery_q.pt`
  - `experiments/curriculum_easy/recovery_q.summary.json`
  - `experiments/curriculum_easy/benchmark_curriculum/evaluation_summary.json`
  - `experiments/curriculum_easy/benchmark_curriculum/evaluation_grid_summary.json`
- Main result:
  - training finished cleanly at `episode=3000`, with `num_updates=21071`, `total_steps=21570`
  - the saved curriculum-grid validation history was non-monotonic but materially stronger than the misleading reference slice: grid `final_anc_mean` moved through `0.428` (`ep=200`), `0.514` (`ep=800`), `0.525` (`ep=1400`), and finished at `0.432` (`ep=3000`)
  - corresponding grid `threshold_hit_mean` moved between `0.500` and `0.667`, finishing at `0.583`
  - broader holdout over both curriculum cells: RL mean `final_anc=0.458`, `threshold_hit=0.603`, `rounds=3.617`, `solved_fraction=0.400`
- Regime-bucket summary:
  - both evaluated cells were labeled `decision-sensitive`
  - overall / `decision-sensitive` RL minus best heuristic: `-0.166`
  - bucket winner counts: `degree` won both cells
- RL minus best heuristic:
  - at `pfail=0.08`, RL reached `final_anc=0.538`, which beat `random`, `risk`, and `greedy`, but still trailed `degree=0.702` and `betweenness=0.692`
  - at `pfail=0.10`, RL reached `final_anc=0.377`, again ahead of `random`, `risk`, and `greedy`, but behind `degree=0.546` and `betweenness=0.535`
  - RL did not win either curriculum cell on the broader holdout
- Interpretation:
  - this curriculum run is better than the terminal reference slice suggested; the saved grid metrics show the policy did learn something meaningful on its actual training distribution
  - however, the broader evaluation still shows a clear gap to the strongest structural heuristics, so the curriculum run is an improved but still non-competitive RL baseline rather than a success case
  - the internal grid history is notably non-monotonic, which supports continuing to rely on saved summaries and post-hoc evaluation rather than on terminal prints alone
- Next action:
  - compare this `curriculum_easy` checkpoint directly against the imitation-warmstarted run on the same two-cell curriculum holdout
  - if imitation warmstart does not close the heuristic gap, prioritize checkpoint selection and training-objective improvements before expanding to harder ablations

### Implementation note: evaluation-script caveat

- Date: 2026-03-28
- Finding:
  - `scripts/evaluate_policy.py` prints a top-level benchmark summary from the config's built-in reference benchmark, while CLI overrides such as `--num-graphs`, `--alpha-values`, and `--pfail-values` are applied to the generated grid evaluation stored in `evaluation_grid_summary.json`
- Impact:
  - the console output can appear unchanged even when a broader or different regime-grid evaluation was requested
  - when checking targeted robustness cells, treat `evaluation_grid_summary.json` as the authoritative artifact

### Implementation note: imitation warmstart added

- Date: 2026-03-28
- Change summary:
  - added `ImitationSample`, `generate_imitation_data()`, and `pretrain_by_imitation()` to `src/cascading_rl/training/trainer.py`
  - added warmstart config fields: `use_imitation_warmstart`, `imitation_graphs`, `imitation_seeds`, `imitation_epochs`
  - when enabled, `train_recovery_agent()` now generates BA graphs, rolls out the degree policy to collect supervision, pretrains the Q-network with masked cross-entropy behavioral cloning, logs per-epoch imitation loss, and then continues with normal RL
  - added `scripts/action_comparison.py` to inspect whether RL is imitating or diverging from degree on shared rollouts
- Verification:
  - added a held-out agreement test: after pretraining on `50` graphs x `3` seeds with the degree policy, the pretrained model exceeds `60%` agreement on `10` held-out graphs
  - `pytest tests/test_training.py -q` -> `7 passed`
  - `pytest tests/ -x -q` -> `34 passed`
- Research use:
  - this enables a direct test of whether a degree-initialized policy is easier to refine with RL than learning from scratch in the cascading-failure setting

## Immediate Next Runs

1. Run one imitation-warmstarted training job and compare it against the current MC baseline on the same broader holdout cells.
2. Evaluate intermediate and final checkpoints from the robust-default and warmstarted runs on the same regime grid.
3. Add best-checkpoint selection based on validation-grid metrics before the next long run if instability persists.
4. Only after that, start Stage B and Stage C ablations.
