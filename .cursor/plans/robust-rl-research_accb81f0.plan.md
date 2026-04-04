---
name: robust-rl-research
overview: Create a step-by-step research workflow for Cascading-RL that separates trivial, recoverable, and hopeless regimes; compares RL against heuristics fairly; and records progress in a dedicated research README.
todos:
  - id: research-readme
    content: Create a dedicated `README_RESEARCH.md` with sections for hypotheses, experiment log, findings, and next steps.
    status: completed
  - id: regime-taxonomy
    content: Use current regime-map outputs and thresholds to define trivial, hopeless, and decision-sensitive buckets.
    status: completed
  - id: grid-eval
    content: Unify the evaluation scripts so a single checkpoint can be compared against heuristics across a configurable regime grid.
    status: completed
  - id: ablation-program
    content: Design a staged experiment matrix covering environment difficulty, observability/action constraints, heterogeneity, and model/training choices.
    status: completed
  - id: robustness-reporting
    content: Report every experiment by regime bucket and heuristic gap, not just raw mean ANC.
    status: completed
isProject: false
---

# Robust Cascading-RL Research Plan

## Goal

Build a research workflow around the existing Cascading-RL code so you can answer: which environment/model choices actually improve robustness across regimes, and when does RL beat strong heuristics rather than only doing well on easy settings.

## Research framing

The biggest confounder in this repo is that some episodes are effectively trivial and some are unrecoverable. Before optimizing the learner, the project should classify regime difficulty and evaluate policies inside comparable buckets.

Current code already gives useful building blocks:

- Training samples per-episode `(alpha, pfail)` from grids in [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\scripts\train_policy.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\scripts\train_policy.py) and [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\src\cascading_rl\training\trainer.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\src\cascading_rl\training\trainer.py).
- Heuristic comparisons and regime sweeps already exist in [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\scripts\evaluate_policy.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\scripts\evaluate_policy.py), [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\scripts\evaluate_hard_regime.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\scripts\evaluate_hard_regime.py), and [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\scripts\map_regime.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\scripts\map_regime.py).
- Environment semantics that most affect conclusions live in [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\src\cascading_rl\envs\recovery.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\src\cascading_rl\envs\recovery.py): reward is based on ANC improvement after reactivation and before the next cascade wave.

## Proposed workflow

1. Add a dedicated research log file, preferably `README_RESEARCH.md`, to track hypotheses, experiment IDs, config choices, outcomes, and next decisions.
2. Define a regime taxonomy using the existing regime-map outputs:
  - `trivial`: heuristics and RL all near ceiling.
  - `hopeless`: all methods fail badly.
  - `decision-sensitive`: medium-difficulty cells where policy choice matters.
   This should reuse [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\config\default.yaml](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\config\default.yaml) thresholds and [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\src\cascading_rl\evaluation\regime.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\src\cascading_rl\evaluation\regime.py).
3. Standardize the evaluation protocol so every experiment reports the same things:
  - final ANC

n   - solved / threshold-hit fraction

- rounds used
- gap to best heuristic
- performance broken out by regime bucket, not only global averages

1. Expand evaluation so one trained checkpoint can be tested on a full `(alpha, pfail)` grid, not only a single validation regime. Reuse the existing heuristic evaluation pipeline instead of inventing a new benchmark path.
2. Run a staged ablation program rather than changing many things at once:
  - Stage A: environment difficulty controls only (`alpha`, `pfail`, budget, max rounds, graph size).
  - Stage B: observability and action-space choices (`obs_hops`, frontier/adjacent action spaces).
  - Stage C: environment heterogeneity (`capacity_noise`, `failure_bias`).
  - Stage D: model/training choices (virtual node, hidden/embed size, mixed-regime training vs narrow-regime training).
3. Promote robustness as the main claim:
  - compare in-distribution performance on training-like regimes,
  - nearby out-of-distribution regimes,
  - and structurally shifted environments where graph size or stochastic failure pattern changes.
4. After each experiment batch, update the research README with: hypothesis, exact config, artifact paths, key results, interpretation, and next action.

## Implementation focus

### Phase 1: Research bookkeeping

Create a lightweight experiment log structure in `README_RESEARCH.md` with sections for:

- research question
- regime taxonomy
- experiment table
- key findings
- open problems / next runs

### Phase 2: Fair regime-aware benchmarking

Adapt the current evaluation scripts so they can evaluate one checkpoint across a configurable grid and summarize results by difficulty bucket. The most relevant existing code paths are:

- [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\scripts\evaluate_hard_regime.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\scripts\evaluate_hard_regime.py)
- [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\scripts\evaluate_policy.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\scripts\evaluate_policy.py)
- [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\src\cascading_rl\evaluation\benchmarks.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\src\cascading_rl\evaluation\benchmarks.py)
- [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\src\cascading_rl\evaluation\regime.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\src\cascading_rl\evaluation\regime.py)

### Phase 3: Controlled training studies

Use the current trainer in [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\src\cascading_rl\training\trainer.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\src\cascading_rl\training\trainer.py) and config entrypoints in [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\scripts\train_policy.py](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\scripts\train_policy.py) / [c:\Users\Ramonavscode\Seminar_ML\Cascading-RL\config\default.yaml](c:\Users\Ramona.vscode\Seminar_ML\Cascading-RL\config\default.yaml) to run only a small set of well-motivated experiment families:

- narrow-regime vs multi-regime training
- easy/mixed/hard training distributions
- full observability vs partial observability
- default action space vs constrained action spaces
- homogeneous vs heterogeneous capacities / failure bias

### Phase 4: Interpretation discipline

For each result, separate these cases explicitly:

- RL wins because the regime is genuinely decision-sensitive.
- RL ties heuristics because the regime is trivial.
- Everyone fails because the regime is hopeless.
This prevents overclaiming and makes the final thesis/story much stronger.

## Suggested first milestone

Start by building the research README and a regime-aware benchmark pass. Before training more variants, establish a reliable answer to: "On which cells of the `(alpha, pfail, budget)` space is this problem trivial, hopeless, or actually worth learning?" That gives you a defensible basis for every later model comparison.