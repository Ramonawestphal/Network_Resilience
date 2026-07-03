# Cascading-RL

This project studies reinforcement learning for budget-constrained recovery from
cascading failures on graphs. Nodes fail exogenously at `t = 0`, load
redistributes locally to active neighbours, and each round an agent may
reactivate up to a fixed per-round repair budget `B` before the next cascade
wave is advanced. A GNN-based DQN agent (GDQ-N) is trained to choose which
failed nodes to repair so as to maximise pairwise connectivity over a fixed
horizon, and is compared against degree, betweenness, risk, greedy, and
random repair heuristics. The full motivation, methodology, and evaluation
are described in the accompanying paper (`paper.tex`, kept outside this
repository).

> **Reference of truth.** When the README and the paper disagree, the
> canonical reference is the code in `src/cascading_rl/` together with
> `config/default.yaml`. The paper is the second reference. This README is a
> short pointer, not the spec.

## Repository Layout

```
config/default.yaml   canonical configuration (graph, training, regime-mapping)
src/cascading_rl/      core package
  envs/recovery.py      the recovery Gym-style environment
  dynamics/cascade.py   cascade/load-redistribution dynamics
  graph/generation.py   synthetic graph generation (BA, ER, WS, ...)
  models/               GNN encoder + Q-network
  policies/             heuristic policies (degree, betweenness, risk, greedy, random)
  training/             replay buffer + trainer
  evaluation/           benchmark rollouts, regime mapping, budget search
  metrics/connectivity.py  pairwise-connectivity metric
  reproducibility.py    run-metadata / config-hash helpers used by all scripts
scripts/               CLI entry points (train, evaluate, regime-map, plot, ...)
config/, eval_sets/    fixed evaluation instances checked into git
experiments/           experiment outputs (mostly gitignored; see below)
tests/                 pytest suite
docs/                  architecture and dynamics notes
data/                  real-world / external graph datasets (not committed)
```

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
Eq. (3) and Eq. (4) of the paper. See `docs/dynamics_interpretation.md` for
the full narrative description of the failure/cascade/recovery round
structure, and `docs/architecture.md` for how the model, ablation, and
evaluation code fit together.

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
- device: CPU by default; `training.device` in `config/default.yaml` and
  `Trainer` fall back to CUDA automatically when `torch.cuda.is_available()`
  (`src/cascading_rl/training/trainer.py:126`), but training was developed
  and is reproducible on CPU alone — no GPU is required.

## Installation

Requires **Python ≥ 3.9** (developed against CPython 3.13).

```bash
git clone <this-repo-url>
cd Cascading-RL
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

This installs the `cascading_rl` package in editable mode plus `pytest` /
`pytest-cov`. `requirements.txt` pins the same runtime dependencies
(`networkx`, `torch`, `PyYAML`, `pandas`, `numpy`, `matplotlib`, `seaborn`,
`scipy`, `pyarrow`, `tqdm`) if you prefer `pip install -r requirements.txt`
over the editable install.

No GPU, external services, or credentials are required for any part of the
pipeline.

## Quick Start

All commands are run from the repository root and use `config/default.yaml`
unless overridden with `--config`.

```bash
# 1. Sanity-check the environment + heuristics on a small parameter sweep.
python scripts/map_regime.py

# 2. Train a fresh GDQ-N checkpoint (writes to experiments/learner/ by default).
python scripts/train_policy.py --episodes 2000   # use full 20000 for the reported results

# 3. Evaluate a checkpoint against the heuristic baselines.
python scripts/evaluate_policy.py --checkpoint experiments/learner/recovery_q.pt

# 4. Run the full parameter-generalisation / topology-ablation / OOD evaluation suite.
python scripts/run_full_evaluation.py --checkpoint experiments/learner/recovery_q.pt

# 5. Plot results.
python scripts/plot_evaluation.py --summary experiments/learner_benchmark/evaluation_summary_global.json
```

`--help` on any script lists its full override surface (regime grid, seeds,
graph-size range, budget scaling, output directory, etc.). No trained
checkpoint is committed to this repository (`*.pt` is gitignored) — you must
run `scripts/train_policy.py` yourself to produce one before evaluation.

### Real-world / out-of-distribution data

`scripts/evaluate_real_world.py` benchmarks against out-of-distribution
graphs that are not generated synthetically. Fetch them first:

```bash
python scripts/download_real_world_data.py
```

This downloads the IEEE 300-bus power-grid test case (PGLIB-OPF, CC-BY 4.0)
and deterministically generates a Watts–Strogatz small-world graph
(`n=300, k=4, p=0.1, seed=42`) into `data/processed/`. Nothing under `data/`
is committed to git; regenerate it locally with this script.

### Fixed evaluation sets

`eval_sets/ds_validation.{json,pkl}` and `eval_sets/large_graph_large.{json,pkl}`
are committed, pre-generated evaluation instances used for validation during
training and for the large-graph OOD tier. Regenerate or extend them with
`scripts/generate_eval_set.py` / `scripts/generate_large_graph_eval_set.py`;
see `src/cascading_rl/evaluation/saved_eval_sets.py` for the (de)serialisation
format.

## Main Entry Points

- `python scripts/map_regime.py` — heuristic-only regime map.
- `python scripts/map_regime_comprehensive.py` — full regime map over the
  extended parameter grid (see `README_RESEARCH.md`).
- `python scripts/train_policy.py` — train a fresh GDQ-N checkpoint.
- `python scripts/evaluate_policy.py` — canonical checkpoint evaluation.
- `python scripts/run_full_evaluation.py` — orchestrates the full
  parameter-generalisation, topology-ablation, and OOD evaluation suites.
- `python scripts/run_ablation.py` — feature/virtual-node ablation sweep
  (see `docs/architecture.md` §3).
- `python scripts/download_real_world_data.py` — fetch/generate OOD graphs.
- `python -m pytest tests/ -x -q` — full verification suite.

## Reproducibility Notes

- **Config-driven, not flag-driven.** Every script reads `config/default.yaml`
  by default and accepts `--config` to point at an alternate file; most
  regime/graph/training parameters can also be overridden per-invocation via
  CLI flags (see `--help`). Treat the config file, not ad hoc flags, as the
  source of truth for a given run.
- **Deterministic seeding.** Graph generation, training-episode graph draws,
  and evaluation graph batches are all seeded explicitly (`training.seed`,
  `regime_mapping.graph_seed`, `validation_seeds`, `benchmark_seeds`, etc. in
  `config/default.yaml`). The regime-comprehensive sweep additionally derives
  per-graph seeds from a single `MASTER_SEED` (see `README_RESEARCH.md`).
- **Run metadata is written automatically.** `src/cascading_rl/reproducibility.py`
  (`build_run_metadata` / `write_run_metadata`) stamps every experiment
  output directory with a `run_metadata.json` containing the git commit hash,
  the resolved config file's SHA-256, the full (repo-relative) CLI invocation,
  Python/PyTorch versions, and platform info. Use this file to confirm which
  code and config actually produced a given artifact.
- **No trained weights are committed.** `*.pt` / `*.pth` checkpoints are
  gitignored; `experiments/learner/` only tracks its `run_metadata.json`.
  Reproducing reported RL numbers requires re-running
  `scripts/train_policy.py` (or resuming from a checkpoint you trained
  yourself — the trainer supports checkpoint resume).
- **Fixed vs. regenerated experiment outputs.** Small, hand-curated artifacts
  (`eval_sets/*.json`/`*.pkl`, some `experiments/eval_*` and
  `experiments/learner_benchmark/*.json` summaries) are committed so results
  can be inspected without rerunning anything. Large sweep outputs
  (`experiments/regime_comprehensive*/`, generated `ablation_comparison.json`,
  per-round CSVs) are gitignored by design — see `.gitignore` and
  `README_RESEARCH.md` for exactly which directories to regenerate and with
  which script.
- **CPU-only reproducibility.** Training defaults to `device: cpu` in
  `config/default.yaml`; CUDA is used automatically if available but is never
  required to reproduce results, and no run depends on nondeterministic GPU
  kernels.
- **Heuristic policies are the reproducibility floor.** Because heuristics
  (`degree`, `betweenness`, `risk`, `greedy`, `random`) are deterministic
  given a seed, `scripts/map_regime.py` / `map_regime_comprehensive.py` are
  the cheapest way to confirm your environment reproduces the paper's
  baseline numbers before investing in a full RL training run.

## Testing

```bash
python -m pytest tests/ -x -q
```

The suite (22 test modules under `tests/`) covers the environment dynamics,
cascade/graph generation, budgeting, policies, training loop, evaluation
pipeline, regime mapping, and the CLI scripts' JSON/CSV output contracts. It
runs on CPU and does not require network access or downloaded data
(`test_data.py` covers the data-loading contract, not the real-world
download itself).

## Notes

`README_RESEARCH.md` is the running experiment log (current retained regime
map results and their interpretation). `docs/architecture.md` documents how
the model/ablation/evaluation code fits together for anyone extending the
codebase.

## License

MIT — see [LICENSE](LICENSE).
