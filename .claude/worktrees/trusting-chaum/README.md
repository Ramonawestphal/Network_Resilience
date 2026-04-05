# Cascading-RL

Controlled research codebase for **budget-constrained recovery** on networks under **cascading failures**. It couples explicit cascade dynamics (loads, capacities, failure spread) with a **graph observation** for learning: a **GNN-based DQN** trains recovery policies, while **heuristic baselines** support benchmarking and regime characterization.

The repo is structured for **reproducible experiments**: YAML config, scripted entry points, checkpoints, JSON artifacts, and `run_metadata.json` manifests under `experiments/`.

## Features

- **Recovery environment** (`RecoveryEnv`): round-based reactivations, post-action cascades, optional observation masking and action-space variants.
- **Learning**: `RecoveryQNetwork` with optional global readout and virtual node; replay buffer, target updates, epsilon schedule (`scripts/train_policy.py`).
- **Evaluation**: matched-seed rollouts, regime grids, hard-regime sweeps, scaling and budget search scripts.
- **Analysis**: regime maps (coarse and resumable comprehensive), plots, CSV/JSON outputs.

## Requirements

- **Python** 3.9+
- **PyTorch** 2.6+ (see `pyproject.toml`)
- **NetworkX**, **PyYAML** (core); **matplotlib** / **seaborn** for plotting scripts; **pandas** useful for tabular outputs

GPU is optional; set `device` in training config to `cuda` when available.

## Installation

From the repository root:

```bash
pip install -e ".[dev,plot]"
```

This installs the `cascading_rl` package from `src/`, development tools (`pytest`), and plotting extras.

Alternatively:

```bash
pip install -r requirements.txt
pip install -e .
```

(Use `pip install -e .[dev]` if you only need tests, without plot dependencies.)

## Quick start

1. **Train** a Q-network checkpoint (paths and hyperparameters come from `config/default.yaml` → `training` section):

   ```bash
   python scripts/train_policy.py --config config/default.yaml
   ```

2. **Evaluate** the trained policy against baselines (adjust `--checkpoint` to your saved `.pt`):

   ```bash
   python scripts/evaluate_policy.py --config config/default.yaml --checkpoint experiments/learner/recovery_q.pt
   ```

3. **Run tests**:

   ```bash
   pytest
   ```

## Scripts

| Script | Purpose |
|--------|--------|
| `scripts/train_policy.py` | Train the recovery Q-network; writes checkpoint and run metadata. |
| `scripts/evaluate_policy.py` | Benchmark learned vs heuristic policies on configured graphs/regimes. |
| `scripts/map_regime.py` | Coarse regime sweep and recommendations. |
| `scripts/map_regime_comprehensive.py` | Larger, **resumable** regime sweep (checkpoint fingerprint avoids mismatched resumes). |
| `scripts/evaluate_hard_regime.py` | Focused evaluation on “hard” (alpha, pfail) grids. |
| `scripts/run_budget_search.py` | Budget sensitivity search for a fixed checkpoint. |
| `scripts/evaluate_scaling.py` | Scaling-related evaluation experiments. |
| `scripts/run_ablation.py` | Model / feature ablations. |
| `scripts/action_comparison.py` | Compare action-selection behavior. |
| `scripts/visualize_cascade.py` | Visualize cascade dynamics. |
| `scripts/plot_regime.py` | Plotting helpers used by mapping scripts. |

Most scripts accept `--config` pointing at a YAML file. Override output locations via the config (e.g. `regime_mapping.output_dir`) or script-specific CLI flags where provided.

## Configuration

- **Default stack**: `config/default.yaml` — graph generator (`n_range`, `m`), cascade (`alpha`, `pfail`), recovery budget and `max_rounds`, evaluation budgets/seeds, training block, regime mapping, and artifact directories.
- **Variants**: `config/sensitivity/` and other trees for study-specific settings; see nested READMEs where present.

Training reads the `training:` section; mapping scripts use `regime_mapping`, `graph`, `evaluation`, and related keys. Keeping a single YAML per “paper figure” or experiment reduces drift between commands.

## Project layout

- `src/cascading_rl/` — library code: graph generation, cascade dynamics, envs, metrics, GNN Q-network, policies, replay, training, evaluation.
- `config/` — YAML defaults and experiment configs.
- `scripts/` — CLI entry points (all intended to be run from repo root or with `PYTHONPATH` set via editable install).
- `tests/` — unit and smoke tests.
- `experiments/` — generated results; maintained folders and reproduction commands are described in [`experiments/README.md`](experiments/README.md).
- `docs/` — design notes (e.g. architecture, dynamics interpretation).
- `notebooks/` — exploratory analysis (see `notebooks/README.md`).

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — observation features, `step` vs `step_batch`, and training/eval alignment.
- [`docs/dynamics_interpretation.md`](docs/dynamics_interpretation.md) — short interpretation notes for cascade dynamics.

## Reproducibility

Scripts record **`run_metadata.json`** (command line, environment, config-relevant fields) alongside outputs where implemented. For a full regeneration checklist and canonical artifact folders (`learner/`, `regime_map/`, `regime_comprehensive/`, etc.), see [`experiments/README.md`](experiments/README.md).

## License

This project is released under the **MIT License** — see [`LICENSE`](LICENSE).
