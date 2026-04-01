# Cascading-RL

Research code for budget-constrained recovery in networks subject to cascading
failure. The repository now includes the merged training, analysis, and result
reproduction workflow in one branch.

## Install

Recommended:

```bash
pip install -e .[dev,plot]
```

Alternative:

```bash
pip install -r requirements.txt
```

## Main Workflow

Train a learner checkpoint:

```bash
python scripts/train_policy.py --config config/default.yaml
```

Generate the maintained analysis outputs:

```bash
python scripts/map_regime.py --config config/default.yaml
python scripts/map_regime_comprehensive.py --config config/default.yaml
python scripts/evaluate_hard_regime.py --config config/default.yaml
python scripts/evaluate_policy.py --config config/default.yaml --checkpoint experiments/learner/recovery_q.pt
python scripts/evaluate_policy.py --config config/default.yaml --checkpoint experiments/learner/recovery_q.pt --grid-source training --output-dir experiments/reference_regime
python scripts/run_budget_search.py --config config/default.yaml --checkpoint experiments/learner/recovery_q.pt
```

Each maintained result directory writes a `run_metadata.json` manifest so the
artifacts under `experiments/` can be traced back to the exact config and
command that generated them.

## Repository Layout

- `src/cascading_rl/`: package code for graph generation, dynamics, environments, models, policies, training, and evaluation
- `config/`: default and study-specific settings
- `scripts/`: reproducible entry points for training and analysis
- `tests/`: unit and integration tests for the merged workflow
- `experiments/`: generated result folders and reproduction notes
- `docs/`: architecture and dynamics notes
- `notebooks/`: exploratory analysis notebooks
