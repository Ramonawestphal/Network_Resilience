# Experiments

This directory stores reproducible analysis artifacts generated from
`config/default.yaml` and fixed seeds. Each maintained result folder should
contain a `run_metadata.json` manifest recording the command, config hash, git
commit, Python environment, and relevant runtime options that produced it.

## Canonical Folders

- `learner/`: training summaries and checkpoint metadata. Regenerate locally; avoid committing large `.pt` files unless Git LFS is adopted.
- `learner_benchmark/`: benchmark outputs from `scripts/evaluate_policy.py`.
- `regime_map/`: coarse heuristic regime sweep from `scripts/map_regime.py`.
- `regime_comprehensive/`: resumable regime sweep from `scripts/map_regime_comprehensive.py`.
- `hard_regime/`: hard-cell grid evaluation from `scripts/evaluate_hard_regime.py`.
- `reference_regime/`: reference-cell evaluation and budget-search outputs.

## Reproduction Commands

Use one of these supported install paths first:

```bash
pip install -e .[dev,plot]
```

or

```bash
pip install -r requirements.txt
```

Then regenerate the maintained outputs with:

```bash
python scripts/map_regime.py --config config/default.yaml
python scripts/map_regime_comprehensive.py --config config/default.yaml
python scripts/evaluate_hard_regime.py --config config/default.yaml
python scripts/evaluate_policy.py --config config/default.yaml --checkpoint experiments/learner/recovery_q.pt
python scripts/evaluate_policy.py --config config/default.yaml --checkpoint experiments/learner/recovery_q.pt --grid-source training --output-dir experiments/reference_regime
python scripts/run_budget_search.py --config config/default.yaml --checkpoint experiments/learner/recovery_q.pt
```

If you retrain the learner first, regenerate `learner_benchmark/` and
`reference_regime/` immediately afterward so the JSON outputs stay aligned with
the checkpoint contract.
