# Recommended RL Regime

- `minimum_budget_solved_target` (for b\* search in evaluate scripts): 0.8
- env stopping: `abandonment_anc_threshold` = 0.3
- fixed graph instances per cell: 3
- matched seeds per graph: 6

## Recommendation

Start RL training and baseline comparison around `alpha=0.1`, `pfail=0.15`, and `budget=3`.

## Why This Cell

- regime label: `decision-sensitive`
- interestingness score: `0.879`
- final ANC spread across heuristics: `0.606`
- solved-fraction spread across heuristics: `0.722`
- budget sensitivity at this `(alpha, pfail)`: `0.724`
- current best heuristic in this cell: `greedy`
- current weakest heuristic in this cell: `random`
