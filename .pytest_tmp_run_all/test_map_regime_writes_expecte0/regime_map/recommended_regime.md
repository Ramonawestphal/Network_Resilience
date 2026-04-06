# Recommended RL Regime

- `minimum_budget_solved_target` (for b\* search in evaluate scripts): 0.8
- env stopping: `abandonment_anc_threshold` = None
- fixed graph instances per cell: 1
- matched seeds per graph: 1

## Recommendation

Start RL training and baseline comparison around `alpha=0.2`, `pfail=0.05`, and `budget=1`.

## Why This Cell

- regime label: `trivial`
- interestingness score: `0.000`
- final ANC spread across heuristics: `0.000`
- solved-fraction spread across heuristics: `0.000`
- budget sensitivity at this `(alpha, pfail)`: `0.000`
- current best heuristic in this cell: `random`
- current weakest heuristic in this cell: `random`

## Caveat

The coarse sweep found only a small spread between heuristic baselines in this cell. This still marks the strongest policy-sensitive setting observed so far, but it suggests the regime map should be refined further before assuming there is large headroom for RL.
