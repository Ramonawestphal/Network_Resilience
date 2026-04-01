# Recommended RL Regime

- `tau`: 0.8
- fixed graph instances per cell: 3
- matched seeds per graph: 6

## Recommendation

Start RL training and baseline comparison around `alpha=0.2`, `pfail=0.1`, and `budget=1`.

## Why This Cell

- regime label: `decision-sensitive`
- interestingness score: `0.551`
- final ANC spread across policies: `0.223`
- threshold-hit spread across policies: `0.111`
- budget sensitivity at this `(alpha, pfail)`: `0.387`
- best heuristic in this cell: `greedy`
- best overall policy in this cell: `greedy`
- weakest overall policy in this cell: `random`
