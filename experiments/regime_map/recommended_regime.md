# Recommended RL Regime

- `tau`: 0.8
- fixed graph instances per cell: 3
- matched seeds per graph: 6

## Recommendation

Start RL training and baseline comparison around `alpha=0.2`, `pfail=0.05`, and `budget=4`.

## Why This Cell

- regime label: `interesting`
- interestingness score: `0.319`
- final ANC spread across heuristics: `0.006`
- threshold-hit spread across heuristics: `0.000`
- budget sensitivity at this `(alpha, pfail)`: `0.017`
- current best heuristic in this cell: `greedy`
- current weakest heuristic in this cell: `random`

## Caveat

The coarse sweep found only a small spread between heuristic baselines in this cell. This still marks the strongest policy-sensitive setting observed so far, but it suggests the regime map should be refined further before assuming there is large headroom for RL.
