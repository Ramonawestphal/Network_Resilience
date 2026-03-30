# Recommended RL Regime

- `tau`: 0.8
- fixed graph instances per cell: 3
- matched seeds per graph: 6

## Recommendation

Start RL training and baseline comparison around `alpha=0.1`, `pfail=0.15`, and `budget=4`.

## Why This Cell

- regime label: `decision-sensitive`
- interestingness score: `0.616`
- final ANC spread across heuristics: `0.567`
- threshold-hit spread across heuristics: `0.278`
- budget sensitivity at this `(alpha, pfail)`: `0.821`
- current best heuristic in this cell: `degree`
- current weakest heuristic in this cell: `random`
