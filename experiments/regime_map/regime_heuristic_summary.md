# Regime heuristic rollup

Per-policy means over all grid cells (alpha x p_fail x budget).

“Unsolved low-ANC” counts episodes with remaining failed nodes and final ANC strictly below **0.3** (from `abandonment_anc_threshold` when set in config, else 0.3).


| policy | mean solved fraction | mean unsolved low-ANC fraction |
| --- | ---: | ---: |
| random | 0.5937 | 0.4063 |
| degree | 0.8342 | 0.1658 |
| risk | 0.6641 | 0.3359 |
| greedy | 0.8585 | 0.1380 |
| betweenness | 0.8264 | 0.1727 |
