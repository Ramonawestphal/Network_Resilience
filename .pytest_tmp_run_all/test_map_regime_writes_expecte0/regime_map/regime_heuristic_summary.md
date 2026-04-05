# Regime heuristic rollup

Per-policy means over all grid cells (alpha x p_fail x budget).

“Unsolved low-ANC” counts episodes with remaining failed nodes and final ANC strictly below **0.3** (from `abandonment_anc_threshold` when set in config, else 0.3).


| policy | mean solved fraction | mean unsolved low-ANC fraction |
| --- | ---: | ---: |
| random | 1.0000 | 0.0000 |
| degree | 1.0000 | 0.0000 |
