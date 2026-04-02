from __future__ import annotations

DEFAULT_REFERENCE_N = 40


def compute_scaled_budget(
    reference_budget: int,
    *,
    num_nodes: int,
    reference_n: int = DEFAULT_REFERENCE_N,
    enabled: bool = True,
) -> int:
    """Return the per-graph recovery budget from a canonical reference budget."""
    if reference_budget < 1:
        raise ValueError("reference_budget must be at least 1.")
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")
    if reference_n < 1:
        raise ValueError("reference_n must be at least 1.")
    if not enabled:
        return reference_budget

    beta = reference_budget / reference_n
    return max(1, round(beta * num_nodes))
