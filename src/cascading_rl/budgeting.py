from __future__ import annotations


DEFAULT_REFERENCE_N = 40


def compute_scaled_budget(
    reference_budget: int,
    *,
    num_nodes: int,
    reference_n: int = DEFAULT_REFERENCE_N,
    enabled: bool = True,
) -> int:
    """Return the per-graph recovery budget.

    When scaling is enabled, the configured budget is interpreted as a reference
    budget at ``reference_n`` nodes and scaled linearly with graph size.
    """
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


def compute_scaled_max_rounds(
    reference_max_rounds: int,
    *,
    num_nodes: int,
    reference_n: int = DEFAULT_REFERENCE_N,
    enabled: bool = True,
) -> int:
    """Scale ``max_rounds`` linearly with graph size (same reference rule as ``compute_scaled_budget``)."""
    if reference_max_rounds < 1:
        raise ValueError("reference_max_rounds must be at least 1.")
    if num_nodes < 1:
        raise ValueError("num_nodes must be at least 1.")
    if reference_n < 1:
        raise ValueError("reference_n must be at least 1.")
    if not enabled:
        return reference_max_rounds
    gamma = reference_max_rounds / reference_n
    return max(1, round(gamma * num_nodes))
