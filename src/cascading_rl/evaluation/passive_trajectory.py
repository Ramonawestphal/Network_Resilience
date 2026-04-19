"""Cascade-only (no recovery) NC trajectories for reference curves."""

from __future__ import annotations

from cascading_rl.dynamics.cascade import CascadeState, advance_cascade_round
from cascading_rl.metrics.connectivity import normalized_connectivity


def passive_nc_trajectory(state: CascadeState, *, max_rounds: int) -> list[float]:
    """Normalized connectivity after each cascade wave with no repairs.

    Index ``0`` is NC immediately after the initial exogenous failures (same state
    as ``RecoveryEnv.reset``). For ``k`` in ``1 .. max_rounds``, index ``k`` is NC
    after the ``k``-th ``advance_cascade_round`` call (one wave per round, matching
    the single cascade at the end of each recovery round).

    Parameters
    ----------
    state
        A copy of the cascade state (will not be mutated).
    max_rounds
        Number of passive cascade waves after the initial snapshot (length of
        output is ``max_rounds + 1``).
    """
    if max_rounds < 0:
        raise ValueError("max_rounds must be non-negative.")
    s = state.copy()
    out: list[float] = [normalized_connectivity(s.graph, s.active)]
    for _ in range(max_rounds):
        advance_cascade_round(s)
        out.append(normalized_connectivity(s.graph, s.active))
    return out
