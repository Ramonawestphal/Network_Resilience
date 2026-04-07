"""Episode-level and aggregate recovery metrics for saved eval sets."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean as _mean, stdev as _stdev
from typing import Sequence


@dataclass
class EpisodeMetrics:
    recovered: bool
    rounds_to_recovery: int | None  # None if episode failed
    rounds_to_termination: int  # always set
    anc_per_round: list[float]  # ANC after each round, length == rounds_to_termination
    mean_anc_conditional: float | None  # mean over recovery rounds; None if failed
    mean_anc_unconditional: float  # mean over all rounds; always set


def compute_episode_metrics(
    anc_trajectory: list[float],
    recovered: bool,
) -> EpisodeMetrics:
    """Build per-episode metrics from a recorded ANC trajectory.

    Parameters
    ----------
    anc_trajectory:
        ANC value recorded after each repair-cascade wave (one entry per round).
        Length equals the actual number of rounds played.
    recovered:
        Whether the episode ended with full network recovery.
    """
    n = len(anc_trajectory)
    return EpisodeMetrics(
        recovered=recovered,
        rounds_to_recovery=n if recovered else None,
        rounds_to_termination=n,
        anc_per_round=list(anc_trajectory),
        mean_anc_conditional=_mean(anc_trajectory) if (recovered and anc_trajectory) else None,
        mean_anc_unconditional=_mean(anc_trajectory) if anc_trajectory else 0.0,
    )


@dataclass
class AggregateMetrics:
    n_episodes: int
    recovered_fraction: float  # fraction of episodes where recovered=True
    mean_rounds_to_recovery: float | None  # mean over recovered episodes; None if none recovered
    std_rounds_to_recovery: float | None
    mean_rounds_to_termination_failed: float | None  # mean over failed episodes; None if none failed
    std_rounds_to_termination_failed: float | None
    mean_anc_conditional: float | None  # mean ANC over recovered episodes; None if none recovered
    stderr_anc_conditional: float | None
    mean_anc_unconditional: float  # mean ANC over all episodes
    stderr_anc_unconditional: float
    mean_anc_per_round: list[float]  # mean ANC at each round index (aligned to max length)
    n_per_round: list[int]  # number of episodes that reached each round index

    # per-round curves split by outcome
    mean_nc_per_round_recovered: list[float]   # averaged over recovered episodes only
    n_episodes_per_round_recovered: list[int]
    mean_nc_per_round_failed: list[float]      # averaged over failed episodes only
    n_episodes_per_round_failed: list[int]


def compute_aggregate_metrics(episodes: Sequence[EpisodeMetrics]) -> AggregateMetrics:
    """Aggregate a list of EpisodeMetrics into summary statistics.

    For ``mean_anc_per_round``: episodes may have different lengths. At each round
    index the mean is taken only over episodes that reached that round — no padding
    with zeros or ones.

    ``stderr = stdev / sqrt(n)`` where n is the number of contributing episodes.
    """
    n = len(episodes)

    # --- recovery fraction ---
    recovered_episodes = [ep for ep in episodes if ep.recovered]
    failed_episodes = [ep for ep in episodes if not ep.recovered]
    recovered_fraction = len(recovered_episodes) / n if n else 0.0

    # --- rounds to recovery (recovered episodes only) ---
    if recovered_episodes:
        rtr = [float(ep.rounds_to_termination) for ep in recovered_episodes]
        mean_rtr = _mean(rtr)
        std_rtr = _stdev(rtr) if len(rtr) > 1 else 0.0
    else:
        mean_rtr = None
        std_rtr = None

    # --- rounds to termination (failed episodes only) ---
    if failed_episodes:
        rtf = [float(ep.rounds_to_termination) for ep in failed_episodes]
        mean_rtf = _mean(rtf)
        std_rtf = _stdev(rtf) if len(rtf) > 1 else 0.0
    else:
        mean_rtf = None
        std_rtf = None

    # --- conditional ANC (recovered episodes only) ---
    cond_values = [ep.mean_anc_conditional for ep in recovered_episodes if ep.mean_anc_conditional is not None]
    if cond_values:
        mean_anc_cond = _mean(cond_values)
        stderr_anc_cond = (_stdev(cond_values) / sqrt(len(cond_values))) if len(cond_values) > 1 else 0.0
    else:
        mean_anc_cond = None
        stderr_anc_cond = None

    # --- unconditional ANC (all episodes) ---
    uncond_values = [ep.mean_anc_unconditional for ep in episodes]
    if uncond_values:
        mean_anc_uncond = _mean(uncond_values)
        stderr_anc_uncond = (_stdev(uncond_values) / sqrt(len(uncond_values))) if len(uncond_values) > 1 else 0.0
    else:
        mean_anc_uncond = 0.0
        stderr_anc_uncond = 0.0

    # --- per-round ANC alignment (all episodes) ---
    max_len = max((len(ep.anc_per_round) for ep in episodes), default=0)
    mean_anc_per_round: list[float] = []
    n_per_round: list[int] = []
    for i in range(max_len):
        values = [ep.anc_per_round[i] for ep in episodes if len(ep.anc_per_round) > i]
        mean_anc_per_round.append(_mean(values) if values else 0.0)
        n_per_round.append(len(values))

    # --- per-round curves split by outcome ---
    max_len_recovered = max((len(ep.anc_per_round) for ep in recovered_episodes), default=0)
    mean_nc_per_round_recovered: list[float] = []
    n_episodes_per_round_recovered: list[int] = []
    for i in range(max_len_recovered):
        vals = [ep.anc_per_round[i] for ep in recovered_episodes if len(ep.anc_per_round) > i]
        mean_nc_per_round_recovered.append(_mean(vals) if vals else 0.0)
        n_episodes_per_round_recovered.append(len(vals))

    max_len_failed = max((len(ep.anc_per_round) for ep in failed_episodes), default=0)
    mean_nc_per_round_failed: list[float] = []
    n_episodes_per_round_failed: list[int] = []
    for i in range(max_len_failed):
        vals = [ep.anc_per_round[i] for ep in failed_episodes if len(ep.anc_per_round) > i]
        mean_nc_per_round_failed.append(_mean(vals) if vals else 0.0)
        n_episodes_per_round_failed.append(len(vals))

    return AggregateMetrics(
        n_episodes=n,
        recovered_fraction=recovered_fraction,
        mean_rounds_to_recovery=mean_rtr,
        std_rounds_to_recovery=std_rtr,
        mean_rounds_to_termination_failed=mean_rtf,
        std_rounds_to_termination_failed=std_rtf,
        mean_anc_conditional=mean_anc_cond,
        stderr_anc_conditional=stderr_anc_cond,
        mean_anc_unconditional=mean_anc_uncond,
        stderr_anc_unconditional=stderr_anc_uncond,
        mean_anc_per_round=mean_anc_per_round,
        n_per_round=n_per_round,
        mean_nc_per_round_recovered=mean_nc_per_round_recovered,
        n_episodes_per_round_recovered=n_episodes_per_round_recovered,
        mean_nc_per_round_failed=mean_nc_per_round_failed,
        n_episodes_per_round_failed=n_episodes_per_round_failed,
    )
