"""Bar charts: fraction of instances fully recovered per heuristic for **decision-sensitive** instances only.

By default writes **one PNG per** ``(alpha, p_fail)`` combination (only cells that have at
least one DS instance). Use ``--grid-figure`` for a single multi-panel image instead.
Reads ``regime_instances.parquet`` (preferred) or ``regime_instances.csv`` from a
``map_regime_comprehensive`` output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import pyarrow  # noqa: F401

    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False

POLICY_ORDER = ("degree", "random", "betweenness", "greedy", "risk")


def _fmt_regime_tag(alpha: float, pfail: float) -> str:
    return f"alpha_{alpha:g}_pfail_{pfail:g}"


def load_instances(data_dir: Path) -> pd.DataFrame:
    parquet_path = data_dir / "regime_instances.parquet"
    csv_path = data_dir / "regime_instances.csv"
    if HAS_PARQUET and parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if len(df) <= 10_000:
            print(
                "Warning: using truncated regime_instances.csv (first 10k rows only). "
                "Install pyarrow and use regime_instances.parquet for full runs.",
                file=sys.stderr,
            )
        return df
    raise FileNotFoundError(
        f"No regime_instances.parquet or regime_instances.csv under {data_dir}"
    )


def plot_recovery_fraction_ds_by_alpha_pfail(
    df: pd.DataFrame,
    *,
    budget_ref: int,
    output_path: Path,
    instance_label: str = "decision_sensitive",
) -> None:
    at_budget = df.loc[df["budget_ref"] == budget_ref].copy()
    if at_budget.empty:
        raise ValueError(f"No rows with budget_ref={budget_ref}")

    alphas = sorted(at_budget["alpha"].unique(), key=float)
    pfails = sorted(at_budget["pfail"].unique(), key=float)

    work = at_budget.loc[at_budget["instance_label"] == instance_label].copy()
    if work.empty:
        raise ValueError(
            f"No rows with instance_label={instance_label!r} and budget_ref={budget_ref}"
        )

    if work["solved"].dtype == object:
        work["solved"] = work["solved"].map(
            {"True": True, "False": False, True: True, False: False}
        )
    work["solved"] = work["solved"].astype(bool)
    n_r, n_c = len(pfails), len(alphas)

    fig_w = max(10.0, 1.55 * n_c)
    fig_h = max(8.0, 1.35 * n_r)
    fig, axes = plt.subplots(
        n_r,
        n_c,
        figsize=(fig_w, fig_h),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if n_r == 1 and n_c == 1:
        axes_array = np.array([[axes]])
    elif n_r == 1:
        axes_array = np.array([axes])
    elif n_c == 1:
        axes_array = np.array([[ax] for ax in axes])
    else:
        axes_array = np.array(axes)

    for pi, pfail in enumerate(pfails):
        for ai, alpha in enumerate(alphas):
            cell = work.loc[(work["alpha"] == alpha) & (work["pfail"] == pfail)]
            ax = axes_array[pi, ai]
            fracs: list[float] = []
            for pol in POLICY_ORDER:
                pol_rows = cell.loc[cell["policy"] == pol, "solved"]
                fracs.append(float(pol_rows.mean()) if len(pol_rows) else float("nan"))
            n_inst = cell.groupby(["graph_id", "seed_index"], observed=True).ngroups
            if n_inst == 0:
                ax.set_facecolor("#f2f2f2")
                ax.text(0.5, 0.5, "no DS\ninstances", ha="center", va="center", fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([0, 0.5, 1.0])
                continue
            x = np.arange(len(POLICY_ORDER))
            mask = [np.isfinite(f) for f in fracs]
            plot_vals = [f if m else 0.0 for f, m in zip(fracs, mask, strict=True)]
            ax.bar(x, plot_vals, color="steelblue", edgecolor="black", linewidth=0.4)
            ax.set_xticks(x, [])
            ax.tick_params(axis="y", labelsize=7)
            if n_inst <= 500:
                tiny = n_r * n_c <= 24
                for xi, f in enumerate(fracs):
                    if np.isfinite(f) and (f > 0 or tiny):
                        ax.text(
                            xi,
                            min(f + 0.04, 0.98),
                            f"{f:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=5 if not tiny else 6,
                        )

    for pi in range(n_r):
        for ai in range(n_c):
            axes_array[pi, ai].set_ylim(0.0, 1.0)

    for ai, alpha in enumerate(alphas):
        axes_array[n_r - 1, ai].set_xticks(
            range(len(POLICY_ORDER)), list(POLICY_ORDER), rotation=55, ha="right", fontsize=7
        )

    for pi, pfail in enumerate(pfails):
        axes_array[pi, 0].set_ylabel(f"p_fail={pfail:.2f}\nfrac.", fontsize=8)

    for ai, alpha in enumerate(alphas):
        axes_array[0, ai].set_title(f"α={alpha:.2f}", fontsize=9)

    fig.suptitle(
        f"Decision-sensitive instances only — fraction fully recovered per heuristic "
        f"(budget_ref = {budget_ref})",
        fontsize=12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_recovery_fraction_ds_separate_panels(
    df: pd.DataFrame,
    *,
    budget_ref: int,
    output_dir: Path,
    instance_label: str = "decision_sensitive",
    include_empty_cells: bool = False,
) -> list[Path]:
    """One bar-chart PNG per (alpha, p_fail); skips empty cells unless ``include_empty_cells``."""
    at_budget = df.loc[df["budget_ref"] == budget_ref].copy()
    if at_budget.empty:
        raise ValueError(f"No rows with budget_ref={budget_ref}")

    alphas = sorted(at_budget["alpha"].unique(), key=float)
    pfails = sorted(at_budget["pfail"].unique(), key=float)

    work = at_budget.loc[at_budget["instance_label"] == instance_label].copy()
    if work.empty:
        raise ValueError(
            f"No rows with instance_label={instance_label!r} and budget_ref={budget_ref}"
        )

    if work["solved"].dtype == object:
        work["solved"] = work["solved"].map(
            {"True": True, "False": False, True: True, False: False}
        )
    work["solved"] = work["solved"].astype(bool)

    saved: list[Path] = []
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for pfail in pfails:
        for alpha in alphas:
            cell = work.loc[(work["alpha"] == alpha) & (work["pfail"] == pfail)]
            n_inst = cell.groupby(["graph_id", "seed_index"], observed=True).ngroups
            if n_inst == 0 and not include_empty_cells:
                continue

            fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
            if n_inst == 0:
                ax.set_facecolor("#f2f2f2")
                ax.text(0.5, 0.5, "no DS instances", ha="center", va="center", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([0, 0.5, 1.0])
                ax.set_ylim(0, 1)
            else:
                fracs: list[float] = []
                for pol in POLICY_ORDER:
                    pol_rows = cell.loc[cell["policy"] == pol, "solved"]
                    fracs.append(
                        float(pol_rows.mean()) if len(pol_rows) else float("nan")
                    )
                x = np.arange(len(POLICY_ORDER))
                mask = [np.isfinite(f) for f in fracs]
                plot_vals = [f if m else 0.0 for f, m in zip(fracs, mask, strict=True)]
                ax.bar(x, plot_vals, color="steelblue", edgecolor="black", linewidth=0.5)
                ax.set_xticks(x, list(POLICY_ORDER), rotation=40, ha="right", fontsize=9)
                ax.set_ylabel("Fraction fully recovered", fontsize=10)
                ax.set_ylim(0, 1)
                for xi, f in enumerate(fracs):
                    if np.isfinite(f) and f > 0:
                        ax.text(
                            xi,
                            min(f + 0.04, 0.96),
                            f"{f:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

            tag = _fmt_regime_tag(float(alpha), float(pfail))
            out_path = output_dir / f"recovery_fraction_DS_B{budget_ref}_{tag}.png"
            fig.suptitle(
                f"Decision-sensitive — α={float(alpha):.3g}, p_fail={float(pfail):.3g} "
                f"({n_inst} graph×seed; budget_ref={budget_ref})",
                fontsize=11,
            )
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            saved.append(out_path)

    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot recovery fraction per heuristic for decision-sensitive instances, "
        "faceted by (alpha, p_fail)."
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "path" / "to" / "your" / "output",
        help="map_regime_comprehensive output directory",
    )
    p.add_argument(
        "--budget-ref",
        type=int,
        default=2,
        help="Reference budget level (default: 2)",
    )
    p.add_argument(
        "--instance-label",
        type=str,
        default="decision_sensitive",
        help="Instance regime filter (default: decision_sensitive)",
    )
    p.add_argument(
        "--grid-figure",
        action="store_true",
        help="Single multi-panel PNG instead of one file per (alpha, p_fail)",
    )
    p.add_argument(
        "--include-empty-cells",
        action="store_true",
        help="With separate PNGs, also write panels for (α,p_fail) with no DS instances",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG path (--grid-figure) or directory for separate PNGs (default: auto)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    df = load_instances(data_dir)
    if args.grid_figure:
        out = args.output or (
            data_dir / "plots" / f"recovery_fraction_DS_B{args.budget_ref}_alpha_pfail.png"
        )
        plot_recovery_fraction_ds_by_alpha_pfail(
            df,
            budget_ref=args.budget_ref,
            output_path=out,
            instance_label=args.instance_label,
        )
        print(f"Saved {out}")
    else:
        out_dir = args.output or (
            data_dir / "plots" / f"recovery_fraction_DS_B{args.budget_ref}_by_regime"
        )
        paths = plot_recovery_fraction_ds_separate_panels(
            df,
            budget_ref=args.budget_ref,
            output_dir=out_dir,
            instance_label=args.instance_label,
            include_empty_cells=args.include_empty_cells,
        )
        print(f"Saved {len(paths)} figure(s) under {out_dir}")
        for pth in paths:
            print(f"  {pth}")


if __name__ == "__main__":
    main()
