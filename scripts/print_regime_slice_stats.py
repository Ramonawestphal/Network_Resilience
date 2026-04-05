"""Print summary statistics for one (budget_ref, alpha, pfail) regime slice."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "path" / "to" / "your" / "output"
    budget_ref = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    alpha = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15
    pfail = float(sys.argv[4]) if len(sys.argv) > 4 else 0.18

    parquet_path = data_dir / "regime_instances.parquet"
    csv_path = data_dir / "regime_instances.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        source = parquet_path
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        source = csv_path
        if len(df) <= 10_000:
            print(
                "Warning: regime_instances.csv may be truncated (≤10k rows). "
                "Use regime_instances.parquet for full statistics.\n",
                file=sys.stderr,
            )
    else:
        print(f"No regime_instances under {data_dir}", file=sys.stderr)
        sys.exit(1)

    if df["solved"].dtype == object:
        df["solved"] = df["solved"].map(
            {"True": True, "False": False, True: True, False: False}
        )

    mask = (
        (df["budget_ref"] == budget_ref)
        & np.isclose(df["alpha"].astype(float), alpha)
        & np.isclose(df["pfail"].astype(float), pfail)
    )
    sub = df.loc[mask]
    policies = ["degree", "random", "betweenness", "greedy", "risk"]

    print("Source:", source)
    print(f"Filter: budget_ref={budget_ref}, alpha={alpha}, pfail={pfail}")
    print("Rows (all policies):", len(sub))
    if sub.empty:
        print("(No rows match — check data path or use full parquet.)")
        sys.exit(0)

    inst = sub.groupby(["graph_id", "seed_index"], observed=True).first().reset_index()
    print("Unique graph_id × seed_index instances:", len(inst))
    print()
    print("instance_label counts (one label per instance):")
    vc = inst["instance_label"].value_counts()
    for k, v in vc.items():
        print(f"  {k}: {v}")
    print()

    print("Per-policy: n rows, frac solved, mean/median rounds_when_solved (solved only)")
    for pol in policies:
        p = sub.loc[sub["policy"] == pol]
        n = len(p)
        frac = float(p["solved"].mean()) if n else float("nan")
        solved = p.loc[p["solved"], "rounds_when_solved"]
        rmean = float(solved.mean()) if len(solved) else float("nan")
        rmed = float(solved.median()) if len(solved) else float("nan")
        print(
            f"  {pol:12s} n={n:5d}  frac_solved={frac:.4f}  "
            f"rounds_mean={rmean:.4g}  rounds_median={rmed:.4g}"
        )

    print()
    ds = sub.loc[sub["instance_label"] == "decision_sensitive"]
    print("Decision-sensitive rows:", len(ds))
    if len(ds):
        print("DS per-policy frac solved:")
        for pol in policies:
            p = ds.loc[ds["policy"] == pol]
            print(f"  {pol:12s} {float(p['solved'].mean()):.4f} (n={len(p)})")

    print()
    print("Numeric summaries (all policies in slice):")
    for c in ("final_pr", "n_active_final", "rounds_when_solved", "spread_vs_random"):
        if c not in sub.columns:
            continue
        s = pd.to_numeric(sub[c], errors="coerce")
        print(
            f"  {c}: mean={s.mean():.4g} std={s.std():.4g} "
            f"min={s.min():.4g} max={s.max():.4g}"
        )


if __name__ == "__main__":
    main()
