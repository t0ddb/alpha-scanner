from __future__ import annotations

"""
audit_rs_fine_grained.py — Fine-grained RS bucket analysis.

Tests the hypothesis that RS forward returns are non-monotonic — peak at
90-95 then dip at 95-99 before recovering at 99+. Reports mean, median,
and win rate at 2-percentile resolution.
"""

import argparse
import numpy as np
import pandas as pd


def main(path: str, target: str):
    df = pd.read_parquet(path)
    df = df.dropna(subset=[target, "rs_percentile"]).copy()
    print(f"loaded {len(df):,} rows with non-null {target}")

    print(f"\n  Fine-grained RS percentile → forward return ({target})")
    print(f"  {'pctl':<10} {'n':>6}  {'mean':>9}  {'median':>9}  {'win %':>7}")
    print(f"  {'─'*10} {'─'*6}  {'─'*9}  {'─'*9}  {'─'*7}")

    edges = list(range(0, 90, 10)) + list(range(90, 100, 1)) + [100]
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = df[(df["rs_percentile"] >= lo) & (df["rs_percentile"] < hi)]
        if len(sub) == 0:
            continue
        vals = sub[target].dropna()
        print(f"  [{lo:>3}, {hi:>3})  {len(sub):>6,}  "
              f"{vals.mean():>+8.2%}  {vals.median():>+8.2%}  "
              f"{(vals > 0).mean():>6.1%}")

    # Special: compare 90-95 vs 95-99 for held-position semantics
    # i.e., what's the distribution of 63d xspy returns?
    print(f"\n  Distribution comparison: 90-95 vs 95-99 (within-bucket detail)")
    for lo, hi, label in [(90, 95, "90-95"), (95, 99, "95-99"), (99, 100, "99+")]:
        sub = df[(df["rs_percentile"] >= lo) & (df["rs_percentile"] < hi)][target]
        sub = sub.dropna()
        print(f"  {label:>6}: n={len(sub):>5}  "
              f"p10={sub.quantile(0.10):>+7.2%}  "
              f"p25={sub.quantile(0.25):>+7.2%}  "
              f"p50={sub.median():>+7.2%}  "
              f"p75={sub.quantile(0.75):>+7.2%}  "
              f"p90={sub.quantile(0.90):>+7.2%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    args = ap.parse_args()
    main(args.input, args.target)
