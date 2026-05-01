from __future__ import annotations

"""
audit_rs_full_range.py — RS bucket analysis at uniform 2-pt resolution
across the FULL 0-100 percentile range.
"""

import argparse
import pandas as pd


def main(path: str, target: str):
    df = pd.read_parquet(path)
    df = df.dropna(subset=[target, "rs_percentile"]).copy()
    print(f"loaded {len(df):,} rows")
    print(f"\n  RS percentile → forward {target}, 2-pt buckets across full range")
    print(f"  {'pctl':<10} {'n':>6}  {'mean':>10}  {'median':>10}  {'win %':>7}")
    print(f"  {'─'*10} {'─'*6}  {'─'*10}  {'─'*10}  {'─'*7}")

    for lo in range(0, 100, 2):
        hi = lo + 2
        sub = df[(df["rs_percentile"] >= lo) & (df["rs_percentile"] < hi)]
        if len(sub) < 50:
            continue
        vals = sub[target].dropna()
        marker = ""
        if vals.median() > 0.03:
            marker = " ← strong median"
        elif vals.median() < -0.01:
            marker = " ← negative median"
        print(f"  [{lo:>3}, {hi:>3})  {len(sub):>6,}  "
              f"{vals.mean():>+9.2%}  {vals.median():>+9.2%}  "
              f"{(vals > 0).mean():>6.1%}{marker}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    args = ap.parse_args()
    main(args.input, args.target)
