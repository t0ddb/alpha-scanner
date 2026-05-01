from __future__ import annotations

"""
audit_scheme_d_distribution.py — Re-rescore audit_dataset.parquet under
Scheme D's gradient curves; report new score distribution + matched-
selectivity threshold (i.e. what threshold gives ~5% of cells passing,
the same selectivity Scheme C had at 9.0).

Also reports score concentration at the top — verifying the goal of
spreading the 9.9-cluster.
"""

import argparse
import numpy as np
import pandas as pd
from indicators import (
    gradient_score,
    RS_GRADIENT_ANCHORS,
    ROC_GRADIENT_ANCHORS,
    ATR_GRADIENT_ANCHORS,
    ICHIMOKU_GRADIENT_ANCHORS,
    DUAL_TF_GRADIENT_ANCHORS,
    HIGHER_LOWS_GRADIENT_ANCHORS,
)


def scheme_d_score(row: pd.Series) -> float:
    s = 0.0
    s += gradient_score(row["rs_percentile"], RS_GRADIENT_ANCHORS)
    s += gradient_score(row["higher_lows_count"], HIGHER_LOWS_GRADIENT_ANCHORS)
    s += gradient_score(row["ichimoku_score"], ICHIMOKU_GRADIENT_ANCHORS)
    s += gradient_score(row["roc_value"], ROC_GRADIENT_ANCHORS)
    s += gradient_score(row["atr_percentile"], ATR_GRADIENT_ANCHORS)
    # Dual-TF: only score if EITHER cond_a or cond_b is met (qualification),
    # then gradient by max(63d, 21d) percentile.
    rs_63 = row.get("rs_63d_pctl", 0) or 0
    rs_21 = row.get("rs_21d_pctl", 0) or 0
    rs_126 = row.get("rs_126d_pctl", 0) or 0
    cond_a = (rs_126 >= 70) and (rs_63 > rs_126)
    cond_b = (rs_63 >= 80) and (rs_21 >= 80)
    if cond_a or cond_b:
        strength = max(rs_63, rs_21)
        s += gradient_score(strength, DUAL_TF_GRADIENT_ANCHORS)
    return s


def main(path: str):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    n = len(df)
    print(f"loaded {n:,} rows, {df['ticker'].nunique()} tickers")
    print(f"date range: {df['date'].min().date()} → {df['date'].max().date()}")

    # Vectorize the per-row score
    print("\nComputing Scheme D scores...")
    df["scheme_d_score"] = df.apply(scheme_d_score, axis=1)

    # Old (Scheme C) score for comparison — already in df['score']
    print("\n" + "=" * 70)
    print("  SCORE DISTRIBUTION COMPARISON")
    print("=" * 70)
    buckets = [
        (9.5, 10.01, "9.5 - 10.0"),
        (9.0, 9.5,  "9.0 - 9.4"),
        (8.5, 9.0,  "8.5 - 8.9"),
        (8.0, 8.5,  "8.0 - 8.4"),
        (7.0, 8.0,  "7.0 - 7.9"),
        (5.0, 7.0,  "5.0 - 6.9"),
        (0.0, 5.0,  "< 5.0"),
    ]
    print(f"  {'Score range':<14}  {'Scheme C (n / pct)':<24}  {'Scheme D (n / pct)':<24}")
    print(f"  {'─'*14}  {'─'*24}  {'─'*24}")
    for lo, hi, label in buckets:
        c = ((df["score"] >= lo) & (df["score"] < hi)).sum()
        d = ((df["scheme_d_score"] >= lo) & (df["scheme_d_score"] < hi)).sum()
        print(f"  {label:<14}  {c:>8,} ({100*c/n:>5.2f}%)        "
              f"{d:>8,} ({100*d/n:>5.2f}%)")

    # Find the Scheme D threshold that matches Scheme C's 9.0 selectivity
    sc_at_9 = (df["score"] >= 9.0).sum()
    target = sc_at_9 / n
    print(f"\n  Scheme C @ 9.0 selectivity: {sc_at_9:,} ({100*target:.2f}%)")

    # Find quantile cut for Scheme D matching that selectivity
    sd_thresh = df["scheme_d_score"].quantile(1 - target)
    n_at_sd = (df["scheme_d_score"] >= sd_thresh).sum()
    print(f"  Scheme D matched-selectivity threshold: {sd_thresh:.2f}  "
          f"(passes {n_at_sd:,}, {100*n_at_sd/n:.2f}%)")

    # Top-end concentration analysis
    print("\n" + "=" * 70)
    print("  TOP-END CONCENTRATION (the original 9.9-cluster problem)")
    print("=" * 70)
    print(f"\n  Cells within 0.1 of max score:")
    c_top = ((df["score"] >= 9.9)).sum()
    d_top = ((df["scheme_d_score"] >= 9.9)).sum()
    print(f"    Scheme C @ ≥9.9:  {c_top:>6,} ({100*c_top/n:.3f}%)")
    print(f"    Scheme D @ ≥9.9:  {d_top:>6,} ({100*d_top/n:.3f}%)")
    print(f"\n  Cells within 0.5 of max score:")
    c_top = ((df["score"] >= 9.5)).sum()
    d_top = ((df["scheme_d_score"] >= 9.5)).sum()
    print(f"    Scheme C @ ≥9.5:  {c_top:>6,} ({100*c_top/n:.3f}%)")
    print(f"    Scheme D @ ≥9.5:  {d_top:>6,} ({100*d_top/n:.3f}%)")

    # Recent-day comparison (today-style)
    print("\n" + "=" * 70)
    print("  RECENT-DAY COMPARISON (last 5 trading days in dataset)")
    print("=" * 70)
    recent_dates = sorted(df["date"].unique())[-5:]
    for d in recent_dates:
        sub = df[df["date"] == d]
        c99 = (sub["score"] >= 9.9).sum()
        d99 = (sub["scheme_d_score"] >= 9.9).sum()
        c95 = (sub["score"] >= 9.5).sum()
        d95 = (sub["scheme_d_score"] >= 9.5).sum()
        print(f"  {pd.to_datetime(d).date()}  "
              f"≥9.9: C={c99:>3} D={d99:>3}    ≥9.5: C={c95:>3} D={d95:>3}")

    # Save the rescored data for downstream analyses
    out = "backtest_results/audit_dataset_scheme_d_scored.parquet"
    df.to_parquet(out, index=False)
    print(f"\n  Wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    args = ap.parse_args()
    main(args.input)
