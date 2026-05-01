from __future__ import annotations

"""
audit_topN_comparison.py — Compare top-N selection quality between
Scheme C and Scheme D, bypassing threshold-passing dynamics entirely.

For each date, take the top-N stocks by Scheme C score and the top-N
by Scheme D score. Measure mean / median forward return of each
selection. If Scheme D's top-N outperforms Scheme C's at the same N,
the gradient is genuinely better at ranking, regardless of how the
threshold mechanic interacts with the position cap.

Why this matters: Scheme C has many tickers tied at high scores
(e.g. 9 tickers at 9.9 today). When the production system fills 12
slots, it picks among ties effectively at random (alphabetical fallback).
Scheme D's finer ranking may pick BETTER stocks within the same tier,
even if its threshold-pass rate looks worse in raw backtest.
"""

import argparse
import numpy as np
import pandas as pd

from indicators import (
    gradient_score,
    RS_GRADIENT_ANCHORS, ROC_GRADIENT_ANCHORS, ATR_GRADIENT_ANCHORS,
    ICHIMOKU_GRADIENT_ANCHORS, DUAL_TF_GRADIENT_ANCHORS,
    HIGHER_LOWS_GRADIENT_ANCHORS,
)


def scheme_c_score_row(row) -> float:
    """Reconstruct Scheme C score from parquet's continuous + binary columns.
    The parquet was rebuilt with Scheme D in indicators.py, so its 'score'
    column is Scheme D — we have to recompute Scheme C from raw inputs."""
    s = 0.0
    # RS — Scheme C bucketed gradient
    rs = row["rs_percentile"] if pd.notna(row["rs_percentile"]) else 0
    for min_p, pts in [(90, 3.0), (80, 2.4), (70, 1.8), (60, 1.2), (50, 0.6)]:
        if rs >= min_p: s += pts; break
    # Higher Lows — Scheme C bucketed
    hl = row["higher_lows_count"] if pd.notna(row["higher_lows_count"]) else 0
    for min_c, pts in [(5, 0.5), (4, 0.375), (3, 0.25), (2, 0.125)]:
        if hl >= min_c: s += pts; break
    # Binary indicators (use *_fired columns from parquet)
    if row.get("ichimoku_fired", 0): s += 2.0
    if row.get("roc_fired", 0):      s += 1.5
    if row.get("dual_tf_rs_fired", 0): s += 2.5
    if row.get("atr_fired", 0):      s += 0.5
    # CMF dropped (weight 0) per Scheme C
    return round(s, 2)


def scheme_d_score_row(row) -> float:
    s = 0.0
    s += gradient_score(row["rs_percentile"], RS_GRADIENT_ANCHORS)
    s += gradient_score(row["higher_lows_count"], HIGHER_LOWS_GRADIENT_ANCHORS)
    s += gradient_score(row["ichimoku_score"], ICHIMOKU_GRADIENT_ANCHORS)
    s += gradient_score(row["roc_value"], ROC_GRADIENT_ANCHORS)
    s += gradient_score(row["atr_percentile"], ATR_GRADIENT_ANCHORS)
    rs_63 = row.get("rs_63d_pctl", 0) or 0
    rs_21 = row.get("rs_21d_pctl", 0) or 0
    rs_126 = row.get("rs_126d_pctl", 0) or 0
    cond_a = (rs_126 >= 70) and (rs_63 > rs_126)
    cond_b = (rs_63 >= 80) and (rs_21 >= 80)
    if cond_a or cond_b:
        s += gradient_score(max(rs_63, rs_21), DUAL_TF_GRADIENT_ANCHORS)
    return s


def main(path: str, target: str):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])

    # The parquet was rebuilt with Scheme D in indicators.py, so its 'score'
    # column reflects Scheme D. Reconstruct both schemes explicitly.
    print("computing Scheme C scores from parquet inputs...")
    df["scheme_c_score"] = df.apply(scheme_c_score_row, axis=1)
    print("computing Scheme D scores from parquet inputs...")
    df["scheme_d_score"] = df.apply(scheme_d_score_row, axis=1)
    df = df.dropna(subset=[target]).copy()
    print(f"loaded {len(df):,} rows with non-null {target}")
    print(f"  Scheme C score range: {df['scheme_c_score'].min():.2f} → {df['scheme_c_score'].max():.2f}")
    print(f"  Scheme D score range: {df['scheme_d_score'].min():.2f} → {df['scheme_d_score'].max():.2f}")

    # ── Top-N comparison ───────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"  TOP-N PER-DAY SELECTION (forward 63d xSPY return)")
    print(f"  Scheme C score vs Scheme D score, same N, no threshold")
    print(f"{'=' * 90}")
    print(f"  {'N':<5} {'C n_picks':>11} {'C mean':>10} {'C median':>10} {'C win%':>8}  "
          f"{'D n_picks':>11} {'D mean':>10} {'D median':>10} {'D win%':>8}  "
          f"{'D - C mean':>11}")
    print(f"  {'─'*5} {'─'*11} {'─'*10} {'─'*10} {'─'*8}  "
          f"{'─'*11} {'─'*10} {'─'*10} {'─'*8}  {'─'*11}")

    for N in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30]:
        # For each date, take top-N by each score
        c_picks = df.groupby("date", group_keys=False).apply(
            lambda g: g.nlargest(N, "scheme_c_score"))
        d_picks = df.groupby("date", group_keys=False).apply(
            lambda g: g.nlargest(N, "scheme_d_score"))

        c_vals = c_picks[target].dropna()
        d_vals = d_picks[target].dropna()

        c_mean = c_vals.mean()
        d_mean = d_vals.mean()
        delta = d_mean - c_mean

        print(f"  {N:<5} "
              f"{len(c_picks):>11,} {c_mean:>+9.2%} {c_vals.median():>+9.2%} "
              f"{(c_vals > 0).mean():>7.1%}   "
              f"{len(d_picks):>11,} {d_mean:>+9.2%} {d_vals.median():>+9.2%} "
              f"{(d_vals > 0).mean():>7.1%}   "
              f"{delta:>+10.2%}")

    # ── Tie-breaking analysis: how often do schemes disagree on top-12? ──
    print(f"\n{'=' * 90}")
    print(f"  AGREEMENT BETWEEN SCHEMES")
    print(f"{'=' * 90}")
    overlap_counts = []
    for date, g in df.groupby("date"):
        if len(g) < 12:
            continue
        c_top12 = set(g.nlargest(12, "scheme_c_score")["ticker"])
        d_top12 = set(g.nlargest(12, "scheme_d_score")["ticker"])
        overlap_counts.append(len(c_top12 & d_top12))
    overlaps = pd.Series(overlap_counts)
    print(f"  Days with top-12 comparison: {len(overlaps):,}")
    print(f"  Mean overlap (out of 12):    {overlaps.mean():.2f}  "
          f"(perfect agreement = 12.0)")
    print(f"  Median overlap:              {overlaps.median():.0f}")
    print(f"  Days with full agreement:    {(overlaps == 12).sum():,} "
          f"({100*(overlaps == 12).mean():.1f}%)")
    print(f"  Days with ≤6 overlap:        {(overlaps <= 6).sum():,} "
          f"({100*(overlaps <= 6).mean():.1f}%)")

    # ── Tied-score concentration in Scheme C ──
    print(f"\n{'=' * 90}")
    print(f"  TIES AT THE TOP: how many tickers at score >= max(score) - 0.05 per day")
    print(f"{'=' * 90}")
    c_tie_counts = []
    d_tie_counts = []
    for date, g in df.groupby("date"):
        c_max = g["scheme_c_score"].max()
        c_tied = (g["scheme_c_score"] >= c_max - 0.05).sum()
        d_max = g["scheme_d_score"].max()
        d_tied = (g["scheme_d_score"] >= d_max - 0.05).sum()
        c_tie_counts.append(c_tied)
        d_tie_counts.append(d_tied)
    c_ties = pd.Series(c_tie_counts)
    d_ties = pd.Series(d_tie_counts)
    print(f"  {'metric':<30} {'Scheme C':>10} {'Scheme D':>10}")
    print(f"  {'mean ties at top':<30} {c_ties.mean():>10.2f} {d_ties.mean():>10.2f}")
    print(f"  {'median ties':<30} {c_ties.median():>10.0f} {d_ties.median():>10.0f}")
    print(f"  {'days with >5 ties at top':<30} "
          f"{(c_ties > 5).sum():>10,} {(d_ties > 5).sum():>10,}")
    print(f"  {'days with >12 ties at top':<30} "
          f"{(c_ties > 12).sum():>10,} {(d_ties > 12).sum():>10,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    args = ap.parse_args()
    main(args.input, args.target)
