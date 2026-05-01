from __future__ import annotations

"""
audit_score_streaks.py — Forward-return analysis by SCORE-based streak length.

Production semantics: entry threshold is score >= 9.0 with 3-day persistence.
This analysis tests the streak hypothesis at thresholds matching how the
system actually evaluates entries — score >= 7.0, 8.0, 9.0, 9.5.

For each (date, ticker), compute:
  - "Days streak with score >= T" — how long has the score held above T?
Then bucket forward returns by streak length, separately for each T.
"""

import argparse
import numpy as np
import pandas as pd


def streak_above_threshold(s: pd.Series, k: float) -> pd.Series:
    """Per-day count of consecutive days where s >= k, ending at this day."""
    is_above = (s >= k).astype(int)
    groups = (is_above != is_above.shift()).cumsum()
    streak = is_above.groupby(groups).cumsum()
    return streak.where(is_above == 1, 0)


def main(path: str, target: str):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows")

    # The 'score' column in the parquet was computed by score_all() at
    # rebuild time. indicators.py is currently Scheme D — which we don't
    # want here. Reconstruct Scheme C scores from binary firings + RS.
    print("computing Scheme C scores from raw inputs...")

    def scheme_c_score(row) -> float:
        s = 0.0
        # RS (bucketed Scheme C: 50→0.6, 60→1.2, 70→1.8, 80→2.4, 90→3.0)
        rs = row.get("rs_percentile", 0) or 0
        for thresh, pts in [(90, 3.0), (80, 2.4), (70, 1.8), (60, 1.2), (50, 0.6)]:
            if rs >= thresh:
                s += pts; break
        # HL bucketed
        hl = row.get("higher_lows_count", 0) or 0
        for c, pts in [(5, 0.5), (4, 0.375), (3, 0.25), (2, 0.125)]:
            if hl >= c:
                s += pts; break
        if row.get("ichimoku_fired", 0): s += 2.0
        if row.get("roc_fired", 0):      s += 1.5
        if row.get("dual_tf_rs_fired", 0): s += 2.5
        if row.get("atr_fired", 0):      s += 0.5
        return s

    df["scheme_c_score"] = df.apply(scheme_c_score, axis=1)

    # ── Compute streak features per threshold ──
    for T in [7.0, 8.0, 9.0]:
        df[f"streak_ge_{T}"] = df.groupby("ticker")["scheme_c_score"].transform(
            lambda s: streak_above_threshold(s, T))

    # ── Bucket forward returns by streak length, for each threshold ──
    for T in [7.0, 8.0, 9.0]:
        col = f"streak_ge_{T}"
        # Only consider rows where the streak is currently active (≥1)
        active = df[df[col] >= 1].copy()
        print(f"\n  Streak length where score >= {T} (production threshold = 9.0)")
        print(f"  Active streak rows: {len(active):,}")
        print(f"  {'streak':<14} {'n':>7} {'mean':>10} {'median':>9} {'win %':>7}")
        print(f"  {'─'*14} {'─'*7} {'─'*10} {'─'*9} {'─'*7}")
        for lo, hi, label in [
            (1, 2, "1 day (new)"),
            (2, 3, "2 days"),
            (3, 4, "3 days (= production persistence)"),
            (4, 6, "4-5 days"),
            (6, 11, "6-10 days"),
            (11, 21, "11-20 days"),
            (21, 41, "21-40 days"),
            (41, 9999, "41+ days"),
        ]:
            mask = (active[col] >= lo) & (active[col] < hi)
            sub = active.loc[mask, target].dropna()
            if len(sub) > 0:
                print(f"  {label:<14} {len(sub):>7,} "
                      f"{sub.mean():>+9.2%} {sub.median():>+8.2%} "
                      f"{(sub > 0).mean():>6.1%}")

    # ── Critical comparison: at production threshold 9.0, does
    # entering after a LONGER pre-9.0 streak help? Two angles:
    # (a) the streak length AT score >= 9.0 (what persistence_days controls)
    # (b) the streak length AT score >= 7.0 (longer-horizon "trend" signal)
    print(f"\n  ===========================================================")
    print(f"  CONDITIONAL: among rows with score >= 9.0 today,")
    print(f"  bucket by ALSO having longer streaks at LOWER thresholds.")
    print(f"  ===========================================================")
    hi_score = df[df["scheme_c_score"] >= 9.0].copy()
    print(f"  Total score≥9.0 rows: {len(hi_score):,}")

    print(f"\n  Among score≥9.0, bucket by streak length at score >= 7.0")
    print(f"  (i.e. how long has this stock been at least 'warm'?)")
    print(f"  {'streak ≥7':<14} {'n':>7} {'mean':>10} {'median':>9} {'win %':>7}")
    print(f"  {'─'*14} {'─'*7} {'─'*10} {'─'*9} {'─'*7}")
    for lo, hi, label in [
        (1, 6, "1-5 days"), (6, 11, "6-10 days"), (11, 21, "11-20 days"),
        (21, 41, "21-40 days"), (41, 9999, "41+ days"),
    ]:
        mask = (hi_score["streak_ge_7.0"] >= lo) & (hi_score["streak_ge_7.0"] < hi)
        sub = hi_score.loc[mask, target].dropna()
        if len(sub) > 0:
            print(f"  {label:<14} {len(sub):>7,} "
                  f"{sub.mean():>+9.2%} {sub.median():>+8.2%} "
                  f"{(sub > 0).mean():>6.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    args = ap.parse_args()
    main(args.input, args.target)
