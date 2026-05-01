from __future__ import annotations

"""
audit_firing_sequence.py — Test whether the TIMING and SEQUENCE of
indicator firings predicts forward returns.

Three tests:
  1. Freshness: how many days has each indicator been firing? Are FRESH
     fires (just turned on) different from STALE fires (firing for weeks)?
  2. Acceleration: # firing today vs # firing N days ago. Stocks that are
     accelerating into the breakout vs decelerating out of it.
  3. First-firer: which indicator turned on FIRST in this current breakout
     setup? Different starting indicators → different forward returns?
"""

import argparse
import numpy as np
import pandas as pd


FIRE_COLS = [
    "rs_fired", "ichimoku_fired", "higher_lows_fired",
    "roc_fired", "atr_fired", "dual_tf_rs_fired",
]


def main(path: str, target: str):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows")

    # ── Compute firing-count features per (date, ticker) ──
    # Fire count today
    df["fire_count"] = df[FIRE_COLS].sum(axis=1)

    # 5-day-ago fire count, per ticker
    df["fire_count_5d_ago"] = df.groupby("ticker")["fire_count"].shift(5)
    df["fire_count_10d_ago"] = df.groupby("ticker")["fire_count"].shift(10)

    # Acceleration = today minus 5-days-ago
    df["fire_accel_5d"] = df["fire_count"] - df["fire_count_5d_ago"]
    df["fire_accel_10d"] = df["fire_count"] - df["fire_count_10d_ago"]

    # Days since first FULL setup (≥ 4 indicators firing)
    # Track per-ticker streak length where fire_count >= 4
    def streak_above_threshold(s: pd.Series, k: int) -> pd.Series:
        is_above = (s >= k).astype(int)
        # Reset when below threshold
        groups = (is_above != is_above.shift()).cumsum()
        streak = is_above.groupby(groups).cumsum()
        return streak.where(is_above == 1, 0)

    df["days_4plus_firing"] = df.groupby("ticker")["fire_count"].transform(
        lambda s: streak_above_threshold(s, 4))

    # ── Bucket by acceleration ──
    print(f"\n  TEST 2: Firing acceleration (fire_count today − 5 days ago)")
    print(f"  Hypothesis: accelerating breakouts > decelerating ones")
    print(f"  {'accel':<12} {'n':>7} {'mean_fwd':>10} {'median':>9} {'win %':>7}")
    print(f"  {'─'*12} {'─'*7} {'─'*10} {'─'*9} {'─'*7}")
    for accel_lo, accel_hi, label in [
        (-6, -3, "≤−3 (sharp dec)"), (-3, -1, "−3 to −1"),
        (-1, 0, "−1 to 0"), (0, 1, "0 to 1 (flat)"),
        (1, 3, "+1 to +3"), (3, 7, "+3+ (sharp acc)"),
    ]:
        mask = (df["fire_accel_5d"] >= accel_lo) & (df["fire_accel_5d"] < accel_hi)
        sub = df.loc[mask, target].dropna()
        if len(sub) > 0:
            print(f"  {label:<14} {len(sub):>7,} "
                  f"{sub.mean():>+9.2%} {sub.median():>+8.2%} {(sub > 0).mean():>6.1%}")

    # ── Bucket by freshness (days streak ≥4 indicators firing) ──
    print(f"\n  TEST 1: Freshness (days streak with ≥4 indicators firing)")
    print(f"  Hypothesis: FRESH setups (1-5 days) > stale (30+ days)")
    print(f"  {'streak':<12} {'n':>7} {'mean_fwd':>10} {'median':>9} {'win %':>7}")
    print(f"  {'─'*12} {'─'*7} {'─'*10} {'─'*9} {'─'*7}")
    for lo, hi, label in [
        (1, 3, "1-2 days"), (3, 6, "3-5 days"), (6, 11, "6-10 days"),
        (11, 21, "11-20 days"), (21, 41, "21-40 days"), (41, 9999, "40+ days"),
    ]:
        mask = (df["days_4plus_firing"] >= lo) & (df["days_4plus_firing"] < hi)
        sub = df.loc[mask, target].dropna()
        if len(sub) > 0:
            print(f"  {label:<12} {len(sub):>7,} "
                  f"{sub.mean():>+9.2%} {sub.median():>+8.2%} {(sub > 0).mean():>6.1%}")

    # ── Within high-conviction (Scheme-C-equivalent ≥4 firing): does freshness matter? ──
    print(f"\n  TEST 1b: Freshness, conditioned on currently ≥5 indicators firing")
    print(f"  This isolates the 'extension' effect from 'low-quality setup'")
    print(f"  {'streak':<12} {'n':>7} {'mean_fwd':>10} {'median':>9} {'win %':>7}")
    print(f"  {'─'*12} {'─'*7} {'─'*10} {'─'*9} {'─'*7}")
    hi_conv = df[df["fire_count"] >= 5].copy()
    hi_conv["days_5plus"] = hi_conv.groupby("ticker")["fire_count"].transform(
        lambda s: streak_above_threshold(s, 5))
    for lo, hi, label in [
        (1, 3, "1-2 days"), (3, 6, "3-5 days"), (6, 11, "6-10 days"),
        (11, 21, "11-20 days"), (21, 9999, "20+ days"),
    ]:
        mask = (hi_conv["days_5plus"] >= lo) & (hi_conv["days_5plus"] < hi)
        sub = hi_conv.loc[mask, target].dropna()
        if len(sub) > 0:
            print(f"  {label:<12} {len(sub):>7,} "
                  f"{sub.mean():>+9.2%} {sub.median():>+8.2%} {(sub > 0).mean():>6.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    args = ap.parse_args()
    main(args.input, args.target)
