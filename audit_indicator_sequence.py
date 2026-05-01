from __future__ import annotations

"""
audit_indicator_sequence.py — Does the ORDER in which indicators fire
predict forward returns?

For each (date, ticker) row, compute each indicator's current streak length
(how many consecutive days the indicator has been firing, ending at today).
Then for rows where score >= 9.0:

  - Identify the FIRST-firer (longest streak): which indicator started
    the breakout setup?
  - Identify the LAST-firer (shortest streak, but >0): which indicator
    most recently confirmed?

Bucket forward returns by these labels to detect patterns.
"""

import argparse
from collections import Counter

import numpy as np
import pandas as pd


FIRE_COLS = [
    ("rs",         "rs_fired"),
    ("ichimoku",   "ichimoku_fired"),
    ("higher_lows","higher_lows_fired"),
    ("cmf",        "cmf_fired"),       # weight 0 in Scheme C but include for sequence
    ("roc",        "roc_fired"),
    ("atr",        "atr_fired"),
    ("dual_tf_rs", "dual_tf_rs_fired"),
]


def streak_above(s: pd.Series) -> pd.Series:
    """Per-day streak length: consecutive days where the binary value == 1."""
    is_on = s.astype(int)
    groups = (is_on != is_on.shift()).cumsum()
    streak = is_on.groupby(groups).cumsum()
    return streak.where(is_on == 1, 0)


def main(path: str, target: str):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows")

    # Reconstruct Scheme C score (production-equivalent)
    print("computing Scheme C scores...")
    def scheme_c_score(row) -> float:
        s = 0.0
        rs = row.get("rs_percentile", 0) or 0
        for thresh, pts in [(90, 3.0), (80, 2.4), (70, 1.8), (60, 1.2), (50, 0.6)]:
            if rs >= thresh:
                s += pts; break
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

    # Per-indicator streak length
    print("computing per-indicator streak lengths...")
    for label, col in FIRE_COLS:
        df[f"streak_{label}"] = df.groupby("ticker")[col].transform(streak_above)

    # Filter to production-equivalent signals
    sig = df[df["scheme_c_score"] >= 9.0].copy()
    print(f"\nsignal rows (score >= 9.0): {len(sig):,}")

    # Identify first-firer (longest current streak) and last-firer (shortest, but > 0)
    streak_cols = {label: f"streak_{label}" for label, _ in FIRE_COLS}

    def first_firer(row) -> str | None:
        # Pick label with max streak (must be > 0)
        candidates = [(lbl, row[c]) for lbl, c in streak_cols.items() if row[c] > 0]
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[1])[0]

    def last_firer(row) -> str | None:
        candidates = [(lbl, row[c]) for lbl, c in streak_cols.items() if row[c] > 0]
        if not candidates:
            return None
        return min(candidates, key=lambda x: x[1])[0]

    sig["first_firer"] = sig.apply(first_firer, axis=1)
    sig["last_firer"] = sig.apply(last_firer, axis=1)

    # ── First-firer analysis ──
    print(f"\n  FIRST-FIRER (which indicator started this setup?)")
    print(f"  All 7 indicators included. dual_tf_rs structurally cannot")
    print(f"  be 'first' because its trigger requires higher percentile")
    print(f"  thresholds than rs_fired's (50th); rs_fired always lights")
    print(f"  up before dual_tf_rs in any uptrend. Hence n=0.")
    print(f"  {'indicator':<14} {'n':>6} {'mean':>9} {'median':>9} {'win %':>7}  {'avg streak':>11}")
    print(f"  {'─'*14} {'─'*6} {'─'*9} {'─'*9} {'─'*7}  {'─'*11}")
    for label in [lbl for lbl, _ in FIRE_COLS]:
        sub = sig[sig["first_firer"] == label]
        n = len(sub)
        if n == 0:
            print(f"  {label:<14} {0:>6} (never first — see note above)")
            continue
        vals = sub[target].dropna()
        avg_streak = sub[streak_cols[label]].mean()
        print(f"  {label:<14} {n:>6,} "
              f"{vals.mean():>+8.2%} {vals.median():>+8.2%} "
              f"{(vals > 0).mean():>6.1%}  {avg_streak:>10.1f}")

    # ── Last-firer analysis ──
    print(f"\n  LAST-FIRER (which indicator most recently confirmed?)")
    print(f"  {'indicator':<14} {'n':>6} {'mean':>9} {'median':>9} {'win %':>7}")
    print(f"  {'─'*14} {'─'*6} {'─'*9} {'─'*9} {'─'*7}")
    for label in [lbl for lbl, _ in FIRE_COLS]:
        sub = sig[sig["last_firer"] == label]
        if len(sub) < 30:
            continue
        vals = sub[target].dropna()
        print(f"  {label:<14} {len(sub):>6,} "
              f"{vals.mean():>+8.2%} {vals.median():>+8.2%} "
              f"{(vals > 0).mean():>6.1%}")

    # ── Common (first → last) sequence pairs ──
    print(f"\n  TOP (first-firer, last-firer) PAIRS (n >= 50)")
    print(f"  {'first':<14} {'last':<14} {'n':>6} {'mean':>9} {'median':>9} {'win %':>7}")
    print(f"  {'─'*14} {'─'*14} {'─'*6} {'─'*9} {'─'*9} {'─'*7}")
    pair_counts = sig.groupby(["first_firer", "last_firer"]).size().reset_index(name="n")
    pair_counts = pair_counts[pair_counts["n"] >= 50].sort_values("n", ascending=False)
    for _, r in pair_counts.iterrows():
        sub = sig[(sig["first_firer"] == r["first_firer"]) &
                  (sig["last_firer"] == r["last_firer"])]
        vals = sub[target].dropna()
        if len(vals) > 0:
            print(f"  {r['first_firer']:<14} {r['last_firer']:<14} {len(sub):>6,} "
                  f"{vals.mean():>+8.2%} {vals.median():>+8.2%} "
                  f"{(vals > 0).mean():>6.1%}")

    # ── How fresh is the LAST-firer? (days streak of last-firer)
    print(f"\n  LAST-FIRER FRESHNESS — streak of the most-recently-confirmed indicator")
    print(f"  Hypothesis: shorter (1-3 days) = fresh confirmation; longer = stale")
    print(f"  {'streak':<10} {'n':>6} {'mean':>9} {'median':>9} {'win %':>7}")
    print(f"  {'─'*10} {'─'*6} {'─'*9} {'─'*9} {'─'*7}")
    sig["last_firer_streak"] = sig.apply(
        lambda r: r[streak_cols[r["last_firer"]]] if r["last_firer"] else None, axis=1)
    for lo, hi, label in [(1, 2, "1 day"), (2, 4, "2-3 days"), (4, 8, "4-7 days"),
                          (8, 21, "8-20 days"), (21, 9999, "21+ days")]:
        mask = (sig["last_firer_streak"] >= lo) & (sig["last_firer_streak"] < hi)
        sub = sig.loc[mask, target].dropna()
        if len(sub) > 0:
            print(f"  {label:<10} {len(sub):>6,} "
                  f"{sub.mean():>+8.2%} {sub.median():>+8.2%} "
                  f"{(sub > 0).mean():>6.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    args = ap.parse_args()
    main(args.input, args.target)
