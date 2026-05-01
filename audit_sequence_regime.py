from __future__ import annotations

"""
audit_sequence_regime.py — Regime + horizon stability of sequence findings.

Builds on audit_sequence_robustness.py. The Part 1 finding there was that
first-firer win rates flip across time periods (RS-first 32% pre-2024,
74% in 2025Q2+). This script asks two follow-ups:

  A. Per-regime baseline + DELTA: in each window, baseline win rate is
     not 50% — bull periods have everything winning. Compute first-firer
     win rate as a DELTA above the period's baseline. Does the relative
     ranking hold once we control for "everything wins in 2025Q2+"?

  B. Pair stability per regime: do the killer pairs (ROC→DTF, HL→Ich,
     CMF→HL) hold up in each window, or are they pooled artifacts?

  C. Horizon robustness: same analysis at fwd_21d_xspy. If the pattern
     is real, it should appear at multiple horizons.
"""

import argparse
import numpy as np
import pandas as pd

FIRE_COLS = [
    ("rs", "rs_fired"), ("ichimoku", "ichimoku_fired"),
    ("higher_lows", "higher_lows_fired"), ("cmf", "cmf_fired"),
    ("roc", "roc_fired"), ("atr", "atr_fired"),
    ("dual_tf_rs", "dual_tf_rs_fired"),
]


def streak_above(s: pd.Series) -> pd.Series:
    is_on = s.astype(int)
    groups = (is_on != is_on.shift()).cumsum()
    streak = is_on.groupby(groups).cumsum()
    return streak.where(is_on == 1, 0)


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


def first_firer(row, streak_cols):
    candidates = [(lbl, row[c]) for lbl, c in streak_cols.items() if row[c] > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def last_firer(row, streak_cols):
    candidates = [(lbl, row[c]) for lbl, c in streak_cols.items() if row[c] > 0]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])[0]


WINDOWS = [
    ("pre-2024", pd.Timestamp("2000-01-01"), pd.Timestamp("2024-01-01")),
    ("2024H1",   pd.Timestamp("2024-01-01"), pd.Timestamp("2024-09-01")),
    ("2024H2-2025Q1", pd.Timestamp("2024-09-01"), pd.Timestamp("2025-05-01")),
    ("2025Q2+",  pd.Timestamp("2025-05-01"), pd.Timestamp("2030-01-01")),
]


def per_regime_with_baseline(sig: pd.DataFrame, target: str) -> None:
    print(f"  {'window':<16} {'baseline':>9} {'first':<14} {'n':>5} {'win %':>7} {'Δ vs base':>10}")
    print(f"  {'─'*16} {'─'*9} {'─'*14} {'─'*5} {'─'*7} {'─'*10}")
    for wname, lo, hi in WINDOWS:
        sub = sig[(sig["date"] >= lo) & (sig["date"] < hi)]
        if len(sub) < 50:
            continue
        base = (sub[target].dropna() > 0).mean()
        first_line = True
        for label in [lbl for lbl, _ in FIRE_COLS]:
            cell = sub[sub["first_firer"] == label]
            vals = cell[target].dropna()
            if len(vals) < 30:
                continue
            wr = (vals > 0).mean()
            delta = wr - base
            wlabel = wname if first_line else ""
            blabel = f"{base:.1%}" if first_line else ""
            arrow = "↑↑" if delta > 0.10 else ("↑" if delta > 0.05 else
                    ("↓↓" if delta < -0.10 else ("↓" if delta < -0.05 else " ")))
            print(f"  {wlabel:<16} {blabel:>9} {label:<14} {len(vals):>5,} {wr:>6.1%} {delta:>+9.1%} {arrow}")
            first_line = False
        print()


def per_regime_pair(sig: pd.DataFrame, target: str, pairs: list[tuple[str, str]]) -> None:
    """For each named pair, compute its win rate per regime + delta vs regime baseline."""
    print(f"  {'pair':<28} {'window':<16} {'n':>5} {'win %':>7} {'base':>7} {'Δ':>7}")
    print(f"  {'─'*28} {'─'*16} {'─'*5} {'─'*7} {'─'*7} {'─'*7}")
    for first, last in pairs:
        pair_label = f"{first}→{last}"
        for wname, lo, hi in WINDOWS:
            sub = sig[(sig["date"] >= lo) & (sig["date"] < hi)]
            if len(sub) < 50:
                continue
            base = (sub[target].dropna() > 0).mean()
            cell = sub[(sub["first_firer"] == first) & (sub["last_firer"] == last)]
            vals = cell[target].dropna()
            if len(vals) < 10:
                print(f"  {pair_label:<28} {wname:<16} {len(vals):>5} (n<10)")
                continue
            wr = (vals > 0).mean()
            delta = wr - base
            arrow = "↑↑" if delta > 0.10 else ("↑" if delta > 0.05 else
                    ("↓↓" if delta < -0.10 else ("↓" if delta < -0.05 else " ")))
            print(f"  {pair_label:<28} {wname:<16} {len(vals):>5,} {wr:>6.1%} {base:>6.1%} {delta:>+6.1%} {arrow}")
        print()


def main(path: str):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows  (date range {df['date'].min().date()} → {df['date'].max().date()})")

    print("computing scores + streaks...")
    df["scheme_c_score"] = df.apply(scheme_c_score, axis=1)
    for label, col in FIRE_COLS:
        df[f"streak_{label}"] = df.groupby("ticker")[col].transform(streak_above)
    streak_cols = {label: f"streak_{label}" for label, _ in FIRE_COLS}

    sig = df[df["scheme_c_score"] >= 9.0].copy()
    print(f"signal rows (score >= 9.0): {len(sig):,}\n")

    print("computing first/last firers...")
    sig["first_firer"] = sig.apply(lambda r: first_firer(r, streak_cols), axis=1)
    sig["last_firer"]  = sig.apply(lambda r: last_firer(r, streak_cols), axis=1)

    KILLER_PAIRS = [
        ("roc", "dual_tf_rs"),
        ("higher_lows", "ichimoku"),
        ("cmf", "higher_lows"),
        ("ichimoku", "higher_lows"),
        ("ichimoku", "dual_tf_rs"),
        ("rs", "higher_lows"),     # RS-first, big sample, low win rate
        ("rs", "cmf"),              # RS-first, low win rate
        ("atr", "higher_lows"),     # AVOID — 31% win pooled
    ]

    for target, htag in [("fwd_63d_xspy", "63d xSPY"), ("fwd_21d_xspy", "21d xSPY")]:
        if target not in df.columns:
            continue
        local = sig.dropna(subset=[target]).copy()
        print("\n" + "=" * 78)
        print(f"HORIZON: {htag}   (n={len(local):,} signals with valid forward return)")
        print("=" * 78)

        print("\n— Per-regime first-firer win rate (Δ vs regime baseline) —\n")
        per_regime_with_baseline(local, target)

        print("\n— Per-regime PAIR stability (Δ vs regime baseline) —\n")
        per_regime_pair(local, target, KILLER_PAIRS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    args = ap.parse_args()
    main(args.input)
