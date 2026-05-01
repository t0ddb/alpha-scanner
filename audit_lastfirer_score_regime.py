from __future__ import annotations

"""
audit_lastfirer_score_regime.py — Two robustness checks across regimes:

  1. LAST-FIRER FRESHNESS per regime: pooled data showed 8-20 day last-firer
     streaks win 55.7% vs 47-52% for shorter/longer. Does that hold per
     regime, or is it another pooling artifact?

  2. SCORE-STREAK SWEET SPOT per regime: pooled data showed day 1 (just
     crossed >=9.0) wins 53.7% and days 11-20 wins 55.0%, while days 3-5
     win only 49.6%. This was the basis for G1 (persistence=1). If THIS
     pattern is regime-stable, G1 is the only robust improvement available.
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

LAST_FIRER_BUCKETS = [
    (1, 2, "1 day"),
    (2, 4, "2-3 days"),
    (4, 8, "4-7 days"),
    (8, 21, "8-20 days"),
    (21, 9999, "21+ days"),
]

SCORE_STREAK_BUCKETS = [
    (1, 2, "day 1 (just crossed)"),
    (2, 3, "day 2"),
    (3, 4, "day 3 (= production p)"),
    (4, 6, "days 4-5"),
    (6, 11, "days 6-10"),
    (11, 21, "days 11-20"),
    (21, 41, "days 21-40"),
    (41, 99999, "days 41+"),
]


def main(path: str, target: str):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows  (date range {df['date'].min().date()} → {df['date'].max().date()})")

    print("computing scores + streaks...")
    df["scheme_c_score"] = df.apply(scheme_c_score, axis=1)
    for label, col in FIRE_COLS:
        df[f"streak_{label}"] = df.groupby("ticker")[col].transform(streak_above)
    streak_cols = {label: f"streak_{label}" for label, _ in FIRE_COLS}

    sig = df[df["scheme_c_score"] >= 9.0].copy()
    print(f"signal rows (score >= 9.0): {len(sig):,}")

    sig["last_firer"] = sig.apply(lambda r: last_firer(r, streak_cols), axis=1)
    sig["last_firer_streak"] = sig.apply(
        lambda r: r[streak_cols[r["last_firer"]]] if r["last_firer"] else None, axis=1)

    # Score-streak: per ticker, count consecutive days with score >= 9.0
    df["above"] = (df["scheme_c_score"] >= 9.0).astype(int)
    df["score_streak"] = df.groupby("ticker")["above"].transform(streak_above)
    sig = sig.merge(df[["ticker", "date", "score_streak"]], on=["ticker", "date"], how="left")

    # ─────────────────────────────────────────────────────────
    # PART 1 — Last-firer freshness per regime
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PART 1: LAST-FIRER FRESHNESS PER REGIME")
    print("=" * 78)
    print("Pooled finding: 8-20 day last-firer streaks win 55.7% vs ~47-52% else.")
    print("Test: does this hold per regime as Δ vs regime baseline?\n")

    print(f"  {'window':<16} {'baseline':>9} {'streak':<22} {'n':>5} {'win %':>7} {'Δ vs base':>10}")
    print(f"  {'─'*16} {'─'*9} {'─'*22} {'─'*5} {'─'*7} {'─'*10}")
    for wname, lo, hi in WINDOWS:
        sub = sig[(sig["date"] >= lo) & (sig["date"] < hi)]
        if len(sub) < 50:
            continue
        base = (sub[target].dropna() > 0).mean()
        first = True
        for slo, shi, slabel in LAST_FIRER_BUCKETS:
            cell = sub[(sub["last_firer_streak"] >= slo) & (sub["last_firer_streak"] < shi)]
            vals = cell[target].dropna()
            if len(vals) < 30:
                continue
            wr = (vals > 0).mean()
            delta = wr - base
            arrow = "↑↑" if delta > 0.07 else ("↑" if delta > 0.03 else
                    ("↓↓" if delta < -0.07 else ("↓" if delta < -0.03 else " ")))
            wlabel = wname if first else ""
            blabel = f"{base:.1%}" if first else ""
            print(f"  {wlabel:<16} {blabel:>9} {slabel:<22} {len(vals):>5,} {wr:>6.1%} {delta:>+9.1%} {arrow}")
            first = False
        print()

    # ─────────────────────────────────────────────────────────
    # PART 2 — Score-streak (the G1 motivating finding) per regime
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("PART 2: SCORE-STREAK SWEET SPOT PER REGIME (the G1 evidence)")
    print("=" * 78)
    print("Pooled finding: day 1 = 53.7% win, days 4-5 = 49.6%, days 11-20 = 55.0%.")
    print("If THIS holds per regime, G1 (persistence=1) is the robust improvement.\n")

    print(f"  {'window':<16} {'baseline':>9} {'score-streak':<22} {'n':>5} {'win %':>7} {'Δ vs base':>10}")
    print(f"  {'─'*16} {'─'*9} {'─'*22} {'─'*5} {'─'*7} {'─'*10}")
    for wname, lo, hi in WINDOWS:
        sub = sig[(sig["date"] >= lo) & (sig["date"] < hi)]
        if len(sub) < 50:
            continue
        base = (sub[target].dropna() > 0).mean()
        first = True
        for slo, shi, slabel in SCORE_STREAK_BUCKETS:
            cell = sub[(sub["score_streak"] >= slo) & (sub["score_streak"] < shi)]
            vals = cell[target].dropna()
            if len(vals) < 30:
                continue
            wr = (vals > 0).mean()
            delta = wr - base
            arrow = "↑↑" if delta > 0.07 else ("↑" if delta > 0.03 else
                    ("↓↓" if delta < -0.07 else ("↓" if delta < -0.03 else " ")))
            wlabel = wname if first else ""
            blabel = f"{base:.1%}" if first else ""
            print(f"  {wlabel:<16} {blabel:>9} {slabel:<22} {len(vals):>5,} {wr:>6.1%} {delta:>+9.1%} {arrow}")
            first = False
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    args = ap.parse_args()
    main(args.input, args.target)
