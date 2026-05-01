from __future__ import annotations

"""
audit_sequence_position.py — Forward returns by indicator × position-rank.

For each signal row (score >= 9.0), each currently-firing indicator has a
rank in the firing-sequence (1 = first to fire, sorted by streak length).
Bucket forward returns by (indicator, rank) cell. This is a more
interpretable view than full sequences — shows "where in the sequence
does X tend to live for the BEST signals?"
"""

import argparse
import pandas as pd


FIRE_COLS = [
    ("rs",  "rs_fired"), ("ich", "ichimoku_fired"), ("hl", "higher_lows_fired"),
    ("cmf", "cmf_fired"), ("roc", "roc_fired"), ("atr", "atr_fired"),
    ("dtf", "dual_tf_rs_fired"),
]


def streak_above(s: pd.Series) -> pd.Series:
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

    # Per-indicator streak
    streak_cols = {}
    for label, col in FIRE_COLS:
        sname = f"streak_{label}"
        df[sname] = df.groupby("ticker")[col].transform(streak_above)
        streak_cols[label] = sname

    sig = df[df["scheme_c_score"] >= 9.0].copy()
    print(f"signal rows (score >= 9.0): {len(sig):,}")

    # ── For each row, compute rank of each indicator (1 = first to fire) ──
    print("computing per-row sequence positions...")
    indicators = [lbl for lbl, _ in FIRE_COLS]

    for ind in indicators:
        sig[f"rank_{ind}"] = None

    def assign_ranks(row):
        firing = [(lbl, row[scol]) for lbl, scol in streak_cols.items() if row[scol] > 0]
        firing.sort(key=lambda x: -x[1])  # longest streak first
        rank_map = {lbl: r + 1 for r, (lbl, _) in enumerate(firing)}
        return pd.Series({f"rank_{ind}": rank_map.get(ind) for ind in indicators})

    rank_df = sig.apply(assign_ranks, axis=1)
    for ind in indicators:
        sig[f"rank_{ind}"] = rank_df[f"rank_{ind}"]

    # ── For each indicator, bucket forward returns by rank ──
    print(f"\n  Forward {target} by (indicator, position-rank)")
    print(f"  Rank 1 = first to fire / longest streak; high rank = most recent")
    print()
    for ind in indicators:
        col = f"rank_{ind}"
        if sig[col].isna().all():
            continue
        ind_data = sig[sig[col].notna()].copy()
        ind_data[col] = ind_data[col].astype(int)
        n_rows = len(ind_data)
        print(f"  {ind} (fires in {100*n_rows/len(sig):.1f}% of signals)")
        print(f"  {'rank':<6} {'n':>5} {'mean':>9} {'median':>9} {'win %':>7}")
        for r in range(1, 8):
            mask = ind_data[col] == r
            if mask.sum() < 30:
                continue
            vals = ind_data.loc[mask, target].dropna()
            print(f"    #{r:<5} {len(vals):>5,} "
                  f"{vals.mean():>+8.2%} {vals.median():>+8.2%} "
                  f"{(vals > 0).mean():>6.1%}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    args = ap.parse_args()
    main(args.input, args.target)
