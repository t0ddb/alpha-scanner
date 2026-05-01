from __future__ import annotations

"""
audit_full_sequence.py — Full firing-sequence analysis.

For each (date, ticker) with score >= 9.0, compute the full ordering of
currently-firing indicators by streak length descending. The first
element is the indicator that started firing first; the last is the
most-recently-confirmed.

Reports forward returns by full sequence pattern.
"""

import argparse
import pandas as pd


FIRE_COLS = [
    ("rs",         "rs_fired"),
    ("ich",        "ichimoku_fired"),
    ("hl",         "higher_lows_fired"),
    ("cmf",        "cmf_fired"),
    ("roc",        "roc_fired"),
    ("atr",        "atr_fired"),
    ("dtf",        "dual_tf_rs_fired"),
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

    # Reconstruct Scheme C score
    print("computing Scheme C scores + per-indicator streaks...")
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

    # Filter to production signals
    sig = df[df["scheme_c_score"] >= 9.0].copy()
    print(f"signal rows (score >= 9.0): {len(sig):,}")

    # ── Per-row sequence: order indicators by streak length descending ──
    def build_sequence(row) -> str:
        firing = [(lbl, row[scol]) for lbl, scol in streak_cols.items() if row[scol] > 0]
        firing.sort(key=lambda x: -x[1])
        return "→".join(lbl for lbl, _ in firing)

    sig["sequence"] = sig.apply(build_sequence, axis=1)
    sig["seq_length"] = sig["sequence"].str.count("→") + 1

    # ── Distribution by sequence length ──
    print(f"\n  Sequence length distribution (# indicators currently firing)")
    print(f"  {'length':<8} {'n':>6} {'mean':>9} {'median':>9} {'win %':>7}")
    print(f"  {'─'*8} {'─'*6} {'─'*9} {'─'*9} {'─'*7}")
    for L in range(1, 8):
        sub = sig[sig["seq_length"] == L]
        if len(sub) == 0:
            continue
        vals = sub[target].dropna()
        print(f"  {L:<8} {len(sub):>6,} "
              f"{vals.mean():>+8.2%} {vals.median():>+8.2%} "
              f"{(vals > 0).mean():>6.1%}")

    # ── Top sequences by frequency (n >= 30) ──
    print(f"\n  Top 30 full sequences (sorted by n, n >= 30)")
    print(f"  {'sequence':<55} {'n':>5} {'mean':>9} {'median':>9} {'win %':>7}")
    print(f"  {'─'*55} {'─'*5} {'─'*9} {'─'*9} {'─'*7}")
    seq_groups = sig.groupby("sequence")
    seq_counts = seq_groups.size().reset_index(name="n")
    seq_counts = seq_counts[seq_counts["n"] >= 30].sort_values("n", ascending=False)
    for _, r in seq_counts.head(30).iterrows():
        sub = sig[sig["sequence"] == r["sequence"]]
        vals = sub[target].dropna()
        print(f"  {r['sequence'][:55]:<55} {len(sub):>5,} "
              f"{vals.mean():>+8.2%} {vals.median():>+8.2%} "
              f"{(vals > 0).mean():>6.1%}")

    # ── Best sequences by win rate (n >= 30) ──
    print(f"\n  Top 15 sequences by WIN RATE (n >= 30)")
    print(f"  {'sequence':<55} {'n':>5} {'mean':>9} {'median':>9} {'win %':>7}")
    print(f"  {'─'*55} {'─'*5} {'─'*9} {'─'*9} {'─'*7}")
    seq_metrics = []
    for seq, sub in sig.groupby("sequence"):
        if len(sub) < 30:
            continue
        vals = sub[target].dropna()
        if len(vals) == 0:
            continue
        seq_metrics.append({
            "sequence": seq, "n": len(sub),
            "mean": vals.mean(), "median": vals.median(),
            "win_rate": (vals > 0).mean(),
        })
    seq_df = pd.DataFrame(seq_metrics).sort_values("win_rate", ascending=False)
    for _, r in seq_df.head(15).iterrows():
        print(f"  {r['sequence'][:55]:<55} {r['n']:>5,} "
              f"{r['mean']:>+8.2%} {r['median']:>+8.2%} "
              f"{r['win_rate']:>6.1%}")

    # ── Worst sequences by win rate (n >= 30) ──
    print(f"\n  Bottom 10 sequences by WIN RATE (n >= 30)")
    print(f"  {'sequence':<55} {'n':>5} {'mean':>9} {'median':>9} {'win %':>7}")
    print(f"  {'─'*55} {'─'*5} {'─'*9} {'─'*9} {'─'*7}")
    for _, r in seq_df.sort_values("win_rate").head(10).iterrows():
        print(f"  {r['sequence'][:55]:<55} {r['n']:>5,} "
              f"{r['mean']:>+8.2%} {r['median']:>+8.2%} "
              f"{r['win_rate']:>6.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    args = ap.parse_args()
    main(args.input, args.target)
