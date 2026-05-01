from __future__ import annotations

"""
audit_sequence_total_return.py — Re-rank sequence patterns by MEAN forward
return (not win rate).

Why: prior sequence analysis ranked patterns by win rate. In heavy-tail
regimes, this systematically excludes patterns containing mega-winners
(which have lower win rate but massive mean return). The win-rate-based
penalties in Scheme I+ v1.0/v1.1 backtest underperformed because they
penalized patterns like ICH→RS that have +88% mean return despite 62.5% win.

This script re-derives Layer 2 candidates using:
  1. mean fwd_63d_xspy as the primary ranking metric
  2. median (still useful as robust check)
  3. n (sample size)
  4. p25 / p75 (variance picture)

Output: per-pattern table sorted by mean return, with proposed v2-revised
Layer 2 magnitudes for use in a re-built scoring scheme.
"""

import argparse
import numpy as np
import pandas as pd

FIRE_DEFS = [
    ("rs", "rs_fired_v2"), ("ich", "ichimoku_fired"), ("hl", "hl_fired_v2"),
    ("cmf", "cmf_fired"), ("roc", "roc_fired"), ("atr", "atr_fired"),
    ("dtf", "dual_tf_rs_fired"),
]
LABELS = [lbl for lbl, _ in FIRE_DEFS]

INDICATOR_TYPE = {
    "rs": "TREND", "hl": "TREND", "ich": "TREND",
    "roc": "MOMENTUM", "dtf": "MOMENTUM",
    "cmf": "VOL", "atr": "VOL",
}


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


def ordered_firers(row, streak_cols) -> list[str]:
    firing = [(lbl, row[c]) for lbl, c in streak_cols.items() if row[c] > 0]
    firing.sort(key=lambda x: (-x[1], x[0]))
    return [lbl for lbl, _ in firing]


def first_two_distinct_types(types):
    seen = []
    for t in types:
        if not seen or seen[-1] != t:
            seen.append(t)
        if len(seen) == 2: break
    return tuple(seen)


def fmt_pattern_table(df: pd.DataFrame, group_col: str, target: str,
                      population_mean: float, min_n: int, out: list,
                      label_col_w: int = 28):
    """Print a per-pattern table sorted by MEAN return."""
    grp = df.groupby(group_col, observed=True).agg(
        n=(target, "count"),
        mean=(target, "mean"),
        median=(target, "median"),
        p25=(target, lambda s: s.quantile(0.25)),
        p75=(target, lambda s: s.quantile(0.75)),
        win_pct=(target, lambda s: (s > 0).mean()),
    ).reset_index()
    grp = grp[grp["n"] >= min_n].sort_values("mean", ascending=False)

    out.append(f"  {'pattern':<{label_col_w}} {'n':>5} "
               f"{'mean':>9} {'median':>9} {'p25':>9} {'p75':>9} "
               f"{'win%':>6}  {'Δ vs pop mean':>15}")
    out.append(f"  {'─'*label_col_w} {'─'*5} {'─'*9} {'─'*9} {'─'*9} {'─'*9} "
               f"{'─'*6}  {'─'*15}")
    for _, r in grp.iterrows():
        delta_mean = r["mean"] - population_mean
        arrow = ("↑↑" if delta_mean > 0.30 else
                 "↑"  if delta_mean > 0.10 else
                 "↓↓" if delta_mean < -0.20 else
                 "↓"  if delta_mean < -0.05 else "  ")
        out.append(f"  {str(r[group_col]):<{label_col_w}} {int(r['n']):>5,} "
                   f"{r['mean']:>+8.1%} {r['median']:>+8.1%} "
                   f"{r['p25']:>+8.1%} {r['p75']:>+8.1%} "
                   f"{r['win_pct']:>5.1%}  {delta_mean:>+12.1%} {arrow}")


def main(path: str, target: str, output: str, start_date: str = "2025-05-01"):
    out: list[str] = []
    out.append("=" * 110)
    out.append("audit_sequence_total_return.py — Patterns ranked by MEAN return")
    out.append("=" * 110)

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # Compute fire flags + streaks across full history
    print("computing scores + streaks...")
    df["scheme_c_score"] = df.apply(scheme_c_score, axis=1)
    df["rs_fired_v2"] = (df["rs_percentile"].fillna(0) >= 90).astype(int)
    df["hl_fired_v2"] = (df["higher_lows_count"].fillna(0) >= 4).astype(int)
    fire_col_for = {lbl: col for lbl, col in FIRE_DEFS}
    for lbl in LABELS:
        df[f"streak_{lbl}"] = df.groupby("ticker")[fire_col_for[lbl]].transform(streak_above)
    streak_cols = {lbl: f"streak_{lbl}" for lbl in LABELS}

    # Filter to regime + score>=9
    sd = pd.Timestamp(start_date)
    sig = df[(df["scheme_c_score"] >= 9.0) & (df["date"] >= sd)].copy()

    out.append(f"\n*** REGIME-FILTERED to date >= {start_date} ***")
    out.append(f"signal rows (score >= 9.0): {len(sig):,}")

    print("computing sequences...")
    sig["seq"] = sig.apply(lambda r: ordered_firers(r, streak_cols), axis=1)
    sig = sig[sig["seq"].apply(len) > 0].copy()
    sig["first_firer"] = sig["seq"].apply(lambda s: s[0])
    sig["last_firer"]  = sig["seq"].apply(lambda s: s[-1])
    sig["first_two_pair"] = sig["seq"].apply(
        lambda s: f"{s[0]}→{s[1]}" if len(s) >= 2 else None)
    sig["first_three_pat"] = sig["seq"].apply(
        lambda s: f"{s[0]}→{s[1]}→{s[2]}" if len(s) >= 3 else None)
    sig["last_two"] = sig["seq"].apply(
        lambda s: f"{s[-2]}→{s[-1]}" if len(s) >= 2 else None)
    sig["first_two_types"] = sig["seq"].apply(
        lambda s: first_two_distinct_types([INDICATOR_TYPE[i] for i in s]))

    pop_mean = sig[target].mean()
    pop_median = sig[target].median()
    pop_win = (sig[target] > 0).mean()
    out.append(f"\nPopulation stats (signals at score>=9.0 in regime):")
    out.append(f"  mean fwd_63d_xspy:   {pop_mean:>+7.1%}")
    out.append(f"  median fwd_63d_xspy: {pop_median:>+7.1%}")
    out.append(f"  win rate:            {pop_win:>5.1%}\n")

    out.append("Δ vs pop mean legend: ↑↑ > +30pp, ↑ > +10pp, ↓ < -5pp, ↓↓ < -20pp\n")

    # ─── Tables ─────────────────────────────────────────────────
    out.append("\n" + "=" * 110)
    out.append("FIRST-FIRER (single label) — ranked by mean return")
    out.append("=" * 110 + "\n")
    fmt_pattern_table(sig, "first_firer", target, pop_mean, min_n=30, out=out, label_col_w=14)

    out.append("\n" + "=" * 110)
    out.append("LAST-FIRER (single label) — ranked by mean return")
    out.append("=" * 110 + "\n")
    fmt_pattern_table(sig, "last_firer", target, pop_mean, min_n=30, out=out, label_col_w=14)

    out.append("\n" + "=" * 110)
    out.append("FIRST-2 PAIRS — ranked by mean return")
    out.append("=" * 110 + "\n")
    fmt_pattern_table(sig, "first_two_pair", target, pop_mean, min_n=30, out=out, label_col_w=20)

    out.append("\n" + "=" * 110)
    out.append("FIRST-3 PATTERNS — ranked by mean return")
    out.append("=" * 110 + "\n")
    fmt_pattern_table(sig, "first_three_pat", target, pop_mean, min_n=50, out=out, label_col_w=24)

    out.append("\n" + "=" * 110)
    out.append("LAST-2 PAIRS — ranked by mean return")
    out.append("=" * 110 + "\n")
    fmt_pattern_table(sig, "last_two", target, pop_mean, min_n=30, out=out, label_col_w=20)

    # First-2 distinct TYPE
    sig["type_pair"] = sig["first_two_types"].apply(lambda t: "→".join(t) if len(t) == 2 else "")
    out.append("\n" + "=" * 110)
    out.append("FIRST-2 DISTINCT TYPES — ranked by mean return")
    out.append("=" * 110 + "\n")
    fmt_pattern_table(sig, "type_pair", target, pop_mean, min_n=30, out=out, label_col_w=24)

    # n_led_ich
    def n_led(row):
        ich_streak = row["streak_ich"]
        if ich_streak <= 0: return None
        return sum(1 for lbl in LABELS
                   if lbl != "ich" and row[f"streak_{lbl}"] > ich_streak)
    sig["n_led_ich"] = sig.apply(n_led, axis=1)
    sig_ich = sig.dropna(subset=["n_led_ich"]).copy()
    sig_ich["n_led_label"] = sig_ich["n_led_ich"].astype(int).astype(str) + " led Ich"
    out.append("\n" + "=" * 110)
    out.append("ICHIMOKU LEAD-LAG — ranked by mean return")
    out.append("=" * 110 + "\n")
    fmt_pattern_table(sig_ich, "n_led_label", target, pop_mean, min_n=30, out=out, label_col_w=14)

    # ─── Path C Layer 2 candidates ──────────────────────────────
    out.append("\n" + "=" * 110)
    out.append("PATH C LAYER 2 CANDIDATES (mean-return-based)")
    out.append("=" * 110)
    out.append("""
Selection criteria for v2-revised Layer 2:

  PENALTIES (skip / discount):
    Mean return < pop_mean - 20pp AND median < 0 AND n >= 50
    (Pattern is consistently below population AND median negative)

  BONUSES (small reward):
    Mean return > pop_mean + 20pp AND n >= 50
    (Pattern's mean is meaningfully above population)

  HEAVY-TAIL CAPTURE BONUSES (large reward, even with lower win rate):
    Mean return > pop_mean + 50pp AND n >= 30
    (Pattern catches mega-winners; reward even with mediocre median)

Use the tables above to identify candidates that satisfy these criteria.
The audit script does not auto-select penalties — manual review needed because
mean-return rankings can be skewed by single mega-outliers in small-n cells.
""")

    text = "\n".join(out)
    print(text)
    with open(output, "w") as f:
        f.write(text + "\n")
    print(f"\n[wrote {output}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    ap.add_argument("--output", default="backtest_results/audit_sequence_total_return.txt")
    ap.add_argument("--start-date", default="2025-05-01")
    args = ap.parse_args()
    main(args.input, args.target, args.output, args.start_date)
