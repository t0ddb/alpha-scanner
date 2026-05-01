from __future__ import annotations

"""
audit_indicator_curves.py — Fine-grained empirical curves per indicator.

For each scored indicator, computes win rate by raw-value bucket on the
FULL dataset (not just signal rows). For each bucket, shows:
  - bucket range
  - n, mean fwd return, median, win %
  - Δ vs full-dataset baseline win rate
  - current Scheme C points awarded for this bucket
  - AUTO-PROPOSED Scheme I points (positive-only) — max_weight at the peak
    bucket, scaled linearly to 0 at baseline.
  - AUTO-PROPOSED Scheme I points (with penalties) — same as above but
    negative scoring for buckets ≥2pp below baseline.
  - ASCII bar chart of win-rate delta for visual curve inspection.

Includes coverage of CMF (currently weight 0 in Scheme C) and Higher Lows
(currently gradient but no empirical data captured before). Also splits
Dual-TF into its two underlying conditions (acceleration vs sustained).

Output: backtest_results/audit_indicator_curves.txt
"""

import argparse
import numpy as np
import pandas as pd

# Current Scheme C scoring — for side-by-side comparison
CURRENT = {
    "rs":   {"max": 3.0, "rule": "gradient: 90→3.0, 80→2.4, 70→1.8, 60→1.2, 50→0.6"},
    "hl":   {"max": 0.5, "rule": "gradient: 5→0.5, 4→0.375, 3→0.25, 2→0.125"},
    "ich":  {"max": 2.0, "rule": "binary: above+bullish (2/3) → 2.0"},
    "roc":  {"max": 1.5, "rule": "binary: roc > 5% → 1.5"},
    "cmf":  {"max": 0.0, "rule": "DROPPED (was 1.5 binary at >0.05)"},
    "dtf":  {"max": 2.5, "rule": "binary: cond_a OR cond_b → 2.5"},
    "atr":  {"max": 0.5, "rule": "binary: pctl ≥ 80 → 0.5"},
}


def current_pts(name: str, lo: float, hi: float | None) -> float:
    """Compute Scheme C points for a value in [lo, hi)."""
    mid = (lo + hi) / 2 if hi is not None else lo
    if name == "rs":
        for thresh, pts in [(90, 3.0), (80, 2.4), (70, 1.8), (60, 1.2), (50, 0.6)]:
            if mid >= thresh:
                return pts
        return 0.0
    if name == "hl":
        # mid is float-ish but values are integers
        v = int(round(mid))
        for c, pts in [(5, 0.5), (4, 0.375), (3, 0.25), (2, 0.125)]:
            if v >= c:
                return pts
        return 0.0
    if name == "ich":
        v = int(round(mid))
        return 2.0 if v >= 2 else 0.0
    if name == "roc":
        return 1.5 if mid > 5.0 else 0.0
    if name == "cmf":
        return 0.0  # dropped in Scheme C
    if name == "atr":
        return 0.5 if mid >= 80 else 0.0
    return 0.0


def ascii_bar(delta: float, scale: float = 0.20) -> str:
    """Bar chart for delta in [-scale, +scale]. Width 20 chars."""
    if pd.isna(delta):
        return ""
    width = 20
    units = max(-width // 2, min(width // 2, int(round(delta / scale * (width / 2)))))
    if units >= 0:
        return " " * (width // 2) + "│" + "█" * units + " " * (width // 2 - units)
    else:
        return " " * (width // 2 + units) + "█" * (-units) + "│" + " " * (width // 2)


def auto_score(win_pct: float, base: float, max_pos_delta: float, max_weight: float,
               with_penalty: bool, max_neg_delta: float = 0.0) -> float:
    """Auto-proposed score for a single bucket."""
    delta = win_pct - base
    if delta > 0.01:
        if max_pos_delta <= 0:
            return 0.0
        return round(max_weight * min(1.0, delta / max_pos_delta), 2)
    if with_penalty and delta < -0.02 and max_neg_delta > 0:
        # Negative scoring: cap at -0.5 * max_weight
        return round(-0.5 * max_weight * min(1.0, (-delta) / max_neg_delta), 2)
    return 0.0


def analyze_indicator(name: str, label: str, data: pd.Series, target_vals: pd.Series,
                      bins: list[float], bin_labels: list[str],
                      baseline: float, max_weight: float, out: list,
                      include_below_min: bool = True):
    """Bucket an indicator and print its empirical curve + scoring proposal."""
    out.append("\n" + "=" * 90)
    out.append(f"INDICATOR: {label}  (parquet col: {data.name})")
    out.append("=" * 90)
    cur = CURRENT[name]
    out.append(f"  Current Scheme C: {cur['rule']}  (max {cur['max']:.1f} pts)")
    out.append(f"  Proposed max weight for Scheme I: {max_weight:.1f} pts")
    out.append("")

    # Build bucket assignments
    df_local = pd.DataFrame({"v": data.values, "fwd": target_vals.values}).dropna(subset=["v"])
    df_local["bucket"] = pd.cut(df_local["v"], bins=bins, labels=bin_labels,
                                 right=False, include_lowest=True)

    # Stats per bucket
    grouped = df_local.groupby("bucket", observed=True).agg(
        n=("fwd", "count"),
        mean_fwd=("fwd", "mean"),
        median_fwd=("fwd", "median"),
        wins=("fwd", lambda s: (s > 0).sum()),
    ).reset_index()
    grouped["win_pct"] = grouped["wins"] / grouped["n"]
    grouped["delta"] = grouped["win_pct"] - baseline

    # Compute deltas for auto-scoring
    pos_deltas = grouped[grouped["delta"] > 0.01]["delta"]
    neg_deltas = grouped[grouped["delta"] < -0.02]["delta"]
    max_pos_delta = pos_deltas.max() if len(pos_deltas) > 0 else 0.0
    max_neg_delta = -neg_deltas.min() if len(neg_deltas) > 0 else 0.0

    out.append(f"  {'bucket':<14} {'n':>7} {'mean':>8} {'median':>8} {'win %':>7} "
               f"{'Δ base':>7}  {'cur pts':>7}  {'I+':>5}  {'I±':>5}  bar")
    out.append(f"  {'─'*14} {'─'*7} {'─'*8} {'─'*8} {'─'*7} {'─'*7}  {'─'*7}  "
               f"{'─'*5}  {'─'*5}  {'─'*22}")

    proposed_pos = []
    proposed_pen = []

    # Get the bucket boundaries as numeric pairs
    bin_pairs = list(zip(bins[:-1], bins[1:]))

    for i, row in grouped.iterrows():
        bucket_label = str(row["bucket"])
        n = int(row["n"])
        if n == 0:
            continue
        mean = row["mean_fwd"]
        med = row["median_fwd"]
        wp = row["win_pct"]
        dp = row["delta"]
        # Find the corresponding bin pair to compute current/proposed pts
        # Match label index to bin index (they're 1:1)
        idx = bin_labels.index(row["bucket"])
        lo, hi = bin_pairs[idx]
        cur_pts = current_pts(name, lo, hi)
        i_pos = auto_score(wp, baseline, max_pos_delta, max_weight,
                           with_penalty=False)
        i_pen = auto_score(wp, baseline, max_pos_delta, max_weight,
                           with_penalty=True, max_neg_delta=max_neg_delta)
        proposed_pos.append((bucket_label, lo, hi, i_pos))
        proposed_pen.append((bucket_label, lo, hi, i_pen))
        bar = ascii_bar(dp)
        out.append(f"  {bucket_label:<14} {n:>7,} {mean:>+7.2%} {med:>+7.2%} "
                   f"{wp:>6.1%} {dp:>+6.1%}  {cur_pts:>6.2f}   "
                   f"{i_pos:>5.2f}  {i_pen:>+5.2f}  {bar}")

    out.append(f"\n  Baseline win rate (full dataset): {baseline:.1%}")
    out.append(f"  Max positive Δ: +{max_pos_delta:.1%}   Max negative Δ: -{max_neg_delta:.1%}")
    out.append(f"\n  PROPOSED Scheme I scoring (positive-only, max {max_weight:.1f} pts):")
    for lbl, lo, hi, pts in proposed_pos:
        if pts > 0:
            out.append(f"    {lbl:<14} → {pts:.2f} pts")
    out.append(f"\n  PROPOSED Scheme I scoring (with penalties, range -{0.5*max_weight:.2f} to +{max_weight:.1f}):")
    for lbl, lo, hi, pts in proposed_pen:
        if pts != 0:
            out.append(f"    {lbl:<14} → {pts:+.2f} pts")


def main(path: str, target: str, output: str, start_date: str | None = None):
    out: list[str] = []
    out.append("=" * 90)
    out.append("audit_indicator_curves.py — Empirical scoring curves (Scheme I draft)")
    out.append("=" * 90)

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target]).copy()
    if start_date:
        sd = pd.Timestamp(start_date)
        before = len(df)
        df = df[df["date"] >= sd].copy()
        out.append(f"\n*** REGIME-FILTERED to date >= {start_date} ***")
        out.append(f"    {before:,} → {len(df):,} rows after filter")
    out.append(f"\nloaded {len(df):,} rows  ({df['date'].min().date()} → {df['date'].max().date()})")
    out.append(f"target: {target}")

    # Baseline win rate (across full dataset, not just signal rows)
    baseline = (df[target] > 0).mean()
    out.append(f"baseline win rate (full dataset, all rows with valid fwd return): {baseline:.1%}\n")

    # ─── RS ──────────────────────────────────────────────────────
    bins = [0, 50, 55, 60, 65, 70, 75, 80, 85, 88, 90, 92, 94, 96, 98, 100.0001]
    labels = ["<50", "[50,55)", "[55,60)", "[60,65)", "[65,70)",
              "[70,75)", "[75,80)", "[80,85)", "[85,88)", "[88,90)",
              "[90,92)", "[92,94)", "[94,96)", "[96,98)", "[98,100]"]
    analyze_indicator("rs", "Relative Strength (RS percentile)",
                      df["rs_percentile"], df[target], bins, labels,
                      baseline, max_weight=3.0, out=out)

    # ─── Higher Lows ─────────────────────────────────────────────
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 100]
    labels = ["0", "1", "2", "3", "4", "5", "6", "7-8", "9-11", "12+"]
    analyze_indicator("hl", "Higher Lows (consecutive count)",
                      df["higher_lows_count"], df[target], bins, labels,
                      baseline, max_weight=1.0, out=out)
    out.append("\n  NOTE: max weight bumped from 0.5 to 1.0 if empirical curve supports it.")
    out.append("  Final max weight to be set after curve review.")

    # ─── Ichimoku composite ──────────────────────────────────────
    bins = [0, 1, 2, 3, 4]
    labels = ["0/3", "1/3", "2/3", "3/3"]
    analyze_indicator("ich", "Ichimoku Cloud (composite 0-3)",
                      df["ichimoku_score"], df[target], bins, labels,
                      baseline, max_weight=2.5, out=out)
    out.append("\n  NOTE: max weight bumped from 2.0 to 2.5 to reflect 3/3 → 4pp lift over 2/3.")

    # ─── ROC ─────────────────────────────────────────────────────
    bins = [-100, -10, -5, 0, 3, 5, 7.5, 10, 15, 20, 25, 35, 50, 75, 100, 1000]
    labels = ["≤-10", "[-10,-5)", "[-5,0)", "[0,3)", "[3,5)", "[5,7.5)",
              "[7.5,10)", "[10,15)", "[15,20)", "[20,25)", "[25,35)",
              "[35,50)", "[50,75)", "[75,100)", "≥100"]
    analyze_indicator("roc", "Rate of Change (21d %)",
                      df["roc_value"], df[target], bins, labels,
                      baseline, max_weight=1.5, out=out)

    # ─── CMF ─────────────────────────────────────────────────────
    bins = [-1, -0.20, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.30, 1]
    labels = ["≤-0.20", "[-0.20,-0.10)", "[-0.10,-0.05)", "[-0.05,0)",
              "[0,0.05)", "[0.05,0.10)", "[0.10,0.15)", "[0.15,0.20)",
              "[0.20,0.30)", "≥0.30"]
    analyze_indicator("cmf", "Chaikin Money Flow (20-day)",
                      df["cmf_value"], df[target], bins, labels,
                      baseline, max_weight=1.0, out=out)
    out.append("\n  NOTE: CMF currently dropped (0 pts in Scheme C). If empirical curve")
    out.append("  shows signal, propose re-adding at recommended max weight.")

    # ─── ATR ─────────────────────────────────────────────────────
    bins = [0, 30, 50, 60, 70, 75, 80, 85, 90, 95, 100.0001]
    labels = ["<30", "[30,50)", "[50,60)", "[60,70)", "[70,75)",
              "[75,80)", "[80,85)", "[85,90)", "[90,95)", "[95,100]"]
    analyze_indicator("atr", "ATR Expansion (percentile in 50d range)",
                      df["atr_percentile"], df[target], bins, labels,
                      baseline, max_weight=0.5, out=out)

    # ─── Dual-TF: split into 3 component analyses ────────────────
    out.append("\n" + "=" * 90)
    out.append("INDICATOR: Dual-TF RS  (composite, currently 2.5 pts on cond_a OR cond_b)")
    out.append("=" * 90)
    out.append("  Current Scheme C: binary on cond_a OR cond_b → 2.5 pts")
    out.append("  Cond_a: 126d RS ≥ 70 AND 63d > 126d (acceleration)")
    out.append("  Cond_b: 63d ≥ 80 AND 21d ≥ 80 (sustained short-term)")
    out.append("\n  Splitting analysis into the 3 underlying numeric drivers:\n")

    # 126d percentile
    bins = [0, 50, 60, 70, 75, 80, 85, 90, 95, 100.0001]
    labels = ["<50", "[50,60)", "[60,70)", "[70,75)", "[75,80)",
              "[80,85)", "[85,90)", "[90,95)", "[95,100]"]
    analyze_indicator("dtf", "Dual-TF — 126d RS percentile",
                      df["rs_126d_pctl"], df[target], bins, labels,
                      baseline, max_weight=2.5, out=out)

    # 63d percentile
    analyze_indicator("dtf", "Dual-TF — 63d RS percentile",
                      df["rs_63d_pctl"], df[target], bins, labels,
                      baseline, max_weight=2.5, out=out)

    # Acceleration spread (63d - 126d) — using full data, not conditioned on 126d≥70
    df["dtf_accel"] = df["rs_63d_pctl"] - df["rs_126d_pctl"]
    bins = [-100, -20, -10, -5, 0, 5, 10, 20, 100]
    labels = ["≤-20", "[-20,-10)", "[-10,-5)", "[-5,0)", "[0,5)",
              "[5,10)", "[10,20)", "≥20"]
    analyze_indicator("dtf", "Dual-TF — Acceleration spread (63d − 126d)",
                      df["dtf_accel"], df[target], bins, labels,
                      baseline, max_weight=2.5, out=out)

    # ─────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────
    out.append("\n" + "=" * 90)
    out.append("SUMMARY — Scheme I draft scoring spec")
    out.append("=" * 90)
    out.append("Manual review needed. Use the per-indicator 'PROPOSED' tables above to")
    out.append("decide on final scoring formula. Total max weight should target ~10.0")
    out.append("(matching Scheme C's 10.0 max).\n")

    out.append("Suggested starting weights (subject to review):")
    out.append(f"  RS         max 3.0  (peak in mid-90s, dip in 95-97 zone)")
    out.append(f"  Ichimoku   max 2.5  (3/3 worth more than 2/3)")
    out.append(f"  Dual-TF    max 2.5  (recompose: drop accel requirement, score sustained)")
    out.append(f"  ROC        max 1.5  (cap at 50% — collapses above)")
    out.append(f"  HL         max 1.0? (TBD per curve)")
    out.append(f"  CMF        max ?    (re-include if curve supports)")
    out.append(f"  ATR        max 0.5? (or invert / drop — empirically anti-correlated)")
    out.append("")
    out.append("After reviewing this output, hand-craft a Scheme I scoring spec and")
    out.append("implement in indicators.py for backtesting.")

    text = "\n".join(out)
    print(text)
    with open(output, "w") as f:
        f.write(text + "\n")
    print(f"\n[wrote {output}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    ap.add_argument("--output", default="backtest_results/audit_indicator_curves.txt")
    ap.add_argument("--start-date", default=None,
                    help="If set, filter to rows where date >= this (YYYY-MM-DD)")
    args = ap.parse_args()
    main(args.input, args.target, args.output, args.start_date)
