from __future__ import annotations

"""
conditional_edge_analysis.py — Tests incremental edge of each indicator
AFTER controlling for Relative Strength.

Question: When RS is already strong, does each other indicator add
independent value, or is it just a noisy proxy for the same signal?

Usage:
    python3 conditional_edge_analysis.py
"""

import pandas as pd
import numpy as np
from config import load_config, get_all_tickers
from data_fetcher import fetch_batch
from gradient_analysis import collect_continuous_metrics, METRIC_LABELS, CURRENT_THRESHOLDS

FORWARD_WINDOWS = [10, 21, 42, 63]


def analyze_conditional_edge(df: pd.DataFrame) -> None:
    """
    For each indicator, compare:
    1. Edge when RS is strong (≥75th pctl) AND indicator fires
    2. Edge when RS is strong (≥75th pctl) AND indicator does NOT fire
    3. The incremental edge = (1) - (2)

    If incremental edge is near zero, the indicator is redundant with RS.
    """
    rs_threshold = 75
    rs_strong = df[df["rs_percentile"] >= rs_threshold]
    rs_weak = df[df["rs_percentile"] < rs_threshold]

    print(f"\n{'=' * 100}")
    print(f"  CONDITIONAL EDGE ANALYSIS — Does each indicator add value BEYOND Relative Strength?")
    print(f"{'=' * 100}")
    print(f"\n  Universe: {len(df)} total observations")
    print(f"  RS Strong (≥{rs_threshold}th pctl): {len(rs_strong)} observations ({len(rs_strong)/len(df)*100:.1f}%)")
    print(f"  RS Weak (<{rs_threshold}th pctl): {len(rs_weak)} observations ({len(rs_weak)/len(df)*100:.1f}%)")

    # Define indicator conditions (using continuous metrics)
    indicators = {
        "Ichimoku Cloud (score ≥ 2)": ("ichimoku_score", lambda x: x >= 2),
        "Ichimoku Cloud (score = 3)": ("ichimoku_score", lambda x: x == 3),
        "Higher Lows (≥ 4)":          ("higher_lows_count", lambda x: x >= 4),
        "Higher Lows (≥ 3)":          ("higher_lows_count", lambda x: x >= 3),
        "MA Alignment (above 50)":    ("ma_pct_above_50", lambda x: x > 0),
        "MA Alignment (above 200)":   ("ma_pct_above_200", lambda x: x > 0),
        "MA: >10% above 200":        ("ma_pct_above_200", lambda x: x > 10),
        "ROC (> 5%)":                 ("roc_value", lambda x: x > 5),
        "ROC (> 10%)":                ("roc_value", lambda x: x > 10),
        "ROC (> 15%)":                ("roc_value", lambda x: x > 15),
        "CMF (> 0.05)":               ("cmf_value", lambda x: x > 0.05),
        "CMF (> 0.15)":               ("cmf_value", lambda x: x > 0.15),
        "CMF (> 0.25)":               ("cmf_value", lambda x: x > 0.25),
        "Near 52w High (≥ -2%)":      ("pct_from_52w_high", lambda x: x >= -0.02),
        "ATR Expansion (≥ 80th)":     ("atr_percentile", lambda x: x >= 80),
        "ATR Expansion (≥ 90th)":     ("atr_percentile", lambda x: x >= 90),
    }

    # ── SECTION 1: Incremental edge when RS is strong ──
    print(f"\n\n{'─' * 100}")
    print(f"  SECTION 1: INCREMENTAL EDGE WHEN RS IS STRONG (≥ {rs_threshold}th percentile)")
    print(f"  Question: Does firing this indicator ALSO improve returns beyond RS alone?")
    print(f"{'─' * 100}\n")

    col = "fwd_63d"
    results = []

    for label, (metric, condition) in indicators.items():
        valid = rs_strong[metric].dropna()
        if len(valid) == 0:
            continue

        mask = rs_strong[metric].apply(condition)
        fired = rs_strong[mask][col].dropna()
        not_fired = rs_strong[~mask][col].dropna()

        if len(fired) < 20 or len(not_fired) < 20:
            continue

        fired_avg = fired.mean() * 100
        fired_wr = (fired > 0).mean() * 100
        not_fired_avg = not_fired.mean() * 100
        not_fired_wr = (not_fired > 0).mean() * 100
        incr_edge = fired_avg - not_fired_avg

        results.append({
            "indicator": label,
            "fired_n": len(fired),
            "fired_avg": fired_avg,
            "fired_wr": fired_wr,
            "not_fired_n": len(not_fired),
            "not_fired_avg": not_fired_avg,
            "not_fired_wr": not_fired_wr,
            "incr_edge": incr_edge,
        })

    results.sort(key=lambda x: -x["incr_edge"])

    print(f"  {'Indicator':<30} {'Fired':>7} {'Avg Ret':>9} {'WR':>7}  │  "
          f"{'¬Fired':>7} {'Avg Ret':>9} {'WR':>7}  │  {'Incr Edge':>10}")
    print(f"  {'─' * 95}")

    for r in results:
        emoji = "✅" if r["incr_edge"] > 3 else "🔶" if r["incr_edge"] > 0 else "❌"
        print(f"  {r['indicator']:<30} {r['fired_n']:>7} {r['fired_avg']:>+8.1f}% "
              f"{r['fired_wr']:>5.1f}%  │  {r['not_fired_n']:>7} "
              f"{r['not_fired_avg']:>+8.1f}% {r['not_fired_wr']:>5.1f}%  │  "
              f"{r['incr_edge']:>+9.1f}% {emoji}")

    # ── SECTION 2: Same but when RS is weak ──
    print(f"\n\n{'─' * 100}")
    print(f"  SECTION 2: INCREMENTAL EDGE WHEN RS IS WEAK (< {rs_threshold}th percentile)")
    print(f"  Question: Can any indicator rescue a weak RS signal?")
    print(f"{'─' * 100}\n")

    results_weak = []

    for label, (metric, condition) in indicators.items():
        valid = rs_weak[metric].dropna()
        if len(valid) == 0:
            continue

        mask = rs_weak[metric].apply(condition)
        fired = rs_weak[mask][col].dropna()
        not_fired = rs_weak[~mask][col].dropna()

        if len(fired) < 20 or len(not_fired) < 20:
            continue

        fired_avg = fired.mean() * 100
        fired_wr = (fired > 0).mean() * 100
        not_fired_avg = not_fired.mean() * 100
        not_fired_wr = (not_fired > 0).mean() * 100
        incr_edge = fired_avg - not_fired_avg

        results_weak.append({
            "indicator": label,
            "fired_n": len(fired),
            "fired_avg": fired_avg,
            "fired_wr": fired_wr,
            "not_fired_n": len(not_fired),
            "not_fired_avg": not_fired_avg,
            "not_fired_wr": not_fired_wr,
            "incr_edge": incr_edge,
        })

    results_weak.sort(key=lambda x: -x["incr_edge"])

    print(f"  {'Indicator':<30} {'Fired':>7} {'Avg Ret':>9} {'WR':>7}  │  "
          f"{'¬Fired':>7} {'Avg Ret':>9} {'WR':>7}  │  {'Incr Edge':>10}")
    print(f"  {'─' * 95}")

    for r in results_weak:
        emoji = "✅" if r["incr_edge"] > 3 else "🔶" if r["incr_edge"] > 0 else "❌"
        print(f"  {r['indicator']:<30} {r['fired_n']:>7} {r['fired_avg']:>+8.1f}% "
              f"{r['fired_wr']:>5.1f}%  │  {r['not_fired_n']:>7} "
              f"{r['not_fired_avg']:>+8.1f}% {r['not_fired_wr']:>5.1f}%  │  "
              f"{r['incr_edge']:>+9.1f}% {emoji}")

    # ── SECTION 3: Correlation matrix between indicators ──
    print(f"\n\n{'─' * 100}")
    print(f"  SECTION 3: INDICATOR CORRELATION MATRIX")
    print(f"  High correlation = redundant signals measuring the same thing")
    print(f"{'─' * 100}\n")

    corr_metrics = [
        "rs_percentile", "ichimoku_score", "higher_lows_count",
        "roc_value", "cmf_value", "atr_percentile",
        "ma_pct_above_50", "ma_pct_above_200",
    ]
    corr_labels = [
        "RS Pctl", "Ichimoku", "H.Lows",
        "ROC", "CMF", "ATR Pctl",
        "vs 50SMA", "vs 200SMA",
    ]

    corr_df = df[corr_metrics].dropna()
    corr_matrix = corr_df.corr(method="spearman")

    # Print header
    header = f"  {'':>12}"
    for label in corr_labels:
        header += f"  {label:>8}"
    print(header)
    print(f"  {'─' * (12 + 10 * len(corr_labels))}")

    for i, (metric, label) in enumerate(zip(corr_metrics, corr_labels)):
        line = f"  {label:>12}"
        for j, metric2 in enumerate(corr_metrics):
            val = corr_matrix.loc[metric, metric2]
            if i == j:
                line += f"  {'   —':>8}"
            else:
                marker = " ⚠" if abs(val) > 0.5 else ""
                line += f"  {val:>+6.2f}{marker}"
        print(line)

    print(f"\n  ⚠ = correlation > |0.50| (potential redundancy)")

    # ── SECTION 4: Summary verdict ──
    print(f"\n\n{'=' * 100}")
    print(f"  VERDICT: KEEP, MODIFY, OR DROP?")
    print(f"{'=' * 100}\n")

    print("  Based on incremental edge (when RS ≥ 75th) and correlation analysis:\n")

    # Summarise
    for r in results:
        if r["incr_edge"] > 5:
            verdict = "KEEP ✅ — Strong independent signal"
        elif r["incr_edge"] > 2:
            verdict = "KEEP 🔶 — Moderate independent signal"
        elif r["incr_edge"] > 0:
            verdict = "QUESTIONABLE ⚠ — Marginal incremental value"
        else:
            verdict = "DROP ❌ — No incremental edge, likely redundant with RS"
        print(f"  {r['indicator']:<30} Incr edge: {r['incr_edge']:>+6.1f}%  →  {verdict}")


if __name__ == "__main__":
    cfg = load_config()

    print("=" * 100)
    print("  CONDITIONAL EDGE ANALYSIS")
    print("  Testing indicator independence from Relative Strength")
    print("=" * 100)
    print()

    # Fetch data
    print("  Fetching 3 years of data...\n")
    all_tickers = ["SPY"] + get_all_tickers(cfg)
    data = fetch_batch(all_tickers, period="3y", verbose=True)

    if not data:
        print("  [ERROR] No data fetched.")
        exit(1)

    # Collect continuous metrics (reuses gradient_analysis collection)
    print("\n  Collecting continuous metrics...\n")
    events_df = collect_continuous_metrics(data, cfg, test_frequency=5)

    if events_df.empty:
        print("  [ERROR] No events collected.")
        exit(1)

    print(f"\n  Collected {len(events_df)} observations.\n")

    # Run analysis
    analyze_conditional_edge(events_df)

    print(f"\n{'=' * 100}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 100}\n")
