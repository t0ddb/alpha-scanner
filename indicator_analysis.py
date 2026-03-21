from __future__ import annotations

"""
indicator_analysis.py — Measures the predictive power of each indicator.

Answers the question: "When indicator X fires, what happens to the stock
over the next N days? And how does that compare to when it doesn't fire?"

This lets you determine whether all 5 indicators deserve equal weight,
or whether some are much more predictive than others.

Usage:
    python3 indicator_analysis.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

from config import (
    load_config, get_all_tickers, get_indicator_config,
    get_ticker_metadata,
)
from data_fetcher import fetch_batch
from indicators import compute_all_indicators, score_ticker
from backtester import score_as_of_date, compute_forward_returns


INDICATOR_NAMES = [
    "volume_spike",
    "near_52w_high",
    "bollinger_bands",
    "relative_strength",
    "moving_averages",
]

INDICATOR_LABELS = {
    "volume_spike": "Volume Spike",
    "near_52w_high": "Near 52w High",
    "bollinger_bands": "BB Squeeze",
    "relative_strength": "Relative Strength",
    "moving_averages": "MA Alignment",
}

FORWARD_WINDOWS = [10, 21, 42, 63]


def collect_indicator_events(
    data: dict[str, pd.DataFrame],
    cfg: dict,
    test_frequency: int = 5,
) -> pd.DataFrame:
    """
    Walk through historical dates and record, for each ticker on each date:
    - Which indicators fired
    - Forward returns at each window

    Returns a DataFrame with one row per ticker per test date.
    """
    benchmark_ticker = cfg["benchmark"]["ticker"]
    benchmark_df = data.get(benchmark_ticker)
    if benchmark_df is None:
        print("  [ERROR] No benchmark data.")
        return pd.DataFrame()

    max_forward = max(FORWARD_WINDOWS)
    warmup = 220
    total_days = len(benchmark_df)

    if total_days < warmup + max_forward:
        print(f"  [ERROR] Not enough data. Have {total_days} days, need {warmup + max_forward}+.")
        return pd.DataFrame()

    start_idx = warmup
    end_idx = total_days - max_forward
    test_indices = list(range(start_idx, end_idx, test_frequency))

    metadata = get_ticker_metadata(cfg)
    ind_cfg = get_indicator_config(cfg)
    rs_period = ind_cfg["relative_strength"]["period"]

    print(f"  Analyzing {len(test_indices)} dates across {len(data) - 1} tickers...")
    print(f"  Window: {benchmark_df.index[start_idx].strftime('%Y-%m-%d')} -> {benchmark_df.index[end_idx].strftime('%Y-%m-%d')}")
    print()

    all_rows = []

    for count, idx in enumerate(test_indices, 1):
        if count % 10 == 0:
            print(f"  [{count}/{len(test_indices)}] {benchmark_df.index[idx].strftime('%Y-%m-%d')}...")

        bench_slice = benchmark_df.iloc[:idx + 1]

        # Compute RS values for percentile ranking
        raw_rs = {}
        for ticker, full_df in data.items():
            if ticker == benchmark_ticker:
                continue
            df = full_df.iloc[:idx + 1]
            if len(df) < rs_period + 1 or len(bench_slice) < rs_period + 1:
                continue
            stock_ret = (df["Close"].iloc[-1] / df["Close"].iloc[-rs_period - 1]) - 1
            bench_ret = (bench_slice["Close"].iloc[-1] / bench_slice["Close"].iloc[-rs_period - 1]) - 1
            if bench_ret != 0:
                raw_rs[ticker] = stock_ret / bench_ret if bench_ret > 0 else stock_ret - bench_ret
            else:
                raw_rs[ticker] = 0
        all_rs_values = list(raw_rs.values())

        # Score each ticker and record indicator states + forward returns
        for ticker, full_df in data.items():
            if ticker == benchmark_ticker:
                continue

            df = full_df.iloc[:idx + 1]
            if len(df) < warmup:
                continue

            indicators = compute_all_indicators(df, bench_slice, cfg, all_rs_values=all_rs_values)

            # Get forward returns
            target_date = benchmark_df.index[idx]
            ticker_indices = full_df.index.get_indexer([target_date], method="nearest")
            ticker_idx = ticker_indices[0]
            fwd = compute_forward_returns(full_df, ticker_idx, windows=FORWARD_WINDOWS)

            meta = metadata.get(ticker, {})
            row = {
                "date": benchmark_df.index[idx].strftime("%Y-%m-%d"),
                "ticker": ticker,
                "sector": meta.get("sector_name", ""),
                "subsector": meta.get("subsector_name", ""),
            }

            # Record each indicator's triggered state
            for ind_name in INDICATOR_NAMES:
                row[ind_name] = indicators[ind_name]["triggered"]

            # Record forward returns
            for w in FORWARD_WINDOWS:
                row[f"fwd_{w}d"] = fwd.get(w)

            # Record overall score
            scoring = score_ticker(indicators)
            row["score"] = scoring["score"]

            all_rows.append(row)

    return pd.DataFrame(all_rows)


def analyze_individual_indicators(df: pd.DataFrame) -> None:
    """
    For each indicator, compare forward returns when it fires vs. when it doesn't.
    """
    print(f"\n{'='*90}")
    print(f"  INDIVIDUAL INDICATOR ANALYSIS")
    print(f"  'When this indicator fires, what happens next?'")
    print(f"{'='*90}")

    summary_rows = []

    for ind_name in INDICATOR_NAMES:
        label = INDICATOR_LABELS[ind_name]
        fired = df[df[ind_name] == True]
        not_fired = df[df[ind_name] == False]

        fire_rate = len(fired) / len(df) * 100 if len(df) > 0 else 0

        print(f"\n  {'─'*80}")
        print(f"  {label}")
        print(f"  Fired: {len(fired)} times ({fire_rate:.1f}%) | Did not fire: {len(not_fired)} times")
        print(f"  {'─'*80}")

        print(f"\n  {'Window':<12} {'Fired→Win%':>12} {'Fired→Avg':>12} {'NotFired→Win%':>15} {'NotFired→Avg':>14} {'Edge':>10}")
        print(f"  {'':─<12} {'':─>12} {'':─>12} {'':─>15} {'':─>14} {'':─>10}")

        for w in FORWARD_WINDOWS:
            col = f"fwd_{w}d"

            fired_valid = fired[col].dropna()
            not_fired_valid = not_fired[col].dropna()

            if len(fired_valid) == 0 or len(not_fired_valid) == 0:
                continue

            fired_wr = (fired_valid > 0).mean() * 100
            fired_avg = fired_valid.mean() * 100
            nf_wr = (not_fired_valid > 0).mean() * 100
            nf_avg = not_fired_valid.mean() * 100
            edge = fired_avg - nf_avg

            print(f"  {w:>2}-day      {fired_wr:>10.1f}%  {fired_avg:>+10.2f}%  {nf_wr:>13.1f}%  {nf_avg:>+12.2f}%  {edge:>+8.2f}%")

            if w == 63:  # Use 63-day for summary ranking
                summary_rows.append({
                    "indicator": label,
                    "fire_rate": fire_rate,
                    "fired_win_rate_63d": fired_wr,
                    "fired_avg_return_63d": fired_avg,
                    "not_fired_avg_return_63d": nf_avg,
                    "edge_63d": edge,
                })

    # --- Summary ranking ---
    print(f"\n\n{'='*90}")
    print(f"  INDICATOR RANKING BY 63-DAY PREDICTIVE EDGE")
    print(f"  (Edge = avg return when fired minus avg return when not fired)")
    print(f"{'='*90}\n")

    summary_df = pd.DataFrame(summary_rows).sort_values("edge_63d", ascending=False)

    for i, row in summary_df.iterrows():
        bar_len = max(0, int(row["edge_63d"] / 2))
        bar = "█" * bar_len
        print(f"  {row['indicator']:<20s}  Edge: {row['edge_63d']:>+7.2f}%  |  Win rate: {row['fired_win_rate_63d']:>5.1f}%  |  Fire rate: {row['fire_rate']:>5.1f}%  {bar}")

    return summary_df


def analyze_indicator_combinations(df: pd.DataFrame) -> None:
    """
    Analyze how pairs of indicators perform together vs. alone.
    """
    print(f"\n\n{'='*90}")
    print(f"  INDICATOR PAIR ANALYSIS (63-DAY FORWARD RETURNS)")
    print(f"  'Which combinations produce the best results?'")
    print(f"{'='*90}\n")

    col = "fwd_63d"

    print(f"  {'Pair':<45s} {'Events':>8} {'Win%':>8} {'Avg Ret':>10}")
    print(f"  {'':─<45} {'':─>8} {'':─>8} {'':─>10}")

    pair_results = []

    for i, ind1 in enumerate(INDICATOR_NAMES):
        for ind2 in INDICATOR_NAMES[i + 1:]:
            both = df[(df[ind1] == True) & (df[ind2] == True)]
            valid = both[col].dropna()

            if len(valid) < 5:
                continue

            wr = (valid > 0).mean() * 100
            avg = valid.mean() * 100
            label = f"{INDICATOR_LABELS[ind1]} + {INDICATOR_LABELS[ind2]}"

            pair_results.append({
                "pair": label,
                "events": len(valid),
                "win_rate": wr,
                "avg_return": avg,
            })

    pair_results.sort(key=lambda x: -x["avg_return"])

    for p in pair_results:
        print(f"  {p['pair']:<45s} {p['events']:>8} {p['win_rate']:>7.1f}% {p['avg_return']:>+9.2f}%")

    # --- Best and worst pairs ---
    if pair_results:
        best = pair_results[0]
        worst = pair_results[-1]
        print(f"\n  Best pair:  {best['pair']} → {best['avg_return']:+.2f}% avg (win rate: {best['win_rate']:.1f}%)")
        print(f"  Worst pair: {worst['pair']} → {worst['avg_return']:+.2f}% avg (win rate: {worst['win_rate']:.1f}%)")


def analyze_without_indicator(df: pd.DataFrame) -> None:
    """
    Test what happens to overall system performance if you remove each indicator.
    Simulates a 'leave one out' approach.
    """
    print(f"\n\n{'='*90}")
    print(f"  LEAVE-ONE-OUT ANALYSIS (63-DAY)")
    print(f"  'If we removed this indicator, would the system get better or worse?'")
    print(f"{'='*90}\n")

    col = "fwd_63d"

    # Baseline: score >= 3 with all indicators
    baseline = df[df["score"] >= 3][col].dropna()
    baseline_wr = (baseline > 0).mean() * 100
    baseline_avg = baseline.mean() * 100

    print(f"  Baseline (all 5 indicators, score >= 3): Win rate: {baseline_wr:.1f}% | Avg: {baseline_avg:+.2f}% | N={len(baseline)}")
    print()

    print(f"  {'Removed Indicator':<25s} {'Win%':>8} {'Avg Ret':>10} {'N':>6} {'vs Baseline':>14}")
    print(f"  {'':─<25} {'':─>8} {'':─>10} {'':─>6} {'':─>14}")

    for ind_name in INDICATOR_NAMES:
        label = INDICATOR_LABELS[ind_name]

        # Recalculate score without this indicator
        remaining = [n for n in INDICATOR_NAMES if n != ind_name]
        df["adjusted_score"] = df[remaining].sum(axis=1)

        # Filter with adjusted score >= 3 (now out of 4 max)
        # Use >= 2 since max is now 4, to keep a similar selectivity ratio
        adjusted_threshold = 2
        subset = df[df["adjusted_score"] >= adjusted_threshold][col].dropna()

        if len(subset) == 0:
            continue

        wr = (subset > 0).mean() * 100
        avg = subset.mean() * 100
        diff = avg - baseline_avg

        print(f"  {label:<25s} {wr:>7.1f}% {avg:>+9.2f}% {len(subset):>6} {diff:>+12.2f}%")

    df.drop(columns=["adjusted_score"], inplace=True, errors="ignore")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    cfg = load_config()

    print("=" * 90)
    print("  INDICATOR PREDICTIVE POWER ANALYSIS")
    print("=" * 90)
    print()
    print("  This analysis measures each indicator's individual contribution")
    print("  to predicting future returns. It answers:")
    print("    1. Which indicators have the most predictive edge?")
    print("    2. Which pairs of indicators work best together?")
    print("    3. Would removing any indicator improve the system?")
    print()

    # Fetch data
    print("  Fetching 2 years of data...\n")
    all_tickers = ["SPY"] + get_all_tickers(cfg)
    data = fetch_batch(all_tickers, period="2y", verbose=True)

    if not data:
        print("  [ERROR] No data fetched.")
        exit(1)

    # Collect events
    print("\n  Collecting indicator events across all dates and tickers...\n")
    events_df = collect_indicator_events(data, cfg, test_frequency=5)

    if events_df.empty:
        print("  [ERROR] No events collected.")
        exit(1)

    print(f"\n  Collected {len(events_df)} ticker-date observations.\n")

    # Run analyses
    summary = analyze_individual_indicators(events_df)
    analyze_indicator_combinations(events_df)
    analyze_without_indicator(events_df)

    print(f"\n{'='*90}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*90}\n")
