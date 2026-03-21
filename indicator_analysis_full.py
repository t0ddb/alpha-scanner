from __future__ import annotations

"""
indicator_analysis_full.py — Tests all 16 indicators for predictive power.

Runs the same 3 analyses as before but across all 16 indicators:
    1. Individual indicator edge (fired vs not fired)
    2. Top indicator pairs
    3. Leave-one-out from the best N indicators

Usage:
    python3 indicator_analysis_full.py
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
from indicators_expanded import (
    compute_all_expanded, INDICATOR_LABELS,
)


ALL_INDICATOR_NAMES = list(INDICATOR_LABELS.keys())
FORWARD_WINDOWS = [10, 21, 42, 63]


def _compute_rs_at_period(data, benchmark_ticker, bench_slice, idx, period):
    """Compute raw RS values for all tickers at a given period length."""
    raw_rs = {}
    for ticker, full_df in data.items():
        if ticker == benchmark_ticker:
            continue
        df = full_df.iloc[:idx + 1]
        if len(df) < period + 1 or len(bench_slice) < period + 1:
            continue
        stock_ret = (df["Close"].iloc[-1] / df["Close"].iloc[-period - 1]) - 1
        bench_ret = (bench_slice["Close"].iloc[-1] / bench_slice["Close"].iloc[-period - 1]) - 1
        if bench_ret != 0:
            raw_rs[ticker] = stock_ret / bench_ret if bench_ret > 0 else stock_ret - bench_ret
        else:
            raw_rs[ticker] = 0
    return raw_rs


def _percentile_rank(value, all_values):
    """Compute percentile rank of value within all_values."""
    if not all_values:
        return 0
    arr = np.array(all_values)
    return float(np.sum(arr <= value) / len(arr) * 100)


def collect_all_indicator_events(
    data: dict[str, pd.DataFrame],
    cfg: dict,
    test_frequency: int = 5,
) -> pd.DataFrame:
    """
    Walk through historical dates and record all 22 indicator states
    plus forward returns for every ticker on each test date.
    """
    benchmark_ticker = cfg["benchmark"]["ticker"]
    benchmark_df = data.get(benchmark_ticker)
    if benchmark_df is None:
        print("  [ERROR] No benchmark data.")
        return pd.DataFrame()

    max_forward = max(FORWARD_WINDOWS)
    warmup = 260  # increased from 220 for 200-day MA + 30-day slope lookback
    total_days = len(benchmark_df)

    if total_days < warmup + max_forward:
        print(f"  [ERROR] Not enough data. Have {total_days}, need {warmup + max_forward}+.")
        return pd.DataFrame()

    start_idx = warmup
    end_idx = total_days - max_forward
    test_indices = list(range(start_idx, end_idx, test_frequency))

    metadata = get_ticker_metadata(cfg)
    ind_cfg = get_indicator_config(cfg)

    # RS periods for multi-timeframe analysis
    rs_periods = {"21d": 21, "63d": 63, "126d": 126}

    print(f"  Analyzing {len(test_indices)} dates across {len(data) - 1} tickers...")
    print(f"  Testing {len(ALL_INDICATOR_NAMES)} indicators per ticker per date")
    print(f"  Window: {benchmark_df.index[start_idx].strftime('%Y-%m-%d')} -> {benchmark_df.index[end_idx].strftime('%Y-%m-%d')}")
    print()

    all_rows = []

    for count, idx in enumerate(test_indices, 1):
        if count % 10 == 0:
            print(f"  [{count}/{len(test_indices)}] {benchmark_df.index[idx].strftime('%Y-%m-%d')}...")

        bench_slice = benchmark_df.iloc[:idx + 1]

        # Compute RS values at multiple timeframes for percentile ranking
        rs_by_period = {}
        for label, period in rs_periods.items():
            rs_by_period[label] = _compute_rs_at_period(
                data, benchmark_ticker, bench_slice, idx, period
            )

        all_rs_values_63d = list(rs_by_period["63d"].values())

        for ticker, full_df in data.items():
            if ticker == benchmark_ticker:
                continue

            df = full_df.iloc[:idx + 1]
            if len(df) < warmup:
                continue

            # Compute multi-timeframe RS percentiles for this ticker
            multi_tf_rs = {}
            for label in rs_periods:
                rs_val = rs_by_period[label].get(ticker, 0)
                all_vals = list(rs_by_period[label].values())
                multi_tf_rs[f"rs_{label}_pctl"] = _percentile_rank(rs_val, all_vals)

            # Compute all 22 indicators
            indicators = compute_all_expanded(
                df, bench_slice, cfg,
                all_rs_values=all_rs_values_63d,
                multi_tf_rs=multi_tf_rs,
            )

            # Get forward returns
            target_date = benchmark_df.index[idx]
            ticker_indices = full_df.index.get_indexer([target_date], method="nearest")
            ticker_idx = ticker_indices[0]

            fwd = {}
            entry_price = full_df["Close"].iloc[ticker_idx]
            for w in FORWARD_WINDOWS:
                future_idx = ticker_idx + w
                if future_idx < len(full_df):
                    future_price = full_df["Close"].iloc[future_idx]
                    fwd[w] = round((future_price - entry_price) / entry_price, 4)
                else:
                    fwd[w] = None

            meta = metadata.get(ticker, {})
            row = {
                "date": benchmark_df.index[idx].strftime("%Y-%m-%d"),
                "ticker": ticker,
                "sector": meta.get("sector_name", ""),
            }

            # Record each indicator's triggered state
            for ind_name in ALL_INDICATOR_NAMES:
                if ind_name in indicators:
                    row[ind_name] = indicators[ind_name].get("triggered", False)
                else:
                    row[ind_name] = False

            # Record forward returns
            for w in FORWARD_WINDOWS:
                row[f"fwd_{w}d"] = fwd.get(w)

            all_rows.append(row)

    return pd.DataFrame(all_rows)


def analyze_individual(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze each indicator individually and rank by 63-day edge."""

    print(f"\n{'='*90}")
    print(f"  ALL {len(ALL_INDICATOR_NAMES)} INDICATORS — INDIVIDUAL ANALYSIS")
    print(f"{'='*90}")

    summary_rows = []

    for ind_name in ALL_INDICATOR_NAMES:
        label = INDICATOR_LABELS[ind_name]
        fired = df[df[ind_name] == True]
        not_fired = df[df[ind_name] == False]

        fire_rate = len(fired) / len(df) * 100 if len(df) > 0 else 0

        row_data = {
            "indicator": label,
            "ind_key": ind_name,
            "fire_rate": fire_rate,
            "fired_count": len(fired),
        }

        for w in FORWARD_WINDOWS:
            col = f"fwd_{w}d"
            fired_valid = fired[col].dropna()
            not_fired_valid = not_fired[col].dropna()

            if len(fired_valid) > 0 and len(not_fired_valid) > 0:
                row_data[f"fired_wr_{w}d"] = (fired_valid > 0).mean() * 100
                row_data[f"fired_avg_{w}d"] = fired_valid.mean() * 100
                row_data[f"nf_wr_{w}d"] = (not_fired_valid > 0).mean() * 100
                row_data[f"nf_avg_{w}d"] = not_fired_valid.mean() * 100
                row_data[f"edge_{w}d"] = row_data[f"fired_avg_{w}d"] - row_data[f"nf_avg_{w}d"]
            else:
                row_data[f"fired_wr_{w}d"] = 0
                row_data[f"fired_avg_{w}d"] = 0
                row_data[f"nf_avg_{w}d"] = 0
                row_data[f"edge_{w}d"] = 0

        summary_rows.append(row_data)

    summary_df = pd.DataFrame(summary_rows)

    # Print detailed results per indicator
    for _, row in summary_df.iterrows():
        print(f"\n  {'─'*80}")
        print(f"  {row['indicator']}")
        print(f"  Fired: {row['fired_count']:.0f} times ({row['fire_rate']:.1f}%)")
        print(f"  {'─'*80}")
        print(f"  {'Window':<12} {'Fired→Win%':>12} {'Fired→Avg':>12} {'NotFired→Avg':>14} {'Edge':>10}")

        for w in FORWARD_WINDOWS:
            print(f"  {w:>2}-day      {row[f'fired_wr_{w}d']:>10.1f}%  {row[f'fired_avg_{w}d']:>+10.2f}%  {row[f'nf_avg_{w}d']:>+12.2f}%  {row[f'edge_{w}d']:>+8.2f}%")

    # --- RANKING ---
    print(f"\n\n{'='*90}")
    print(f"  INDICATOR RANKING BY 63-DAY PREDICTIVE EDGE")
    print(f"{'='*90}\n")

    ranked = summary_df.sort_values("edge_63d", ascending=False)

    for _, row in ranked.iterrows():
        bar_len = max(0, int(row["edge_63d"] / 1.5))
        bar = "█" * bar_len
        edge_str = f"{row['edge_63d']:>+7.2f}%"
        wr_str = f"{row['fired_wr_63d']:>5.1f}%"
        fr_str = f"{row['fire_rate']:>5.1f}%"
        print(f"  {row['indicator']:<22s}  Edge: {edge_str}  |  Win rate: {wr_str}  |  Fire rate: {fr_str}  {bar}")

    return ranked


def analyze_pairs(df: pd.DataFrame, top_n: int = 20) -> None:
    """Analyze pairs of indicators — show the best and worst combos."""

    print(f"\n\n{'='*90}")
    print(f"  TOP INDICATOR PAIRS (63-DAY FORWARD RETURNS)")
    print(f"{'='*90}\n")

    col = "fwd_63d"
    pair_results = []

    for i, ind1 in enumerate(ALL_INDICATOR_NAMES):
        for ind2 in ALL_INDICATOR_NAMES[i + 1:]:
            both = df[(df[ind1] == True) & (df[ind2] == True)]
            valid = both[col].dropna()

            if len(valid) < 10:
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

    print(f"  {'Pair':<50s} {'Events':>8} {'Win%':>8} {'Avg Ret':>10}")
    print(f"  {'':─<50} {'':─>8} {'':─>8} {'':─>10}")

    # Top pairs
    print(f"\n  TOP {top_n} PAIRS:")
    for p in pair_results[:top_n]:
        print(f"  {p['pair']:<50s} {p['events']:>8} {p['win_rate']:>7.1f}% {p['avg_return']:>+9.2f}%")

    # Bottom pairs
    print(f"\n  BOTTOM 10 PAIRS:")
    for p in pair_results[-10:]:
        print(f"  {p['pair']:<50s} {p['events']:>8} {p['win_rate']:>7.1f}% {p['avg_return']:>+9.2f}%")

    if pair_results:
        print(f"\n  Best:  {pair_results[0]['pair']} → {pair_results[0]['avg_return']:+.2f}%")
        print(f"  Worst: {pair_results[-1]['pair']} → {pair_results[-1]['avg_return']:+.2f}%")


def analyze_optimal_stack(df: pd.DataFrame, ranked: pd.DataFrame) -> None:
    """
    Starting from the best indicator, progressively add the next best
    and measure how combined performance changes.
    """

    print(f"\n\n{'='*90}")
    print(f"  BUILDING THE OPTIMAL STACK")
    print(f"  (Adding indicators one at a time, from best to worst)")
    print(f"{'='*90}\n")

    col = "fwd_63d"
    indicator_order = ranked["ind_key"].tolist()

    print(f"  {'Stack':<55s} {'Min Score':>10} {'Events':>8} {'Win%':>8} {'Avg Ret':>10}")
    print(f"  {'':─<55} {'':─>10} {'':─>8} {'':─>8} {'':─>10}")

    for n_indicators in range(1, len(indicator_order) + 1):
        current_stack = indicator_order[:n_indicators]
        stack_label = " + ".join(INDICATOR_LABELS[k] for k in current_stack[:3])
        if n_indicators > 3:
            stack_label += f" + {n_indicators - 3} more"

        # Compute score as count of triggered indicators in current stack
        df["stack_score"] = df[current_stack].sum(axis=1)

        # Try requiring at least 2 from the stack, or 1 if stack is size 1
        min_score = min(2, n_indicators)

        subset = df[df["stack_score"] >= min_score][col].dropna()

        if len(subset) < 10:
            continue

        wr = (subset > 0).mean() * 100
        avg = subset.mean() * 100

        marker = ""
        if n_indicators <= 6:
            marker = " ◀" if avg == max(
                df[df[indicator_order[:i + 1]].sum(axis=1) >= min(2, i + 1)][col].dropna().mean() * 100
                for i in range(n_indicators)
                if len(df[df[indicator_order[:i + 1]].sum(axis=1) >= min(2, i + 1)][col].dropna()) >= 10
            ) else ""

        print(f"  {stack_label:<55s} {min_score:>10} {len(subset):>8} {wr:>7.1f}% {avg:>+9.2f}%")

    df.drop(columns=["stack_score"], inplace=True, errors="ignore")

    # --- Also test: require top 3 indicators specifically ---
    print(f"\n  {'─'*80}")
    print(f"  FOCUSED TEST: Top 3 indicators only (Require 2+ of 3)")
    print(f"  {'─'*80}")

    top3 = indicator_order[:3]
    top3_label = " + ".join(INDICATOR_LABELS[k] for k in top3)
    print(f"  Stack: {top3_label}\n")

    for min_req in [1, 2, 3]:
        df["top3_score"] = df[top3].sum(axis=1)
        subset = df[df["top3_score"] >= min_req][col].dropna()

        if len(subset) < 10:
            continue

        wr = (subset > 0).mean() * 100
        avg = subset.mean() * 100

        print(f"    Require {min_req}/3:  Events: {len(subset):>6}  |  Win rate: {wr:.1f}%  |  Avg return: {avg:+.2f}%")

    df.drop(columns=["top3_score"], inplace=True, errors="ignore")

    # --- Top 5 test ---
    print(f"\n  {'─'*80}")
    print(f"  FOCUSED TEST: Top 5 indicators (Require N+ of 5)")
    print(f"  {'─'*80}")

    top5 = indicator_order[:5]
    top5_label = " + ".join(INDICATOR_LABELS[k] for k in top5)
    print(f"  Stack: {top5_label}\n")

    for min_req in [2, 3, 4, 5]:
        df["top5_score"] = df[top5].sum(axis=1)
        subset = df[df["top5_score"] >= min_req][col].dropna()

        if len(subset) < 10:
            continue

        wr = (subset > 0).mean() * 100
        avg = subset.mean() * 100

        print(f"    Require {min_req}/5:  Events: {len(subset):>6}  |  Win rate: {wr:.1f}%  |  Avg return: {avg:+.2f}%")

    df.drop(columns=["top5_score"], inplace=True, errors="ignore")

    # --- Top 8 test ---
    print(f"\n  {'─'*80}")
    print(f"  FOCUSED TEST: Top 8 indicators (Require N+ of 8)")
    print(f"  {'─'*80}")

    top8 = indicator_order[:8]
    top8_label = " + ".join(INDICATOR_LABELS[k] for k in top8[:4]) + f" + {4} more"
    print(f"  Stack: {top8_label}\n")

    for min_req in [3, 4, 5, 6]:
        df["top8_score"] = df[top8].sum(axis=1)
        subset = df[df["top8_score"] >= min_req][col].dropna()

        if len(subset) < 10:
            continue

        wr = (subset > 0).mean() * 100
        avg = subset.mean() * 100

        print(f"    Require {min_req}/8:  Events: {len(subset):>6}  |  Win rate: {wr:.1f}%  |  Avg return: {avg:+.2f}%")

    df.drop(columns=["top8_score"], inplace=True, errors="ignore")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    cfg = load_config()

    print("=" * 90)
    print("  FULL INDICATOR ANALYSIS — 22 INDICATORS")
    print("=" * 90)
    print()
    print("  Testing all 22 indicators across the full ticker universe.")
    print("  This will take 15-25 minutes.")
    print()

    # Fetch data
    print("  Fetching 3 years of data...\n")
    all_tickers = ["SPY"] + get_all_tickers(cfg)
    data = fetch_batch(all_tickers, period="3y", verbose=True)

    if not data:
        print("  [ERROR] No data fetched.")
        exit(1)

    # Collect events
    print("\n  Collecting all indicator events...\n")
    events_df = collect_all_indicator_events(data, cfg, test_frequency=5)

    if events_df.empty:
        print("  [ERROR] No events collected.")
        exit(1)

    print(f"\n  Collected {len(events_df)} observations.\n")

    # Run analyses
    ranked = analyze_individual(events_df)
    analyze_pairs(events_df)
    analyze_optimal_stack(events_df, ranked)

    print(f"\n{'='*90}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'='*90}\n")
