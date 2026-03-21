from __future__ import annotations

"""
backtester.py — Validates breakout signals against historical outcomes.

Usage:
    from backtester import run_backtest, backtest_summary

    # Run a backtest over the past 6 months
    results = run_backtest(cfg, period="2y", lookback_days=126)

    # Print summary statistics
    backtest_summary(results)

The backtester asks: "On each historical date, what did our system flag?
And what happened to those stocks over the next N days?"

This lets you validate whether high-scoring stocks actually broke out.
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from config import (
    load_config, get_all_tickers, get_indicator_config,
    get_scoring_config, get_ticker_metadata,
)
from indicators import compute_all_indicators, score_ticker


# =============================================================
# FORWARD RETURN: What happened after a signal?
# =============================================================
def compute_forward_returns(
    df: pd.DataFrame,
    signal_date_idx: int,
    windows: list[int] = None,
) -> dict:
    """
    Compute forward returns from a given index position in the DataFrame.

    Args:
        df:              Price DataFrame
        signal_date_idx: Integer index of the signal date
        windows:         List of forward-looking periods in trading days

    Returns:
        {
            5: 0.032,    # +3.2% after 5 days
            10: 0.058,   # +5.8% after 10 days
            21: -0.012,  # -1.2% after 21 days
        }
    """
    if windows is None:
        windows = [10, 21, 42, 63]  # ~2wk, 1mo, 2mo, 3mo

    entry_price = df["Close"].iloc[signal_date_idx]
    returns = {}

    for w in windows:
        future_idx = signal_date_idx + w
        if future_idx < len(df):
            future_price = df["Close"].iloc[future_idx]
            returns[w] = round((future_price - entry_price) / entry_price, 4)
        else:
            returns[w] = None  # not enough future data

    return returns


# =============================================================
# SINGLE-DATE SCORING: Score all tickers as of a specific date
# =============================================================
def score_as_of_date(
    data: dict[str, pd.DataFrame],
    cfg: dict,
    as_of_idx: int,
) -> list[dict]:
    """
    Run the indicator engine on all tickers using data up to (and including)
    the given index position. This simulates what the system would have
    flagged on that historical date.

    Args:
        data:       dict of {ticker: full DataFrame}
        cfg:        config dict
        as_of_idx:  integer index — score using data[:as_of_idx+1]

    Returns:
        List of scored ticker dicts (same format as indicators.score_all)
    """
    benchmark_ticker = cfg["benchmark"]["ticker"]
    benchmark_full = data.get(benchmark_ticker)
    if benchmark_full is None:
        return []

    benchmark_df = benchmark_full.iloc[:as_of_idx + 1]
    metadata = get_ticker_metadata(cfg)
    ind_cfg = get_indicator_config(cfg)

    # --- Pass 1: RS values for percentile ranking ---
    rs_period = ind_cfg["relative_strength"]["period"]
    raw_rs = {}
    for ticker, full_df in data.items():
        if ticker == benchmark_ticker:
            continue
        df = full_df.iloc[:as_of_idx + 1]
        if len(df) < rs_period + 1 or len(benchmark_df) < rs_period + 1:
            continue
        stock_ret = (df["Close"].iloc[-1] / df["Close"].iloc[-rs_period - 1]) - 1
        bench_ret = (benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[-rs_period - 1]) - 1
        if bench_ret != 0:
            raw_rs[ticker] = stock_ret / bench_ret if bench_ret > 0 else stock_ret - bench_ret
        else:
            raw_rs[ticker] = 0

    all_rs_values = list(raw_rs.values())

    # --- Pass 2: Score each ticker ---
    results = []
    for ticker, full_df in data.items():
        if ticker == benchmark_ticker:
            continue

        df = full_df.iloc[:as_of_idx + 1]
        meta = metadata.get(ticker, {})
        indicators = compute_all_indicators(df, benchmark_df, cfg, all_rs_values=all_rs_values)
        scoring = score_ticker(indicators)

        results.append({
            "ticker": ticker,
            "name": meta.get("name", ""),
            "sector": meta.get("sector_name", ""),
            "subsector": meta.get("subsector_name", ""),
            "score": scoring["score"],
            "signals": scoring["signals"],
            "indicators": indicators,
        })

    return results


# =============================================================
# MAIN BACKTEST: Run signals across multiple historical dates
# =============================================================
def run_backtest(
    cfg: dict,
    data: dict[str, pd.DataFrame] = None,
    period: str = "2y",
    test_frequency: int = 5,
    min_score: int = 3,
    forward_windows: list[int] = None,
    verbose: bool = True,
) -> list[dict]:
    """
    Walk through historical dates, score all tickers, and measure
    forward returns for high-scoring stocks.

    Args:
        cfg:             Config dict
        data:            Pre-fetched data (if None, fetches fresh)
        period:          How much history to fetch (needs enough for
                         200-day MA + forward returns)
        test_frequency:  Score every N trading days (5 = weekly)
        min_score:       Only track forward returns for scores >= this
        forward_windows: Days to measure forward performance [5, 10, 21]

    Returns:
        List of signal events:
        {
            "date": "2025-06-15",
            "ticker": "NVDA",
            "score": 4,
            "signals": [...],
            "forward_returns": {5: 0.03, 10: 0.05, 21: 0.08},
        }
    """
    if forward_windows is None:
        forward_windows = [10, 21, 42, 63]  # ~2wk, 1mo, 2mo, 3mo

    # --- Fetch data if not provided ---
    if data is None:
        from data_fetcher import fetch_all
        if verbose:
            print(f"  Fetching {period} of data for all tickers...\n")
        data = fetch_all(cfg, period=period, verbose=verbose)

    if not data:
        print("  [ERROR] No data fetched.")
        return []

    # --- Use the benchmark's index as our date spine ---
    benchmark_ticker = cfg["benchmark"]["ticker"]
    benchmark_df = data.get(benchmark_ticker)
    if benchmark_df is None:
        print(f"  [ERROR] Benchmark {benchmark_ticker} not in data.")
        return []

    # We need at least 200 days of warmup for the MA indicator,
    # plus forward window for measuring outcomes
    max_forward = max(forward_windows)
    warmup = 220  # slightly more than 200 to be safe
    total_days = len(benchmark_df)

    if total_days < warmup + max_forward:
        print(f"  [ERROR] Not enough data. Have {total_days} days, need {warmup + max_forward}+.")
        print(f"  Try fetching with period='2y' or longer.")
        return []

    # Determine test dates (every N-th trading day after warmup)
    start_idx = warmup
    end_idx = total_days - max_forward  # leave room for forward returns
    test_indices = list(range(start_idx, end_idx, test_frequency))

    if verbose:
        start_date = benchmark_df.index[start_idx].strftime("%Y-%m-%d")
        end_date = benchmark_df.index[end_idx].strftime("%Y-%m-%d")
        print(f"\n  Backtest window: {start_date} -> {end_date}")
        print(f"  Testing {len(test_indices)} dates (every {test_frequency} trading days)")
        print(f"  Minimum score to track: {min_score}+")
        print(f"  Forward return windows: {forward_windows} days")
        print()

    # --- Walk through each test date ---
    all_events = []
    for count, idx in enumerate(test_indices, 1):
        date_str = benchmark_df.index[idx].strftime("%Y-%m-%d")

        if verbose and count % 10 == 0:
            print(f"  [{count}/{len(test_indices)}] Scoring as of {date_str}...")

        scores = score_as_of_date(data, cfg, as_of_idx=idx)

        for s in scores:
            if s["score"] >= min_score:
                # Compute forward returns for this ticker from this date
                ticker_df = data.get(s["ticker"])
                if ticker_df is None:
                    continue

                # Find the corresponding index in this ticker's DataFrame
                # (may differ from benchmark index due to crypto/metals trading days)
                target_date = benchmark_df.index[idx]
                ticker_indices = ticker_df.index.get_indexer([target_date], method="nearest")
                ticker_idx = ticker_indices[0]

                fwd_returns = compute_forward_returns(
                    ticker_df, ticker_idx, windows=forward_windows
                )

                all_events.append({
                    "date": date_str,
                    "ticker": s["ticker"],
                    "name": s["name"],
                    "sector": s["sector"],
                    "subsector": s["subsector"],
                    "score": s["score"],
                    "signals": s["signals"],
                    "forward_returns": fwd_returns,
                })

    if verbose:
        print(f"\n  Done. {len(all_events)} signal events recorded.\n")

    return all_events


# =============================================================
# ANALYSIS: Summarize backtest results
# =============================================================
def backtest_summary(events: list[dict]) -> None:
    """Print summary statistics from backtest results."""
    if not events:
        print("  No signal events to summarize.")
        return

    df = pd.DataFrame(events)

    # Extract forward returns into columns
    windows = sorted(events[0]["forward_returns"].keys())
    for w in windows:
        df[f"fwd_{w}d"] = df["forward_returns"].apply(lambda x: x.get(w))

    print(f"\n{'='*80}")
    print(f"  BACKTEST SUMMARY")
    print(f"{'='*80}")
    print(f"  Total signal events: {len(df)}")
    print(f"  Unique tickers flagged: {df['ticker'].nunique()}")
    print(f"  Date range: {df['date'].min()} -> {df['date'].max()}")

    # --- Overall forward return stats ---
    print(f"\n  {'─'*60}")
    print(f"  FORWARD RETURNS (all signals, score >= min threshold)")
    print(f"  {'─'*60}")

    for w in windows:
        col = f"fwd_{w}d"
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        win_rate = (valid > 0).mean() * 100
        avg_ret = valid.mean() * 100
        median_ret = valid.median() * 100
        print(f"    {w:>2}-day:  Win rate: {win_rate:.1f}%  |  Avg: {avg_ret:+.2f}%  |  Median: {median_ret:+.2f}%  |  N={len(valid)}")

    # --- By score level ---
    print(f"\n  {'─'*60}")
    print(f"  FORWARD RETURNS BY SCORE")
    print(f"  {'─'*60}")

    for score in sorted(df["score"].unique(), reverse=True):
        subset = df[df["score"] == score]
        print(f"\n    Score = {score}  ({len(subset)} events)")
        for w in windows:
            col = f"fwd_{w}d"
            valid = subset[col].dropna()
            if len(valid) == 0:
                continue
            win_rate = (valid > 0).mean() * 100
            avg_ret = valid.mean() * 100
            print(f"      {w:>2}-day:  Win rate: {win_rate:.1f}%  |  Avg: {avg_ret:+.2f}%  |  N={len(valid)}")

    # --- By sector ---
    print(f"\n  {'─'*60}")
    print(f"  SIGNALS BY SECTOR")
    print(f"  {'─'*60}")

    sector_counts = df.groupby("sector").agg(
        events=("ticker", "count"),
        unique_tickers=("ticker", "nunique"),
        avg_score=("score", "mean"),
    ).sort_values("events", ascending=False)

    for sector, row in sector_counts.iterrows():
        print(f"    {sector[:35]:35s}  Events: {row['events']:>4}  |  Tickers: {row['unique_tickers']:>3}  |  Avg score: {row['avg_score']:.1f}")

    # --- Top tickers by frequency ---
    print(f"\n  {'─'*60}")
    print(f"  MOST FREQUENTLY FLAGGED TICKERS")
    print(f"  {'─'*60}")

    top_tickers = df.groupby(["ticker", "name"]).agg(
        times_flagged=("date", "count"),
        avg_score=("score", "mean"),
    ).sort_values("times_flagged", ascending=False).head(15)

    for (ticker, name), row in top_tickers.iterrows():
        fwd_col = f"fwd_{windows[-1]}d"
        ticker_events = df[df["ticker"] == ticker][fwd_col].dropna()
        avg_fwd = ticker_events.mean() * 100 if len(ticker_events) > 0 else 0
        print(f"    {ticker:8s} {name[:25]:25s}  Flagged: {row['times_flagged']:>3}x  |  Avg score: {row['avg_score']:.1f}  |  Avg {windows[-1]}d return: {avg_fwd:+.2f}%")

    print(f"\n{'='*80}\n")


# =============================================================
# Quick test
# =============================================================
if __name__ == "__main__":
    cfg = load_config()

    print("=" * 80)
    print("  BACKTESTER — SMOKE TEST (small ticker set)")
    print("=" * 80)
    print()
    print("  This fetches 2 years of data for a small set of tickers")
    print("  and runs the backtest. Full universe takes longer.\n")

    # Small test set — use 2 years for enough warmup + forward window
    from data_fetcher import fetch_batch
    test_tickers = [
        "SPY",                          # benchmark
        "NVDA", "AMD", "AVGO", "MU",    # AI chips
        "GLD", "NEM", "AG",             # metals
        "BTC-USD", "ETH-USD",           # crypto
    ]

    print("  Fetching data...\n")
    data = fetch_batch(test_tickers, period="2y", verbose=True)

    if data:
        events = run_backtest(
            cfg,
            data=data,
            test_frequency=5,   # score weekly
            min_score=2,        # lower threshold for small test set
            forward_windows=[10, 21, 42, 63],  # ~2wk, 1mo, 2mo, 3mo
            verbose=True,
        )

        if events:
            backtest_summary(events)
        else:
            print("  No signals found. Try lowering min_score.")
