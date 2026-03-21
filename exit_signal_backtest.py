from __future__ import annotations

"""
exit_signal_backtest.py — Comprehensive backtest of exit strategies.

Tests 10+ exit strategies against historical data to find the optimal
approach for maximizing returns when riding subsector breakout waves.

Entry: Buy when ticker scores >= 7 (breakout signal)
Exit:  Various strategies tested head-to-head

Usage:
    python3 exit_signal_backtest.py
"""

import pandas as pd
import numpy as np
from datetime import timedelta
from config import load_config, get_all_tickers, get_indicator_config, get_ticker_metadata
from indicators import compute_all_indicators, score_ticker, check_cmf, check_ichimoku
from data_fetcher import fetch_all
from backtester import score_as_of_date


# =============================================================
# EXIT STRATEGY DEFINITIONS
# =============================================================

def exit_score_below(scores_series, entry_idx, threshold, max_hold):
    """Exit when composite score drops below threshold."""
    for day in range(1, max_hold + 1):
        idx = entry_idx + day
        if idx >= len(scores_series):
            return max_hold  # hold to end
        if scores_series[idx] < threshold:
            return day
    return max_hold


def exit_score_drop(scores_series, entry_idx, drop_amount, max_hold):
    """Exit when score drops by X points from entry score."""
    entry_score = scores_series[entry_idx]
    for day in range(1, max_hold + 1):
        idx = entry_idx + day
        if idx >= len(scores_series):
            return max_hold
        if scores_series[idx] < entry_score - drop_amount:
            return day
    return max_hold


def exit_price_below_sma(df, entry_idx, sma_period, max_hold):
    """Exit when price closes below N-day SMA."""
    close = df["Close"].values
    for day in range(1, max_hold + 1):
        idx = entry_idx + day
        if idx >= len(close) or idx < sma_period:
            return max_hold
        sma = np.mean(close[idx - sma_period:idx])
        if close[idx] < sma:
            return day
    return max_hold


def exit_ichimoku_break(df, entry_idx, max_hold):
    """Exit when price drops below Ichimoku cloud."""
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    tenkan = 9
    kijun = 26

    for day in range(1, max_hold + 1):
        idx = entry_idx + day
        if idx >= len(close) or idx < 52 + kijun:
            continue

        # Compute cloud at this point
        h_slice = high[:idx + 1]
        l_slice = low[:idx + 1]

        tenkan_val = (pd.Series(h_slice).rolling(tenkan).max().iloc[-1] +
                      pd.Series(l_slice).rolling(tenkan).min().iloc[-1]) / 2
        kijun_val = (pd.Series(h_slice).rolling(kijun).max().iloc[-1] +
                     pd.Series(l_slice).rolling(kijun).min().iloc[-1]) / 2
        senkou_a = (tenkan_val + kijun_val) / 2
        senkou_b = (pd.Series(h_slice).rolling(52).max().iloc[-1] +
                    pd.Series(l_slice).rolling(52).min().iloc[-1]) / 2

        cloud_top = max(senkou_a, senkou_b)

        if close[idx] < cloud_top:
            return day
    return max_hold


def exit_cmf_negative(df, entry_idx, max_hold, period=20):
    """Exit when CMF turns negative (distribution)."""
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values
    volume = df["Volume"].values

    for day in range(1, max_hold + 1):
        idx = entry_idx + day
        if idx >= len(close) or idx < period:
            continue

        # Compute CMF
        h = high[idx - period:idx + 1]
        l = low[idx - period:idx + 1]
        c = close[idx - period:idx + 1]
        v = volume[idx - period:idx + 1]

        hl_range = h - l
        hl_range[hl_range == 0] = np.nan
        mf_mult = ((c - l) - (h - c)) / hl_range
        mf_mult = np.nan_to_num(mf_mult)
        mf_vol = mf_mult * v
        vol_sum = np.sum(v)
        if vol_sum > 0:
            cmf = np.sum(mf_vol) / vol_sum
        else:
            cmf = 0

        if cmf < 0:
            return day
    return max_hold


def exit_trailing_stop_atr(df, entry_idx, atr_multiplier, max_hold, atr_period=14):
    """Exit when price drops X * ATR from highest close since entry."""
    close = df["Close"].values
    high = df["High"].values
    low = df["Low"].values

    if entry_idx < atr_period + 1:
        return max_hold

    highest_close = close[entry_idx]

    for day in range(1, max_hold + 1):
        idx = entry_idx + day
        if idx >= len(close):
            return max_hold

        # Update highest close
        highest_close = max(highest_close, close[idx])

        # Compute ATR at current point
        tr_vals = []
        for i in range(max(1, idx - atr_period), idx + 1):
            tr = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
            tr_vals.append(tr)
        atr = np.mean(tr_vals)

        # Check if price dropped enough from peak
        if close[idx] < highest_close - (atr_multiplier * atr):
            return day
    return max_hold


def exit_fixed_stop_loss(df, entry_idx, stop_pct, max_hold):
    """Exit when price drops X% from entry."""
    entry_price = df["Close"].values[entry_idx]
    stop_price = entry_price * (1 - stop_pct)

    for day in range(1, max_hold + 1):
        idx = entry_idx + day
        if idx >= len(df):
            return max_hold
        if df["Close"].values[idx] < stop_price:
            return day
    return max_hold


def exit_time_based(entry_idx, hold_days, max_hold):
    """Fixed hold period exit."""
    return min(hold_days, max_hold)


def exit_rs_breakdown(df, benchmark_df, entry_idx, max_hold, rs_period=21):
    """Exit when short-term RS vs SPY turns negative."""
    close = df["Close"].values
    bench_close = benchmark_df["Close"].values

    for day in range(1, max_hold + 1):
        idx = entry_idx + day
        if idx >= len(close) or idx >= len(bench_close) or idx < rs_period + 1:
            continue

        stock_ret = (close[idx] / close[idx - rs_period]) - 1
        bench_ret = (bench_close[idx] / bench_close[idx - rs_period]) - 1

        # Exit if stock is underperforming SPY over last N days
        if stock_ret < bench_ret:
            return day
    return max_hold


# =============================================================
# MAIN BACKTEST ENGINE
# =============================================================
def run_exit_backtest(
    cfg: dict,
    data: dict = None,
    entry_min_score: float = 7.0,
    test_frequency: int = 5,
    max_hold: int = 63,
    verbose: bool = True,
):
    """
    For each historical entry signal (score >= entry_min_score),
    simulate holding under each exit strategy and compare results.
    """
    if data is None:
        if verbose:
            print("  Fetching 2y of data for all tickers...\n")
        data = fetch_all(cfg, period="2y", verbose=verbose)

    if not data:
        print("  [ERROR] No data fetched.")
        return {}

    benchmark_ticker = cfg["benchmark"]["ticker"]
    benchmark_df = data.get(benchmark_ticker)
    if benchmark_df is None:
        print(f"  [ERROR] Benchmark {benchmark_ticker} not in data.")
        return {}

    # We need warmup + max_hold room
    warmup = 220
    total_days = len(benchmark_df)
    if total_days < warmup + max_hold:
        print(f"  [ERROR] Not enough data ({total_days} days).")
        return {}

    start_idx = warmup
    end_idx = total_days - max_hold
    test_indices = list(range(start_idx, end_idx, test_frequency))

    # Define all exit strategies to test
    strategies = {
        # Score-based exits
        "Score < 5":        lambda df, bench_df, idx, scores: exit_score_below(scores, 0, 5, max_hold),
        "Score < 6":        lambda df, bench_df, idx, scores: exit_score_below(scores, 0, 6, max_hold),
        "Score drops 2pts": lambda df, bench_df, idx, scores: exit_score_drop(scores, 0, 2.0, max_hold),
        "Score drops 3pts": lambda df, bench_df, idx, scores: exit_score_drop(scores, 0, 3.0, max_hold),

        # Price/technical exits
        "Price < 20 SMA":   lambda df, bench_df, idx, scores: exit_price_below_sma(df, idx, 20, max_hold),
        "Price < 50 SMA":   lambda df, bench_df, idx, scores: exit_price_below_sma(df, idx, 50, max_hold),
        "Ichimoku break":   lambda df, bench_df, idx, scores: exit_ichimoku_break(df, idx, max_hold),
        "CMF negative":     lambda df, bench_df, idx, scores: exit_cmf_negative(df, idx, max_hold),

        # Stop losses
        "5% stop loss":     lambda df, bench_df, idx, scores: exit_fixed_stop_loss(df, idx, 0.05, max_hold),
        "10% stop loss":    lambda df, bench_df, idx, scores: exit_fixed_stop_loss(df, idx, 0.10, max_hold),
        "15% stop loss":    lambda df, bench_df, idx, scores: exit_fixed_stop_loss(df, idx, 0.15, max_hold),

        # Trailing stops
        "2x ATR trail":     lambda df, bench_df, idx, scores: exit_trailing_stop_atr(df, idx, 2.0, max_hold),
        "3x ATR trail":     lambda df, bench_df, idx, scores: exit_trailing_stop_atr(df, idx, 3.0, max_hold),

        # RS-based exits
        "RS < SPY (21d)":   lambda df, bench_df, idx, scores: exit_rs_breakdown(df, bench_df, idx, max_hold, 21),
        "RS < SPY (10d)":   lambda df, bench_df, idx, scores: exit_rs_breakdown(df, bench_df, idx, max_hold, 10),

        # Time-based (baselines)
        "Hold 10 days":     lambda df, bench_df, idx, scores: exit_time_based(idx, 10, max_hold),
        "Hold 21 days":     lambda df, bench_df, idx, scores: exit_time_based(idx, 21, max_hold),
        "Hold 42 days":     lambda df, bench_df, idx, scores: exit_time_based(idx, 42, max_hold),
        "Hold 63 days":     lambda df, bench_df, idx, scores: exit_time_based(idx, 63, max_hold),
    }

    # Results storage: strategy -> list of trade outcomes
    results = {name: [] for name in strategies}
    # Also track a "Buy & hold 63d" baseline for every entry
    all_entries = []

    if verbose:
        start_date = benchmark_df.index[start_idx].strftime("%Y-%m-%d")
        end_date = benchmark_df.index[end_idx].strftime("%Y-%m-%d")
        print(f"\n{'='*80}")
        print(f"  EXIT SIGNAL BACKTEST")
        print(f"{'='*80}")
        print(f"  Entry rule: Score >= {entry_min_score}")
        print(f"  Max hold: {max_hold} trading days")
        print(f"  Test window: {start_date} -> {end_date}")
        print(f"  Testing {len(test_indices)} dates (every {test_frequency} trading days)")
        print(f"  Strategies: {len(strategies)}")
        print()

    # Pre-compute forward scores for score-based exits
    # We need to score tickers at future dates too, which is expensive
    # Instead, we'll compute daily scores for each ticker across the test window
    if verbose:
        print("  Phase 1: Scoring all tickers across test dates...")

    # Build a score time series for each ticker: {ticker: {date_idx: score}}
    ticker_score_series = {}
    all_score_dates = set()

    for count, idx in enumerate(test_indices, 1):
        if verbose and count % 10 == 0:
            print(f"    [{count}/{len(test_indices)}] Scoring as of {benchmark_df.index[idx].strftime('%Y-%m-%d')}...")

        scores = score_as_of_date(data, cfg, as_of_idx=idx)
        all_score_dates.add(idx)

        for s in scores:
            ticker = s["ticker"]
            if ticker not in ticker_score_series:
                ticker_score_series[ticker] = {}
            ticker_score_series[ticker][idx] = s["score"]

    # Also score at intermediate dates for score-based exits
    # (we need daily-ish scores between test dates)
    # Score every day between test dates for accuracy on score-based exits
    if verbose:
        print("\n  Phase 2: Computing daily scores for score-based exits...")

    # For efficiency, only compute daily scores for tickers that had entries
    entry_tickers = set()
    for idx in test_indices:
        for ticker, score_dict in ticker_score_series.items():
            if score_dict.get(idx, 0) >= entry_min_score:
                entry_tickers.add(ticker)

    if verbose:
        print(f"    {len(entry_tickers)} tickers had score >= {entry_min_score} at some point")

    # Fill in daily scores for entry tickers across the full range
    daily_score_indices = list(range(start_idx, total_days, 1))  # every day
    daily_count = 0
    for count, idx in enumerate(daily_score_indices, 1):
        if idx in all_score_dates:
            continue  # already scored
        if verbose and count % 50 == 0:
            print(f"    [{count}/{len(daily_score_indices)}] Day {benchmark_df.index[idx].strftime('%Y-%m-%d')}...")

        scores = score_as_of_date(data, cfg, as_of_idx=idx)
        daily_count += 1

        for s in scores:
            if s["ticker"] in entry_tickers:
                if s["ticker"] not in ticker_score_series:
                    ticker_score_series[s["ticker"]] = {}
                ticker_score_series[s["ticker"]][idx] = s["score"]

    if verbose:
        print(f"    Computed {daily_count} additional daily scoring passes")

    # Phase 3: For each entry, simulate all exit strategies
    if verbose:
        print(f"\n  Phase 3: Simulating {len(strategies)} exit strategies...\n")

    trade_count = 0
    for idx in test_indices:
        for ticker in entry_tickers:
            score = ticker_score_series.get(ticker, {}).get(idx, 0)
            if score < entry_min_score:
                continue

            ticker_df = data.get(ticker)
            if ticker_df is None:
                continue

            # Map benchmark index to ticker index
            target_date = benchmark_df.index[idx]
            ticker_indices = ticker_df.index.get_indexer([target_date], method="nearest")
            ticker_idx = ticker_indices[0]

            if ticker_idx < 0 or ticker_idx + max_hold >= len(ticker_df):
                continue

            entry_price = ticker_df["Close"].iloc[ticker_idx]

            # Build score array for this ticker from entry forward
            score_array = []
            for d in range(max_hold + 1):
                future_bench_idx = idx + d
                s = ticker_score_series.get(ticker, {}).get(future_bench_idx, None)
                if s is not None:
                    score_array.append(s)
                else:
                    # Interpolate: use last known score
                    score_array.append(score_array[-1] if score_array else score)

            # Map benchmark to ticker for benchmark df alignment
            bench_idx = idx  # benchmark index

            entry_info = {
                "date": benchmark_df.index[idx].strftime("%Y-%m-%d"),
                "ticker": ticker,
                "entry_score": score,
                "entry_price": entry_price,
            }
            all_entries.append(entry_info)

            # Test each exit strategy
            for strat_name, strat_fn in strategies.items():
                try:
                    hold_days = strat_fn(ticker_df, benchmark_df, ticker_idx, score_array)
                except Exception:
                    hold_days = max_hold

                hold_days = min(hold_days, max_hold)
                exit_idx = ticker_idx + hold_days
                if exit_idx >= len(ticker_df):
                    exit_idx = len(ticker_df) - 1
                    hold_days = exit_idx - ticker_idx

                exit_price = ticker_df["Close"].iloc[exit_idx]
                ret = (exit_price - entry_price) / entry_price

                # SPY return over same period
                spy_entry_idx = idx
                spy_exit_idx = min(idx + hold_days, len(benchmark_df) - 1)
                spy_entry_price = benchmark_df["Close"].iloc[spy_entry_idx]
                spy_exit_price = benchmark_df["Close"].iloc[spy_exit_idx]
                spy_ret = (spy_exit_price - spy_entry_price) / spy_entry_price

                results[strat_name].append({
                    "date": entry_info["date"],
                    "ticker": ticker,
                    "entry_score": score,
                    "return": ret,
                    "spy_return": spy_ret,
                    "alpha": ret - spy_ret,
                    "hold_days": hold_days,
                })

            trade_count += 1

    if verbose:
        print(f"  Total trades simulated: {trade_count}")
        print(f"  Total strategy evaluations: {trade_count * len(strategies)}")

    return results


# =============================================================
# ANALYSIS: Compare exit strategies
# =============================================================
def print_exit_comparison(results: dict) -> None:
    """Print head-to-head comparison of all exit strategies."""

    if not results:
        print("  No results to analyze.")
        return

    print(f"\n{'='*100}")
    print(f"  EXIT STRATEGY COMPARISON")
    print(f"{'='*100}")

    # Build summary table
    rows = []
    for name, trades in results.items():
        if not trades:
            continue
        df = pd.DataFrame(trades)
        avg_ret = df["return"].mean() * 100
        avg_alpha = df["alpha"].mean() * 100
        win_rate = (df["return"] > 0).mean() * 100
        alpha_win_rate = (df["alpha"] > 0).mean() * 100
        avg_hold = df["hold_days"].mean()
        median_ret = df["return"].median() * 100
        max_loss = df["return"].min() * 100
        max_gain = df["return"].max() * 100
        sharpe = df["return"].mean() / df["return"].std() if df["return"].std() > 0 else 0
        # Annualize: scale by sqrt(252 / avg_hold)
        ann_sharpe = sharpe * np.sqrt(252 / max(avg_hold, 1))
        n_trades = len(df)

        rows.append({
            "Strategy": name,
            "Avg Return": avg_ret,
            "Avg Alpha": avg_alpha,
            "Win Rate": win_rate,
            "Alpha Win%": alpha_win_rate,
            "Median Ret": median_ret,
            "Avg Hold": avg_hold,
            "Max Loss": max_loss,
            "Max Gain": max_gain,
            "Ann Sharpe": ann_sharpe,
            "N": n_trades,
        })

    summary = pd.DataFrame(rows).sort_values("Avg Alpha", ascending=False)

    # Print main comparison
    print(f"\n  {'Strategy':<20s} {'Avg Ret':>8s} {'Alpha':>8s} {'Win%':>6s} {'αWin%':>6s} {'Median':>8s} {'Hold':>5s} {'MaxLoss':>8s} {'Sharpe':>7s} {'N':>5s}")
    print(f"  {'─'*95}")

    for _, row in summary.iterrows():
        print(f"  {row['Strategy']:<20s} {row['Avg Return']:>+7.2f}% {row['Avg Alpha']:>+7.2f}% {row['Win Rate']:>5.1f}% {row['Alpha Win%']:>5.1f}% {row['Median Ret']:>+7.2f}% {row['Avg Hold']:>5.1f} {row['Max Loss']:>+7.1f}% {row['Ann Sharpe']:>6.2f} {row['N']:>5d}")

    # Group analysis
    print(f"\n\n  {'─'*95}")
    print(f"  STRATEGY GROUPS")
    print(f"  {'─'*95}")

    groups = {
        "Score-based": ["Score < 5", "Score < 6", "Score drops 2pts", "Score drops 3pts"],
        "Technical": ["Price < 20 SMA", "Price < 50 SMA", "Ichimoku break", "CMF negative"],
        "Stop losses": ["5% stop loss", "10% stop loss", "15% stop loss"],
        "Trailing stops": ["2x ATR trail", "3x ATR trail"],
        "RS-based": ["RS < SPY (21d)", "RS < SPY (10d)"],
        "Time-based": ["Hold 10 days", "Hold 21 days", "Hold 42 days", "Hold 63 days"],
    }

    for group_name, strat_names in groups.items():
        group_rows = summary[summary["Strategy"].isin(strat_names)]
        if group_rows.empty:
            continue
        best = group_rows.iloc[0]  # already sorted by alpha
        print(f"\n  {group_name}:")
        print(f"    Best: {best['Strategy']} → Alpha: {best['Avg Alpha']:+.2f}%, Win: {best['Win Rate']:.1f}%, Hold: {best['Avg Hold']:.0f}d")
        for _, row in group_rows.iterrows():
            marker = " ◀ BEST" if row["Strategy"] == best["Strategy"] else ""
            print(f"      {row['Strategy']:<20s}  α={row['Avg Alpha']:+.2f}%  Win={row['Win Rate']:.1f}%  Hold={row['Avg Hold']:.0f}d{marker}")

    # Overall winner
    print(f"\n\n  {'='*95}")
    best_overall = summary.iloc[0]
    print(f"  🏆 BEST EXIT STRATEGY: {best_overall['Strategy']}")
    print(f"     Avg Return: {best_overall['Avg Return']:+.2f}%")
    print(f"     Alpha vs SPY: {best_overall['Avg Alpha']:+.2f}%")
    print(f"     Win Rate: {best_overall['Win Rate']:.1f}%")
    print(f"     Avg Hold Period: {best_overall['Avg Hold']:.0f} days")
    print(f"     Annualized Sharpe: {best_overall['Ann Sharpe']:.2f}")
    print(f"  {'='*95}")

    # Combination analysis
    print(f"\n\n  {'─'*95}")
    print(f"  HYBRID STRATEGIES (combining best from each category)")
    print(f"  {'─'*95}")

    # Test logical combos: take the earlier exit of two strategies
    # We can do this by finding the minimum hold days across strategy pairs
    best_score = summary[summary["Strategy"].isin(["Score < 5", "Score < 6", "Score drops 2pts", "Score drops 3pts"])].iloc[0]["Strategy"]
    best_technical = summary[summary["Strategy"].isin(["Price < 20 SMA", "Price < 50 SMA", "Ichimoku break", "CMF negative"])].iloc[0]["Strategy"]
    best_stop = summary[summary["Strategy"].isin(["5% stop loss", "10% stop loss", "15% stop loss", "2x ATR trail", "3x ATR trail"])].iloc[0]["Strategy"]

    combos = [
        (f"{best_score} + {best_technical}", best_score, best_technical),
        (f"{best_score} + {best_stop}", best_score, best_stop),
        (f"{best_technical} + {best_stop}", best_technical, best_stop),
    ]

    for combo_name, strat_a, strat_b in combos:
        trades_a = {(t["date"], t["ticker"]): t for t in results[strat_a]}
        trades_b = {(t["date"], t["ticker"]): t for t in results[strat_b]}

        combo_trades = []
        for key in trades_a:
            if key not in trades_b:
                continue
            ta = trades_a[key]
            tb = trades_b[key]
            # Take the earlier exit
            if ta["hold_days"] <= tb["hold_days"]:
                combo_trades.append(ta)
            else:
                combo_trades.append(tb)

        if combo_trades:
            cdf = pd.DataFrame(combo_trades)
            avg_ret = cdf["return"].mean() * 100
            avg_alpha = cdf["alpha"].mean() * 100
            win_rate = (cdf["return"] > 0).mean() * 100
            avg_hold = cdf["hold_days"].mean()
            print(f"\n  {combo_name}:")
            print(f"    Avg Return: {avg_ret:+.2f}%  |  Alpha: {avg_alpha:+.2f}%  |  Win: {win_rate:.1f}%  |  Hold: {avg_hold:.0f}d  |  N={len(cdf)}")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    cfg = load_config()

    print("=" * 100)
    print("  EXIT SIGNAL BACKTEST — Finding the optimal exit strategy")
    print("=" * 100)
    print()

    results = run_exit_backtest(
        cfg,
        entry_min_score=7.0,
        test_frequency=5,
        max_hold=63,
        verbose=True,
    )

    if results:
        print_exit_comparison(results)
