from __future__ import annotations

"""
exit_combo_backtest.py — Tests the recommended exit combo:
  - 15% stop loss (crash protection)
  - Score < 5 (breakout over signal)
  - No fixed time limit (hold as long as conditions allow)

Also tests variations to find the optimal parameters.
"""

import pandas as pd
import numpy as np
from config import load_config, get_indicator_config, get_ticker_metadata
from data_fetcher import fetch_all
from backtester import score_as_of_date


def run_combo_backtest(
    cfg: dict,
    data: dict = None,
    entry_min_score: float = 7.0,
    test_frequency: int = 5,
    max_hold: int = 126,       # 6 months cap (just for data bounds)
    verbose: bool = True,
):
    if data is None:
        if verbose:
            print("  Fetching 2y of data...\n")
        data = fetch_all(cfg, period="2y", verbose=verbose)

    if not data:
        print("  [ERROR] No data.")
        return

    benchmark_ticker = cfg["benchmark"]["ticker"]
    benchmark_df = data.get(benchmark_ticker)
    if benchmark_df is None:
        print(f"  [ERROR] No benchmark.")
        return

    warmup = 220
    total_days = len(benchmark_df)
    if total_days < warmup + max_hold:
        print(f"  [ERROR] Not enough data ({total_days} days).")
        return

    start_idx = warmup
    end_idx = total_days - max_hold
    test_indices = list(range(start_idx, end_idx, test_frequency))

    if verbose:
        start_date = benchmark_df.index[start_idx].strftime("%Y-%m-%d")
        end_date = benchmark_df.index[end_idx].strftime("%Y-%m-%d")
        print(f"\n{'='*90}")
        print(f"  EXIT COMBO BACKTEST — Stop Loss + Score Exit (No Time Limit)")
        print(f"{'='*90}")
        print(f"  Entry: Score >= {entry_min_score}")
        print(f"  Max hold cap: {max_hold} trading days (data boundary only)")
        print(f"  Window: {start_date} -> {end_date}")
        print(f"  Test dates: {len(test_indices)} (every {test_frequency} trading days)")
        print()

    # Phase 1: Score all tickers at test dates
    if verbose:
        print("  Phase 1: Scoring at test dates...")

    ticker_score_series = {}
    all_score_dates = set()

    for count, idx in enumerate(test_indices, 1):
        if verbose and count % 10 == 0:
            print(f"    [{count}/{len(test_indices)}] {benchmark_df.index[idx].strftime('%Y-%m-%d')}...")

        scores = score_as_of_date(data, cfg, as_of_idx=idx)
        all_score_dates.add(idx)
        for s in scores:
            if s["ticker"] not in ticker_score_series:
                ticker_score_series[s["ticker"]] = {}
            ticker_score_series[s["ticker"]][idx] = s["score"]

    # Find entry tickers
    entry_tickers = set()
    for idx in test_indices:
        for ticker, scores in ticker_score_series.items():
            if scores.get(idx, 0) >= entry_min_score:
                entry_tickers.add(ticker)

    if verbose:
        print(f"\n  {len(entry_tickers)} tickers had entries")

    # Phase 2: Daily scores for entry tickers
    if verbose:
        print("  Phase 2: Daily scores for entry tickers...")

    daily_indices = list(range(start_idx, total_days, 1))
    daily_count = 0
    for count, idx in enumerate(daily_indices, 1):
        if idx in all_score_dates:
            continue
        if verbose and count % 50 == 0:
            print(f"    [{count}/{len(daily_indices)}] {benchmark_df.index[idx].strftime('%Y-%m-%d')}...")

        scores = score_as_of_date(data, cfg, as_of_idx=idx)
        daily_count += 1
        for s in scores:
            if s["ticker"] in entry_tickers:
                if s["ticker"] not in ticker_score_series:
                    ticker_score_series[s["ticker"]] = {}
                ticker_score_series[s["ticker"]][idx] = s["score"]

    if verbose:
        print(f"    {daily_count} daily scoring passes")

    # Phase 3: Test all combo variations
    if verbose:
        print(f"\n  Phase 3: Simulating exit combos...\n")

    # Define variations to test
    combos = [
        # (name, stop_loss_pct, score_exit_threshold)
        ("15% stop + Score<5 (no cap)",     0.15, 5.0),
        ("15% stop + Score<4 (no cap)",     0.15, 4.0),
        ("15% stop + Score<6 (no cap)",     0.15, 6.0),
        ("10% stop + Score<5 (no cap)",     0.10, 5.0),
        ("20% stop + Score<5 (no cap)",     0.20, 5.0),
        ("15% stop + Score<5 + 63d cap",    0.15, 5.0),  # for comparison
        ("15% stop only (no cap)",          0.15, None),
        ("Score<5 only (no cap)",           None, 5.0),
        # Score drop variants
        ("15% stop + ScoreDrop2 (no cap)",  0.15, "drop2"),
        ("15% stop + ScoreDrop3 (no cap)",  0.15, "drop3"),
        # Baseline
        ("Hold 63d (baseline)",             None, None),
    ]

    results = {}

    for combo_name, stop_pct, score_exit in combos:
        trades = []

        for idx in test_indices:
            for ticker in entry_tickers:
                entry_score = ticker_score_series.get(ticker, {}).get(idx, 0)
                if entry_score < entry_min_score:
                    continue

                ticker_df = data.get(ticker)
                if ticker_df is None:
                    continue

                target_date = benchmark_df.index[idx]
                ticker_indices = ticker_df.index.get_indexer([target_date], method="nearest")
                ticker_idx = ticker_indices[0]

                if ticker_idx < 0 or ticker_idx + 10 >= len(ticker_df):
                    continue

                entry_price = ticker_df["Close"].iloc[ticker_idx]
                close_vals = ticker_df["Close"].values

                # Determine hold period for this combo
                actual_max = max_hold if "63d cap" not in combo_name else 63
                if combo_name == "Hold 63d (baseline)":
                    actual_max = 63

                hold_days = actual_max  # default: hold to cap

                for day in range(1, actual_max + 1):
                    exit_ticker_idx = ticker_idx + day
                    if exit_ticker_idx >= len(close_vals):
                        hold_days = day - 1
                        break

                    current_price = close_vals[exit_ticker_idx]
                    bench_day_idx = idx + day

                    # Check stop loss
                    if stop_pct is not None:
                        pct_change = (current_price - entry_price) / entry_price
                        if pct_change <= -stop_pct:
                            hold_days = day
                            break

                    # Check score exit
                    if score_exit is not None and bench_day_idx < total_days:
                        current_score = ticker_score_series.get(ticker, {}).get(bench_day_idx, None)
                        if current_score is None:
                            # Use nearest known score
                            known = ticker_score_series.get(ticker, {})
                            nearest = None
                            for offset in range(0, 6):
                                if bench_day_idx - offset in known:
                                    nearest = known[bench_day_idx - offset]
                                    break
                            current_score = nearest

                        if current_score is not None:
                            if score_exit == "drop2":
                                if current_score < entry_score - 2.0:
                                    hold_days = day
                                    break
                            elif score_exit == "drop3":
                                if current_score < entry_score - 3.0:
                                    hold_days = day
                                    break
                            elif isinstance(score_exit, (int, float)):
                                if current_score < score_exit:
                                    hold_days = day
                                    break

                    # For baseline hold — just hold to cap
                    if combo_name == "Hold 63d (baseline)":
                        continue

                # Calculate returns
                hold_days = max(1, hold_days)
                exit_ticker_idx = min(ticker_idx + hold_days, len(close_vals) - 1)
                exit_price = close_vals[exit_ticker_idx]
                ret = (exit_price - entry_price) / entry_price

                # SPY return
                spy_exit_idx = min(idx + hold_days, len(benchmark_df) - 1)
                spy_ret = (benchmark_df["Close"].iloc[spy_exit_idx] -
                          benchmark_df["Close"].iloc[idx]) / benchmark_df["Close"].iloc[idx]

                trades.append({
                    "date": benchmark_df.index[idx].strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "entry_score": entry_score,
                    "return": ret,
                    "spy_return": spy_ret,
                    "alpha": ret - spy_ret,
                    "hold_days": hold_days,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                })

        results[combo_name] = trades

    # Print results
    print(f"\n{'='*100}")
    print(f"  EXIT COMBO COMPARISON")
    print(f"{'='*100}")
    print(f"\n  {'Strategy':<38s} {'AvgRet':>8s} {'Alpha':>8s} {'Win%':>6s} {'αWin%':>6s} {'Median':>8s} {'Hold':>5s} {'MaxLoss':>8s} {'Sharpe':>7s} {'N':>5s}")
    print(f"  {'─'*105}")

    rows = []
    for name, trades in results.items():
        if not trades:
            continue
        df = pd.DataFrame(trades)
        avg_ret = df["return"].mean() * 100
        avg_alpha = df["alpha"].mean() * 100
        win_rate = (df["return"] > 0).mean() * 100
        alpha_win = (df["alpha"] > 0).mean() * 100
        median_ret = df["return"].median() * 100
        avg_hold = df["hold_days"].mean()
        max_loss = df["return"].min() * 100
        max_gain = df["return"].max() * 100
        sharpe = df["return"].mean() / df["return"].std() if df["return"].std() > 0 else 0
        ann_sharpe = sharpe * np.sqrt(252 / max(avg_hold, 1))

        rows.append({
            "name": name, "avg_ret": avg_ret, "alpha": avg_alpha,
            "win": win_rate, "alpha_win": alpha_win, "median": median_ret,
            "hold": avg_hold, "max_loss": max_loss, "max_gain": max_gain,
            "sharpe": ann_sharpe, "n": len(df),
        })

    rows.sort(key=lambda x: -x["alpha"])
    for r in rows:
        marker = " ◀" if r["name"] == "15% stop + Score<5 (no cap)" else ""
        print(f"  {r['name']:<38s} {r['avg_ret']:>+7.2f}% {r['alpha']:>+7.2f}% {r['win']:>5.1f}% {r['alpha_win']:>5.1f}% {r['median']:>+7.2f}% {r['hold']:>5.1f} {r['max_loss']:>+7.1f}% {r['sharpe']:>6.2f} {r['n']:>5d}{marker}")

    # Detailed analysis of recommended combo
    print(f"\n\n{'='*100}")
    print(f"  DEEP DIVE: 15% Stop + Score<5 (No Time Cap)")
    print(f"{'='*100}")

    rec = results.get("15% stop + Score<5 (no cap)", [])
    if rec:
        df = pd.DataFrame(rec)

        # By exit reason
        stop_exits = df[df["return"] <= -0.14]  # roughly hit stop
        score_exits = df[(df["return"] > -0.14) & (df["hold_days"] < max_hold)]
        cap_exits = df[df["hold_days"] >= max_hold]

        print(f"\n  Exit reason breakdown:")
        print(f"    Stop loss hit:    {len(stop_exits):>4d} trades ({len(stop_exits)/len(df)*100:.1f}%)  Avg return: {stop_exits['return'].mean()*100:+.1f}%")
        print(f"    Score < 5:        {len(score_exits):>4d} trades ({len(score_exits)/len(df)*100:.1f}%)  Avg return: {score_exits['return'].mean()*100 if len(score_exits) > 0 else 0:+.1f}%")
        print(f"    Hit max hold:     {len(cap_exits):>4d} trades ({len(cap_exits)/len(df)*100:.1f}%)  Avg return: {cap_exits['return'].mean()*100 if len(cap_exits) > 0 else 0:+.1f}%")

        # Hold period distribution
        print(f"\n  Hold period distribution:")
        for bucket, label in [(10, "≤10d"), (21, "11-21d"), (42, "22-42d"), (63, "43-63d"), (126, "64-126d")]:
            prev = {10: 0, 21: 10, 42: 21, 63: 42, 126: 63}[bucket]
            subset = df[(df["hold_days"] > prev) & (df["hold_days"] <= bucket)]
            if len(subset) > 0:
                print(f"    {label:>8s}: {len(subset):>4d} trades  |  Avg ret: {subset['return'].mean()*100:+.2f}%  |  Win: {(subset['return']>0).mean()*100:.0f}%")

        # By entry score
        print(f"\n  By entry score:")
        for score_min in [9, 8, 7]:
            score_max = score_min + 1 if score_min < 9 else 11
            subset = df[(df["entry_score"] >= score_min) & (df["entry_score"] < score_max)]
            if len(subset) > 0:
                print(f"    Score {score_min}-{score_max}: {len(subset):>4d} trades  |  Avg ret: {subset['return'].mean()*100:+.2f}%  |  Alpha: {subset['alpha'].mean()*100:+.2f}%  |  Win: {(subset['return']>0).mean()*100:.0f}%  |  Hold: {subset['hold_days'].mean():.0f}d")

        # Top/bottom trades
        print(f"\n  Top 5 trades:")
        top = df.nlargest(5, "return")
        for _, t in top.iterrows():
            print(f"    {t['ticker']:>8s}  {t['date']}  Entry: ${t['entry_price']:.2f}  Exit: ${t['exit_price']:.2f}  Return: {t['return']*100:+.1f}%  Hold: {t['hold_days']:.0f}d")

        print(f"\n  Bottom 5 trades:")
        bottom = df.nsmallest(5, "return")
        for _, t in bottom.iterrows():
            print(f"    {t['ticker']:>8s}  {t['date']}  Entry: ${t['entry_price']:.2f}  Exit: ${t['exit_price']:.2f}  Return: {t['return']*100:+.1f}%  Hold: {t['hold_days']:.0f}d")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    cfg = load_config()
    print("=" * 100)
    print("  EXIT COMBO BACKTEST")
    print("=" * 100)
    run_combo_backtest(cfg, entry_min_score=7.0, test_frequency=5, max_hold=126, verbose=True)
