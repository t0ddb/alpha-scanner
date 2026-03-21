from __future__ import annotations

"""
first_hit_analysis.py — For each ticker, find the FIRST time it scores 8+
on the new 10-point scale, then measure forward returns from that date.
"""

import pandas as pd
import numpy as np
from config import load_config, get_all_tickers, get_ticker_metadata
from data_fetcher import fetch_batch
from indicators import score_all, MAX_SCORE

FORWARD_WINDOWS = [10, 21, 42, 63]
SCORE_THRESHOLD = 8.0
WARMUP = 260  # enough for 200-day SMA + RS + multi-TF


def run_first_hit_analysis():
    cfg = load_config()
    metadata = get_ticker_metadata(cfg)

    print("=" * 90)
    print("  FIRST-HIT ANALYSIS: Forward returns from FIRST time a ticker scores 8+")
    print("=" * 90)
    print(f"\n  Threshold: {SCORE_THRESHOLD}/{MAX_SCORE}")
    print(f"  Forward windows: {FORWARD_WINDOWS} trading days\n")

    # Fetch data
    print("  Fetching 3 years of data...\n")
    all_tickers = ["SPY"] + get_all_tickers(cfg)
    data = fetch_batch(all_tickers, period="3y", verbose=True)

    if not data:
        print("  [ERROR] No data fetched.")
        return

    # Find the common date range
    benchmark = data.get("SPY")
    if benchmark is None:
        print("  [ERROR] No SPY data.")
        return

    dates = benchmark.index
    max_fwd = max(FORWARD_WINDOWS)

    # Walk through dates at 5-day frequency
    start_idx = WARMUP
    end_idx = len(dates) - max_fwd
    test_indices = list(range(start_idx, end_idx, 5))

    print(f"\n  Walking {len(test_indices)} test dates from {dates[start_idx].date()} to {dates[end_idx].date()}")
    print(f"  Tickers: {len(data) - 1}\n")

    # Track first hit per ticker
    first_hits = {}  # ticker -> {date_idx, score, signals, fwd_returns}

    for step, idx in enumerate(test_indices):
        if (step + 1) % 10 == 0:
            print(f"  [{step+1}/{len(test_indices)}] {dates[idx].date()}...")

        # Slice all data up to this date
        sliced = {}
        for ticker, df in data.items():
            # Find rows up to this date
            mask = df.index <= dates[idx]
            sub = df[mask]
            if len(sub) >= WARMUP:
                sliced[ticker] = sub

        if "SPY" not in sliced:
            continue

        # Score all tickers at this date
        results = score_all(sliced, cfg)

        for r in results:
            ticker = r["ticker"]
            if ticker in first_hits:
                continue  # already recorded first hit

            if r["score"] >= SCORE_THRESHOLD:
                # Compute forward returns from this date
                ticker_df = data.get(ticker)
                if ticker_df is None:
                    continue

                # Find the position in the ticker's own index
                current_date = dates[idx]
                ticker_dates = ticker_df.index
                pos_arr = ticker_dates.get_indexer([current_date], method="ffill")
                pos = pos_arr[0]
                if pos < 0:
                    continue

                current_close = ticker_df["Close"].iloc[pos]
                fwd = {}
                for w in FORWARD_WINDOWS:
                    fwd_pos = pos + w
                    if fwd_pos < len(ticker_df):
                        fwd_close = ticker_df["Close"].iloc[fwd_pos]
                        fwd[w] = (fwd_close / current_close) - 1
                    else:
                        fwd[w] = None

                first_hits[ticker] = {
                    "date": current_date.date(),
                    "score": r["score"],
                    "signals": r["signals"],
                    "signal_weights": r["signal_weights"],
                    "close": current_close,
                    "fwd_returns": fwd,
                    "name": metadata.get(ticker, {}).get("name", ""),
                    "sector": metadata.get(ticker, {}).get("sector_name", ""),
                    "subsector": metadata.get(ticker, {}).get("subsector_name", ""),
                }

    # ── Results ──
    print(f"\n\n{'=' * 90}")
    print(f"  RESULTS: {len(first_hits)} tickers hit {SCORE_THRESHOLD}+ at least once")
    print(f"{'=' * 90}\n")

    if not first_hits:
        print("  No tickers reached the threshold.")
        return

    # Sort by date of first hit
    sorted_hits = sorted(first_hits.items(), key=lambda x: x[1]["date"])

    # Print individual ticker results
    print(f"  {'Ticker':<10} {'Name':<25} {'Date':>12} {'Score':>6} "
          f"{'10d':>8} {'21d':>8} {'42d':>8} {'63d':>8}  Signals")
    print(f"  {'─' * 115}")

    for ticker, h in sorted_hits:
        fwd = h["fwd_returns"]
        fwd_strs = []
        for w in FORWARD_WINDOWS:
            val = fwd.get(w)
            if val is not None:
                fwd_strs.append(f"{val*100:>+7.1f}%")
            else:
                fwd_strs.append(f"{'—':>8}")

        signals_str = ", ".join(
            f"{s}({h['signal_weights'].get(s, 0)})"
            for s in h["signals"]
        )

        print(f"  {ticker:<10} {h['name'][:24]:<25} {str(h['date']):>12} "
              f"{h['score']:>5.1f} {fwd_strs[0]} {fwd_strs[1]} {fwd_strs[2]} {fwd_strs[3]}  {signals_str}")

    # ── Aggregate statistics ──
    print(f"\n\n{'─' * 90}")
    print(f"  AGGREGATE FORWARD RETURNS (from first 8+ score)")
    print(f"{'─' * 90}\n")

    for w in FORWARD_WINDOWS:
        returns = [h["fwd_returns"][w] for _, h in sorted_hits if h["fwd_returns"].get(w) is not None]
        if returns:
            avg = np.mean(returns) * 100
            med = np.median(returns) * 100
            wr = sum(1 for r in returns if r > 0) / len(returns) * 100
            print(f"  {w:>3}d:  N={len(returns):>3}  |  Avg: {avg:>+7.2f}%  |  Median: {med:>+7.2f}%  |  Win rate: {wr:>5.1f}%")

    # ── By sector ──
    print(f"\n\n{'─' * 90}")
    print(f"  BY SECTOR (63-day returns)")
    print(f"{'─' * 90}\n")

    sector_groups = {}
    for ticker, h in sorted_hits:
        sector = h["sector"] or "Unknown"
        if sector not in sector_groups:
            sector_groups[sector] = []
        if h["fwd_returns"].get(63) is not None:
            sector_groups[sector].append(h["fwd_returns"][63])

    for sector in sorted(sector_groups, key=lambda s: -np.mean(sector_groups[s]) if sector_groups[s] else 0):
        rets = sector_groups[sector]
        if rets:
            avg = np.mean(rets) * 100
            wr = sum(1 for r in rets if r > 0) / len(rets) * 100
            print(f"  {sector:<40} N={len(rets):>3}  |  Avg 63d: {avg:>+7.2f}%  |  WR: {wr:>5.1f}%")

    # ── Score distribution ──
    print(f"\n\n{'─' * 90}")
    print(f"  SCORE DISTRIBUTION AT FIRST HIT")
    print(f"{'─' * 90}\n")

    scores = [h["score"] for _, h in sorted_hits]
    for bracket_low, bracket_high in [(8, 8.5), (8.5, 9), (9, 9.5), (9.5, 10.1)]:
        in_bracket = [(t, h) for t, h in sorted_hits
                      if bracket_low <= h["score"] < bracket_high]
        if in_bracket:
            rets_63 = [h["fwd_returns"][63] for _, h in in_bracket if h["fwd_returns"].get(63) is not None]
            if rets_63:
                avg = np.mean(rets_63) * 100
                wr = sum(1 for r in rets_63 if r > 0) / len(rets_63) * 100
                label = f"{bracket_low:.1f}-{bracket_high:.1f}" if bracket_high <= 10 else f"{bracket_low:.1f}+"
                print(f"  Score {label:<10}  N={len(in_bracket):>3}  |  Avg 63d: {avg:>+7.2f}%  |  WR: {wr:>5.1f}%")

    print(f"\n{'=' * 90}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    run_first_hit_analysis()
