from __future__ import annotations

"""
run_full_backtest.py — Runs the backtest across the entire ticker universe.

This will take several minutes due to:
- Fetching 2 years of data for ~80 tickers
- Scoring all tickers across ~44 weekly test dates

Usage:
    python3 run_full_backtest.py
"""

from config import load_config
from data_fetcher import fetch_all
from backtester import run_backtest, backtest_summary

if __name__ == "__main__":
    cfg = load_config()

    print("=" * 80)
    print("  FULL UNIVERSE BACKTEST")
    print("=" * 80)
    print()
    print("  This will fetch 2 years of data for all tickers and run")
    print("  the full backtest. Expect this to take 5-10 minutes.")
    print()

    # --- Step 1: Fetch all data ---
    print("  STEP 1: Fetching data...\n")
    data = fetch_all(cfg, period="2y", verbose=True)

    if not data:
        print("  [ERROR] No data fetched. Exiting.")
        exit(1)

    print(f"\n  Successfully fetched {len(data)} tickers.\n")

    # --- Step 2: Run backtest ---
    print("  STEP 2: Running backtest...\n")
    events = run_backtest(
        cfg,
        data=data,
        test_frequency=5,               # score weekly
        min_score=3,                     # only track strong signals
        forward_windows=[10, 21, 42, 63],  # ~2wk, 1mo, 2mo, 3mo
        verbose=True,
    )

    # --- Step 3: Print results ---
    if events:
        backtest_summary(events)
    else:
        print("  No signal events found. Try lowering min_score.")
        print("  You can edit this file and change min_score=3 to min_score=2.")
