from __future__ import annotations

"""
backfill_subsector.py — Populate 6 months of subsector breakout history.

Fetches 2yr of OHLCV data, walks dates at 5-day frequency, and at each date:
  1. Slices data up to that date (simulating running on that day)
  2. Runs score_all() on the slice
  3. Computes subsector metrics
  4. Inserts into SQLite
  5. Runs breakout state machine

This builds the historical foundation needed for z-scores, acceleration, and
state machine warm-up.

Usage:
    python3 backfill_subsector.py               # backfill 6 months
    python3 backfill_subsector.py --days 90      # backfill 90 days
    python3 backfill_subsector.py --frequency 3  # every 3 trading days

Runtime: ~10-20 min depending on ticker count and frequency.
"""

import sys
import time
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

from config import load_config, get_all_tickers
from data_fetcher import fetch_batch
from indicators import score_all
from subsector_store import init_db, get_latest_date
from subsector_breakout import (
    compute_subsector_metrics,
    compute_derived_metrics,
    detect_breakout_state,
    print_breakout_summary,
    run_breakout_detection,
)
from subsector_store import (
    upsert_daily,
    upsert_ticker_scores,
    get_history,
    get_breakout_states,
    update_breakout_state,
)


def slice_data_to_date(
    data: dict[str, pd.DataFrame],
    target_date: pd.Timestamp,
    min_bars: int = 252,
) -> dict[str, pd.DataFrame]:
    """
    Slice all DataFrames to only include data up to target_date.
    Requires at least min_bars of data for meaningful indicator calculation.
    """
    sliced = {}
    for ticker, df in data.items():
        mask = df.index <= target_date
        sub = df[mask]
        if len(sub) >= min_bars:
            sliced[ticker] = sub
    return sliced


def get_backfill_dates(
    data: dict[str, pd.DataFrame],
    backfill_days: int = 180,
    frequency: int = 5,
) -> list[pd.Timestamp]:
    """
    Get list of dates to backfill at, based on available data.
    Uses the SPY (or first ticker's) date index as reference.
    """
    # Find a reference ticker with good data
    ref_df = data.get("SPY")
    if ref_df is None:
        ref_df = next(iter(data.values()))

    all_dates = ref_df.index.sort_values()

    # We need at least 252 bars of history for indicators, so start after that
    if len(all_dates) < 252 + backfill_days:
        start_idx = 252
    else:
        start_idx = len(all_dates) - backfill_days

    # Ensure we don't go before the minimum
    start_idx = max(start_idx, 252)

    # Sample every `frequency` trading days
    dates = []
    for i in range(start_idx, len(all_dates), frequency):
        dates.append(all_dates[i])

    return dates


def run_backfill(
    backfill_days: int = 180,
    frequency: int = 5,
    verbose: bool = True,
) -> None:
    """
    Run the full backfill process.
    """
    cfg = load_config()

    print("=" * 80)
    print(f"  SUBSECTOR BACKFILL — {backfill_days} days, every {frequency} trading days")
    print("=" * 80)

    # Step 1: Fetch 2 years of data
    print("\n  Step 1: Fetching 2 years of data...")
    all_tickers = ["SPY"] + get_all_tickers(cfg)
    data = fetch_batch(all_tickers, period="2y", verbose=verbose)

    if not data or "SPY" not in data:
        print("  [ERROR] Failed to fetch data (SPY missing).")
        return

    print(f"  Fetched {len(data)} tickers successfully.")

    # Step 2: Determine backfill dates
    dates = get_backfill_dates(data, backfill_days=backfill_days, frequency=frequency)
    print(f"\n  Step 2: Will backfill {len(dates)} dates")
    if dates:
        print(f"  Range: {dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")

    if not dates:
        print("  [ERROR] No dates to backfill. Not enough data?")
        return

    # Step 3: Initialize database
    conn = init_db()
    print(f"\n  Step 3: Database initialized.")

    # Step 4: Walk each date
    print(f"\n  Step 4: Processing {len(dates)} dates...\n")

    prev_states = {}
    errors = 0
    start_time = time.time()

    for i, target_date in enumerate(dates):
        date_str = target_date.strftime("%Y-%m-%d")

        try:
            # Slice data to this date
            sliced = slice_data_to_date(data, target_date)

            if len(sliced) < 10:
                if verbose:
                    print(f"  [{i+1}/{len(dates)}] {date_str} — skipped (only {len(sliced)} tickers)")
                continue

            # Score all tickers with sliced data
            results = score_all(sliced, cfg)

            if not results:
                continue

            # Compute subsector metrics
            metrics = compute_subsector_metrics(results, cfg)

            # Persist daily metrics (subsector + individual ticker scores)
            upsert_daily(conn, date_str, metrics)
            upsert_ticker_scores(conn, date_str, results)

            # Run state machine for each subsector
            for record in metrics:
                sub_key = record["subsector"]

                # Load history up to this date
                history = get_history(conn, sub_key, days=90)

                # Compute derived metrics
                bd_cfg = cfg.get("breakout_detection", {})
                lookback = bd_cfg.get("lookback_days", 60)
                derived = compute_derived_metrics(history, lookback_days=lookback)

                # Get previous state
                prev = prev_states.get(sub_key)

                # Detect state
                new_state = detect_breakout_state(
                    sub_key, derived, record, prev, cfg, current_date=date_str
                )

                # Update state tracking
                prev_states[sub_key] = new_state
                update_breakout_state(conn, sub_key, new_state)

            # Progress
            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(dates) - i - 1) / rate if rate > 0 else 0

                # Count states
                state_counts = {}
                for s in prev_states.values():
                    st = s.get("status", "quiet")
                    state_counts[st] = state_counts.get(st, 0) + 1

                print(f"  [{i+1}/{len(dates)}] {date_str}  "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)  "
                      f"States: {state_counts}")

        except Exception as e:
            errors += 1
            if verbose:
                print(f"  [{i+1}/{len(dates)}] {date_str} — ERROR: {e}")
            if errors > 20:
                print(f"\n  [ABORT] Too many errors ({errors}). Stopping.")
                break

    elapsed = time.time() - start_time
    conn.close()

    # Step 5: Summary
    print(f"\n{'='*80}")
    print(f"  BACKFILL COMPLETE")
    print(f"{'='*80}")
    print(f"  Dates processed: {len(dates)}")
    print(f"  Errors: {errors}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print()

    # Show final state summary
    print("  Final breakout states:")
    state_counts = {}
    for sub_key, state in sorted(prev_states.items()):
        st = state.get("status", "quiet")
        state_counts[st] = state_counts.get(st, 0) + 1
        if st != "quiet":
            print(f"    {sub_key:<30} {st:>12} (since {state.get('status_since', '?')})")

    print(f"\n  Totals: {state_counts}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill subsector breakout history")
    parser.add_argument("--days", type=int, default=180, help="Days to backfill (default: 180)")
    parser.add_argument("--frequency", type=int, default=5, help="Sample every N trading days (default: 5)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    run_backfill(
        backfill_days=args.days,
        frequency=args.frequency,
        verbose=not args.quiet,
    )
