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
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Slice all DataFrames to only include data up to target_date.
    Requires at least min_bars of data for meaningful indicator calculation.

    Returns (sliced, missing_tickers) where missing_tickers is the list of
    tickers EXCLUDED from the slice because their last available row does
    NOT match target_date. This guards against the silent-corruption bug
    documented in DECISIONS.md 2026-05-19: if yfinance has an intermittent
    data gap for ticker T at target_date, scoring T with its (target_date-1)
    bar and keying the result to target_date would write a stale score
    that misrepresents the day's actual indicator state. We'd rather drop T
    from this target_date entirely than write a misleading score.
    """
    sliced: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    target_norm = pd.Timestamp(target_date).normalize()
    for ticker, df in data.items():
        mask = df.index <= target_date
        sub = df[mask]
        if len(sub) < min_bars:
            continue
        # Verify the slice's last row actually corresponds to target_date.
        last_norm = sub.index[-1].normalize()
        # Normalize tz so we compare apples to apples.
        if last_norm.tz is not None and target_norm.tz is None:
            cmp_target = target_norm.tz_localize(last_norm.tz)
        elif last_norm.tz is None and target_norm.tz is not None:
            cmp_target = target_norm.tz_convert(None)
        else:
            cmp_target = target_norm
        if last_norm != cmp_target:
            missing.append(ticker)
            continue
        sliced[ticker] = sub
    return sliced, missing


def date_already_scored(conn, date_str: str) -> int:
    """
    Return the number of tickers already scored for `date_str` in the DB.
    Used by the backfill to skip dates the trade-executor has populated.
    Trade-exec is the source of truth for live decisions; backfill only
    fills genuine gaps (typically dates where trade-exec failed to commit).
    """
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ticker_scores WHERE date = ?", (date_str,))
    return int(cur.fetchone()[0] or 0)


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
    period: str = "2y",
    target_dates: list[str] = None,
    verbose: bool = True,
) -> None:
    """
    Run the full backfill process.
    """
    cfg = load_config()

    print("=" * 80)
    if target_dates:
        print(f"  SUBSECTOR BACKFILL — targeted dates: {', '.join(target_dates)}")
    else:
        print(f"  SUBSECTOR BACKFILL — {backfill_days} days, every {frequency} trading days")
    print("=" * 80)

    # Step 1: Fetch price data
    print(f"\n  Step 1: Fetching {period} of data...")
    all_tickers = ["SPY"] + get_all_tickers(cfg)
    data = fetch_batch(all_tickers, period=period, verbose=verbose)

    if not data or "SPY" not in data:
        print("  [ERROR] Failed to fetch data (SPY missing).")
        return

    print(f"  Fetched {len(data)} tickers successfully.")

    # Step 2: Determine backfill dates
    if target_dates:
        # Use explicit list of dates; match them against the SPY index
        ref_df = data.get("SPY")
        ref_dates = set(ref_df.index.normalize())
        dates = []
        for d_str in target_dates:
            ts = pd.Timestamp(d_str)
            # Normalize to match index (strip tz if present)
            if ts.tz is None and ref_df.index.tz is not None:
                ts = ts.tz_localize(ref_df.index.tz)
            ts_norm = ts.normalize()
            if ts_norm in ref_dates:
                # Find the exact timestamp in the index
                matches = ref_df.index[ref_df.index.normalize() == ts_norm]
                if len(matches) > 0:
                    dates.append(matches[0])
            else:
                print(f"  [WARN] {d_str} not found in price data (likely not a trading day)")
    else:
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

    skipped_already_scored = 0
    skipped_missing_data = 0
    coverage_warned: list[str] = []  # (date_str, n_missing)
    # Threshold above which we abort a date because the slice has too many
    # tickers with missing target-date data. 5% of the universe (~9 tickers
    # of 184) is plenty — at that point the cross-sectional RS percentile
    # is materially distorted by absent peers, so the surviving tickers'
    # scores can't be trusted as a same-day comparison.
    MISSING_ABORT_PCT = 0.05

    for i, target_date in enumerate(dates):
        date_str = target_date.strftime("%Y-%m-%d")

        try:
            # ── Option 3 guard: don't overwrite trade-exec-written scores ──
            # If the date already has scores in the DB, the trade-executor
            # (or a prior backfill) already populated it. Trade-exec is the
            # source of truth for live decisions — backfill only fills
            # genuine gaps. Skipping prevents the LUNR-2026-05-15-style
            # silent overwrite documented in DECISIONS.md.
            existing = date_already_scored(conn, date_str)
            if existing > 0:
                if verbose:
                    print(f"  [{i+1}/{len(dates)}] {date_str} — already scored "
                          f"({existing} tickers); preserving prior value (no overwrite).")
                skipped_already_scored += 1
                continue

            # ── Option 2 guard: only score tickers that actually have data
            # for the target date. slice_data_to_date now returns a list of
            # tickers it excluded because their last row pre-dates the
            # target — see DECISIONS.md 2026-05-19 for the LUNR case study.
            sliced, missing = slice_data_to_date(data, target_date)

            if len(sliced) < 10:
                if verbose:
                    print(f"  [{i+1}/{len(dates)}] {date_str} — skipped "
                          f"(only {len(sliced)} tickers with data)")
                continue

            # Abort the date if too many tickers are missing target-date
            # bars — partial coverage distorts cross-sectional RS rankings.
            total_eligible = len(sliced) + len(missing)
            if missing and total_eligible > 0:
                missing_pct = len(missing) / total_eligible
                if missing_pct > MISSING_ABORT_PCT:
                    print(f"  [{i+1}/{len(dates)}] {date_str} — ABORTED: "
                          f"{len(missing)} of {total_eligible} tickers "
                          f"({missing_pct:.1%}) have data ending before target. "
                          f"Cross-sectional RS would be distorted. "
                          f"Missing examples: {missing[:5]}")
                    skipped_missing_data += 1
                    continue
                else:
                    coverage_warned.append((date_str, len(missing)))
                    if verbose:
                        print(f"  [{i+1}/{len(dates)}] {date_str} — note: "
                              f"{len(missing)} ticker(s) missing target-date data "
                              f"({', '.join(missing[:10])}{', …' if len(missing) > 10 else ''})")

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
    print(f"  Dates processed:        {len(dates)}")
    print(f"    Skipped (already scored, no overwrite): {skipped_already_scored}")
    print(f"    Aborted (too much missing data):        {skipped_missing_data}")
    print(f"    Coverage warnings (partial missing):    {len(coverage_warned)}")
    if coverage_warned:
        # Show a few examples
        examples = ", ".join(f"{d} ({n} missing)" for d, n in coverage_warned[:5])
        more = f" + {len(coverage_warned)-5} more" if len(coverage_warned) > 5 else ""
        print(f"      e.g., {examples}{more}")
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
    parser.add_argument("--period", type=str, default="2y", help="yfinance data period (default: 2y, use 5y for longer)")
    parser.add_argument("--dates", type=str, default=None, help="Comma-separated list of specific dates to backfill (YYYY-MM-DD)")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    target_dates = None
    if args.dates:
        target_dates = [d.strip() for d in args.dates.split(",") if d.strip()]

    run_backfill(
        backfill_days=args.days,
        frequency=args.frequency,
        period=args.period,
        target_dates=target_dates,
        verbose=not args.quiet,
    )
