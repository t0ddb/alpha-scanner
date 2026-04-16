"""
cold_start_catchup.py — One-off diagnostic to find tickers the system would
currently be holding if it had been running for the last 60 days.

The live trading system went live on April 10, 2026. This script walks
the historical score DB to identify tickers that:
  1. Had 3+ consecutive days scoring ≥ 8.5 (would have passed persistence)
  2. Never subsequently dropped below 5.0 (would not have been score-exited)
  3. Never hit a 15% stop loss from the qualification price
  4. Still score ≥ 8.5 today (or are flagged as "drifting" if 5.0–8.49)

Output is a formatted table for manual review — no trades are placed.

Usage:
    python cold_start_catchup.py              # default 60-day lookback
    python cold_start_catchup.py --days 90    # custom lookback window
"""

from __future__ import annotations

import argparse
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yfinance as yf

from config import load_config, get_all_tickers
import trade_log

DB_PATH = Path(__file__).parent / "breakout_tracker.db"
PT_TZ = ZoneInfo("America/Los_Angeles")

# Trading thresholds (must match trade_executor.py defaults)
ENTRY_THRESHOLD = 8.5
EXIT_THRESHOLD = 5.0
PERSISTENCE_DAYS = 3
STOP_LOSS_PCT = 0.15


def get_held_tickers() -> set[str]:
    """Return tickers currently held based on trade_history.json."""
    trades = trade_log._load()["trades"]
    held: set[str] = set()
    for t in trades:
        if t["side"] == "buy":
            held.add(t["ticker"])
        elif t["side"] == "sell":
            held.discard(t["ticker"])
    return held


def get_ticker_scores(db_conn, ticker: str, since: str) -> list[tuple[str, float]]:
    """
    Return [(date_str, score), ...] ordered oldest-first for a ticker
    from `since` date onward.
    """
    cur = db_conn.cursor()
    cur.execute(
        """SELECT date, score FROM ticker_scores
           WHERE ticker = ? AND date >= ?
           ORDER BY date ASC""",
        (ticker, since),
    )
    return cur.fetchall()


def find_first_qualifying_streak(
    scores: list[tuple[str, float]],
    threshold: float,
    streak_len: int,
) -> int | None:
    """
    Find the index of the first row where `streak_len` consecutive scores
    are all >= threshold. Returns the index of the final day of the streak
    (i.e., the qualification date), or None.
    """
    run = 0
    for i, (_, score) in enumerate(scores):
        if score >= threshold:
            run += 1
            if run >= streak_len:
                return i  # this is day `streak_len` of the streak
        else:
            run = 0
    return None


def check_stop_loss(
    ticker: str,
    qual_date: str,
    qual_price: float,
    today: date,
) -> bool:
    """
    Check if the ticker ever hit the 15% stop loss after the qualification
    date. Uses yfinance daily data (Low column).

    Returns True if the stop loss was hit (should be excluded).
    """
    stop_price = qual_price * (1 - STOP_LOSS_PCT)

    try:
        # Fetch from day after qualification to today
        start = (datetime.strptime(qual_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        end = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return False  # can't determine — be optimistic

        # Check if any day's low breached the stop
        lows = df["Low"]
        if hasattr(lows, "columns"):
            # Multi-ticker download wraps in extra level
            lows = lows.iloc[:, 0] if len(lows.columns) == 1 else lows[ticker]
        return bool((lows < stop_price).any())
    except Exception:
        return False  # can't determine — be optimistic


def get_current_price(ticker: str) -> float | None:
    """Fetch the latest close price from yfinance."""
    try:
        df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
        if df is not None and not df.empty:
            close = df["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0] if len(close.columns) == 1 else close[ticker]
            return float(close.iloc[-1])
    except Exception:
        pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Cold start catch-up: find tickers the system would be holding"
    )
    parser.add_argument(
        "--days", type=int, default=60,
        help="Lookback window in calendar days (default: 60)",
    )
    args = parser.parse_args()

    today = datetime.now(PT_TZ).date()
    since_date = (today - timedelta(days=args.days)).strftime("%Y-%m-%d")

    # Load config and tickers
    cfg = load_config()
    all_tickers = get_all_tickers(cfg)
    held = get_held_tickers()

    db_conn = sqlite3.connect(str(DB_PATH))

    candidates = []
    excluded_by_exit = 0
    excluded_by_stop = 0
    excluded_insufficient = 0
    excluded_no_streak = 0

    print(f"Scanning {len(all_tickers)} tickers with {args.days}-day lookback (since {since_date})...")
    print(f"Already held: {', '.join(sorted(held)) or '(none)'}")
    print()

    for ticker in sorted(all_tickers):
        if ticker in held:
            continue

        # Step 1: Pull scores
        scores = get_ticker_scores(db_conn, ticker, since_date)
        if len(scores) < 10:
            excluded_insufficient += 1
            continue

        # Step 2: Find qualifying streak
        qual_idx = find_first_qualifying_streak(scores, ENTRY_THRESHOLD, PERSISTENCE_DAYS)
        if qual_idx is None:
            excluded_no_streak += 1
            continue

        qual_date, qual_score = scores[qual_idx]

        # Step 3: Check for score exit after qualification
        post_qual_scores = scores[qual_idx:]  # includes qual date
        min_score_since = min(s for _, s in post_qual_scores)

        if min_score_since < EXIT_THRESHOLD:
            excluded_by_exit += 1
            continue

        # Step 3b: Check stop loss (need the close price on qual date)
        qual_price = get_current_price(ticker)  # we'll use this for stop loss ref below
        # Actually we need the price on the qualification date, not today.
        # Use yfinance to get the close on qual_date.
        try:
            end_d = (datetime.strptime(qual_date, "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d")
            df = yf.download(ticker, start=qual_date, end=end_d, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                close_col = df["Close"]
                if hasattr(close_col, "columns"):
                    close_col = close_col.iloc[:, 0] if len(close_col.columns) == 1 else close_col[ticker]
                qual_close = float(close_col.iloc[0])
            else:
                qual_close = None
        except Exception:
            qual_close = None

        if qual_close is not None:
            if check_stop_loss(ticker, qual_date, qual_close, today):
                excluded_by_stop += 1
                continue

        # Step 4: Check current score
        current_date, current_score = scores[-1]

        # Get current price
        current_price = get_current_price(ticker)

        days_since_qual = (today - datetime.strptime(qual_date, "%Y-%m-%d").date()).days

        # Flags
        flags = []
        if min_score_since < 6.0:
            flags.append("⚠️ Min score near exit")
        if current_score < ENTRY_THRESHOLD:
            flags.append("⚠️ Score drifting")

        candidates.append({
            "ticker": ticker,
            "qual_date": qual_date,
            "days_since": days_since_qual,
            "current_score": current_score,
            "current_price": current_price,
            "min_score": min_score_since,
            "flags": " | ".join(flags),
        })

    db_conn.close()

    # ── Output ──
    print("=" * 120)
    print(f"  Cold Start Catch-Up Candidates")
    print(f"  Date: {today.isoformat()}")
    print("=" * 120)
    print()

    if not candidates:
        print("  No candidates found.")
    else:
        # Sort by current score descending
        candidates.sort(key=lambda c: -c["current_score"])

        hdr = (f"  {'Ticker':<8s} {'Qualified Date':<16s} {'Days Since Qual':>16s} "
               f"{'Current Score':>14s} {'Current Price':>14s} {'Min Score':>10s} {'Flag'}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        for c in candidates:
            price_str = f"${c['current_price']:.2f}" if c["current_price"] else "N/A"
            print(f"  {c['ticker']:<8s} {c['qual_date']:<16s} {c['days_since']:>16d} "
                  f"{c['current_score']:>14.1f} {price_str:>14s} {c['min_score']:>10.1f} {c['flags']}")

    print()
    print(f"  Found {len(candidates)} candidates out of {len(all_tickers)} tickers "
          f"({len(held)} already held, "
          f"{excluded_by_exit} score-exited, "
          f"{excluded_by_stop} stop-lossed, "
          f"{excluded_no_streak} no qualifying streak, "
          f"{excluded_insufficient} insufficient data)")
    print()


if __name__ == "__main__":
    main()
