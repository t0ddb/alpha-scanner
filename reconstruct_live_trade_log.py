"""
reconstruct_live_trade_log.py — One-time recovery script.

Rebuilds trade_history_live.json from Alpaca's order history after the
live workflow's first run committed locally but failed to push (workflow
bug, fixed in commit 4814177).

For each filled BUY in the live Alpaca account that isn't already in
trade_history_live.json, this:
  1. Looks up score_at_entry from breakout_tracker.db for the most recent
     scoring date ≤ the order submission date (the score the system "saw"
     when it decided to enter).
  2. Calls trade_log.log_buy() with the actual fill_avg_price + filled_qty.

Run AFTER orders have filled (so filled_avg_price is populated).

Usage:
    ALPACA_MODE=live python3 reconstruct_live_trade_log.py [--days N] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus, OrderSide

import trade_log


PT_TZ = ZoneInfo("America/Los_Angeles")
DB_PATH = Path(__file__).parent / "breakout_tracker.db"


def _connect_alpaca_live() -> TradingClient:
    api_key = os.getenv("ALPACA_LIVE_API_KEY")
    secret_key = os.getenv("ALPACA_LIVE_SECRET_KEY")
    if not api_key or not secret_key:
        print("[ERROR] ALPACA_LIVE_API_KEY / ALPACA_LIVE_SECRET_KEY not set")
        sys.exit(1)
    client = TradingClient(api_key, secret_key, paper=False)
    acct = client.get_account()
    if str(acct.account_number).startswith("PA"):
        print(f"[ERROR] account {acct.account_number} is a PAPER account; aborting")
        sys.exit(1)
    print(f"[OK] connected to live account {acct.account_number}, equity=${float(acct.equity):,.2f}")
    return client


def _existing_order_ids() -> set[str]:
    """Return set of alpaca_order_id already logged (avoid duplicates)."""
    f_path = trade_log._get_trade_log_file()
    if not f_path.exists():
        return set()
    import json
    with open(f_path) as f:
        data = json.load(f)
    return {t.get("alpaca_order_id") for t in data.get("trades", []) if t.get("alpaca_order_id")}


def _score_at_or_before(conn: sqlite3.Connection, ticker: str, on_date: str) -> float | None:
    """Return score for ticker on the most recent scoring date ≤ on_date."""
    row = conn.execute(
        "SELECT score FROM ticker_scores WHERE ticker = ? AND date <= ? "
        "ORDER BY date DESC LIMIT 1",
        (ticker, on_date),
    ).fetchone()
    return float(row[0]) if row else None


def main(days: int, dry_run: bool):
    load_dotenv()
    if (os.getenv("ALPACA_MODE") or "").lower() != "live":
        print("[ERROR] set ALPACA_MODE=live before running")
        sys.exit(1)

    client = _connect_alpaca_live()
    cutoff_utc = datetime.now(timezone.utc) - timedelta(days=days)
    print(f"[OK] querying orders since {cutoff_utc.isoformat()}")

    req = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,   # filled or canceled
        after=cutoff_utc,
        side=OrderSide.BUY,
        limit=500,
    )
    orders = client.get_orders(req)
    print(f"[OK] {len(orders)} closed BUY order(s) returned")

    existing = _existing_order_ids()
    print(f"[OK] {len(existing)} order(s) already in trade_history_live.json")

    conn = sqlite3.connect(str(DB_PATH))
    new_entries = 0
    skipped = []

    for o in orders:
        oid = str(o.id)
        if oid in existing:
            skipped.append((o.symbol, "already logged"))
            continue
        if str(o.status) not in ("OrderStatus.FILLED", "filled"):
            skipped.append((o.symbol, f"not filled (status={o.status})"))
            continue
        if o.filled_qty is None or float(o.filled_qty) <= 0:
            skipped.append((o.symbol, "filled_qty=0"))
            continue

        ticker = o.symbol
        fill_qty = float(o.filled_qty)
        fill_price = float(o.filled_avg_price) if o.filled_avg_price else float(o.limit_price or 0)
        # submission date in PT (matches DB's score date convention)
        submitted_at_pt = o.submitted_at.astimezone(PT_TZ).date().isoformat()
        filled_at_pt = o.filled_at.astimezone(PT_TZ).date().isoformat() if o.filled_at else submitted_at_pt

        score = _score_at_or_before(conn, ticker, submitted_at_pt)
        if score is None:
            skipped.append((ticker, f"no score in DB on or before {submitted_at_pt}"))
            continue

        print(f"  {'[DRY] ' if dry_run else ''}BUY  {ticker:<6} qty={fill_qty:>5} "
              f"fill=${fill_price:>7.2f} submit={submitted_at_pt} fill={filled_at_pt} "
              f"score={score:.1f} order={oid[:8]}")

        if not dry_run:
            trade_log.log_buy(
                ticker=ticker,
                trade_date=filled_at_pt,
                price=fill_price,
                qty=fill_qty,
                score_at_entry=score,
                reason=f"Reconstructed (live workflow first-run git push failed; submitted {submitted_at_pt})",
                alpaca_order_id=oid,
            )
            new_entries += 1

    conn.close()
    print(f"\n[SUMMARY] {new_entries} entries reconstructed, {len(skipped)} skipped")
    for sym, reason in skipped:
        print(f"  skipped {sym}: {reason}")
    if dry_run:
        print("\n[DRY RUN] no changes written. Re-run without --dry-run to apply.")
    else:
        out = trade_log._get_trade_log_file()
        print(f"\n[OK] {out.name} now contains all reconstructed entries.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5,
                    help="Look back N days (default 5)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be logged, don't write")
    args = ap.parse_args()
    main(args.days, args.dry_run)
