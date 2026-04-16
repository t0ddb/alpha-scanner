"""
backfill_stop_orders.py — One-time utility to place GTC stop orders for
existing positions that were entered before stop order logic was added.

Usage:
    python backfill_stop_orders.py            # place real stop orders
    python backfill_stop_orders.py --dry-run  # show what would be placed
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import StopOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderType
from alpaca.common.exceptions import APIError


STOP_LOSS_PCT = 0.20


def main():
    parser = argparse.ArgumentParser(description="Backfill GTC stop orders for existing positions")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be placed, don't submit")
    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
        sys.exit(1)

    client = TradingClient(api_key, secret_key, paper=True)

    # Safety: must be paper account
    account = client.get_account()
    if not str(account.account_number).startswith("PA"):
        print(f"ERROR: NOT A PAPER ACCOUNT ({account.account_number}) — ABORTING")
        sys.exit(1)

    positions = client.get_all_positions()
    if not positions:
        print("No open positions — nothing to do.")
        return

    # Get all open orders once
    req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
    open_orders = client.get_orders(req)

    # Build set of symbols that already have a stop sell order
    symbols_with_stops = set()
    for order in open_orders:
        if order.type == OrderType.STOP and order.side == OrderSide.SELL:
            symbols_with_stops.add(order.symbol)

    print("Checking existing positions for missing stop orders...")

    placed = 0
    skipped = 0

    for pos in sorted(positions, key=lambda p: p.symbol):
        symbol = pos.symbol
        entry_price = float(pos.avg_entry_price)
        qty = float(pos.qty)
        stop_price = round(entry_price * (1 - STOP_LOSS_PCT), 2)

        if symbol in symbols_with_stops:
            print(f"  {symbol}: already has stop order — skipped")
            skipped += 1
            continue

        if args.dry_run:
            print(f"  {symbol}: entry ${entry_price:.2f}, would place stop at ${stop_price:.2f} ({qty:.0f} shares)")
            placed += 1
        else:
            try:
                order = client.submit_order(StopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                    stop_price=stop_price,
                ))
                print(f"  {symbol}: entry ${entry_price:.2f}, placing stop at ${stop_price:.2f} ({qty:.0f} shares) ✓")
                placed += 1
            except APIError as e:
                print(f"  {symbol}: FAILED — {e}")

    tag = " (dry run)" if args.dry_run else ""
    print(f"\nDone — {placed} stop orders placed{tag}, {skipped} skipped (already had stops)")


if __name__ == "__main__":
    main()
