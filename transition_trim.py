"""
transition_trim.py — ONE-TIME retroactive resize of existing positions to
match the new 12 × 8.33% sizing rule.

Context: the limit-order + 5% cash-floor logic shipped 2026-04-17 cannot
place new entries while the account is over-committed (cash floor blocks
everything). The 5 positions held today were sized under the prior
5 × 20% rule. This script trims each down to what it would have been
under the new rule, preserving accrued P&L proportionally.

Formula (per position):
    new_size_dollars = 0.0833 × 100_000 × (1 + unrealized_pnl_pct)
    target_qty       = floor(new_size_dollars / current_market_price)
    trim_qty         = current_qty - target_qty

Execution per position (only if trim_qty > 0):
  1. Cancel existing GTC stop order (cancel_stop_orders_for_ticker)
  2. Submit market SELL for trim_qty shares
  3. Poll Alpaca until order fills (up to 60s)
  4. Submit new GTC stop for target_qty at original_entry_price × 0.80
  5. Log both actions to trade_history.json

Stop price semantics: the stop price stays fixed at the ORIGINAL entry
× 0.80. We are not rebasing the stop to today's price. Same price,
reduced quantity.

Safety:
  - Refuses to run against non-paper accounts
  - --dry-run shows the full plan; no orders submitted
  - --execute requires market to be open (override: --allow-closed,
    which submits market orders that queue for open — NOT recommended
    because we can't verify fills before script exits)
  - On any failure in steps 2-4, prints a LOUD banner and halts on that
    position; downstream positions are NOT attempted

Usage:
    python transition_trim.py --dry-run
    python transition_trim.py --execute
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError

# Reuse the same helpers used in trade_executor.py so behaviour matches
from trade_executor import (
    cancel_stop_orders_for_ticker,
    submit_stop_order,
    connect_alpaca,
)

TRADE_LOG_FILE = Path(__file__).parent / "trade_history.json"
PT_TZ = ZoneInfo("America/Los_Angeles")

# Hardcoded — these are the reference constants for this one-time migration
STARTING_EQUITY = 100_000.0
NEW_POSITION_PCT = 0.0833   # 1/12 ≈ 8.33%
STOP_LOSS_PCT = 0.20


# ─────────────────────────────────────────────────────────────
# Planning
# ─────────────────────────────────────────────────────────────

def build_plan(client: TradingClient) -> list[dict]:
    """
    Read live positions and compute a trim plan for each.
    Returns one plan dict per held position.
    """
    positions = client.get_all_positions()
    plan = []
    for pos in sorted(positions, key=lambda p: p.symbol):
        symbol = pos.symbol
        qty = float(pos.qty)
        entry_price = float(pos.avg_entry_price)
        current_price = float(pos.current_price) if pos.current_price else 0.0
        # Alpaca returns unrealized_plpc as a decimal (0.29 = +29%)
        pnl_pct = float(pos.unrealized_plpc) if pos.unrealized_plpc is not None else 0.0
        pnl_dollar = float(pos.unrealized_pl) if pos.unrealized_pl is not None else 0.0

        if current_price <= 0:
            plan.append({
                "ticker": symbol, "skip": True,
                "reason": "No current price from Alpaca",
            })
            continue

        new_size = NEW_POSITION_PCT * STARTING_EQUITY * (1 + pnl_pct)
        target_qty = int(math.floor(new_size / current_price))
        trim_qty = int(qty - target_qty)

        plan.append({
            "ticker": symbol,
            "current_qty": int(qty),
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl_pct": pnl_pct,
            "pnl_dollar": pnl_dollar,
            "new_size_dollars": new_size,
            "target_qty": target_qty,
            "trim_qty": trim_qty,
            "stop_price": round(entry_price * (1 - STOP_LOSS_PCT), 2),
            "skip": trim_qty <= 0,
            "reason": "trim ≤ 0 — position already at or below target" if trim_qty <= 0 else None,
        })
    return plan


def print_plan(plan: list[dict]):
    print("\n" + "=" * 92)
    print("  TRANSITION TRIM PLAN — migrate to 12 × 8.33% sizing")
    print("=" * 92)
    print(f"  Formula: new_size_$ = 0.0833 × $100,000 × (1 + unrealized_pnl_pct)\n")

    header = (f"  {'Ticker':<6s}  {'Curr Qty':>8s}  {'Entry $':>9s}  "
              f"{'Curr $':>8s}  {'P&L %':>6s}  {'Target $':>10s}  "
              f"{'Target Qty':>10s}  {'Trim Qty':>9s}  {'Stop $':>8s}")
    print(header)
    print("  " + "─" * 90)

    for p in plan:
        if p.get("skip"):
            note = p.get("reason", "skipped")
            if "current_qty" in p:
                print(f"  {p['ticker']:<6s}  {p['current_qty']:>8d}  "
                      f"${p['entry_price']:>7.2f}  ${p['current_price']:>6.2f}  "
                      f"{p['pnl_pct']*100:>+5.1f}%  {'':>10s}  {'':>10s}  "
                      f"{'SKIP':>9s}  — {note}")
            else:
                print(f"  {p['ticker']:<6s}  {'SKIP':>8s}  — {note}")
            continue
        print(f"  {p['ticker']:<6s}  {p['current_qty']:>8d}  "
              f"${p['entry_price']:>7.2f}  ${p['current_price']:>6.2f}  "
              f"{p['pnl_pct']*100:>+5.1f}%  ${p['new_size_dollars']:>8,.0f}  "
              f"{p['target_qty']:>10d}  {p['trim_qty']:>9d}  ${p['stop_price']:>6.2f}")

    active = [p for p in plan if not p.get("skip")]
    total_trim_dollars = sum(p["trim_qty"] * p["current_price"] for p in active)
    total_target_dollars = sum(p["target_qty"] * p["current_price"] for p in active)
    print("  " + "─" * 90)
    print(f"  {'TOTAL':<6s}  positions to trim: {len(active)} / {len(plan)}")
    print(f"  {'':<6s}  est. proceeds freed : ${total_trim_dollars:,.0f}")
    print(f"  {'':<6s}  est. remaining value: ${total_target_dollars:,.0f}")
    print()


# ─────────────────────────────────────────────────────────────
# Execution helpers
# ─────────────────────────────────────────────────────────────

def submit_market_sell(
    client: TradingClient, ticker: str, qty: int, dry_run: bool,
) -> str | None:
    """Submit a DAY market sell. Returns order_id or None."""
    if dry_run:
        print(f"    [DRY RUN] would submit market SELL {ticker} qty={qty}")
        return None
    try:
        req = MarketOrderRequest(
            symbol=ticker, qty=qty, side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        print(f"    [trim] submitted market SELL {ticker} qty={qty} (order {order.id})")
        return str(order.id)
    except APIError as e:
        print(f"    [ERROR] market sell for {ticker} failed: {e}")
        return None


def wait_for_fill(
    client: TradingClient, order_id: str, timeout_seconds: int = 60,
) -> dict:
    """
    Poll the order until filled, cancelled, or timed out.
    Returns a dict with keys: filled (bool), fill_price (float|None),
    filled_qty (float|None), status (str), error (str|None).
    """
    start = time.time()
    last_status = None
    while time.time() - start < timeout_seconds:
        try:
            order = client.get_order_by_id(order_id)
        except APIError as e:
            return {"filled": False, "error": f"get_order failed: {e}",
                    "status": "unknown", "fill_price": None, "filled_qty": None}
        # Alpaca returns an OrderStatus enum. str() yields "OrderStatus.FILLED"
        # (enum-qualified) — use .value ("filled") when available, falling
        # back to the last-dotted token of str() for safety.
        status = getattr(order.status, "value", None) or str(order.status).split(".")[-1]
        status = status.lower()
        if status != last_status:
            print(f"    [trim] order {order_id} status: {status}")
            last_status = status
        if status == "filled":
            return {
                "filled": True, "status": status,
                "fill_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                "filled_qty": float(order.filled_qty) if order.filled_qty else None,
                "error": None,
            }
        if status in ("canceled", "expired", "rejected", "suspended"):
            return {"filled": False, "error": f"order finalized as {status}",
                    "status": status, "fill_price": None, "filled_qty": None}
        time.sleep(1)
    return {"filled": False, "error": "timeout", "status": last_status or "unknown",
            "fill_price": None, "filled_qty": None}


def log_trim(
    ticker: str, trim_qty: int, fill_price: float, original_entry: float,
    target_qty: int, new_stop_price: float, pnl_pct: float,
    sell_order_id: str, stop_order_id: str | None,
) -> None:
    """Append a transition_trim record to trade_history.json."""
    entry = {
        "ticker": ticker,
        "side": "trim",
        "action": "transition_trim",
        "date": datetime.now(PT_TZ).strftime("%Y-%m-%d"),
        "trim_price": round(float(fill_price), 4),
        "trim_qty": int(trim_qty),
        "proceeds": round(float(fill_price) * int(trim_qty), 2),
        "original_entry_price": round(float(original_entry), 4),
        "pnl_on_trim": round((float(fill_price) - float(original_entry)) * int(trim_qty), 2),
        "pnl_pct_at_trim": round(float(pnl_pct) * 100, 2),
        "shares_remaining": int(target_qty),
        "new_stop_price": round(float(new_stop_price), 2),
        "reason": "Transition trim: 5 × 20% → 12 × 8.33% sizing rule",
        "sell_order_id": sell_order_id,
        "stop_order_id": stop_order_id,
    }
    try:
        if TRADE_LOG_FILE.exists():
            with open(TRADE_LOG_FILE, "r") as f:
                data = json.load(f)
        else:
            data = {"trades": []}
        if "trades" not in data:
            data["trades"] = []
        data["trades"].append(entry)
        with open(TRADE_LOG_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print(f"    [log] wrote transition_trim entry to {TRADE_LOG_FILE.name}")
    except OSError as e:
        print(f"    [WARN] could not append to trade log: {e}")


def _halt(message: str):
    print("\n" + "!" * 92)
    print(f"  HALT: {message}")
    print("  Downstream positions NOT attempted. Manual intervention required.")
    print("!" * 92 + "\n")


# ─────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────

def execute_trim(client: TradingClient, p: dict, dry_run: bool) -> bool:
    """
    Execute one position's trim sequence.
    Returns True on success, False on failure (caller should halt).
    """
    ticker = p["ticker"]
    trim_qty = p["trim_qty"]
    target_qty = p["target_qty"]
    stop_price = p["stop_price"]

    print(f"\n  ── {ticker} ── trim {trim_qty}, keep {target_qty}, "
          f"new stop ${stop_price:.2f} (original entry × 0.80)")

    # Step 1: cancel existing stop
    print(f"    [1/4] cancelling existing GTC stop for {ticker}")
    cancelled = cancel_stop_orders_for_ticker(client, ticker, dry_run=dry_run)
    if not dry_run and cancelled == 0:
        print(f"    [WARN] no existing stop found for {ticker} (proceeding — not fatal)")

    # Step 2: submit market sell for trim_qty
    print(f"    [2/4] submitting market SELL for {trim_qty} shares")
    sell_order_id = submit_market_sell(client, ticker, trim_qty, dry_run=dry_run)
    if not dry_run and sell_order_id is None:
        _halt(f"market sell submission failed for {ticker}. Old stop was already cancelled — "
              f"MANUALLY re-place a stop for {p['current_qty']} shares @ ${stop_price:.2f} NOW.")
        return False

    # Step 3: wait for fill
    fill_result = {"filled": True, "fill_price": p["current_price"], "filled_qty": trim_qty}
    if not dry_run:
        print(f"    [3/4] waiting for fill (timeout 60s)...")
        fill_result = wait_for_fill(client, sell_order_id, timeout_seconds=60)
        if not fill_result["filled"]:
            _halt(f"{ticker} sell order did not fill: {fill_result.get('error')}. "
                  f"Old stop was cancelled. Restore stop manually if needed.")
            return False
        if fill_result["filled_qty"] != trim_qty:
            print(f"    [WARN] filled qty {fill_result['filled_qty']} != requested {trim_qty}")
        print(f"    [3/4] filled @ ${fill_result['fill_price']:.2f}")
    else:
        print(f"    [3/4] [DRY RUN] would wait for fill")

    # Step 4: place new stop for target_qty at original_entry × 0.80
    print(f"    [4/4] placing new GTC stop for {target_qty} shares @ ${stop_price:.2f}")
    stop_order_id = submit_stop_order(client, ticker, target_qty, stop_price, dry_run=dry_run)
    if not dry_run and stop_order_id is None:
        _halt(f"STOP ORDER FAILED to place for {ticker} ({target_qty} shares @ ${stop_price:.2f}). "
              f"Position is UNPROTECTED. Place stop manually IMMEDIATELY.")
        return False

    # Log
    if not dry_run:
        log_trim(
            ticker=ticker, trim_qty=trim_qty,
            fill_price=fill_result["fill_price"] or p["current_price"],
            original_entry=p["entry_price"],
            target_qty=target_qty, new_stop_price=stop_price,
            pnl_pct=p["pnl_pct"], sell_order_id=sell_order_id or "",
            stop_order_id=stop_order_id,
        )

    return True


# ─────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────

def verify_state(client: TradingClient, plan: list[dict]):
    """After execution, verify positions and stops match the plan."""
    print("\n" + "=" * 92)
    print("  POST-EXECUTION VERIFICATION")
    print("=" * 92)

    positions = {p.symbol: p for p in client.get_all_positions()}

    # Stops: fetch open STOP+SELL orders once
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus, OrderType
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = client.get_orders(req)
        stops_by_ticker = {
            o.symbol: {"qty": float(o.qty), "stop_price": float(o.stop_price)}
            for o in open_orders
            if o.type == OrderType.STOP and o.side == OrderSide.SELL
        }
    except APIError as e:
        print(f"  [WARN] could not fetch open orders: {e}")
        stops_by_ticker = {}

    all_ok = True
    for p in plan:
        if p.get("skip"):
            continue
        t = p["ticker"]
        expected_qty = p["target_qty"]
        expected_stop = p["stop_price"]
        actual_pos = positions.get(t)
        actual_stop = stops_by_ticker.get(t)

        pos_qty = float(actual_pos.qty) if actual_pos else 0
        stop_qty = actual_stop["qty"] if actual_stop else 0
        stop_px = actual_stop["stop_price"] if actual_stop else 0

        pos_ok = pos_qty == expected_qty
        stop_ok = actual_stop is not None and stop_qty == expected_qty and abs(stop_px - expected_stop) < 0.01

        status = "✓" if (pos_ok and stop_ok) else "✗"
        print(f"  {status} {t:<6s}  pos qty: {pos_qty:>4.0f} (expected {expected_qty})  "
              f"stop: {stop_qty:>4.0f} @ ${stop_px:>6.2f} (expected {expected_qty} @ ${expected_stop:.2f})")
        if not (pos_ok and stop_ok):
            all_ok = False

    print()
    if all_ok:
        print("  All positions and stops verified OK.")
    else:
        print("  !!! Some positions/stops do NOT match plan — manual review required !!!")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan; submit no orders.")
    ap.add_argument("--execute", action="store_true",
                    help="Actually submit the trims.")
    ap.add_argument("--allow-closed", action="store_true",
                    help="Allow --execute while market is closed (orders will queue; "
                         "fill verification may not complete before script exits).")
    args = ap.parse_args()

    if not args.dry_run and not args.execute:
        print("Specify exactly one of --dry-run or --execute.")
        sys.exit(1)
    if args.dry_run and args.execute:
        print("Specify only one of --dry-run or --execute.")
        sys.exit(1)

    load_dotenv()

    now = datetime.now(PT_TZ)
    print("=" * 92)
    print(f"  TRANSITION TRIM — one-time migration to new sizing  ·  {now:%Y-%m-%d %H:%M %Z}")
    print("=" * 92)
    mode = "DRY RUN" if args.dry_run else "EXECUTE"
    print(f"  MODE: {mode}")

    client = connect_alpaca()  # paper-account check happens inside

    # Market open check (warning in dry-run, gate in execute)
    try:
        clock = client.get_clock()
        is_open = bool(clock.is_open)
        next_open = str(clock.next_open)[:19] if clock.next_open else "?"
        next_close = str(clock.next_close)[:19] if clock.next_close else "?"
        print(f"  Market: {'OPEN' if is_open else 'CLOSED'}  "
              f"(next open {next_open}, next close {next_close})")
    except APIError as e:
        is_open = False
        print(f"  [WARN] could not read Alpaca clock: {e}")

    if args.execute and not is_open and not args.allow_closed:
        print("\n  [ABORT] --execute requires market to be open. Use --allow-closed to override")
        print("          (orders will queue and fill at open; fill verification may not complete).")
        sys.exit(1)

    plan = build_plan(client)
    if not plan:
        print("\n  No open positions — nothing to trim.")
        return

    print_plan(plan)

    if args.dry_run:
        print("  Dry run complete. Re-run with --execute when market is open.")
        return

    # EXECUTE
    active = [p for p in plan if not p.get("skip")]
    if not active:
        print("  No positions need trimming — all already at/below target.")
        return

    print(f"\n  Executing {len(active)} trims...")
    for p in active:
        ok = execute_trim(client, p, dry_run=False)
        if not ok:
            # halt downstream
            return

    verify_state(client, plan)

    print("\n" + "=" * 92)
    print("  TRANSITION TRIM COMPLETE")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    main()
