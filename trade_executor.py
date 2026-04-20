"""
trade_executor.py — Daily paper-trade execution against Alpaca.

Runs after market close. Scores all tickers, evaluates exits on open
positions, and places entry orders on new breakout signals.

Entry config comes from `trade_execution:` in ticker_config.yaml and can
be overridden by environment variables. The current default is
threshold ≥ 8.5 with a 3-day persistence filter (score must have been
at/above the threshold for the prior 3 trading days), exit < 5, and a
20% stop loss (real Alpaca GTC stop orders) — validated by the sizing
comparison backtest (2026-04-16).

Usage:
    python3 trade_executor.py              # live paper trading
    python3 trade_executor.py --dry-run    # show what it WOULD do, no orders

Env vars (store in .env or CI secrets):
    ALPACA_API_KEY       — Alpaca paper trading API key (required)
    ALPACA_SECRET_KEY    — Alpaca paper trading secret (required)
    ENTRY_THRESHOLD      — override config entry score (e.g. "8.5")
    EXIT_THRESHOLD       — override config exit score  (e.g. "5.0")
    PERSISTENCE_DAYS     — override config prior-day persistence (e.g. "3")
    STOP_LOSS_PCT        — override config stop loss   (e.g. "0.20")
    GMAIL_ADDRESS        — Gmail sender address        (optional, for --email)
    GMAIL_APP_PASSWORD   — Gmail app password          (optional, for --email)
    ALERT_EMAIL_TO       — Recipient (defaults to sender if unset)
"""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
from datetime import date, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# Alpaca SDK
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest, LimitOrderRequest, StopOrderRequest, GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus, OrderType
from alpaca.common.exceptions import APIError
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest

# Alpha Scanner pipeline
from config import load_config
from data_fetcher import fetch_all
from indicators import score_all

# Local trade utilities
import wash_sale_tracker
import trade_log
import subsector_store


# ─────────────────────────────────────────────────────────────
# Config loading (config file → env var override → default)
# ─────────────────────────────────────────────────────────────
PT_TZ = ZoneInfo("America/Los_Angeles")

# Defaults if neither config nor env var is set
_DEFAULTS = {
    "entry_threshold": 8.5,
    "exit_threshold": 5.0,
    "persistence_days": 3,
    "stop_loss_pct": 0.20,
    "max_positions": 12,
    "max_position_pct": 0.083,
    "min_position_size": 500,
}


def _env_float(name: str) -> float | None:
    val = os.getenv(name)
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        print(f"  [WARN] Env var {name}={val!r} is not a number — ignoring")
        return None


def _env_int(name: str) -> int | None:
    val = os.getenv(name)
    if val is None or val == "":
        return None
    try:
        return int(val)
    except ValueError:
        print(f"  [WARN] Env var {name}={val!r} is not an int — ignoring")
        return None


def load_trade_config(cfg: dict) -> dict:
    """
    Build the effective trade-execution config. Precedence:
      env var > ticker_config.yaml > hardcoded default
    """
    yaml_cfg = cfg.get("trade_execution", {}) or {}

    def pick(key, caster, env_name):
        env_val = _env_float(env_name) if caster is float else _env_int(env_name)
        if env_val is not None:
            return caster(env_val)
        if key in yaml_cfg and yaml_cfg[key] is not None:
            return caster(yaml_cfg[key])
        return caster(_DEFAULTS[key])

    return {
        "entry_threshold": pick("entry_threshold", float, "ENTRY_THRESHOLD"),
        "exit_threshold": pick("exit_threshold", float, "EXIT_THRESHOLD"),
        "persistence_days": pick("persistence_days", int, "PERSISTENCE_DAYS"),
        "stop_loss_pct": pick("stop_loss_pct", float, "STOP_LOSS_PCT"),
        "max_positions": pick("max_positions", int, "MAX_POSITIONS"),
        "max_position_pct": pick("max_position_pct", float, "MAX_POSITION_PCT"),
        "min_position_size": pick("min_position_size", int, "MIN_POSITION_SIZE"),
    }

# Tickers excluded from Alpaca paper trading
EXCLUDED_SUFFIXES = ("-USD",)          # crypto
EXCLUDED_EXACT = {"GC=F", "SI=F"}      # futures


# ─────────────────────────────────────────────────────────────
# Alpaca helpers
# ─────────────────────────────────────────────────────────────

def connect_alpaca() -> TradingClient:
    """Initialize and validate the Alpaca paper trading client."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        print("  [ERROR] ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
        sys.exit(1)

    client = TradingClient(api_key, secret_key, paper=True)

    # Safety check: must be a paper account
    account = client.get_account()
    if not str(account.account_number).startswith("PA"):
        print(f"  [ERROR] NOT A PAPER ACCOUNT ({account.account_number}) — ABORTING")
        sys.exit(1)

    return client


def connect_alpaca_data() -> StockHistoricalDataClient:
    """Initialize the Alpaca historical data client (stocks)."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    return StockHistoricalDataClient(api_key, secret_key)


def get_alpaca_latest_price(data_client: StockHistoricalDataClient, ticker: str) -> float:
    """
    Fetch the latest trade price from Alpaca's market data API.
    Returns 0.0 if the request fails or the ticker is unknown.
    """
    try:
        req = StockLatestTradeRequest(symbol_or_symbols=ticker)
        resp = data_client.get_stock_latest_trade(req)
        trade = resp.get(ticker) if isinstance(resp, dict) else None
        if trade is not None and getattr(trade, "price", None):
            return float(trade.price)
    except APIError as e:
        print(f"  [WARN] Alpaca latest-trade fetch failed for {ticker}: {e}")
    except Exception as e:
        print(f"  [WARN] Alpaca latest-trade error for {ticker}: {e}")
    return 0.0


def get_account_snapshot(client: TradingClient) -> dict:
    """Grab equity, cash, and open positions in one call."""
    account = client.get_account()
    try:
        raw_positions = client.get_all_positions()
    except APIError:
        raw_positions = []

    positions = {}
    for p in raw_positions:
        lastday_price = float(p.lastday_price) if getattr(p, "lastday_price", None) is not None else 0.0
        change_today = float(p.change_today) * 100 if getattr(p, "change_today", None) is not None else 0.0
        positions[p.symbol] = {
            "qty": float(p.qty),
            "entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price) if p.current_price is not None else 0.0,
            "market_value": float(p.market_value) if p.market_value is not None else 0.0,
            "unrealized_pnl": float(p.unrealized_pl) if p.unrealized_pl is not None else 0.0,
            "unrealized_pnl_pct": float(p.unrealized_plpc) * 100 if p.unrealized_plpc is not None else 0.0,
            "lastday_price": lastday_price,
            "change_today_pct": change_today,
        }

    # last_equity = previous day's closing equity (for day-over-day change)
    last_equity = float(account.last_equity) if getattr(account, "last_equity", None) is not None else None

    return {
        "equity": float(account.equity),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "last_equity": last_equity,
        "positions": positions,
    }


def is_ticker_excluded(ticker: str) -> bool:
    """Filter out crypto and futures tickers we can't trade on Alpaca."""
    if ticker in EXCLUDED_EXACT:
        return True
    return any(ticker.endswith(suf) for suf in EXCLUDED_SUFFIXES)


def is_tradeable_on_alpaca(client: TradingClient, ticker: str) -> bool:
    """Verify the symbol is tradeable on Alpaca."""
    try:
        asset = client.get_asset(ticker)
        return bool(asset.tradable)
    except APIError:
        return False


def submit_market_order(
    client: TradingClient,
    ticker: str,
    qty: float,
    side: OrderSide,
    dry_run: bool = False,
) -> str | None:
    """Place a market order, returning the order ID (or None on failure/dry-run)."""
    if dry_run:
        return None

    try:
        req = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(req)
        return str(order.id)
    except APIError as e:
        print(f"  [ERROR] Order failed for {ticker}: {e}")
        return None


def submit_limit_order(
    client: TradingClient,
    ticker: str,
    qty: float,
    limit_price: float,
    dry_run: bool = False,
) -> str | None:
    """Place a DAY-TIF buy limit order. Returns the order ID or None.

    DAY TIF means the order auto-cancels at market close on the next
    trading day if it hasn't filled. Deferred re-qualification is
    handled naturally by the daily re-scan — no state to track.
    """
    rounded = round(limit_price, 2)
    if dry_run:
        print(f"  [DRY RUN] Would place limit BUY: {ticker} {qty:.0f} @ ${rounded:.2f} (DAY)")
        return None
    try:
        req = LimitOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            limit_price=rounded,
        )
        order = client.submit_order(req)
        print(f"  [limit] Placed BUY limit for {ticker}: {qty:.0f} @ ${rounded:.2f} "
              f"(order {order.id})")
        return str(order.id)
    except APIError as e:
        print(f"  [ERROR] Limit order failed for {ticker}: {e}")
        return None


def ensure_stops_for_positions(
    client: TradingClient,
    snapshot: dict,
    trade_cfg: dict,
    dry_run: bool = False,
) -> int:
    """
    Place GTC stop orders for any held position that doesn't already
    have one. Idempotent — safe to call every run. This is how stops
    land on positions opened via limit orders (we can't place the stop
    at submission time because we don't know if/when the limit fills).
    """
    stop_loss_pct = trade_cfg["stop_loss_pct"]

    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
        open_orders = client.get_orders(req)
    except APIError as e:
        print(f"  [WARN] Failed to query open orders for stop backfill: {e}")
        return 0

    tickers_with_stops = {
        o.symbol for o in open_orders
        if o.type == OrderType.STOP and o.side == OrderSide.SELL
    }

    placed = 0
    for ticker, pos in sorted(snapshot["positions"].items()):
        if ticker in tickers_with_stops:
            continue
        qty = pos["qty"]
        entry_price = pos["entry_price"]
        if qty <= 0 or entry_price <= 0:
            continue
        stop_price = round(entry_price * (1 - stop_loss_pct), 2)
        order_id = submit_stop_order(client, ticker, qty, stop_price, dry_run=dry_run)
        if order_id is not None or dry_run:
            placed += 1
    return placed


def detect_unfilled_limits_since(
    client: TradingClient,
    cutoff_dt,
) -> list[dict]:
    """
    Return BUY limit orders that were submitted after cutoff_dt and
    expired/canceled without filling. Used to surface "limit unfilled"
    context in the email digest on the next run.
    """
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=200)
        orders = client.get_orders(req)
    except APIError as e:
        print(f"  [WARN] Failed to query closed orders for unfilled-limit check: {e}")
        return []

    out = []
    for o in orders:
        try:
            is_buy_limit = (
                o.side == OrderSide.BUY
                and o.type == OrderType.LIMIT
                and str(o.status).lower() in ("expired", "canceled")
            )
        except Exception:
            continue
        if not is_buy_limit:
            continue
        submitted = getattr(o, "submitted_at", None)
        if submitted is None or submitted < cutoff_dt:
            continue
        limit_px = float(o.limit_price) if o.limit_price else 0.0
        filled_qty = float(o.filled_qty) if o.filled_qty else 0.0
        if filled_qty >= float(o.qty):
            continue  # fully filled (even if later canceled — shouldn't happen for buys)
        out.append({
            "ticker": o.symbol,
            "submitted_date": submitted.strftime("%Y-%m-%d"),
            "limit_price": limit_px,
            "status": str(o.status),
        })
    return out


def submit_stop_order(
    client: TradingClient,
    ticker: str,
    qty: float,
    stop_price: float,
    dry_run: bool = False,
) -> str | None:
    """Place a GTC stop-loss sell order. Returns the order ID or None."""
    if dry_run:
        print(f"  [DRY RUN] Would place stop order: {ticker} sell {qty:.0f} shares at ${stop_price:.2f}")
        return None

    try:
        req = StopOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            stop_price=round(stop_price, 2),
        )
        order = client.submit_order(req)
        print(f"  [stop] Placed GTC stop for {ticker}: {qty:.0f} shares @ ${stop_price:.2f} "
              f"(order {order.id})")
        return str(order.id)
    except APIError as e:
        print(f"  [ERROR] Stop order failed for {ticker}: {e}")
        return None


def cancel_stop_orders_for_ticker(
    client: TradingClient,
    ticker: str,
    dry_run: bool = False,
) -> int:
    """Cancel all open stop orders for a ticker. Returns count canceled."""
    try:
        req = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[ticker],
        )
        open_orders = client.get_orders(req)
        canceled = 0
        for order in open_orders:
            if order.type == OrderType.STOP and order.side == OrderSide.SELL:
                if dry_run:
                    print(f"  [DRY RUN] Would cancel stop order {order.id} for {ticker}")
                else:
                    client.cancel_order_by_id(order.id)
                    print(f"  [stop] Canceled stop order {order.id} for {ticker}")
                canceled += 1
        return canceled
    except APIError as e:
        print(f"  [WARN] Failed to query/cancel stop orders for {ticker}: {e}")
        return 0


def detect_filled_stops(
    client: TradingClient,
    snapshot: dict,
    today: date,
    dry_run: bool = False,
) -> list[dict]:
    """
    Check for stop orders that have filled since the last run.
    These are positions that Alpaca auto-sold via the GTC stop order.

    Returns a list of filled-stop dicts suitable for logging as exits.
    """
    filled_stops = []
    try:
        # Get recently closed orders (filled stops will be in here)
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=100,
        )
        closed_orders = client.get_orders(req)

        # Tickers currently tracked in trade_history as held
        held_tickers = set()
        trades = trade_log.get_all_trades()
        for t in trades:
            if t["side"] == "buy":
                held_tickers.add(t["ticker"])
            elif t["side"] == "sell":
                held_tickers.discard(t["ticker"])

        for order in closed_orders:
            if (order.type == OrderType.STOP
                    and order.side == OrderSide.SELL
                    and str(order.status).upper() == "FILLED"
                    and order.symbol in held_tickers
                    and order.symbol not in snapshot["positions"]):
                # This stop filled and the position is no longer in Alpaca
                # but we haven't logged the sell yet
                fill_price = float(order.filled_avg_price) if order.filled_avg_price else 0.0
                fill_qty = float(order.filled_qty) if order.filled_qty else 0.0

                # Look up entry price from trade log
                buys = [t for t in trades if t["ticker"] == order.symbol and t["side"] == "buy"]
                entry_price = buys[-1]["price"] if buys else 0.0
                entry_score = buys[-1].get("score_at_entry", 0.0) if buys else 0.0

                filled_stops.append({
                    "ticker": order.symbol,
                    "qty": fill_qty,
                    "fill_price": fill_price,
                    "entry_price": entry_price,
                    "entry_score": entry_score,
                    "order_id": str(order.id),
                    "filled_at": str(order.filled_at) if order.filled_at else today.isoformat(),
                })

    except APIError as e:
        print(f"  [WARN] Failed to check for filled stops: {e}")

    return filled_stops


# ─────────────────────────────────────────────────────────────
# Decision logic
# ─────────────────────────────────────────────────────────────

def score_lookup(results: list[dict]) -> dict[str, dict]:
    """Transform the score_all() list into a ticker → record dict."""
    return {r["ticker"]: r for r in results}


def evaluate_exits(
    snapshot: dict,
    scores: dict[str, dict],
    trade_cfg: dict,
) -> list[dict]:
    """Determine which open positions should be sold today."""
    exit_threshold = trade_cfg["exit_threshold"]
    stop_loss_pct = trade_cfg["stop_loss_pct"]

    exits = []
    for ticker, pos in snapshot["positions"].items():
        score_rec = scores.get(ticker)
        score = score_rec["score"] if score_rec else None
        entry_price = pos["entry_price"]
        current_price = pos["current_price"]
        stop_price = entry_price * (1 - stop_loss_pct)

        reason = None
        if score is not None and score < exit_threshold:
            reason = f"Score exit (score {score:.1f} < {exit_threshold})"
        elif current_price > 0 and current_price <= stop_price:
            reason = f"Stop loss hit ({stop_loss_pct*100:.0f}%)"

        if reason:
            exits.append({
                "ticker": ticker,
                "qty": pos["qty"],
                "entry_price": entry_price,
                "current_price": current_price,
                "score": score if score is not None else 0.0,
                "unrealized_pnl": pos["unrealized_pnl"],
                "unrealized_pnl_pct": pos["unrealized_pnl_pct"],
                "reason": reason,
            })
    return exits


def check_persistence(
    db_conn,
    ticker: str,
    threshold: float,
    persistence_days: int,
    today: date,
) -> tuple[bool, str]:
    """
    Verify the ticker's score was at/above `threshold` for each of the
    `persistence_days` most recent trading days strictly before `today`.

    Reads from `ticker_scores` in breakout_tracker.db, which is populated
    by (a) this executor at the end of every run and (b) the nightly
    backfill workflow as a safety net.

    Returns (passes, reason_if_not).
    """
    if persistence_days <= 0:
        return True, ""

    cur = db_conn.cursor()
    today_str = today.strftime("%Y-%m-%d")
    cur.execute(
        """SELECT date, score FROM ticker_scores
           WHERE ticker = ? AND date < ?
           ORDER BY date DESC
           LIMIT ?""",
        (ticker, today_str, persistence_days),
    )
    rows = cur.fetchall()

    if len(rows) < persistence_days:
        return False, (
            f"Persistence filter: only {len(rows)} prior scores in DB "
            f"(need {persistence_days})"
        )

    for date_str, prior_score in rows:
        if prior_score is None or prior_score < threshold:
            return False, (
                f"Persistence filter: {date_str} score "
                f"{prior_score:.1f} < {threshold}"
            )

    return True, ""


# Entry execution parameters (validated by entry_mode_backtest.py on
# 2026-04-16: Limit-3% + 5% cash floor + Alpaca-first pricing). Changing
# these without re-running the backtest is risky.
CASH_FLOOR_PCT = 0.05          # reserve 5% of equity uncommitted
LIMIT_ORDER_BUFFER = 0.03      # limit price = sizing_price × 1.03


# Skip-reason categories used by the email digest (keep in sync with
# the allowed values in _build_trade_digest_html).
SKIP_CATEGORY_INSUFFICIENT_CASH = "insufficient cash"
SKIP_CATEGORY_POSITION_CAP = "position cap reached"
SKIP_CATEGORY_WASH_SALE = "wash sale cooldown"
SKIP_CATEGORY_CASH_FLOOR = "cash floor cap"
SKIP_CATEGORY_LIMIT_UNFILLED = "limit unfilled"
SKIP_CATEGORY_PERSISTENCE = "persistence"
SKIP_CATEGORY_OTHER = "other"


def evaluate_entries(
    snapshot: dict,
    scores: dict[str, dict],
    exits: list[dict],
    client: TradingClient,
    data_client: StockHistoricalDataClient,
    today: date,
    trade_cfg: dict,
    db_conn,
    force_entry: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Return (entries, skipped) for today.

    force_entry: list of tickers that bypass the entry threshold and
    persistence filter (cold-start catch-up). All other rules (sizing,
    tradeability, wash sale logging) still apply.

    Each skip dict carries both a verbose ``reason`` (for console) and a
    normalized ``skip_category`` used by the email's Skip Reason column.
    """
    entry_threshold = trade_cfg["entry_threshold"]
    persistence_days = trade_cfg["persistence_days"]
    max_positions = trade_cfg["max_positions"]
    max_position_pct = trade_cfg["max_position_pct"]
    min_position_size = trade_cfg["min_position_size"]
    force_set = set(force_entry or [])

    held = set(snapshot["positions"].keys())
    exiting = {e["ticker"] for e in exits}
    cooldown_map = wash_sale_tracker.get_cooldowns(today)

    # Post-exit position count (for cap check)
    current_position_count = len(held - exiting)

    # Estimate post-exit cash: current cash + proceeds from sells
    projected_cash = snapshot["cash"] + sum(
        e["qty"] * e["current_price"] for e in exits
    )
    equity = snapshot["equity"]

    # Estimated committed dollars (market value locked in existing positions).
    # Used by the 5% cash-floor formula to bound total commitment at 95% of
    # equity. Updated in the loop as each new entry is accepted.
    projected_committed = max(0.0, equity - projected_cash)

    # Filter candidates: normal threshold-based + forced tickers
    candidates = []
    for ticker, rec in scores.items():
        score = rec["score"]
        is_forced = ticker in force_set
        if not is_forced and score < entry_threshold:
            continue
        if ticker in held and ticker not in exiting:
            if is_forced:
                print(f"  [force-entry] {ticker} already held — skipping")
            continue  # already own it
        if is_ticker_excluded(ticker):
            continue
        candidates.append(rec)

    # Process forced tickers first (highest priority), then by score
    candidates.sort(key=lambda r: (0 if r["ticker"] in force_set else 1, -r["score"]))

    entries = []
    skipped = []

    for rec in candidates:
        ticker = rec["ticker"]
        score = rec["score"]
        is_forced = ticker in force_set

        # N-day persistence filter — bypassed for forced entries
        if not is_forced:
            passes, reason = check_persistence(
                db_conn, ticker, entry_threshold, persistence_days, today
            )
            if not passes:
                skipped.append({
                    "ticker": ticker,
                    "score": score,
                    "reason": reason,
                    "skip_category": SKIP_CATEGORY_PERSISTENCE,
                })
                continue

        # Position cap check
        if current_position_count >= max_positions:
            skipped.append({
                "ticker": ticker,
                "score": score,
                "reason": f"Position cap ({max_positions})",
                "skip_category": SKIP_CATEGORY_POSITION_CAP,
            })
            continue

        # Wash sale check — LOG ONLY, DO NOT BLOCK.
        wash_sale_warning = None
        if ticker in cooldown_map:
            cd = cooldown_map[ticker]
            wash_sale_warning = {
                "loss_exit_date": cd["exit_date"],
                "loss_amount": cd["loss_amount"],
                "cooldown_until": cd["cooldown_until"],
            }

        # Tradeability check
        if not is_tradeable_on_alpaca(client, ticker):
            skipped.append({
                "ticker": ticker,
                "score": score,
                "reason": "Not tradeable on Alpaca",
                "skip_category": SKIP_CATEGORY_OTHER,
            })
            continue

        # ── Sizing (8.3% of equity per slot, bounded by 5% cash floor) ──
        # Validated by entry_mode_backtest.py on 2026-04-16: Limit-3% +
        # 5% cash floor is the recommended config (+433.7% return, 3.16
        # Sharpe, zero neg-cash days vs baseline's 70).
        #
        # Formula: per_slot_cap = (equity × 0.95 − committed) / remaining_slots
        # This guarantees total commitment ≤ 95% of equity even when
        # individual fills gap up and cost more than estimated.
        raw_max_position = equity * max_position_pct
        remaining_slots = max(1, max_positions - current_position_count)
        floor_budget = equity * (1 - CASH_FLOOR_PCT) - projected_committed
        per_slot_cap = floor_budget / remaining_slots if floor_budget > 0 else 0.0
        max_position = max(0.0, min(raw_max_position, per_slot_cap))
        target_size = min(projected_cash, max_position)

        if target_size < min_position_size:
            # Distinguish cash-floor cap (commit budget depleted) from raw
            # cash shortage — matters for the email Skip Reason column.
            if per_slot_cap < raw_max_position - 1e-9:
                skipped.append({
                    "ticker": ticker,
                    "score": score,
                    "reason": (f"Cash floor cap: only ${max_position:,.0f} of 95% "
                               f"equity budget remains across {remaining_slots} slot(s)"),
                    "skip_category": SKIP_CATEGORY_CASH_FLOOR,
                })
            else:
                skipped.append({
                    "ticker": ticker,
                    "score": score,
                    "reason": f"Insufficient cash (${projected_cash:,.0f} available, need ${min_position_size})",
                    "skip_category": SKIP_CATEGORY_INSUFFICIENT_CASH,
                })
            continue

        # ── Pricing: prefer Alpaca's latest trade, fall back to yfinance ──
        # Alpaca's latest reflects extended-hours trading (e.g. pre-market
        # moves after news), which is what will drive the next-day open
        # price. yfinance close is a regular-session-only snapshot — the
        # AEHR 2026-04-16 bug (yfinance $73.22 → actual fill $84.58 on a
        # +15.5% overnight gap) slipped through because yfinance was
        # consulted first. Alpaca-first catches this class of slippage.
        est_price = get_alpaca_latest_price(data_client, ticker)
        if not est_price or est_price <= 0:
            est_price = _estimated_price(rec) or 0.0
        if not est_price or est_price <= 0:
            skipped.append({
                "ticker": ticker,
                "score": score,
                "reason": "No price data available (Alpaca + yfinance both failed)",
                "skip_category": SKIP_CATEGORY_OTHER,
            })
            continue

        qty = int(target_size // est_price)  # whole shares only
        if qty <= 0:
            skipped.append({
                "ticker": ticker,
                "score": score,
                "reason": f"Position size ${target_size:,.0f} too small for share price ${est_price:,.2f}",
                "skip_category": SKIP_CATEGORY_INSUFFICIENT_CASH,
            })
            continue

        cost_basis = qty * est_price
        if is_forced:
            reason_str = f"Force entry (cold-start catch-up)"
        else:
            persistence_tag = (
                f" + {persistence_days}d persist"
                if persistence_days > 0 else ""
            )
            reason_str = f"Score ≥ {entry_threshold}{persistence_tag}"

        entries.append({
            "ticker": ticker,
            "qty": qty,
            "est_price": est_price,
            "cost_basis": cost_basis,
            "score": score,
            "reason": reason_str,
            "wash_sale_warning": wash_sale_warning,
        })
        projected_cash -= cost_basis
        projected_committed += cost_basis
        current_position_count += 1

    return entries, skipped


def _estimated_price(score_record: dict) -> float:
    """
    Pull the most recent close price from the indicator dict.
    Prices live under moving_averages.current_close and
    near_52w_high.current_close — try both.
    """
    ind = score_record.get("indicators", {})

    # Preferred: nested indicators that carry the current close
    for sub in ("moving_averages", "near_52w_high"):
        entry = ind.get(sub)
        if isinstance(entry, dict):
            val = entry.get("current_close")
            if val:
                try:
                    return float(val)
                except (TypeError, ValueError):
                    pass

    # Last-resort scan: any nested dict with a price-ish field
    for v in ind.values():
        if isinstance(v, dict):
            for key in ("current_close", "last_close", "close", "price"):
                val = v.get(key)
                if val:
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        continue
    return 0.0


# ─────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────

def execute_exits(
    client: TradingClient,
    exits: list[dict],
    today: date,
    dry_run: bool,
) -> None:
    """Submit sell orders and log them."""
    for e in exits:
        action = "[DRY RUN] " if dry_run else ""
        print(f"  {action}SELL  {e['ticker']:<6s} {e['qty']:>6.0f} shares @ ${e['current_price']:>8.2f}  "
              f"{e['reason']:<35s} P&L: {'+' if e['unrealized_pnl'] >= 0 else ''}${e['unrealized_pnl']:>9,.0f} "
              f"({e['unrealized_pnl_pct']:+.1f}%)")

        order_id = submit_market_order(
            client, e["ticker"], e["qty"], OrderSide.SELL, dry_run=dry_run
        )

        # Cancel the GTC stop order (no longer needed — we're selling via score exit)
        cancel_stop_orders_for_ticker(client, e["ticker"], dry_run=dry_run)

        # Record wash sale if losing
        if e["unrealized_pnl"] < 0:
            wash_sale_tracker.record_loss_exit(e["ticker"], today, e["unrealized_pnl"])
            until = (wash_sale_tracker.get_blocked_tickers(today)
                     .get(e["ticker"], {})
                     .get("cooldown_until", "?"))
            print(f"                                         → Wash sale recorded: blocked until {until}")

        # Trade log
        if not dry_run:
            buy_date = trade_log.get_last_buy_date(e["ticker"])
            hold_days = 0
            if buy_date:
                hold_days = (today - datetime.strptime(buy_date, "%Y-%m-%d").date()).days
            trade_log.log_sell(
                ticker=e["ticker"],
                trade_date=today,
                price=e["current_price"],
                qty=e["qty"],
                entry_price=e["entry_price"],
                score_at_exit=e["score"],
                reason=e["reason"],
                hold_days=hold_days,
                alpaca_order_id=order_id,
            )


def execute_entries(
    client: TradingClient,
    entries: list[dict],
    today: date,
    trade_cfg: dict,
    dry_run: bool,
) -> None:
    """
    Submit DAY-TIF buy limit orders (limit = sizing_price × 1.03) and
    log them. Stops are NOT placed here — they get attached on the next
    run via ``ensure_stops_for_positions`` after Alpaca confirms the
    limit filled. Orders that don't fill by market close on the next
    trading day auto-cancel (DAY TIF); the ticker naturally re-qualifies
    on subsequent runs if its score stays ≥ entry_threshold.
    """
    for e in entries:
        action = "[DRY RUN] " if dry_run else ""
        sizing_price = e["est_price"]
        limit_price = round(sizing_price * (1 + LIMIT_ORDER_BUFFER), 2)
        print(f"  {action}BUY   {e['ticker']:<6s} {e['qty']:>6.0f} shares  "
              f"sizing ${sizing_price:>8.2f}  limit ${limit_price:>8.2f}  "
              f"Score: {e['score']:<5.1f}  Est Cost: ${e['cost_basis']:>9,.0f}")

        # Wash sale warning (log only — trade proceeds regardless)
        ws = e.get("wash_sale_warning")
        if ws:
            print(f"                                         "
                  f"WASH SALE: ${abs(ws['loss_amount']):,.0f} loss on "
                  f"{ws['loss_exit_date']} (cooldown until {ws['cooldown_until']})")
            print(f"                                         "
                  f"→ Trade proceeding; loss deferred to replacement cost basis.")
            # Record the violation for tax awareness
            wash_sale_tracker.record_violation(e["ticker"], today)

        order_id = submit_limit_order(
            client, e["ticker"], e["qty"], limit_price, dry_run=dry_run,
        )

        if not dry_run:
            trade_log.log_buy(
                ticker=e["ticker"],
                trade_date=today,
                price=sizing_price,  # sizing ref; actual fill recorded separately
                qty=e["qty"],
                score_at_entry=e["score"],
                reason=e["reason"],
                alpaca_order_id=order_id,
                # stop will be attached on next run via ensure_stops_for_positions
            )


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def print_header(dry_run: bool) -> None:
    now = datetime.now(PT_TZ)
    mode = "  [DRY RUN MODE — NO ORDERS WILL BE PLACED]\n" if dry_run else ""
    print("=" * 66)
    print("  ALPHA SCANNER — DAILY TRADE EXECUTION")
    print(f"  {now.strftime('%Y-%m-%d %H:%M %Z')}")
    print("=" * 66)
    if mode:
        print(mode)


def print_trade_config(trade_cfg: dict) -> None:
    print("\n  TRADE CONFIG")
    print("  " + "─" * 38)
    print(f"  Entry threshold:   ≥ {trade_cfg['entry_threshold']}")
    print(f"  Persistence:       {trade_cfg['persistence_days']} prior day(s)")
    print(f"  Exit threshold:    < {trade_cfg['exit_threshold']}")
    print(f"  Stop loss:         {trade_cfg['stop_loss_pct']*100:.0f}% (GTC stop orders)")
    print(f"  Max positions:     {trade_cfg['max_positions']}")
    print(f"  Position size:     {trade_cfg['max_position_pct']*100:.1f}% of equity")
    print(f"  Cash floor:        {CASH_FLOOR_PCT*100:.0f}% of equity (max commit {(1-CASH_FLOOR_PCT)*100:.0f}%)")
    print(f"  Entry order:       LIMIT @ sizing × (1 + {LIMIT_ORDER_BUFFER*100:.0f}%) — DAY TIF")
    print(f"  Pricing source:    Alpaca latest-trade (yfinance fallback)")


def print_account_status(snapshot: dict, trade_cfg: dict | None = None) -> None:
    max_pos = trade_cfg["max_positions"] if trade_cfg else "?"
    print("\n  ACCOUNT STATUS")
    print("  " + "─" * 38)
    print(f"  Equity:          ${snapshot['equity']:>12,.2f}")
    print(f"  Cash:            ${snapshot['cash']:>12,.2f}")
    print(f"  Positions:       {len(snapshot['positions']):>7d} / {max_pos}")


def print_section(title: str) -> None:
    print(f"\n  {title}")
    print("  " + "─" * 38)


def print_positions(snapshot: dict, scores: dict[str, dict], today: date) -> None:
    print_section("PORTFOLIO POSITIONS")
    if not snapshot["positions"]:
        print("  (no open positions)")
        return

    print(f"  {'Ticker':<8s} {'Qty':>6s} {'Entry':>9s} {'Current':>9s} {'P&L':>8s} {'Score':>7s} {'Days':>6s}")
    for ticker, pos in sorted(snapshot["positions"].items()):
        score = scores.get(ticker, {}).get("score", 0.0)
        buy_date = trade_log.get_last_buy_date(ticker)
        hold_days = 0
        if buy_date:
            hold_days = (today - datetime.strptime(buy_date, "%Y-%m-%d").date()).days
        print(f"  {ticker:<8s} {pos['qty']:>6.0f} ${pos['entry_price']:>7.2f} ${pos['current_price']:>7.2f} "
              f"{pos['unrealized_pnl_pct']:>+7.1f}% {score:>7.1f} {hold_days:>6d}")


def print_wash_sale_status(today: date) -> None:
    """Print active cooldowns (informational) and recent violations."""
    cooldowns = wash_sale_tracker.get_cooldowns(today)
    violations = wash_sale_tracker.get_violations()

    print_section("WASH SALE COOLDOWNS (informational — trades not blocked)")
    if not cooldowns:
        print("  (none)")
    else:
        for ticker, entry in sorted(cooldowns.items()):
            print(f"  {ticker:<8s} Cooldown until {entry['cooldown_until']}    "
                  f"Loss: ${entry['loss_amount']:,.0f}")

    if violations:
        print_section("WASH SALE VIOLATIONS LOGGED")
        for v in violations:
            print(f"  {v['ticker']:<8s} re-entered {v['reentry_date']} "
                  f"({v['days_between']}d after loss exit)  "
                  f"Disallowed loss: ${abs(v['loss_amount']):,.0f}")
        total = wash_sale_tracker.total_disallowed_loss()
        print(f"\n  Total disallowed losses (deferred): ${total:,.2f}")


# ─────────────────────────────────────────────────────────────
# Email digest
# ─────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    """Return the hex color for a score value per the tier palette."""
    if score >= 9.5:
        return "#15803d"   # dark green
    if score >= 8.5:
        return "#22c55e"   # green
    if score >= 7.0:
        return "#ca8a04"   # amber
    if score >= 5.0:
        return "#ea580c"   # orange
    return "#dc2626"       # red


def _pnl_color(value: float) -> str:
    """Return green/red/default for a P&L value."""
    if value > 0:
        return "#22c55e"
    if value < 0:
        return "#dc2626"
    return "inherit"


def _fmt_signed_pct(value: float, decimals: int = 2) -> str:
    """Format a percentage with '-' only for negatives (no '+' prefix)."""
    return f"{value:.{decimals}f}%"


def _fmt_signed_dollar(value: float) -> str:
    """Format a dollar amount with '-' only for negatives (no '+' prefix)."""
    if value < 0:
        return f"-${abs(value):,.0f}"
    return f"${value:,.0f}"


# Two-letter subsector codes used in the email digest (compact columns).
# Key = full subsector display name as stored on each score record;
# value = 2-letter code. Any subsector not in this dict falls back to
# its full name (see _subsector_code()).
SUBSECTOR_CODES: dict[str, str] = {
    # AI & Tech Capex Cycle
    "Chips — Compute":                      "CC",
    "Chips — Memory":                       "CM",
    "Chips — Networking / Photonics":       "CN",
    "Semiconductor Equipment / Foundry":    "SE",
    "Semiconductor Test / Burn-In":         "ST",
    "Power Semiconductors (GaN/SiC)":       "PS",
    "Power & Energy":                       "PE",
    "Data Center Infrastructure":           "DC",
    "Alternative AI Compute / GPU Hosts":   "AC",
    "Hyperscalers":                         "HY",
    "AI Software / DevTools":               "AS",
    "Enterprise AI Apps":                   "EA",
    "AI Security":                          "AX",
    "Healthcare AI":                        "HA",
    "Physical AI / Robotics":               "PR",
    # Robotics & Automation
    "Industrial Robotics & Automation":     "IR",
    "Surgical / Medical Robotics":          "SR",
    "Subsea / Ocean Robotics":              "SO",
    # Space & Satellite
    "Satellite Communications & Data":      "SC",
    "Launch & Spacecraft":                  "LS",
    # Nuclear & Uranium
    "Nuclear Reactors / SMR":               "NR",
    "Uranium Miners":                       "UM",
    # Metals
    "Gold Miners":                          "GM",
    "Silver Miners":                        "SM",
    "Gold & Silver — Direct Exposure":      "GS",
    # eVTOL & Drones
    "eVTOL / Urban Air Mobility":           "EV",
    # Quantum
    "Quantum Hardware & Software":          "QC",
    # Crypto
    "Crypto Majors":                        "CX",
    "Crypto / AI Crossover Tokens":         "CY",
    # Biotech
    "Gene Editing / CRISPR":                "GE",
    "Synthetic Biology":                    "SB",
}


def _subsector_code(full_name: str) -> str:
    """2-letter code for a subsector. Falls back to full name if missing."""
    if not full_name:
        return "—"
    return SUBSECTOR_CODES.get(full_name, full_name)


def _get_spy_return_since(
    price_data: dict,
    start_date: date,
) -> float | None:
    """
    Compute SPY total return % from ``start_date`` to the most recent
    price available in ``price_data``. Returns None if SPY data isn't
    available or doesn't cover the range.
    """
    import pandas as pd

    spy = price_data.get("SPY") if price_data else None
    if spy is None or spy.empty:
        return None

    # Work in a naive index for comparison but use positional access on
    # the original DataFrame (which may carry a tz-aware index).
    idx_naive = spy.index.tz_localize(None) if spy.index.tz else spy.index
    target = pd.Timestamp(start_date)

    mask = idx_naive >= target
    if not mask.any():
        return None

    pos_start = int(mask.argmax())  # first True position
    pos_end = len(spy) - 1

    try:
        start_price = float(spy.iloc[pos_start]["Close"])
        end_price = float(spy.iloc[pos_end]["Close"])
    except (KeyError, ValueError, IndexError):
        return None

    if start_price <= 0:
        return None
    return (end_price / start_price - 1.0) * 100.0


def _categorize_skip_reason(skip: dict) -> str:
    """Normalize a skip entry's reason into one of the email categories."""
    # Explicit skip_category wins (set by evaluate_entries)
    cat = skip.get("skip_category")
    if cat:
        return cat
    # Fallback text-match for older skip dicts (e.g. from other call sites)
    reason = (skip.get("reason") or "").lower()
    if "cash floor" in reason:
        return SKIP_CATEGORY_CASH_FLOOR
    if "limit unfilled" in reason or "prior limit" in reason:
        return SKIP_CATEGORY_LIMIT_UNFILLED
    if "insufficient cash" in reason or "too small" in reason:
        return SKIP_CATEGORY_INSUFFICIENT_CASH
    if "position cap" in reason:
        return SKIP_CATEGORY_POSITION_CAP
    if "wash sale" in reason:
        return SKIP_CATEGORY_WASH_SALE
    if "persistence" in reason:
        return SKIP_CATEGORY_PERSISTENCE
    return SKIP_CATEGORY_OTHER


def _colored(value_str: str, color: str) -> str:
    """Wrap text in a color span (skip if 'inherit')."""
    if color == "inherit":
        return value_str
    return f"<span style='color:{color}'>{value_str}</span>"


def _count_consecutive_days_above(
    db_conn,
    ticker: str,
    threshold: float,
    today_str: str,
    today_score: float | None = None,
) -> int:
    """
    Count consecutive most-recent trading days (up to and including today)
    where the ticker's score was >= threshold.
    """
    # Start with today's score if available
    count = 0
    if today_score is not None and today_score >= threshold:
        count = 1
    else:
        return 0  # today doesn't meet threshold, streak is 0

    cur = db_conn.cursor()
    cur.execute(
        """SELECT score FROM ticker_scores
           WHERE ticker = ? AND date < ?
           ORDER BY date DESC
           LIMIT 10""",
        (ticker, today_str),
    )
    for (score,) in cur.fetchall():
        if score is not None and score >= threshold:
            count += 1
        else:
            break
    return count


def _count_consecutive_days_below(
    db_conn,
    ticker: str,
    threshold: float,
    today_str: str,
    today_score: float | None = None,
) -> int:
    """
    Count consecutive most-recent trading days (up to and including today)
    where the ticker's score was < threshold.
    """
    count = 0
    if today_score is not None and today_score < threshold:
        count = 1
    else:
        return 0

    cur = db_conn.cursor()
    cur.execute(
        """SELECT score FROM ticker_scores
           WHERE ticker = ? AND date < ?
           ORDER BY date DESC
           LIMIT 30""",
        (ticker, today_str),
    )
    for (score,) in cur.fetchall():
        if score is not None and score < threshold:
            count += 1
        else:
            break
    return count


STARTING_EQUITY = 100_000.0  # Paper account baseline (see Alpaca ph.base_value)


def _build_trade_digest_html(
    snapshot: dict,
    exits: list[dict],
    entries: list[dict],
    skipped: list[dict],
    scores: dict[str, dict],
    today: date,
    dry_run: bool,
    trade_cfg: dict,
    db_conn,
    price_data: dict | None = None,
    account_created: date | None = None,
) -> tuple[str, str]:
    """Build (subject, html) for the daily trade execution digest."""
    equity = snapshot["equity"]
    cash = snapshot["cash"]
    last_equity = snapshot.get("last_equity")
    num_pos = len(snapshot["positions"])
    today_str = today.strftime("%Y-%m-%d")

    # ── Today's P&L (equity delta from yesterday's close) ─────────
    # This single metric covers both unrealized changes and realized
    # P&L from today's fills, so it stays correct once exits begin.
    if last_equity and last_equity > 0:
        today_pnl_dollar = equity - last_equity
        today_pnl_pct = today_pnl_dollar / last_equity * 100
        today_pct_str = _fmt_signed_pct(today_pnl_pct, decimals=2)
    else:
        today_pnl_dollar = 0.0
        today_pnl_pct = 0.0
        today_pct_str = "N/A"

    # ── All-Time P&L (vs starting $100k equity) ───────────────────
    alltime_pnl_dollar = equity - STARTING_EQUITY
    alltime_pnl_pct = alltime_pnl_dollar / STARTING_EQUITY * 100

    # ── vs SPY total (portfolio all-time return − SPY return) ────
    spy_ret_pct = None
    if price_data and account_created:
        spy_ret_pct = _get_spy_return_since(price_data, account_created)
    if spy_ret_pct is not None:
        vs_spy_pct = alltime_pnl_pct - spy_ret_pct
        vs_spy_dollar = vs_spy_pct / 100 * STARTING_EQUITY
    else:
        vs_spy_pct = None
        vs_spy_dollar = None

    dry_tag = " [DRY RUN]" if dry_run else ""
    subject = (f"Alpha Scanner {today.strftime('%m/%d')}: "
               f"{today_pct_str} | "
               f"{len(entries)} buy / {len(exits)} sell{dry_tag}")

    # ── HTML helpers ─────────────────────────────────────────────
    # white-space:nowrap on every <th> prevents mobile clients from
    # breaking single-word headers mid-word (e.g. "Current" → "Cur rent").
    td_left = "padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:left"
    td_center = "padding:6px 10px;border-bottom:1px solid #e5e7eb;text-align:center"
    th_left = ("padding:6px 10px;text-align:left;background:#f3f4f6;"
               "border-bottom:2px solid #d1d5db;white-space:nowrap")
    th_center = ("padding:6px 10px;text-align:center;background:#f3f4f6;"
                 "border-bottom:2px solid #d1d5db;white-space:nowrap")

    def row_mixed(cells: list[str], aligns: list[str]) -> str:
        """Row where first cell is left-aligned, rest are center-aligned."""
        parts = []
        for c, a in zip(cells, aligns):
            style = td_left if a == "left" else td_center
            parts.append(f'<td style="{style}">{c}</td>')
        return "<tr>" + "".join(parts) + "</tr>"

    def header_mixed(cells: list[str], aligns: list[str]) -> str:
        parts = []
        for c, a in zip(cells, aligns):
            style = th_left if a == "left" else th_center
            parts.append(f'<th style="{style}">{c}</th>')
        return "<tr>" + "".join(parts) + "</tr>"

    def table_mixed(header: list[str], rows: list[str], aligns: list[str]) -> str:
        """
        Fixed-layout table with per-column alignment. All columns get
        equal width (100% / n_cols). This ensures two same-column-count
        tables stack aligned — e.g. Entries (5 cols) and Skipped (5 cols)
        share the same horizontal grid. Ticker column gets the same
        width as the others; it just uses left alignment.
        """
        if not rows:
            return "<p style='color:#6b7280'>(none)</p>"
        n_cols = len(aligns)
        col_widths = ""
        if n_cols > 0:
            col_pct = round(100 / n_cols, 2)
            col_widths = f'<col style="width:{col_pct}%"/>' * n_cols
        return (
            "<table style='border-collapse:collapse;width:100%;"
            "font-size:13px;table-layout:fixed'>"
            + col_widths
            + header_mixed(header, aligns)
            + "".join(rows) + "</table>"
        )

    # Back-compat helpers for unchanged tables (Exits, Entries, Cooldowns)
    def row(cells: list[str]) -> str:
        return "<tr>" + "".join(
            f'<td style="{td_left}">{c}</td>' for c in cells
        ) + "</tr>"

    def header_row(cells: list[str]) -> str:
        return "<tr>" + "".join(
            f'<th style="{th_left}">{c}</th>' for c in cells
        ) + "</tr>"

    def table(header: list[str], rows: list[str]) -> str:
        if not rows:
            return "<p style='color:#6b7280'>(none)</p>"
        return ("<table style='border-collapse:collapse;width:100%;font-size:13px'>"
                + header_row(header)
                + "".join(rows) + "</table>")

    # Subsector column uses 2-letter codes; any code we render gets
    # added to used_subsectors so the legend at the bottom only lists
    # codes actually referenced in today's email.
    used_subsectors: dict[str, str] = {}   # code → full name

    def subsector_for(ticker: str) -> str:
        full = scores.get(ticker, {}).get("subsector", "") or ""
        if not full:
            return "—"
        code = _subsector_code(full)
        used_subsectors[code] = full
        return code

    # ── Exits ──
    exit_rows = []
    for e in exits:
        color = _pnl_color(e["unrealized_pnl"])
        pnl_str = (f"{_fmt_signed_dollar(e['unrealized_pnl'])} "
                   f"({_fmt_signed_pct(e['unrealized_pnl_pct'], 1)})")
        exit_rows.append(row([
            f"<b>{e['ticker']}</b>",
            f"{e['qty']:.0f}",
            f"${e['current_price']:.2f}",
            e["reason"],
            _colored(pnl_str, color),
        ]))

    # ── Entries (score color-coded) ──
    entry_aligns = ["left", "center", "center", "center", "center"]
    entry_rows = []
    for e in entries:
        ws_note = ""
        if e.get("wash_sale_warning"):
            ws_note = (f" <span style='color:#f59e0b;font-size:11px'>"
                       f"⚠ wash sale (${abs(e['wash_sale_warning']['loss_amount']):,.0f} deferred)</span>")
        entry_rows.append(row_mixed([
            f"<b>{e['ticker']}</b>{ws_note}",
            f"{e['qty']:.0f}",
            f"${e['est_price']:.2f}",
            _colored(f"{e['score']:.1f}", _score_color(e['score'])),
            f"${e['cost_basis']:,.0f}",
        ], entry_aligns))

    # ── Skipped Signals ──
    # Columns: Ticker | Subsector | Score | Days ≥ threshold | Skip Reason
    # Sorted by days_above DESC (most persistent at top — most likely to enter tomorrow).
    entry_threshold = trade_cfg.get("entry_threshold", 8.5)
    persistence_days = trade_cfg.get("persistence_days", 3)
    skip_records = []
    for s in skipped:
        ticker = s["ticker"]
        score = s["score"]
        days_above = _count_consecutive_days_above(
            db_conn, ticker, entry_threshold, today_str, today_score=score,
        )
        skip_records.append({
            "ticker": ticker,
            "subsector": subsector_for(ticker),
            "score": score,
            "days_above": days_above,
            "category": _categorize_skip_reason(s),
        })
    skip_records.sort(key=lambda r: (-r["days_above"], -r["score"]))

    skip_aligns = ["left", "center", "center", "center", "center"]
    # Denominator is persistence_days + 1 so a full pass (today + N prior
    # days all ≥ threshold) displays as e.g. "4/4" with persistence_days=3.
    streak_denom = persistence_days + 1
    skip_rows = [
        row_mixed([
            f"<b>{r['ticker']}</b>",
            r["subsector"],
            _colored(f"{r['score']:.1f}", _score_color(r["score"])),
            f"{r['days_above']}/{streak_denom}",
            r["category"],
        ], skip_aligns)
        for r in skip_records
    ]

    # ── Current Positions ──
    # Columns: Ticker | Days | Day % | P&L % | P&L $ | Score
    # Sorted by P&L % DESC. Sub Sector column was dropped to free mobile
    # horizontal space (subsector info is still visible in the Skipped
    # Signals table's Sub Sector column, which drives the legend).
    max_positions = trade_cfg.get("max_positions", 12)
    pos_records = []
    exit_watch_rows = []
    stop_loss_pct = trade_cfg.get("stop_loss_pct", 0.20)

    for ticker, pos in snapshot["positions"].items():
        score_val = scores.get(ticker, {}).get("score", 0.0)
        buy_date = trade_log.get_last_buy_date(ticker)
        hold_days = (
            (today - datetime.strptime(buy_date, "%Y-%m-%d").date()).days
            if buy_date else 0
        )
        pos_records.append({
            "ticker": ticker,
            "hold_days": hold_days,
            "current_price": pos["current_price"],
            "day_pct": pos.get("change_today_pct", 0.0),
            "pnl_pct": pos["unrealized_pnl_pct"],
            "pnl_dollar": pos["unrealized_pnl"],
            "score": score_val,
        })

        # ── Exit Watch: score < 7.0 OR P&L% worse than -(stop_loss_pct*100 - 5)% ──
        stop_watch_pct = -(stop_loss_pct * 100 - 5)
        if score_val < 7.0 or pos["unrealized_pnl_pct"] < stop_watch_pct:
            days_below_7 = _count_consecutive_days_below(
                db_conn, ticker, 7.0, today_str, today_score=score_val,
            )
            distance_to_exit = score_val - 5.0
            pnl_pct_v = pos["unrealized_pnl_pct"]
            buffer_color = "#dc2626" if pnl_pct_v < stop_watch_pct else _pnl_color(pnl_pct_v)
            exit_watch_rows.append(row([
                f"<b>{ticker}</b>",
                _colored(f"{score_val:.1f}", _score_color(score_val)),
                f"{days_below_7}",
                f"{distance_to_exit:.1f}",
                _colored(_fmt_signed_pct(pnl_pct_v, 1), buffer_color),
            ]))

    pos_records.sort(key=lambda r: -r["pnl_pct"])

    pos_aligns = ["left", "center", "center", "center", "center", "center"]
    pos_rows = [
        row_mixed([
            f"<b>{r['ticker']}</b>",
            f"{r['hold_days']}",
            _colored(_fmt_signed_pct(r["day_pct"], 1), _pnl_color(r["day_pct"])),
            _colored(_fmt_signed_pct(r["pnl_pct"], 1), _pnl_color(r["pnl_pct"])),
            _colored(_fmt_signed_dollar(r["pnl_dollar"]), _pnl_color(r["pnl_dollar"])),
            _colored(f"{r['score']:.1f}", _score_color(r["score"])),
        ], pos_aligns)
        for r in pos_records
    ]

    exit_watch_html = (
        table(
            ["Ticker", "Current Score", "Days Score &lt; 7", "Distance to Exit", "Stop Loss Buffer"],
            exit_watch_rows,
        )
        if exit_watch_rows
        else "<p style='color:#6b7280'>(none — all positions healthy)</p>"
    )

    # ── Wash Sale Cooldowns ──
    cooldowns = wash_sale_tracker.get_cooldowns(today)
    cooldown_rows = [
        row([f"<b>{t}</b>", e["cooldown_until"], f"${e['loss_amount']:,.0f}"])
        for t, e in sorted(cooldowns.items())
    ]

    # Pre-build tables with headers
    skip_table_html = table_mixed(
        ["Ticker", "Sub Sector", "Score", "Streak", "Skip Reason"],
        skip_rows,
        skip_aligns,
    )
    exit_table_html = table(
        ["Ticker", "Qty", "Price", "Reason", "P&amp;L"], exit_rows,
    )
    entry_table_html = table_mixed(
        ["Ticker", "Qty", "Est Price", "Score", "Cost"], entry_rows,
        entry_aligns,
    )
    pos_table_html = table_mixed(
        ["Ticker", "Days", "Day %", "P&amp;L %", "P&amp;L $", "Score"],
        pos_rows,
        pos_aligns,
    )
    cooldown_table_html = table(
        ["Ticker", "Cooldown Until", "Loss Amount"], cooldown_rows,
    )

    # ── Summary cards (two columns) ──
    today_color = _pnl_color(today_pnl_dollar)
    alltime_color = _pnl_color(alltime_pnl_dollar)
    cash_color = "#dc2626" if cash < 0 else "inherit"

    today_pct_cell = _colored(_fmt_signed_pct(today_pnl_pct, 2), today_color)
    today_dol_cell = _colored(_fmt_signed_dollar(today_pnl_dollar), today_color)
    alltime_pct_cell = _colored(_fmt_signed_pct(alltime_pnl_pct, 2), alltime_color)
    alltime_dol_cell = _colored(_fmt_signed_dollar(alltime_pnl_dollar), alltime_color)

    if vs_spy_pct is not None:
        vs_spy_color = _pnl_color(vs_spy_pct)
        vs_spy_pct_cell = _colored(_fmt_signed_pct(vs_spy_pct, 2), vs_spy_color)
        vs_spy_dol_cell = _colored(_fmt_signed_dollar(vs_spy_dollar), vs_spy_color)
    else:
        vs_spy_pct_cell = "<span style='color:#9ca3af'>n/a</span>"
        vs_spy_dol_cell = "<span style='color:#9ca3af'>—</span>"

    card_base = ("background:#f9fafb;padding:14px 18px;border-radius:8px;"
                 "vertical-align:top")
    kv_label = "padding:3px 12px 3px 0;color:#6b7280"
    kv_num_pct = "padding:3px 10px 3px 0;text-align:right;font-variant-numeric:tabular-nums"
    kv_num_dol = "padding:3px 0;text-align:right;font-variant-numeric:tabular-nums"

    pl_card = f"""
      <td style="{card_base};width:50%">
        <div style="font-weight:600;color:#111;margin-bottom:8px">P&amp;L</div>
        <table style="border-collapse:collapse;font-size:14px;width:100%">
          <tr>
            <td style="{kv_label}">Today</td>
            <td style="{kv_num_pct}">{today_pct_cell}</td>
            <td style="{kv_num_dol}">{today_dol_cell}</td>
          </tr>
          <tr>
            <td style="{kv_label}">All-Time</td>
            <td style="{kv_num_pct}">{alltime_pct_cell}</td>
            <td style="{kv_num_dol}">{alltime_dol_cell}</td>
          </tr>
          <tr><td colspan="3" style="padding:6px 0 0 0;border-top:1px solid #e5e7eb"></td></tr>
          <tr>
            <td style="{kv_label}">vs SPY total</td>
            <td style="{kv_num_pct}">{vs_spy_pct_cell}</td>
            <td style="{kv_num_dol}">{vs_spy_dol_cell}</td>
          </tr>
        </table>
      </td>
    """.strip()

    account_card = f"""
      <td style="{card_base};width:50%">
        <div style="font-weight:600;color:#111;margin-bottom:8px">Account</div>
        <table style="border-collapse:collapse;font-size:14px;width:100%">
          <tr>
            <td style="{kv_label}">Equity</td>
            <td style="{kv_num_dol}"><b>${equity:,.2f}</b></td>
          </tr>
          <tr>
            <td style="{kv_label}">Cash</td>
            <td style="{kv_num_dol}">{_colored(f"${cash:,.2f}", cash_color)}</td>
          </tr>
          <tr>
            <td style="{kv_label}">Positions</td>
            <td style="{kv_num_dol}">{num_pos} / {max_positions}</td>
          </tr>
        </table>
      </td>
    """.strip()

    summary_block = f"""
      <table style="border-collapse:separate;border-spacing:12px 0;width:100%;margin-bottom:20px">
        <tr>{pl_card}{account_card}</tr>
      </table>
    """.strip()

    # ── Subsector legend (only codes that appeared in the email today) ──
    if used_subsectors:
        legend_rows = "".join(
            f"<div style='font-size:12px;color:#374151;margin:2px 0'>"
            f"<b style='display:inline-block;min-width:32px'>{code}</b> — {name}"
            f"</div>"
            for code, name in sorted(used_subsectors.items())
        )
        legend_html = (
            "<h3 style='margin:24px 0 8px 0'>Subsectors referenced</h3>"
            f"<div style='padding:6px 0'>{legend_rows}</div>"
        )
    else:
        legend_html = ""

    html = f"""
    <div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:720px;margin:0 auto;color:#111">
      <h2 style="margin:0 0 8px 0">Alpha Scanner — Daily Summary{dry_tag}</h2>
      <p style="color:#6b7280;margin:0 0 20px 0">{today.strftime('%A, %B %d, %Y')}</p>

      {summary_block}

      <h3 style="margin:24px 0 8px 0">Exits ({len(exits)})</h3>
      {exit_table_html}

      <h3 style="margin:24px 0 8px 0">Entries ({len(entries)})</h3>
      {entry_table_html}

      <h3 style="margin:24px 0 8px 0">Skipped Signals ({len(skipped)})</h3>
      {skip_table_html}

      <h3 style="margin:24px 0 8px 0">Current Positions ({num_pos})</h3>
      {pos_table_html}

      <h3 style="margin:24px 0 8px 0">Exit Watch</h3>
      {exit_watch_html}

      <h3 style="margin:24px 0 8px 0">Wash Sale Cooldowns (informational — trades not blocked)</h3>
      {cooldown_table_html}

      {legend_html}

      <p style="color:#9ca3af;font-size:11px;margin-top:30px">
        Generated by trade_executor.py &bull; Paper trading on Alpaca
      </p>
    </div>
    """.strip()

    return subject, html


def _send_gmail(
    gmail_address: str,
    gmail_app_password: str,
    recipient: str,
    subject: str,
    html_body: str,
    text_body: str = "",
) -> None:
    """
    Send a multipart HTML email via Gmail SMTP (smtp.gmail.com:587, STARTTLS).
    Raises on failure.
    """
    msg = MIMEMultipart("alternative")
    msg["From"] = gmail_address
    msg["To"] = recipient
    msg["Subject"] = subject

    if text_body:
        msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(gmail_address, gmail_app_password)
        server.sendmail(gmail_address, [recipient], msg.as_string())


def send_trade_digest(
    snapshot: dict,
    exits: list[dict],
    entries: list[dict],
    skipped: list[dict],
    scores: dict[str, dict],
    today: date,
    dry_run: bool,
    trade_cfg: dict,
    db_conn,
    price_data: dict | None = None,
    account_created: date | None = None,
) -> None:
    """Send daily trade digest via Gmail SMTP if credentials are configured."""
    gmail_address = os.getenv("GMAIL_ADDRESS", "")
    gmail_app_password = os.getenv("GMAIL_APP_PASSWORD", "")

    if not gmail_address:
        print("\n  [email] GMAIL_ADDRESS not set — skipping digest")
        return
    if not gmail_app_password:
        print("\n  [email] GMAIL_APP_PASSWORD not set — skipping digest")
        return

    # Default recipient is the sender; override with ALERT_EMAIL_TO if set
    recipient = os.getenv("ALERT_EMAIL_TO", gmail_address)

    try:
        subject, html = _build_trade_digest_html(
            snapshot, exits, entries, skipped, scores, today, dry_run,
            trade_cfg, db_conn,
            price_data=price_data,
            account_created=account_created,
        )
        print(f"\n  [email] Sending digest to {recipient}...")
        _send_gmail(
            gmail_address=gmail_address,
            gmail_app_password=gmail_app_password,
            recipient=recipient,
            subject=subject,
            html_body=html,
            text_body=f"{subject}\n\n(HTML version required to view full digest)",
        )
        print(f"  [email] Sent.")
    except Exception as e:
        print(f"\n  [email] Failed to send digest: {e}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Daily Alpaca paper-trade execution")
    parser.add_argument("--dry-run", action="store_true",
                        help="Score everything and show what would be done, but don't place orders")
    parser.add_argument("--email", action="store_true",
                        help="Send an email digest after running (requires GMAIL_ADDRESS and GMAIL_APP_PASSWORD)")
    parser.add_argument("--preview-email", nargs="?", const="email_preview.html", default=None,
                        metavar="PATH",
                        help="Render the email digest to an HTML file (default: email_preview.html) "
                             "without sending — open in a browser to preview layout.")
    parser.add_argument("--force-entry", action="append", default=[], metavar="TICKER",
                        help="Force entry for TICKER (bypasses entry threshold + persistence filter). "
                             "Can be specified multiple times. All other rules (sizing, tradeability, "
                             "wash sale logging) still apply.")
    args = parser.parse_args()
    # Normalize force-entry tickers to uppercase
    args.force_entry = [t.upper() for t in args.force_entry]

    load_dotenv()

    print_header(args.dry_run)

    # 1. Load config (file + env var overrides)
    cfg = load_config()
    trade_cfg = load_trade_config(cfg)
    print_trade_config(trade_cfg)

    # 2. Connect
    client = connect_alpaca()
    data_client = connect_alpaca_data()

    # 2b. Account inception (used for SPY comparison in email digest)
    try:
        _acct = client.get_account()
        account_created = (_acct.created_at.date()
                           if getattr(_acct, "created_at", None) else None)
    except APIError:
        account_created = None

    # 3. Snapshot account
    snapshot = get_account_snapshot(client)
    print_account_status(snapshot, trade_cfg)

    # 4. Cleanup expired wash sale entries
    today = datetime.now(PT_TZ).date()
    wash_sale_tracker.cleanup_expired(today)

    # 5. Detect stop orders that filled since last run
    print_section("STOP ORDER FILLS (since last run)")
    filled_stops = detect_filled_stops(client, snapshot, today, args.dry_run)
    if filled_stops:
        for fs in filled_stops:
            pnl = (fs["fill_price"] - fs["entry_price"]) * fs["qty"]
            pnl_pct = ((fs["fill_price"] / fs["entry_price"]) - 1.0) * 100.0 if fs["entry_price"] else 0.0
            print(f"  STOP FILLED  {fs['ticker']:<6s} {fs['qty']:>6.0f} shares @ ${fs['fill_price']:>8.2f}  "
                  f"P&L: ${pnl:+,.0f} ({pnl_pct:+.1f}%)")

            # Log the exit
            if not args.dry_run:
                buy_date = trade_log.get_last_buy_date(fs["ticker"])
                hold_days = 0
                if buy_date:
                    hold_days = (today - datetime.strptime(buy_date, "%Y-%m-%d").date()).days
                trade_log.log_sell(
                    ticker=fs["ticker"],
                    trade_date=today,
                    price=fs["fill_price"],
                    qty=fs["qty"],
                    entry_price=fs["entry_price"],
                    score_at_exit=0.0,  # score unknown at time of stop fill
                    reason=f"Stop loss hit ({trade_cfg['stop_loss_pct']*100:.0f}%)",
                    hold_days=hold_days,
                    alpaca_order_id=fs["order_id"],
                )

            # Record wash sale if losing
            if pnl < 0:
                wash_sale_tracker.record_loss_exit(fs["ticker"], today, pnl)
    else:
        print("  (none)")

    # Re-snapshot after processing stop fills (positions may have changed)
    if filled_stops and not args.dry_run:
        snapshot = get_account_snapshot(client)

    # 5b. Ensure every held position has a GTC stop order. Needed because
    #     entries now use DAY-TIF limit orders — stops are attached here
    #     after Alpaca confirms the limit filled (i.e. the position shows
    #     up in the snapshot), not at submission time.
    print_section("STOP ORDER BACKFILL")
    placed_stops = ensure_stops_for_positions(client, snapshot, trade_cfg, args.dry_run)
    if placed_stops:
        print(f"  Placed stops for {placed_stops} position(s) missing one")
    else:
        print("  (all held positions already have stops)")

    # 5c. Detect limit orders that didn't fill since last run (for email
    #     context). A 24-48h cutoff captures yesterday's submissions.
    from datetime import timedelta, timezone as _tz
    cutoff_dt = datetime.now(_tz.utc) - timedelta(hours=36)
    unfilled_limits = detect_unfilled_limits_since(client, cutoff_dt)
    if unfilled_limits:
        print_section("UNFILLED LIMIT ORDERS (prior run)")
        for u in unfilled_limits:
            print(f"  {u['ticker']:<6s} submitted {u['submitted_date']}  "
                  f"limit ${u['limit_price']:.2f}  status {u['status']}")

    # 6. Open DB (used for persistence filter + writing today's scores)
    db_conn = subsector_store.init_db()

    # 7. Score all tickers
    print("\n  Scoring tickers...")
    price_data = fetch_all(cfg, period="1y", verbose=False)
    results = score_all(price_data, cfg)
    scores = score_lookup(results)
    print(f"  Scored {len(results)} tickers")

    # 8. Persist today's scores so tomorrow's persistence check has them
    #    (safe to call every run — upsert replaces existing rows)
    today_str = today.strftime("%Y-%m-%d")
    try:
        subsector_store.upsert_ticker_scores(db_conn, today_str, results)
    except Exception as e:
        print(f"  [WARN] Could not persist today's scores to DB: {e}")

    # 9. Evaluate exits (score-based)
    exits = evaluate_exits(snapshot, scores, trade_cfg)
    print_section("EXITS TODAY")
    if exits:
        execute_exits(client, exits, today, args.dry_run)
    else:
        print("  (none)")

    # 10. Evaluate entries
    if args.force_entry:
        print(f"\n  [force-entry] Forcing entry for: {', '.join(args.force_entry)}")
    entries, skipped = evaluate_entries(
        snapshot, scores, exits, client, data_client, today, trade_cfg, db_conn,
        force_entry=args.force_entry,
    )
    print_section("ENTRIES TODAY")
    if entries:
        execute_entries(client, entries, today, trade_cfg, args.dry_run)
    else:
        print("  (none)")

    # 10b. Surface unfilled prior-day limits whose score has now dropped
    #      (i.e. they won't be retried). If the score is still ≥ threshold
    #      they appear naturally in today's entries/skipped, so no need to
    #      add them — only dropped ones get an explicit row.
    entry_threshold = trade_cfg["entry_threshold"]
    entered_or_skipped = (
        {e["ticker"] for e in entries} | {s["ticker"] for s in skipped}
    )
    for u in unfilled_limits:
        tk = u["ticker"]
        if tk in entered_or_skipped:
            continue
        score_rec = scores.get(tk)
        today_score = float(score_rec["score"]) if score_rec else 0.0
        if today_score >= entry_threshold:
            continue  # still eligible — a retry will cover it
        skipped.append({
            "ticker": tk,
            "score": today_score,
            "reason": (f"Prior limit unfilled ({u['submitted_date']} @ ${u['limit_price']:.2f}) "
                       f"and score {today_score:.1f} < {entry_threshold}"),
            "skip_category": SKIP_CATEGORY_LIMIT_UNFILLED,
        })

    # 11. Skipped
    print_section("SKIPPED SIGNALS")
    if skipped:
        for s in skipped:
            print(f"  SKIP  {s['ticker']:<6s} Score: {s['score']:<5.1f} Reason: {s['reason']}")
    else:
        print("  (none)")

    # 12. Re-fetch snapshot after orders (live mode only) for position display
    if not args.dry_run and (exits or entries):
        snapshot = get_account_snapshot(client)

    print_positions(snapshot, scores, today)
    print_wash_sale_status(today)

    # 13. Optional email digest — send via Gmail
    if args.email:
        send_trade_digest(
            snapshot, exits, entries, skipped, scores, today,
            args.dry_run, trade_cfg, db_conn,
            price_data=price_data,
            account_created=account_created,
        )

    # 14. Optional email preview — render HTML to file without sending
    if args.preview_email:
        preview_path = args.preview_email
        subject, html = _build_trade_digest_html(
            snapshot, exits, entries, skipped, scores, today,
            args.dry_run, trade_cfg, db_conn,
            price_data=price_data,
            account_created=account_created,
        )
        with open(preview_path, "w") as f:
            f.write(html)
        print(f"\n  [preview] Subject: {subject}")
        print(f"  [preview] Wrote email HTML to {preview_path}")
        print(f"  [preview] Open it in a browser: open {preview_path}")

    print("\n" + "=" * 66 + "\n")


if __name__ == "__main__":
    main()
