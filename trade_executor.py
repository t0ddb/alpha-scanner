"""
trade_executor.py — Daily paper-trade execution against Alpaca.

Runs after market close. Scores all tickers, evaluates exits on open
positions, and places entry orders on new breakout signals.

Entry config comes from `trade_execution:` in ticker_config.yaml and can
be overridden by environment variables. The current default is
threshold ≥ 8.5 with a 3-day persistence filter (score must have been
at/above the threshold for the prior 3 trading days), exit < 5, and a
15% stop loss — the winning 3yr backtest config.

Usage:
    python3 trade_executor.py              # live paper trading
    python3 trade_executor.py --dry-run    # show what it WOULD do, no orders

Env vars (store in .env or CI secrets):
    ALPACA_API_KEY       — Alpaca paper trading API key (required)
    ALPACA_SECRET_KEY    — Alpaca paper trading secret (required)
    ENTRY_THRESHOLD      — override config entry score (e.g. "8.5")
    EXIT_THRESHOLD       — override config exit score  (e.g. "5.0")
    PERSISTENCE_DAYS     — override config prior-day persistence (e.g. "3")
    STOP_LOSS_PCT        — override config stop loss   (e.g. "0.15")
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
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
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
    "stop_loss_pct": 0.15,
    "max_position_pct": 0.20,
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


def evaluate_entries(
    snapshot: dict,
    scores: dict[str, dict],
    exits: list[dict],
    client: TradingClient,
    data_client: StockHistoricalDataClient,
    today: date,
    trade_cfg: dict,
    db_conn,
) -> tuple[list[dict], list[dict]]:
    """
    Return (entries, skipped) for today.
    """
    entry_threshold = trade_cfg["entry_threshold"]
    persistence_days = trade_cfg["persistence_days"]
    max_position_pct = trade_cfg["max_position_pct"]
    min_position_size = trade_cfg["min_position_size"]

    held = set(snapshot["positions"].keys())
    exiting = {e["ticker"] for e in exits}
    cooldown_map = wash_sale_tracker.get_cooldowns(today)

    # Estimate post-exit cash: current cash + proceeds from sells
    projected_cash = snapshot["cash"] + sum(
        e["qty"] * e["current_price"] for e in exits
    )
    equity = snapshot["equity"]
    max_position = equity * max_position_pct

    # Filter candidates
    candidates = []
    for ticker, rec in scores.items():
        score = rec["score"]
        if score < entry_threshold:
            continue
        if ticker in held and ticker not in exiting:
            continue  # already own it
        if is_ticker_excluded(ticker):
            continue
        candidates.append(rec)

    candidates.sort(key=lambda r: -r["score"])

    entries = []
    skipped = []

    for rec in candidates:
        ticker = rec["ticker"]
        score = rec["score"]

        # N-day persistence filter — score must have been ≥ threshold
        # on each of the prior `persistence_days` trading days
        passes, reason = check_persistence(
            db_conn, ticker, entry_threshold, persistence_days, today
        )
        if not passes:
            skipped.append({
                "ticker": ticker,
                "score": score,
                "reason": reason,
            })
            continue

        # Wash sale check — LOG ONLY, DO NOT BLOCK.
        # Rationale: backtesting showed re-entries after losses often produce
        # strong gains (e.g., VIAV -21% -> same-day re-entry -> +42%). Wash
        # sales defer the tax deduction but don't eliminate it, so we'd
        # rather capture the alpha and pay the deferral cost.
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
            })
            continue

        # Sizing
        target_size = min(projected_cash, max_position)
        if target_size < min_position_size:
            skipped.append({
                "ticker": ticker,
                "score": score,
                "reason": f"Insufficient cash (${projected_cash:,.0f} available, need ${min_position_size})",
            })
            continue

        # Use current score's close price as the estimated share price
        # (Alpaca will fill at next open; this is just for qty sizing).
        # Fall back to Alpaca's latest trade price if the indicator dict
        # doesn't carry one.
        est_price = _estimated_price(rec)
        if not est_price or est_price <= 0:
            est_price = get_alpaca_latest_price(data_client, ticker)

        if not est_price or est_price <= 0:
            skipped.append({
                "ticker": ticker,
                "score": score,
                "reason": "No price data available (yfinance + Alpaca both failed)",
            })
            continue

        qty = int(target_size // est_price)  # whole shares only
        if qty <= 0:
            skipped.append({
                "ticker": ticker,
                "score": score,
                "reason": f"Position size ${target_size:,.0f} too small for share price ${est_price:,.2f}",
            })
            continue

        cost_basis = qty * est_price
        persistence_tag = (
            f" + {persistence_days}d persist"
            if persistence_days > 0 else ""
        )
        entries.append({
            "ticker": ticker,
            "qty": qty,
            "est_price": est_price,
            "cost_basis": cost_basis,
            "score": score,
            "reason": f"Score ≥ {entry_threshold}{persistence_tag}",
            "wash_sale_warning": wash_sale_warning,
        })
        projected_cash -= cost_basis

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
    dry_run: bool,
) -> None:
    """Submit buy orders and log them."""
    for e in entries:
        action = "[DRY RUN] " if dry_run else ""
        print(f"  {action}BUY   {e['ticker']:<6s} {e['qty']:>6.0f} shares @ ${e['est_price']:>8.2f}  "
              f"Score: {e['score']:<5.1f}  Cost: ${e['cost_basis']:>9,.0f}")

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

        order_id = submit_market_order(
            client, e["ticker"], e["qty"], OrderSide.BUY, dry_run=dry_run
        )

        if not dry_run:
            trade_log.log_buy(
                ticker=e["ticker"],
                trade_date=today,
                price=e["est_price"],
                qty=e["qty"],
                score_at_entry=e["score"],
                reason=e["reason"],
                alpaca_order_id=order_id,
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
    print(f"  Stop loss:         {trade_cfg['stop_loss_pct']*100:.0f}%")
    print(f"  Max position:      {trade_cfg['max_position_pct']*100:.0f}% of equity")


def print_account_status(snapshot: dict) -> None:
    print("\n  ACCOUNT STATUS")
    print("  " + "─" * 38)
    print(f"  Equity:          ${snapshot['equity']:>12,.2f}")
    print(f"  Cash:            ${snapshot['cash']:>12,.2f}")
    print(f"  Positions:       {len(snapshot['positions']):>13d}")


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
) -> tuple[str, str]:
    """Build (subject, html) for the daily trade execution digest."""
    equity = snapshot["equity"]
    cash = snapshot["cash"]
    last_equity = snapshot.get("last_equity")
    num_pos = len(snapshot["positions"])
    today_str = today.strftime("%Y-%m-%d")

    # ── Day-over-day equity change for subject line ──
    if last_equity and last_equity > 0:
        equity_change_pct = (equity - last_equity) / last_equity * 100
        equity_change_str = f"{equity_change_pct:+.2f}%"
    else:
        equity_change_pct = 0.0
        equity_change_str = "N/A"

    dry_tag = " [DRY RUN]" if dry_run else ""
    subject = (f"Alpha Scanner {today.isoformat()}: "
               f"{equity_change_str} / "
               f"{len(entries)} buy / {len(exits)} sell{dry_tag}")

    # ── Unrealized P&L across all positions ──
    total_unrealized_pnl = sum(
        pos["unrealized_pnl"] for pos in snapshot["positions"].values()
    )
    total_cost_basis = sum(
        pos["entry_price"] * pos["qty"] for pos in snapshot["positions"].values()
    )
    total_unrealized_pnl_pct = (
        (total_unrealized_pnl / total_cost_basis * 100)
        if total_cost_basis > 0 else 0.0
    )

    # ── HTML helpers ──
    td_style = "padding:6px 10px;border-bottom:1px solid #e5e7eb"
    th_style = "padding:6px 10px;text-align:left;background:#f3f4f6;border-bottom:2px solid #d1d5db"

    def row(cells: list[str]) -> str:
        return "<tr>" + "".join(f'<td style="{td_style}">{c}</td>' for c in cells) + "</tr>"

    def header_row(cells: list[str]) -> str:
        return "<tr>" + "".join(f'<th style="{th_style}">{c}</th>' for c in cells) + "</tr>"

    def table(header: list[str], rows: list[str]) -> str:
        if not rows:
            return "<p style='color:#6b7280'>(none)</p>"
        return ("<table style='border-collapse:collapse;width:100%;font-size:13px'>"
                + header_row(header)
                + "".join(rows) + "</table>")

    # ── Exits ──
    exit_rows = []
    for e in exits:
        color = _pnl_color(e["unrealized_pnl"])
        exit_rows.append(row([
            f"<b>{e['ticker']}</b>",
            f"{e['qty']:.0f}",
            f"${e['current_price']:.2f}",
            e["reason"],
            _colored(f"${e['unrealized_pnl']:+,.0f} ({e['unrealized_pnl_pct']:+.1f}%)", color),
        ]))

    # ── Entries (score color-coded) ──
    entry_rows = []
    for e in entries:
        ws_note = ""
        if e.get("wash_sale_warning"):
            ws_note = (f" <span style='color:#f59e0b;font-size:11px'>"
                       f"⚠ wash sale (${abs(e['wash_sale_warning']['loss_amount']):,.0f} deferred)</span>")
        entry_rows.append(row([
            f"<b>{e['ticker']}</b>{ws_note}",
            f"{e['qty']:.0f}",
            f"${e['est_price']:.2f}",
            _colored(f"{e['score']:.1f}", _score_color(e['score'])),
            f"${e['cost_basis']:,.0f}",
        ]))

    # ── Skipped Signals (simplified: Days >= 8.5 instead of verbose reason) ──
    entry_threshold = trade_cfg.get("entry_threshold", 8.5)
    persistence_days = trade_cfg.get("persistence_days", 3)
    skip_rows = []
    for s in skipped:
        ticker = s["ticker"]
        score = s["score"]
        days_above = _count_consecutive_days_above(
            db_conn, ticker, entry_threshold, today_str, today_score=score,
        )
        skip_rows.append(row([
            f"<b>{ticker}</b>",
            _colored(f"{score:.1f}", _score_color(score)),
            f"{days_above}/{persistence_days}",
        ]))

    # ── Current Positions (restructured) ──
    stop_loss_pct = trade_cfg.get("stop_loss_pct", 0.15)
    pos_rows = []
    exit_watch_rows = []

    for ticker, pos in sorted(snapshot["positions"].items()):
        score_val = scores.get(ticker, {}).get("score", 0.0)

        # Days elapsed (calendar days since entry)
        buy_date = trade_log.get_last_buy_date(ticker)
        if buy_date:
            hold_days = (today - datetime.strptime(buy_date, "%Y-%m-%d").date()).days
        else:
            hold_days = 0

        # Day % from Alpaca
        day_pct = pos.get("change_today_pct", 0.0)

        # P&L
        pnl_pct = pos["unrealized_pnl_pct"]
        pnl_dollar = pos["unrealized_pnl"]

        pos_rows.append(row([
            f"<b>{ticker}</b>",
            f"{hold_days}",
            f"${pos['current_price']:.2f}",
            _colored(f"{day_pct:+.1f}%", _pnl_color(day_pct)),
            _colored(f"{pnl_pct:+.1f}%", _pnl_color(pnl_pct)),
            _colored(f"${pnl_dollar:+,.0f}", _pnl_color(pnl_dollar)),
            _colored(f"{score_val:.1f}", _score_color(score_val)),
        ]))

        # ── Exit Watch: score < 7.0 OR P&L% worse than -12% ──
        if score_val < 7.0 or pnl_pct < -12.0:
            days_below_7 = _count_consecutive_days_below(
                db_conn, ticker, 7.0, today_str, today_score=score_val,
            )
            distance_to_exit = score_val - 5.0
            stop_loss_buffer_pct = pnl_pct  # how close to -15%
            buffer_color = "#dc2626" if stop_loss_buffer_pct < -10.0 else _pnl_color(stop_loss_buffer_pct)

            exit_watch_rows.append(row([
                f"<b>{ticker}</b>",
                _colored(f"{score_val:.1f}", _score_color(score_val)),
                f"{days_below_7}",
                f"{distance_to_exit:+.1f}",
                _colored(f"{stop_loss_buffer_pct:+.1f}%", buffer_color),
            ]))

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
    cooldown_rows = [row([f"<b>{t}</b>", e["cooldown_until"], f"${e['loss_amount']:,.0f}"])
                     for t, e in sorted(cooldowns.items())]

    # ── Header summary: Unrealized P&L styling ──
    upnl_color = _pnl_color(total_unrealized_pnl)
    upnl_html = _colored(
        f"${total_unrealized_pnl:+,.2f} ({total_unrealized_pnl_pct:+.1f}%)",
        upnl_color,
    )

    # Pre-build tables that have dynamic column headers (avoid nested
    # braces inside the main f-string which breaks on Python <3.12).
    skip_col = f"Days &ge; {entry_threshold:.1f}" if entry_threshold % 1 else f"Days &ge; {entry_threshold:.0f}"
    skip_table_html = table(["Ticker", "Score", skip_col], skip_rows)
    exit_table_html = table(["Ticker", "Qty", "Price", "Reason", "P&amp;L"], exit_rows)
    entry_table_html = table(["Ticker", "Qty", "Est Price", "Score", "Cost"], entry_rows)
    pos_table_html = table(
        ["Ticker", "Days Elapsed", "Current", "Day %", "P&amp;L %", "P&amp;L $", "Score"],
        pos_rows,
    )
    cooldown_table_html = table(["Ticker", "Cooldown Until", "Loss Amount"], cooldown_rows)

    html = f"""
    <div style="font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:720px;margin:0 auto;color:#111">
      <h2 style="margin:0 0 8px 0">Alpha Scanner — Daily Trade Execution{dry_tag}</h2>
      <p style="color:#6b7280;margin:0 0 20px 0">{today.strftime('%A, %B %d, %Y')}</p>

      <div style="background:#f9fafb;padding:14px 18px;border-radius:8px;margin-bottom:20px">
        <table style="border-collapse:collapse;font-size:14px">
          <tr><td style="padding:3px 24px 3px 0;color:#6b7280">Equity:</td><td style="text-align:right"><b>${equity:,.2f}</b></td></tr>
          <tr><td style="padding:3px 24px 3px 0;color:#6b7280">Cash:</td><td style="text-align:right">${cash:,.2f}</td></tr>
          <tr><td style="padding:3px 24px 3px 0;color:#6b7280">Positions:</td><td style="text-align:right">{num_pos}</td></tr>
          <tr><td style="padding:3px 24px 3px 0;color:#6b7280">Unrealized P&amp;L:</td><td style="text-align:right">{upnl_html}</td></tr>
        </table>
      </div>

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
    args = parser.parse_args()

    load_dotenv()

    print_header(args.dry_run)

    # 1. Load config (file + env var overrides)
    cfg = load_config()
    trade_cfg = load_trade_config(cfg)
    print_trade_config(trade_cfg)

    # 2. Connect
    client = connect_alpaca()
    data_client = connect_alpaca_data()

    # 3. Snapshot account
    snapshot = get_account_snapshot(client)
    print_account_status(snapshot)

    # 4. Cleanup expired wash sale entries
    today = datetime.now(PT_TZ).date()
    wash_sale_tracker.cleanup_expired(today)

    # 5. Open DB (used for persistence filter + writing today's scores)
    db_conn = subsector_store.init_db()

    # 6. Score all tickers
    print("\n  Scoring tickers...")
    data = fetch_all(cfg, period="1y", verbose=False)
    results = score_all(data, cfg)
    scores = score_lookup(results)
    print(f"  Scored {len(results)} tickers")

    # 7. Persist today's scores so tomorrow's persistence check has them
    #    (safe to call every run — upsert replaces existing rows)
    today_str = today.strftime("%Y-%m-%d")
    try:
        subsector_store.upsert_ticker_scores(db_conn, today_str, results)
    except Exception as e:
        print(f"  [WARN] Could not persist today's scores to DB: {e}")

    # 8. Evaluate exits
    exits = evaluate_exits(snapshot, scores, trade_cfg)
    print_section("EXITS TODAY")
    if exits:
        execute_exits(client, exits, today, args.dry_run)
    else:
        print("  (none)")

    # 9. Evaluate entries
    entries, skipped = evaluate_entries(
        snapshot, scores, exits, client, data_client, today, trade_cfg, db_conn,
    )
    print_section("ENTRIES TODAY")
    if entries:
        execute_entries(client, entries, today, args.dry_run)
    else:
        print("  (none)")

    # 7. Skipped
    print_section("SKIPPED SIGNALS")
    if skipped:
        for s in skipped:
            print(f"  SKIP  {s['ticker']:<6s} Score: {s['score']:<5.1f} Reason: {s['reason']}")
    else:
        print("  (none)")

    # 8. Re-fetch snapshot after orders (live mode only) for position display
    if not args.dry_run and (exits or entries):
        snapshot = get_account_snapshot(client)

    print_positions(snapshot, scores, today)
    print_wash_sale_status(today)

    # 9. Optional email digest
    if args.email:
        send_trade_digest(
            snapshot, exits, entries, skipped, scores, today,
            args.dry_run, trade_cfg, db_conn,
        )

    print("\n" + "=" * 66 + "\n")


if __name__ == "__main__":
    main()
