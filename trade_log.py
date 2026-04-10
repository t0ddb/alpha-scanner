"""
trade_log.py — Persistent JSON log of all buy/sell actions executed by
trade_executor.py. Separate from Alpaca's records so we own our own
history for analysis and auditing.

Schema (trade_history.json):
{
    "trades": [
        {
            "ticker": "AAOI",
            "side": "buy",
            "date": "2026-04-10",
            "price": 45.20,
            "qty": 442,
            "cost_basis": 19978.40,
            "score_at_entry": 9.8,
            "reason": "Score ≥ 9.5",
            "alpaca_order_id": "abc-123"
        },
        {
            "ticker": "AAOI",
            "side": "sell",
            "date": "2026-05-15",
            "price": 38.42,
            "qty": 442,
            "proceeds": 16981.64,
            "pnl": -2996.76,
            "pnl_pct": -15.0,
            "score_at_exit": 6.2,
            "reason": "Stop loss (15%)",
            "hold_days": 35,
            "alpaca_order_id": "def-456"
        }
    ]
}
"""

from __future__ import annotations
import json
from datetime import datetime, date
from pathlib import Path


TRADE_LOG_FILE = Path(__file__).parent / "trade_history.json"


def _iso(d) -> str:
    """Coerce a date-like value to an ISO string."""
    if isinstance(d, str):
        return d
    if isinstance(d, (date, datetime)):
        return d.strftime("%Y-%m-%d")
    raise ValueError(f"Cannot convert {d!r} to date string")


def _load() -> dict:
    """Load the trade log from disk (empty structure if missing)."""
    if not TRADE_LOG_FILE.exists():
        return {"trades": []}
    try:
        with open(TRADE_LOG_FILE, "r") as f:
            data = json.load(f)
            if "trades" not in data:
                data["trades"] = []
            return data
    except (json.JSONDecodeError, OSError):
        return {"trades": []}


def _save(data: dict) -> None:
    """Write the trade log to disk."""
    with open(TRADE_LOG_FILE, "w") as f:
        json.dump(data, f, indent=2)


def log_buy(
    ticker: str,
    trade_date,
    price: float,
    qty: float,
    score_at_entry: float,
    reason: str = "Score ≥ 9.5",
    alpaca_order_id: str | None = None,
) -> dict:
    """Append a buy trade to the log."""
    entry = {
        "ticker": ticker,
        "side": "buy",
        "date": _iso(trade_date),
        "price": round(float(price), 4),
        "qty": round(float(qty), 4),
        "cost_basis": round(float(price) * float(qty), 2),
        "score_at_entry": round(float(score_at_entry), 2),
        "reason": reason,
    }
    if alpaca_order_id:
        entry["alpaca_order_id"] = alpaca_order_id

    data = _load()
    data["trades"].append(entry)
    _save(data)
    return entry


def log_sell(
    ticker: str,
    trade_date,
    price: float,
    qty: float,
    entry_price: float,
    score_at_exit: float,
    reason: str,
    hold_days: int,
    alpaca_order_id: str | None = None,
) -> dict:
    """Append a sell trade to the log, computing proceeds and P&L."""
    price = float(price)
    qty = float(qty)
    entry_price = float(entry_price)

    proceeds = price * qty
    cost = entry_price * qty
    pnl = proceeds - cost
    pnl_pct = ((price / entry_price) - 1.0) * 100.0 if entry_price else 0.0

    entry = {
        "ticker": ticker,
        "side": "sell",
        "date": _iso(trade_date),
        "price": round(price, 4),
        "qty": round(qty, 4),
        "proceeds": round(proceeds, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "score_at_exit": round(float(score_at_exit), 2),
        "reason": reason,
        "hold_days": int(hold_days),
    }
    if alpaca_order_id:
        entry["alpaca_order_id"] = alpaca_order_id

    data = _load()
    data["trades"].append(entry)
    _save(data)
    return entry


def get_all_trades() -> list[dict]:
    """Return the complete trade list."""
    return _load()["trades"]


def get_trades_for_ticker(ticker: str) -> list[dict]:
    """All trades for a given ticker, in order."""
    return [t for t in get_all_trades() if t["ticker"] == ticker]


def get_last_buy_date(ticker: str) -> str | None:
    """Most recent buy date for a ticker (used to compute hold days)."""
    trades = [t for t in get_trades_for_ticker(ticker) if t["side"] == "buy"]
    return trades[-1]["date"] if trades else None


def summary_stats() -> dict:
    """Quick summary of realized trades."""
    trades = get_all_trades()
    buys = [t for t in trades if t["side"] == "buy"]
    sells = [t for t in trades if t["side"] == "sell"]

    if not sells:
        return {
            "total_buys": len(buys),
            "total_sells": 0,
            "realized_pnl": 0.0,
            "win_rate": 0.0,
            "total_wins": 0,
            "total_losses": 0,
        }

    wins = [t for t in sells if t["pnl"] > 0]
    losses = [t for t in sells if t["pnl"] < 0]
    total_pnl = sum(t["pnl"] for t in sells)

    return {
        "total_buys": len(buys),
        "total_sells": len(sells),
        "realized_pnl": round(total_pnl, 2),
        "win_rate": round(len(wins) / len(sells) * 100, 1),
        "total_wins": len(wins),
        "total_losses": len(losses),
    }


if __name__ == "__main__":
    # Quick CLI summary
    stats = summary_stats()
    print("Trade Log Summary")
    print("─" * 40)
    for k, v in stats.items():
        print(f"  {k:<18s}  {v}")
