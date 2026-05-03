from __future__ import annotations

"""
spillover_tracker.py — Hypothetical "spillover" portfolio tracker.

Tracks the P&L of trades that the parent stream (live / paper / Scheme M
shadow) skipped purely for capacity reasons (position cap full, cash
floor exhausted, insufficient cash, daily-cap reached). The spillover
portfolio has NO position cap and NO cash limit — it answers the
question "how much performance are we leaving on the table by being
size-constrained?"

Architecture (mirrors shadow_m.py):
  - Reads pending_spillover_{mode}.json (written by trade_executor.py
    or shadow_m.py).
  - Loads spillover_state_{mode}.json (positions + bookkeeping).
  - For each pending capacity skip, opens a hypothetical position at the
    current close (sized at intended_slot_dollars / current_price, whole
    shares). Re-entries for already-held tickers are no-ops.
  - For each existing position, fetches today's score from the
    appropriate DB table (ticker_scores for live/paper, ticker_scores_m
    for scheme_m). Exits if score < parent_exit_threshold OR if a 20%
    stop is hit.
  - Marks-to-market via yfinance close.
  - Writes spillover_state_{mode}.json, appends to spillover_trades_{mode}.json,
    appends a daily entry to spillover_log_{mode}.json.

Modes (parent stream rules inherited):
  live      — Scheme C: exit < 5.0, stop 20%
  paper     — Scheme C: exit < 5.0, stop 20%
  scheme_m  — Scheme M: exit < 4.5, stop 20%

Usage:
  python3 spillover_tracker.py --mode paper
  python3 spillover_tracker.py --mode live --dry-run
  python3 spillover_tracker.py --mode scheme_m
"""

import argparse
import json
import sqlite3
from datetime import date
from pathlib import Path

from config import load_config
from data_fetcher import fetch_all
import subsector_store as store


REPO_ROOT = Path(__file__).parent
STOP_LOSS_PCT = 0.20

# Per-mode config inherits from parent stream
MODE_CONFIG = {
    "live":     {"exit_threshold": 5.0, "score_table": "ticker_scores"},
    "paper":    {"exit_threshold": 5.0, "score_table": "ticker_scores"},
    "scheme_m": {"exit_threshold": 4.5, "score_table": "ticker_scores_m"},
}


def _state_path(mode: str) -> Path:
    return REPO_ROOT / f"spillover_state_{mode}.json"


def _trades_path(mode: str) -> Path:
    return REPO_ROOT / f"spillover_trades_{mode}.json"


def _log_path(mode: str) -> Path:
    return REPO_ROOT / f"spillover_log_{mode}.json"


def _pending_path(mode: str) -> Path:
    return REPO_ROOT / f"pending_spillover_{mode}.json"


def load_state(mode: str) -> dict:
    path = _state_path(mode)
    if path.exists():
        return json.loads(path.read_text())
    return {"positions": {}, "last_run_date": None, "cumulative_pnl": 0.0}


def save_state(mode: str, state: dict) -> None:
    _state_path(mode).write_text(json.dumps(state, indent=2))


def load_trades(mode: str) -> list:
    path = _trades_path(mode)
    if path.exists():
        return json.loads(path.read_text())
    return []


def save_trades(mode: str, trades: list) -> None:
    _trades_path(mode).write_text(json.dumps(trades, indent=2))


def append_log(mode: str, entry: dict) -> None:
    path = _log_path(mode)
    log = json.loads(path.read_text()) if path.exists() else []
    log.append(entry)
    path.write_text(json.dumps(log, indent=2))


def load_pending(mode: str) -> dict | None:
    path = _pending_path(mode)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_current_score(conn: sqlite3.Connection, table: str, ticker: str, today_str: str) -> float | None:
    cur = conn.cursor()
    cur.execute(
        f"SELECT score FROM {table} WHERE ticker = ? AND date = ?",
        (ticker, today_str),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return float(row[0]) if row[0] is not None else None


def main(mode: str, dry_run: bool = False) -> None:
    if mode not in MODE_CONFIG:
        raise SystemExit(f"unknown mode {mode!r}; expected one of {sorted(MODE_CONFIG)}")

    cfg = MODE_CONFIG[mode]
    today_str = date.today().strftime("%Y-%m-%d")
    print(f"\n=== Spillover tracker [{mode}] — {today_str} ===\n")

    pending = load_pending(mode)
    if pending is None:
        print(f"  [info] no pending_spillover_{mode}.json found — nothing to ingest")
        pending_skips = []
    else:
        pending_skips = pending.get("skips", [])
        if pending.get("date") and pending["date"] != today_str:
            print(f"  [warn] pending file is from {pending['date']}, not today "
                  f"({today_str}) — ingesting anyway")

    state = load_state(mode)
    trades = load_trades(mode)

    # ─── Fetch fresh prices (one yfinance call for the universe) ──
    universe_cfg = load_config()
    print("  fetching market data...")
    data = fetch_all(universe_cfg, period="3mo", verbose=False)
    prices: dict[str, float] = {}
    for ticker, df in data.items():
        if df is not None and len(df) > 0:
            prices[ticker] = float(df["Close"].iloc[-1])
    print(f"  {len(prices)} tickers with prices")

    # Also fetch any held-but-no-longer-in-universe tickers individually.
    # (Rare — only matters if a ticker was removed from ticker_config.yaml
    # while spillover holds it. Skip mark-to-market in that case; carry
    # entry price as proxy.)

    # ─── Connect DB for score lookups ────────────────────────────
    conn = store.init_db()

    log_entry = {
        "date": today_str, "mode": mode,
        "entries": [], "exits": [], "stops_fired": [], "skipped_dup": [],
    }

    # ─── 1. Check exits on existing positions ────────────────────
    to_close: list[tuple[str, str, float]] = []
    for ticker, pos in list(state["positions"].items()):
        cur_px = prices.get(ticker)
        if cur_px is None:
            # No fresh price — skip exit check this run; mark-to-market falls back
            continue

        # Stop-loss check
        stop_price = pos["entry_price"] * (1 - STOP_LOSS_PCT)
        if cur_px <= stop_price:
            to_close.append((ticker, "stop_loss", cur_px))
            log_entry["stops_fired"].append({
                "ticker": ticker, "entry_price": pos["entry_price"],
                "stop_price": round(stop_price, 2), "exit_price": cur_px,
            })
            continue

        # Score-based exit
        score = get_current_score(conn, cfg["score_table"], ticker, today_str)
        if score is not None and score < cfg["exit_threshold"]:
            to_close.append((ticker, "score_exit", cur_px))
            log_entry["exits"].append({
                "ticker": ticker, "entry_price": pos["entry_price"],
                "exit_price": cur_px, "exit_score": score,
            })

    # Execute closes
    for ticker, reason, exit_px in to_close:
        pos = state["positions"][ticker]
        pnl = (exit_px - pos["entry_price"]) * pos["shares"]
        pnl_pct = exit_px / pos["entry_price"] - 1.0
        trades.append({
            "ticker": ticker,
            "entry_date": pos["entry_date"], "exit_date": today_str,
            "entry_price": pos["entry_price"], "exit_price": exit_px,
            "shares": pos["shares"], "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4), "exit_reason": reason,
            "skip_reason_at_entry": pos.get("skip_reason_at_entry", ""),
            "entry_score": pos.get("entry_score", 0.0),
            "intended_slot_dollars": pos.get("intended_slot_dollars", 0.0),
        })
        state["cumulative_pnl"] = round(
            state.get("cumulative_pnl", 0.0) + pnl, 2,
        )
        del state["positions"][ticker]

    # ─── 2. Open new positions from pending skips ────────────────
    for skip in pending_skips:
        ticker = skip["ticker"]
        if ticker in state["positions"]:
            log_entry["skipped_dup"].append({
                "ticker": ticker,
                "reason": "already held in spillover",
            })
            continue
        cur_px = prices.get(ticker)
        if cur_px is None or cur_px <= 0:
            log_entry["skipped_dup"].append({
                "ticker": ticker,
                "reason": "no current price available",
            })
            continue
        intended = float(skip.get("intended_slot_dollars") or 0.0)
        if intended <= 0:
            log_entry["skipped_dup"].append({
                "ticker": ticker,
                "reason": "intended_slot_dollars <= 0",
            })
            continue
        shares = int(intended // cur_px)  # whole shares (floor)
        if shares < 1:
            log_entry["skipped_dup"].append({
                "ticker": ticker,
                "reason": (f"intended ${intended:,.0f} too small for share "
                           f"price ${cur_px:.2f}"),
            })
            continue
        cost = shares * cur_px
        state["positions"][ticker] = {
            "entry_date": today_str,
            "entry_price": cur_px,
            "shares": shares,
            "cost_basis": cost,
            "entry_score": float(skip.get("score", 0.0)),
            "skip_reason_at_entry": skip.get("skip_category", ""),
            "intended_slot_dollars": intended,
        }
        log_entry["entries"].append({
            "ticker": ticker, "score": skip.get("score"),
            "entry_price": cur_px, "shares": shares,
            "cost": round(cost, 2),
            "skip_reason": skip.get("skip_category"),
        })

    # ─── 3. Mark-to-market summary ───────────────────────────────
    open_value = 0.0
    open_unrealized = 0.0
    open_pos_summary = []
    for ticker, pos in state["positions"].items():
        mark = prices.get(ticker, pos["entry_price"])
        mv = mark * pos["shares"]
        unrl = (mark - pos["entry_price"]) * pos["shares"]
        open_value += mv
        open_unrealized += unrl
        open_pos_summary.append({
            "ticker": ticker, "shares": pos["shares"],
            "entry_price": pos["entry_price"], "current_price": mark,
            "unrealized_pnl": round(unrl, 2),
            "unrealized_pnl_pct": round(mark / pos["entry_price"] - 1.0, 4),
        })

    log_entry["open_positions"] = len(state["positions"])
    log_entry["open_value"] = round(open_value, 2)
    log_entry["open_unrealized_pnl"] = round(open_unrealized, 2)
    log_entry["realized_cumulative_pnl"] = round(state.get("cumulative_pnl", 0.0), 2)
    log_entry["total_pnl"] = round(
        state.get("cumulative_pnl", 0.0) + open_unrealized, 2,
    )
    log_entry["positions"] = open_pos_summary

    # ─── 4. Save ──────────────────────────────────────────────────
    state["last_run_date"] = today_str
    if not dry_run:
        save_state(mode, state)
        save_trades(mode, trades)
        append_log(mode, log_entry)
        # Consume the pending file (delete it). Re-running becomes a no-op
        # rather than re-opening the same positions a second time.
        pending_path = _pending_path(mode)
        if pending_path.exists():
            pending_path.unlink()

    # ─── Summary ──────────────────────────────────────────────────
    print(f"\n  Open positions: {len(state['positions'])}")
    print(f"  Open value:     ${open_value:,.2f}")
    print(f"  Open unrealized: ${open_unrealized:+,.2f}")
    print(f"  Realized cum P&L: ${state.get('cumulative_pnl', 0.0):+,.2f}")
    print(f"  Total P&L:      ${log_entry['total_pnl']:+,.2f}")
    print(f"  Today: {len(log_entry['entries'])} entries, "
          f"{len(log_entry['exits'])} score exits, "
          f"{len(log_entry['stops_fired'])} stop-outs, "
          f"{len(log_entry['skipped_dup'])} dup/no-price skips")
    if log_entry["entries"]:
        print(f"\n  Entries:")
        for e in log_entry["entries"]:
            print(f"    BUY  {e['ticker']:<6} {e['shares']:>4} @ ${e['entry_price']:.2f}  "
                  f"reason={e['skip_reason']}  cost=${e['cost']:,.0f}")
    if log_entry["exits"]:
        print(f"\n  Score exits:")
        for x in log_entry["exits"]:
            print(f"    SELL {x['ticker']:<6} @ ${x['exit_price']:.2f}  "
                  f"score={x['exit_score']:.2f}")
    if log_entry["stops_fired"]:
        print(f"\n  Stops fired:")
        for x in log_entry["stops_fired"]:
            print(f"    STOP {x['ticker']:<6} entry ${x['entry_price']:.2f} → "
                  f"exit ${x['exit_price']:.2f}")

    conn.close()
    print("\n[done]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=sorted(MODE_CONFIG.keys()),
                    help="Parent stream whose skips drive this spillover tracker")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write state files; just print decisions")
    args = ap.parse_args()
    main(mode=args.mode, dry_run=args.dry_run)
