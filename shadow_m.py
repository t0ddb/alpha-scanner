from __future__ import annotations

"""
shadow_m.py — Shadow tracker for Scheme M scoring.

Runs daily after trade_executor.py. Computes Scheme M scores for all tickers,
tracks a HYPOTHETICAL portfolio at threshold 7.5, and logs would-have
decisions to JSON files. Does NOT touch Alpaca or any live trading state.

Outputs:
  shadow_m_state.json — current shadow positions, equity, cash
  shadow_m_trades.json — closed-trade history with P&L
  shadow_m_log.json — per-day decision log (entries, exits, skips)

Configuration (sweep-validated; see DECISIONS.md 2026-05-01):
  ENTRY_THRESHOLD = 7.5
  EXIT_THRESHOLD  = 4.5     (Scheme M has wider score range; lower exit optimal)
  PERSISTENCE_DAYS = 3
  MAX_POSITIONS = 12
  POSITION_PCT = 0.0833  (8.33% of equity)
  STOP_LOSS_PCT = 0.20
  STARTING_EQUITY = 100000  (synthetic, mirrors original paper account)

Usage:
  python3 shadow_m.py            # run daily after trade_executor
  python3 shadow_m.py --dry-run  # don't write state, just print
"""

import argparse
import json
import os
import sqlite3
from datetime import datetime, date
from pathlib import Path

from config import load_config
from data_fetcher import fetch_all
from indicators import score_ticker_m
import sequence_overlay as so
import subsector_store as store


# ─── Configuration ────────────────────────────────────────────────
ENTRY_THRESHOLD = float(os.environ.get("SCHEME_M_ENTRY_THRESHOLD", "7.5"))
# Exit threshold validated via _test_scheme_m_exit_threshold.py:
# Cumulative returns are within statistical noise across exit 4.0-5.5
# (+550% to +565%, range 15pp on +550 base = 2.7% relative; path std
# ~11-14%). Path ranges overlap substantially, so any choice in this
# zone is defensible. Picked 4.5 for SHADOW TRACKING because it has
# both lowest path std (10.5%) AND best per-trade quality (mean +76%
# vs 5.5's +51%) — cleaner observation signal for learning. Above 6.0
# returns collapse; below 4.0 path variance rises.
EXIT_THRESHOLD  = float(os.environ.get("SCHEME_M_EXIT_THRESHOLD", "4.5"))
PERSISTENCE_DAYS = int(os.environ.get("SCHEME_M_PERSISTENCE_DAYS", "3"))
MAX_POSITIONS = int(os.environ.get("SCHEME_M_MAX_POSITIONS", "12"))
POSITION_PCT = float(os.environ.get("SCHEME_M_POSITION_PCT", "0.0833"))
STOP_LOSS_PCT = float(os.environ.get("SCHEME_M_STOP_LOSS_PCT", "0.20"))
STARTING_EQUITY = float(os.environ.get("SCHEME_M_STARTING_EQUITY", "100000"))

REPO_ROOT = Path(__file__).parent
STATE_PATH = REPO_ROOT / "shadow_m_state.json"
TRADES_PATH = REPO_ROOT / "shadow_m_trades.json"
LOG_PATH = REPO_ROOT / "shadow_m_log.json"


# ─── Helpers ──────────────────────────────────────────────────────
def _get_indicator_dict_from_score_result(ind_result_per_ticker: dict, ticker: str) -> dict:
    """Build the indicators dict from score_all() raw output."""
    ind = ind_result_per_ticker.get(ticker, {}).get("indicators", {})
    return ind


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {
        "positions": {},  # ticker -> dict
        "cash": STARTING_EQUITY,
        "starting_equity": STARTING_EQUITY,
        "last_run_date": None,
    }


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def load_trades() -> list:
    if TRADES_PATH.exists():
        return json.loads(TRADES_PATH.read_text())
    return []


def save_trades(trades: list) -> None:
    TRADES_PATH.write_text(json.dumps(trades, indent=2))


def append_log(entry: dict) -> None:
    log = []
    if LOG_PATH.exists():
        log = json.loads(LOG_PATH.read_text())
    log.append(entry)
    LOG_PATH.write_text(json.dumps(log, indent=2))


def total_equity(state: dict, prices: dict[str, float]) -> float:
    eq = state["cash"]
    for ticker, pos in state["positions"].items():
        px = prices.get(ticker)
        if px is None:
            # Use entry price as fallback
            px = pos["entry_price"]
        eq += px * pos["shares"]
    return eq


# ─── Main daily logic ─────────────────────────────────────────────
def main(dry_run: bool = False):
    today = date.today().strftime("%Y-%m-%d")
    print(f"\n=== Scheme M shadow tracker — run for {today} ===\n")

    cfg = load_config()
    print("fetching market data...")
    data = fetch_all(cfg, period="2y", verbose=False)
    print(f"  {len(data)} tickers fetched")

    # Get latest closing prices (most recent bar per ticker)
    prices = {}
    for ticker, df in data.items():
        if df is not None and len(df) > 0:
            prices[ticker] = float(df["Close"].iloc[-1])

    # Compute today's indicators + Scheme M score for every ticker
    print("computing indicators + Scheme M scores...")
    from indicators import score_all
    raw_results = score_all(data, cfg)  # list of dicts with 'ticker', 'indicators', etc.

    # Connect to DB once for streak history lookups
    conn = store.init_db()

    # Build records to insert into ticker_scores_m
    m_records = []
    score_by_ticker: dict[str, dict] = {}
    for r in raw_results:
        ticker = r["ticker"]
        ind = r["indicators"]

        # Layer 1
        l1_result = score_ticker_m(ind)
        layer_1 = l1_result["score"]

        # Today's fire flags (v2 thresholds)
        today_flags = so.fire_flags_m_from_indicators(ind)

        # Recent fire flag history (for streak computation)
        # We need streaks "as of today" — append today's flags to history
        history_rows = store.get_fire_flags_history_m(conn, ticker, today, days=120)
        # Append today's row (would be inserted below, but compute with it)
        history_with_today = history_rows + [{
            "date": today, "rs": today_flags["rs"], "ich": today_flags["ich"],
            "hl": today_flags["hl"], "cmf": today_flags["cmf"],
            "roc": today_flags["roc"], "atr": today_flags["atr"],
            "dtf": today_flags["dtf"],
        }]
        streaks = so.compute_streaks_from_history(history_with_today)

        # Layer 2
        features = so.compute_sequence_features(streaks)
        layer_2, tags = so.compute_layer_2_adjustment(features)

        score = round(layer_1 + layer_2, 2)
        score_by_ticker[ticker] = {
            "score": score, "layer_1": layer_1, "layer_2": layer_2,
            "tags": tags, "fire_flags": today_flags, "indicators": ind,
        }

        # DB record
        rs_v = ind.get("relative_strength", {}).get("rs_percentile", 0) or 0
        hl_v = ind.get("higher_lows", {}).get("consecutive_higher_lows", 0) or 0
        ich_v = ind.get("ichimoku_cloud", {})
        ich_score = (int(ich_v.get("above_cloud", False))
                     + int(ich_v.get("cloud_bullish", False))
                     + int(ich_v.get("tenkan_above_kijun", False)))
        roc_v = ind.get("roc", {}).get("roc", 0) or 0
        cmf_v = ind.get("cmf", {}).get("cmf", 0) or 0
        atr_v = ind.get("atr_expansion", {}).get("atr_percentile", 0) or 0
        dtf_v = ind.get("dual_tf_rs", {})
        m_records.append({
            "ticker": ticker, "score": score, "layer_1": layer_1,
            "layer_2": layer_2, "sequence_tags": "|".join(tags),
            "fire_rs": today_flags["rs"], "fire_ich": today_flags["ich"],
            "fire_hl": today_flags["hl"], "fire_cmf": today_flags["cmf"],
            "fire_roc": today_flags["roc"], "fire_atr": today_flags["atr"],
            "fire_dtf": today_flags["dtf"],
            "rs_pctl": rs_v, "hl_count": hl_v, "ich_score": ich_score,
            "roc_value": roc_v, "cmf_value": cmf_v, "atr_pctl": atr_v,
            "dtf_126d_pctl": dtf_v.get("rs_126d_percentile", 0) or 0,
            "dtf_63d_pctl": dtf_v.get("rs_63d_percentile", 0) or 0,
        })

    # Insert today's Scheme M scores into DB
    if not dry_run:
        store.upsert_ticker_scores_m(conn, today, m_records)
        print(f"  inserted {len(m_records)} v2-score rows for {today}")

    # ─── Shadow portfolio update ────────────────────────────────
    state = load_state()
    trades = load_trades()
    log_entry = {
        "date": today, "entries": [], "exits": [], "skipped": [],
        "stops_fired": [], "candidates_total": 0,
    }

    # ─── 1. Update prices + check exits ─────────────────────────
    to_close = []  # (ticker, exit_reason, exit_price)
    for ticker, pos in list(state["positions"].items()):
        cur_px = prices.get(ticker)
        if cur_px is None:
            continue

        # Stop loss check (using current close as proxy for intraday)
        stop_price = pos["entry_price"] * (1 - STOP_LOSS_PCT)
        if cur_px <= stop_price:
            to_close.append((ticker, "stop_loss", cur_px))
            log_entry["stops_fired"].append({
                "ticker": ticker, "entry_price": pos["entry_price"],
                "stop_price": round(stop_price, 2), "exit_price": cur_px,
            })
            continue

        # Score-based exit check
        s = score_by_ticker.get(ticker, {}).get("score", 0)
        if s < EXIT_THRESHOLD:
            to_close.append((ticker, "score_exit", cur_px))
            log_entry["exits"].append({
                "ticker": ticker, "entry_price": pos["entry_price"],
                "exit_price": cur_px, "exit_score": s,
            })

    # Execute closes
    for ticker, reason, exit_px in to_close:
        pos = state["positions"][ticker]
        pnl = (exit_px - pos["entry_price"]) * pos["shares"]
        pnl_pct = (exit_px / pos["entry_price"] - 1.0)
        state["cash"] += exit_px * pos["shares"]
        trades.append({
            "ticker": ticker, "entry_date": pos["entry_date"], "exit_date": today,
            "entry_price": pos["entry_price"], "exit_price": exit_px,
            "shares": pos["shares"], "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4), "exit_reason": reason,
            "entry_score": pos["entry_score"],
            "sequence_tags_at_entry": pos.get("sequence_tags", ""),
        })
        del state["positions"][ticker]

    # ─── 2. Find candidates for new entries ─────────────────────
    candidates: list[tuple[str, float, dict]] = []
    for ticker, info in score_by_ticker.items():
        if ticker in state["positions"]:
            continue
        if info["score"] < ENTRY_THRESHOLD:
            continue
        # Persistence check: prior N days >= ENTRY_THRESHOLD
        prior = store.get_m_scores_for_persistence(conn, ticker, today, days=PERSISTENCE_DAYS)
        if len(prior) < PERSISTENCE_DAYS:
            log_entry["skipped"].append({
                "ticker": ticker, "score": info["score"],
                "reason": f"only {len(prior)} prior days available",
            })
            continue
        if any(p < ENTRY_THRESHOLD for p in prior):
            log_entry["skipped"].append({
                "ticker": ticker, "score": info["score"],
                "reason": f"persistence failed (priors {[round(p, 1) for p in prior]})",
            })
            continue
        candidates.append((ticker, info["score"], info))

    log_entry["candidates_total"] = len(candidates)

    # Sort by score descending; ties broken alphabetically
    candidates.sort(key=lambda x: (-x[1], x[0]))

    # ─── 3. Execute new entries ────────────────────────────────
    eq = total_equity(state, prices)
    target_size_dollars = eq * POSITION_PCT
    capacity_skips: list[dict] = []  # for spillover tracker
    for ticker, score, info in candidates:
        if len(state["positions"]) >= MAX_POSITIONS:
            log_entry["skipped"].append({
                "ticker": ticker, "score": score, "reason": "portfolio full",
            })
            capacity_skips.append({
                "ticker": ticker, "score": score,
                "skip_category": "Full",
                "intended_slot_dollars": target_size_dollars,
            })
            continue
        cur_px = prices.get(ticker)
        if cur_px is None or cur_px <= 0:
            continue
        # Hypothetical fill at current close (next-open in real world)
        if state["cash"] < target_size_dollars * 0.95:  # min 5% cash buffer
            log_entry["skipped"].append({
                "ticker": ticker, "score": score,
                "reason": f"insufficient cash (${state['cash']:.0f} < ${target_size_dollars * 0.95:.0f})",
            })
            capacity_skips.append({
                "ticker": ticker, "score": score,
                "skip_category": "No Cash",
                "intended_slot_dollars": target_size_dollars,
            })
            continue
        shares = int(target_size_dollars // cur_px)
        if shares < 1:
            continue
        cost = shares * cur_px
        state["cash"] -= cost
        state["positions"][ticker] = {
            "entry_date": today, "entry_price": cur_px, "shares": shares,
            "entry_score": score, "cost_basis": cost,
            "sequence_tags": "|".join(info["tags"]),
        }
        log_entry["entries"].append({
            "ticker": ticker, "score": score, "entry_price": cur_px,
            "shares": shares, "cost": round(cost, 2),
            "sequence_tags": "|".join(info["tags"]),
        })

    # ─── 4. Snapshot equity for log ─────────────────────────────
    final_eq = total_equity(state, prices)
    log_entry["equity"] = round(final_eq, 2)
    log_entry["cash"] = round(state["cash"], 2)
    log_entry["positions_count"] = len(state["positions"])
    log_entry["positions"] = [
        {"ticker": t, "shares": p["shares"], "entry_price": p["entry_price"],
         "current_price": prices.get(t, p["entry_price"]),
         "unrealized_pnl_pct": round(prices.get(t, p["entry_price"]) / p["entry_price"] - 1.0, 4)}
        for t, p in state["positions"].items()
    ]

    # ─── 5. Save state + log ───────────────────────────────────
    state["last_run_date"] = today
    if not dry_run:
        save_state(state)
        save_trades(trades)
        append_log(log_entry)

    # ─── 5b. Persist capacity-skip records for the spillover tracker ──
    # Always write (even on dry-run) so the spillover dry-run can pick
    # them up; the file is overwritten each run, so no accumulation risk.
    pending_path = REPO_ROOT / "pending_spillover_scheme_m.json"
    pending_path.write_text(json.dumps(
        {"date": today, "mode": "scheme_m", "skips": capacity_skips},
        indent=2,
    ))
    if capacity_skips:
        print(f"  [spillover] wrote {len(capacity_skips)} pending capacity-skip "
              f"record(s) → {pending_path.name}")

    # ─── Print summary ─────────────────────────────────────────
    print(f"\n=== Daily summary ===")
    print(f"  Total equity: ${final_eq:,.2f}  (cash ${state['cash']:,.2f})")
    print(f"  Positions: {len(state['positions'])}/{MAX_POSITIONS}")
    print(f"  Today: {len(log_entry['entries'])} entries, "
          f"{len(log_entry['exits'])} score exits, "
          f"{len(log_entry['stops_fired'])} stop-outs, "
          f"{len(log_entry['skipped'])} skipped")
    print(f"  Candidates qualifying today: {log_entry['candidates_total']}")
    if log_entry["entries"]:
        print(f"\n  Entries:")
        for e in log_entry["entries"]:
            print(f"    BUY  {e['ticker']:<6} {e['shares']:>4} @ ${e['entry_price']:.2f}  "
                  f"score={e['score']:.2f}  cost=${e['cost']:,.0f}")
    if log_entry["exits"]:
        print(f"\n  Score exits:")
        for x in log_entry["exits"]:
            print(f"    SELL {x['ticker']:<6} @ ${x['exit_price']:.2f}  score={x['exit_score']:.2f}")
    if log_entry["stops_fired"]:
        print(f"\n  Stops fired:")
        for x in log_entry["stops_fired"]:
            print(f"    STOP {x['ticker']:<6} entry ${x['entry_price']:.2f} → exit ${x['exit_price']:.2f}")

    conn.close()
    print("\n[done]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write state files; just print decisions")
    args = ap.parse_args()
    main(dry_run=args.dry_run)
