from __future__ import annotations

"""
portfolio_backtest.py — Portfolio simulation backtest with capital constraints.

Tests entry thresholds, persistence filters, and capital rotation strategies
against real historical data.

Usage:
    python3 portfolio_backtest.py
"""

import sqlite3
import sys
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import pandas as pd
import numpy as np

from config import load_config
from data_fetcher import fetch_all

DB_PATH = Path(__file__).parent / "breakout_tracker.db"

# ============================================================
# DATA LOADING
# ============================================================

def load_score_data() -> pd.DataFrame:
    """Load all ticker scores from SQLite, sorted by date."""
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT date, ticker, score FROM ticker_scores ORDER BY date, ticker",
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_score_lookup(score_df: pd.DataFrame) -> dict:
    """Build {date_str: {ticker: score}} lookup from score dataframe."""
    lookup = {}
    for _, row in score_df.iterrows():
        d = row["date"].strftime("%Y-%m-%d")
        if d not in lookup:
            lookup[d] = {}
        lookup[d][row["ticker"]] = row["score"]
    return lookup


def build_daily_scores(score_df: pd.DataFrame, trading_days: list) -> dict:
    """Build daily score lookup with carry-forward for days between snapshots.
    Returns {date_str: {ticker: score}}."""
    snapshot_dates = sorted(score_df["date"].unique())
    score_lookup = build_score_lookup(score_df)

    daily = {}
    current_scores = {}

    for day in trading_days:
        day_str = day.strftime("%Y-%m-%d")

        # If this is a snapshot date, update current scores
        if day in snapshot_dates or day_str in score_lookup:
            snap = score_lookup.get(day_str, {})
            current_scores.update(snap)

        daily[day_str] = dict(current_scores)

    return daily


def get_trading_days(price_data: dict) -> list:
    """Get sorted list of all trading days from price data (use SPY as reference)."""
    spy_df = price_data.get("SPY")
    if spy_df is not None and not spy_df.empty:
        return sorted(spy_df.index.tz_localize(None) if spy_df.index.tz else spy_df.index)

    # Fallback: use the first ticker with data
    for ticker, df in price_data.items():
        if df is not None and not df.empty:
            idx = df.index.tz_localize(None) if df.index.tz else df.index
            return sorted(idx)
    return []


def get_price(price_data: dict, ticker: str, date, field: str = "Open") -> float | None:
    """Get price for a ticker on a given date. Returns None if not available."""
    df = price_data.get(ticker)
    if df is None or df.empty:
        return None

    idx = df.index.tz_localize(None) if df.index.tz else df.index
    # Find exact or nearest date
    target = pd.Timestamp(date)
    matches = idx[idx == target]
    if len(matches) > 0:
        return float(df.loc[df.index[idx == target][0], field])

    # Try nearest within 3 days
    diffs = abs(idx - target)
    min_idx = diffs.argmin()
    if diffs[min_idx] <= timedelta(days=3):
        return float(df.iloc[min_idx][field])

    return None


def get_spy_return(price_data: dict, start_date, end_date) -> float:
    """Get SPY return between two dates."""
    start_price = get_price(price_data, "SPY", start_date, "Close")
    end_price = get_price(price_data, "SPY", end_date, "Close")
    if start_price and end_price and start_price > 0:
        return (end_price - start_price) / start_price
    return 0.0


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

class Position:
    def __init__(self, ticker: str, entry_date: str, entry_price: float,
                 shares: float, entry_score: float):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.entry_score = entry_score
        self.cost_basis = entry_price * shares

    def current_value(self, current_price: float) -> float:
        return self.shares * current_price

    def pnl(self, current_price: float) -> float:
        return self.current_value(current_price) - self.cost_basis

    def pnl_pct(self, current_price: float) -> float:
        if self.cost_basis > 0:
            return (self.current_value(current_price) - self.cost_basis) / self.cost_basis
        return 0.0


class TradeRecord:
    def __init__(self, ticker: str, entry_date: str, entry_score: float,
                 entry_price: float, cost_basis: float, exit_date: str,
                 exit_price: float, exit_reason: str, pnl: float, pnl_pct: float,
                 hold_days: int):
        self.ticker = ticker
        self.entry_date = entry_date
        self.entry_score = entry_score
        self.entry_price = entry_price
        self.cost_basis = cost_basis
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = exit_reason
        self.pnl = pnl
        self.pnl_pct = pnl_pct
        self.hold_days = hold_days


def run_simulation(
    daily_scores: dict,
    price_data: dict,
    trading_days: list,
    threshold: float = 9.0,
    persistence_filter: bool = False,
    rotation_strategy: str = "none",  # "none", "trim_weakest", "trim_worst_pnl"
    starting_capital: float = 100_000,
    max_position_pct: float = 0.20,
    stop_loss_pct: float = 0.15,
    score_exit_threshold: float = 5.0,
    min_buy: float = 1000,
    verbose: bool = False,
) -> dict:
    """Run a full portfolio simulation. Returns results dict."""

    max_position_size = starting_capital * max_position_pct  # $20k
    cash = starting_capital
    positions: dict[str, Position] = {}
    trades: list[TradeRecord] = []
    history: list[dict] = []
    signals_missed = 0
    rotation_events = 0
    signals_generated = 0
    signals_filtered = 0

    # Track previous day's scores for persistence filter
    prev_scores: dict[str, float] = {}

    # Filter trading days to only those within our score data range
    score_dates = sorted(daily_scores.keys())
    if not score_dates:
        return {"error": "No score data available"}

    start_date = score_dates[0]
    end_date = score_dates[-1]

    filtered_days = [d for d in trading_days
                     if start_date <= d.strftime("%Y-%m-%d") <= end_date]

    if not filtered_days:
        return {"error": "No trading days overlap with score data"}

    for i, day in enumerate(filtered_days):
        day_str = day.strftime("%Y-%m-%d")
        today_scores = daily_scores.get(day_str, {})

        # === 1. CHECK EXITS ===
        tickers_to_exit = []
        for ticker, pos in positions.items():
            current_price = get_price(price_data, ticker, day, "Close")
            current_score = today_scores.get(ticker, 0)

            if current_price is None:
                continue

            exit_reason = None

            # Score exit
            if current_score < score_exit_threshold:
                exit_reason = "score_exit"

            # Stop loss
            elif current_price <= pos.entry_price * (1 - stop_loss_pct):
                exit_reason = "stop_loss"

            if exit_reason:
                # Sell at next day's open if possible, else use today's close
                if i + 1 < len(filtered_days):
                    next_day = filtered_days[i + 1]
                    sell_price = get_price(price_data, ticker, next_day, "Open")
                    sell_date = next_day.strftime("%Y-%m-%d")
                else:
                    sell_price = current_price
                    sell_date = day_str

                if sell_price is None:
                    sell_price = current_price
                    sell_date = day_str

                pnl = pos.pnl(sell_price)
                pnl_pct = pos.pnl_pct(sell_price)
                hold_days = (pd.Timestamp(sell_date) - pd.Timestamp(pos.entry_date)).days

                trades.append(TradeRecord(
                    ticker=ticker, entry_date=pos.entry_date,
                    entry_score=pos.entry_score, entry_price=pos.entry_price,
                    cost_basis=pos.cost_basis, exit_date=sell_date,
                    exit_price=sell_price, exit_reason=exit_reason,
                    pnl=pnl, pnl_pct=pnl_pct, hold_days=hold_days,
                ))

                cash += pos.current_value(sell_price)
                tickers_to_exit.append(ticker)

        for t in tickers_to_exit:
            del positions[t]

        # === 2. CHECK NEW ENTRY SIGNALS ===
        new_signals = []
        for ticker, score in today_scores.items():
            if score >= threshold and ticker not in positions:
                # Persistence filter: require score above threshold on previous snapshot too
                if persistence_filter:
                    signals_generated += 1
                    prev_score = prev_scores.get(ticker, 0)
                    if prev_score < threshold:
                        signals_filtered += 1
                        continue
                else:
                    signals_generated += 1

                new_signals.append((ticker, score))

        # Sort by score descending
        new_signals.sort(key=lambda x: -x[1])

        # === 3. BUY NEW POSITIONS ===
        for ticker, score in new_signals:
            if cash >= min_buy:
                position_size = min(cash, max_position_size)

                # Buy at next day's open
                if i + 1 < len(filtered_days):
                    next_day = filtered_days[i + 1]
                    buy_price = get_price(price_data, ticker, next_day, "Open")
                    buy_date = next_day.strftime("%Y-%m-%d")
                else:
                    continue  # Can't buy on last day

                if buy_price is None or buy_price <= 0:
                    continue

                shares = position_size / buy_price
                positions[ticker] = Position(
                    ticker=ticker, entry_date=buy_date,
                    entry_price=buy_price, shares=shares,
                    entry_score=score,
                )
                cash -= position_size

            elif rotation_strategy != "none" and positions:
                # Try rotation
                if rotation_strategy == "trim_weakest":
                    # Find position with lowest current score
                    weakest_ticker = None
                    weakest_score = float("inf")
                    for t, pos in positions.items():
                        t_score = today_scores.get(t, 0)
                        if t_score < weakest_score:
                            weakest_score = t_score
                            weakest_ticker = t

                    if weakest_ticker and weakest_score < score:
                        pos = positions[weakest_ticker]
                        sell_price = get_price(price_data, weakest_ticker, day, "Close")
                        if sell_price is None:
                            continue

                        pnl = pos.pnl(sell_price)
                        pnl_pct = pos.pnl_pct(sell_price)
                        hold_days = (pd.Timestamp(day_str) - pd.Timestamp(pos.entry_date)).days

                        trades.append(TradeRecord(
                            ticker=weakest_ticker, entry_date=pos.entry_date,
                            entry_score=pos.entry_score, entry_price=pos.entry_price,
                            cost_basis=pos.cost_basis, exit_date=day_str,
                            exit_price=sell_price, exit_reason="rotation",
                            pnl=pnl, pnl_pct=pnl_pct, hold_days=hold_days,
                        ))

                        proceeds = pos.current_value(sell_price)
                        cash += proceeds
                        del positions[weakest_ticker]
                        rotation_events += 1

                        # Now buy
                        position_size = min(cash, max_position_size)
                        if i + 1 < len(filtered_days):
                            next_day = filtered_days[i + 1]
                            buy_price = get_price(price_data, ticker, next_day, "Open")
                            buy_date = next_day.strftime("%Y-%m-%d")
                        else:
                            continue

                        if buy_price and buy_price > 0:
                            shares = position_size / buy_price
                            positions[ticker] = Position(
                                ticker=ticker, entry_date=buy_date,
                                entry_price=buy_price, shares=shares,
                                entry_score=score,
                            )
                            cash -= position_size
                    else:
                        signals_missed += 1

                elif rotation_strategy == "trim_worst_pnl":
                    # Find position with worst unrealized P&L
                    worst_ticker = None
                    worst_pnl = float("inf")
                    for t, pos in positions.items():
                        t_price = get_price(price_data, t, day, "Close")
                        if t_price is None:
                            continue
                        t_pnl = pos.pnl_pct(t_price)
                        if t_pnl < worst_pnl:
                            worst_pnl = t_pnl
                            worst_ticker = t

                    if worst_ticker:
                        pos = positions[worst_ticker]
                        sell_price = get_price(price_data, worst_ticker, day, "Close")
                        if sell_price is None:
                            continue

                        pnl = pos.pnl(sell_price)
                        pnl_pct = pos.pnl_pct(sell_price)
                        hold_days = (pd.Timestamp(day_str) - pd.Timestamp(pos.entry_date)).days

                        trades.append(TradeRecord(
                            ticker=worst_ticker, entry_date=pos.entry_date,
                            entry_score=pos.entry_score, entry_price=pos.entry_price,
                            cost_basis=pos.cost_basis, exit_date=day_str,
                            exit_price=sell_price, exit_reason="rotation",
                            pnl=pnl, pnl_pct=pnl_pct, hold_days=hold_days,
                        ))

                        proceeds = pos.current_value(sell_price)
                        cash += proceeds
                        del positions[worst_ticker]
                        rotation_events += 1

                        # Now buy
                        position_size = min(cash, max_position_size)
                        if i + 1 < len(filtered_days):
                            next_day = filtered_days[i + 1]
                            buy_price = get_price(price_data, ticker, next_day, "Open")
                            buy_date = next_day.strftime("%Y-%m-%d")
                        else:
                            continue

                        if buy_price and buy_price > 0:
                            shares = position_size / buy_price
                            positions[ticker] = Position(
                                ticker=ticker, entry_date=buy_date,
                                entry_price=buy_price, shares=shares,
                                entry_score=score,
                            )
                            cash -= position_size
                    else:
                        signals_missed += 1
            else:
                signals_missed += 1

        # === 4. RECORD DAILY SNAPSHOT ===
        total_position_value = 0
        for t, pos in positions.items():
            p = get_price(price_data, t, day, "Close")
            if p:
                total_position_value += pos.current_value(p)
            else:
                total_position_value += pos.cost_basis  # fallback

        total_value = cash + total_position_value

        history.append({
            "date": day_str,
            "total_value": round(total_value, 2),
            "cash": round(cash, 2),
            "num_positions": len(positions),
            "invested": round(total_position_value, 2),
        })

        # Update prev_scores for persistence filter
        if today_scores:
            prev_scores = dict(today_scores)

    # === CLOSE REMAINING POSITIONS at end ===
    final_day = filtered_days[-1]
    final_day_str = final_day.strftime("%Y-%m-%d")
    for ticker, pos in list(positions.items()):
        sell_price = get_price(price_data, ticker, final_day, "Close")
        if sell_price is None:
            sell_price = pos.entry_price  # fallback

        pnl = pos.pnl(sell_price)
        pnl_pct = pos.pnl_pct(sell_price)
        hold_days = (pd.Timestamp(final_day_str) - pd.Timestamp(pos.entry_date)).days

        trades.append(TradeRecord(
            ticker=ticker, entry_date=pos.entry_date,
            entry_score=pos.entry_score, entry_price=pos.entry_price,
            cost_basis=pos.cost_basis, exit_date=final_day_str,
            exit_price=sell_price, exit_reason="end_of_backtest",
            pnl=pnl, pnl_pct=pnl_pct, hold_days=hold_days,
        ))
        cash += pos.current_value(sell_price)

    positions.clear()

    # === COMPUTE METRICS ===
    final_value = history[-1]["total_value"] if history else starting_capital
    total_return = (final_value - starting_capital) / starting_capital

    spy_return = get_spy_return(price_data, filtered_days[0], filtered_days[-1])
    alpha = total_return - spy_return

    # Max drawdown
    peak = starting_capital
    max_dd = 0
    for h in history:
        if h["total_value"] > peak:
            peak = h["total_value"]
        dd = (h["total_value"] - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Trade stats
    winners = [t for t in trades if t.pnl > 0]
    losers = [t for t in trades if t.pnl <= 0]
    win_rate = len(winners) / len(trades) * 100 if trades else 0
    avg_gain = np.mean([t.pnl_pct for t in trades]) * 100 if trades else 0
    avg_hold = np.mean([t.hold_days for t in trades]) if trades else 0
    max_concurrent = max(h["num_positions"] for h in history) if history else 0

    # Capital utilization
    invested_pcts = [h["invested"] / h["total_value"] * 100 if h["total_value"] > 0 else 0
                     for h in history]
    avg_utilization = np.mean(invested_pcts) if invested_pcts else 0

    # Avg winner / avg loser
    avg_winner_pct = np.mean([t.pnl_pct for t in winners]) * 100 if winners else 0
    avg_loser_pct = np.mean([t.pnl_pct for t in losers]) * 100 if losers else 0
    avg_winner_dollar = np.mean([t.pnl for t in winners]) if winners else 0
    avg_loser_dollar = np.mean([t.pnl for t in losers]) if losers else 0
    win_loss_ratio = abs(avg_winner_pct / avg_loser_pct) if avg_loser_pct != 0 else float("inf")
    total_gains = sum(t.pnl for t in winners)
    total_losses = abs(sum(t.pnl for t in losers))
    profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")

    # Best and worst trades
    best_trade = max(trades, key=lambda t: t.pnl_pct) if trades else None
    worst_trade = min(trades, key=lambda t: t.pnl_pct) if trades else None

    return {
        "threshold": threshold,
        "persistence_filter": persistence_filter,
        "rotation_strategy": rotation_strategy,
        "starting_capital": starting_capital,
        "final_value": final_value,
        "total_return": total_return * 100,
        "spy_return": spy_return * 100,
        "alpha": alpha * 100,
        "max_drawdown": max_dd * 100,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "avg_gain_per_trade": avg_gain,
        "avg_hold_days": avg_hold,
        "max_concurrent_positions": max_concurrent,
        "capital_utilization": avg_utilization,
        "signals_generated": signals_generated,
        "signals_filtered": signals_filtered,
        "signals_missed": signals_missed,
        "rotation_events": rotation_events,
        "trades": trades,
        "history": history,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "avg_winner_pct": avg_winner_pct,
        "avg_loser_pct": avg_loser_pct,
        "avg_winner_dollar": avg_winner_dollar,
        "avg_loser_dollar": avg_loser_dollar,
        "win_loss_ratio": win_loss_ratio,
        "profit_factor": profit_factor,
        "start_date": filtered_days[0].strftime("%Y-%m-%d"),
        "end_date": filtered_days[-1].strftime("%Y-%m-%d"),
    }


# ============================================================
# REPORTING
# ============================================================

def print_section_1(results_by_threshold: dict):
    """Entry Threshold Comparison."""
    thresholds = sorted(results_by_threshold.keys(), reverse=True)
    labels = [f"\u2265 {t}" for t in thresholds]

    print("\n" + "=" * 80)
    print("  SECTION 1: ENTRY THRESHOLD ANALYSIS (No Filter, No Rotation)")
    print("=" * 80)
    print(f"\n  {'':30s}", end="")
    for label in labels:
        print(f"{label:>14s}", end="")
    print()
    print("  " + "\u2500" * 72)

    metrics = [
        ("Total return", "total_return", "+.1f", "%"),
        ("SPY return", "spy_return", "+.1f", "%"),
        ("Alpha", "alpha", "+.1f", "%"),
        ("Max drawdown", "max_drawdown", "+.1f", "%"),
        ("Total trades", "total_trades", "d", ""),
        ("Win rate", "win_rate", ".1f", "%"),
        ("Avg hold (days)", "avg_hold_days", ".0f", ""),
        ("Avg gain/trade", "avg_gain_per_trade", "+.1f", "%"),
        ("Max concurrent pos", "max_concurrent_positions", "d", ""),
        ("Capital utilization", "capital_utilization", ".1f", "%"),
    ]

    for label, key, fmt, suffix in metrics:
        print(f"  {label:30s}", end="")
        for t in thresholds:
            r = results_by_threshold[t]
            val = r[key]
            print(f"{val:{fmt}}{suffix:>14s}"[:14].rjust(14), end="")
        print()

    print("  " + "\u2500" * 72)


def print_section_2(results_no_filter: dict, results_with_filter: dict):
    """Persistence Filter Impact."""
    print("\n" + "=" * 80)
    print("  SECTION 2: PERSISTENCE FILTER IMPACT")
    print("=" * 80)

    for threshold in sorted(results_no_filter.keys(), reverse=True):
        nf = results_no_filter[threshold]
        wf = results_with_filter[threshold]

        print(f"\n  Threshold \u2265 {threshold}")
        print(f"  {'':30s}{'No Filter':>14s}{'2-Day Confirm':>14s}")
        print("  " + "\u2500" * 58)

        rows = [
            ("Total return", f"{nf['total_return']:+.1f}%", f"{wf['total_return']:+.1f}%"),
            ("Signals generated", f"{nf['signals_generated']}", f"{wf['signals_generated']}"),
            ("Signals filtered", "N/A", f"{wf['signals_filtered']} ({wf['signals_filtered']/max(wf['signals_generated'],1)*100:.0f}% removed)"),
            ("Win rate", f"{nf['win_rate']:.1f}%", f"{wf['win_rate']:.1f}%"),
            ("Avg gain/trade", f"{nf['avg_gain_per_trade']:+.1f}%", f"{wf['avg_gain_per_trade']:+.1f}%"),
            ("Total trades", f"{nf['total_trades']}", f"{wf['total_trades']}"),
        ]

        for label, v1, v2 in rows:
            print(f"  {label:30s}{v1:>14s}{v2:>14s}")
        print("  " + "\u2500" * 58)


def print_section_3(results_by_rotation: dict, threshold: float):
    """Capital Rotation Strategy Comparison."""
    print("\n" + "=" * 80)
    print(f"  SECTION 3: ROTATION STRATEGY COMPARISON (Threshold \u2265 {threshold})")
    print("=" * 80)

    strategies = ["none", "trim_weakest", "trim_worst_pnl"]
    labels = ["No Rotate", "Trim Weakest", "Trim Worst PnL"]

    print(f"\n  {'':30s}", end="")
    for label in labels:
        print(f"{label:>16s}", end="")
    print()
    print("  " + "\u2500" * 78)

    metrics = [
        ("Total return", "total_return", "+.1f", "%"),
        ("Alpha vs SPY", "alpha", "+.1f", "%"),
        ("Max drawdown", "max_drawdown", "+.1f", "%"),
        ("Signals missed", "signals_missed", "d", ""),
        ("Rotation events", "rotation_events", "d", ""),
        ("Avg hold (days)", "avg_hold_days", ".0f", ""),
        ("Total trades", "total_trades", "d", ""),
    ]

    for label, key, fmt, suffix in metrics:
        print(f"  {label:30s}", end="")
        for s in strategies:
            r = results_by_rotation[s]
            val = r[key]
            cell = f"{val:{fmt}}{suffix}"
            print(f"{cell:>16s}", end="")
        print()

    print("  " + "\u2500" * 78)


def print_section_4(trades: list):
    """Trade Log (top 30 by P&L)."""
    print("\n" + "=" * 80)
    print("  SECTION 4: TRADE LOG (Top 30 by P&L)")
    print("=" * 80)

    sorted_trades = sorted(trades, key=lambda t: -t.pnl)[:30]

    print(f"\n  {'Ticker':8s} {'Entry Date':12s} {'Score':6s} {'Entry $':>10s} {'Exit Date':12s} {'Exit Reason':14s} {'P&L':>10s} {'P&L %':>8s} {'Hold':>5s}")
    print("  " + "\u2500" * 90)

    for t in sorted_trades:
        print(f"  {t.ticker:8s} {t.entry_date:12s} {t.entry_score:5.1f}  ${t.cost_basis:>9,.0f} {t.exit_date:12s} {t.exit_reason:14s} ${t.pnl:>+9,.0f} {t.pnl_pct*100:>+7.1f}% {t.hold_days:>4d}d")

    # Also worst 10
    print(f"\n  Bottom 10 by P&L:")
    print("  " + "\u2500" * 90)
    worst = sorted(trades, key=lambda t: t.pnl)[:10]
    for t in worst:
        print(f"  {t.ticker:8s} {t.entry_date:12s} {t.entry_score:5.1f}  ${t.cost_basis:>9,.0f} {t.exit_date:12s} {t.exit_reason:14s} ${t.pnl:>+9,.0f} {t.pnl_pct*100:>+7.1f}% {t.hold_days:>4d}d")


def save_equity_curve(history: list, price_data: dict, filepath: str):
    """Save equity curve CSV."""
    spy_start = None
    rows = []

    for h in history:
        spy_price = get_price(price_data, "SPY", h["date"], "Close")
        if spy_start is None and spy_price:
            spy_start = spy_price

        spy_value = 100000 * (spy_price / spy_start) if spy_start and spy_price else 100000

        rows.append({
            "DATE": h["date"],
            "PORTFOLIO_VALUE": h["total_value"],
            "SPY_VALUE": round(spy_value, 2),
            "CASH": h["cash"],
            "NUM_POSITIONS": h["num_positions"],
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"\n  Equity curve saved to: {filepath}")


def save_trade_log(trades: list, filepath: str):
    """Save every trade to CSV."""
    rows = []
    for t in sorted(trades, key=lambda x: x.entry_date):
        rows.append({
            "Ticker": t.ticker,
            "Entry Date": t.entry_date,
            "Entry Score": t.entry_score,
            "Entry Price": round(t.entry_price, 2),
            "Cost Basis": round(t.cost_basis, 2),
            "Exit Date": t.exit_date,
            "Exit Price": round(t.exit_price, 2),
            "Exit Reason": t.exit_reason,
            "P&L ($)": round(t.pnl, 2),
            "P&L (%)": round(t.pnl_pct, 1),
            "Hold Days": t.hold_days,
        })
    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"  Trade log saved to: {filepath}")
    print(f"  Total trades: {len(rows)}")


def print_section_6(result: dict):
    """Summary Statistics."""
    print("\n" + "=" * 80)
    print("  SECTION 6: SUMMARY STATISTICS")
    print("=" * 80)

    bt = result["best_trade"]
    wt = result["worst_trade"]

    if bt:
        print(f"\n  BEST TRADE:     {bt.ticker} {bt.pnl_pct*100:+.1f}% (${bt.pnl:+,.0f}) held {bt.hold_days} days")
    if wt:
        print(f"  WORST TRADE:    {wt.ticker} {wt.pnl_pct*100:+.1f}% (${wt.pnl:+,.0f}) held {wt.hold_days} days ({wt.exit_reason})")

    print(f"  AVG WINNER:     {result['avg_winner_pct']:+.1f}% (${result['avg_winner_dollar']:+,.0f})")
    print(f"  AVG LOSER:      {result['avg_loser_pct']:+.1f}% (${result['avg_loser_dollar']:+,.0f})")
    print(f"  WIN/LOSS RATIO: {result['win_loss_ratio']:.2f}x (avg winner / avg loser)")
    print(f"  PROFIT FACTOR:  {result['profit_factor']:.2f}x (total gains / total losses)")
    print(f"\n  Period: {result['start_date']} \u2192 {result['end_date']}")
    print(f"  Starting: ${result['starting_capital']:,.0f} \u2192 Final: ${result['final_value']:,.0f}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("  PORTFOLIO SIMULATION BACKTEST")
    print("  Alpha Scanner — Capital-Constrained Performance Analysis")
    print("=" * 80)

    # Load data
    print("\n  Loading score data from database...")
    score_df = load_score_data()
    print(f"  {len(score_df)} score records loaded")
    print(f"  Date range: {score_df['date'].min().strftime('%Y-%m-%d')} \u2192 {score_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Unique tickers: {score_df['ticker'].nunique()}")

    print("\n  Fetching 2 years of price data...")
    cfg = load_config()
    price_data = fetch_all(cfg, period="2y", verbose=False)
    print(f"  {len(price_data)} tickers fetched")

    trading_days = get_trading_days(price_data)
    print(f"  {len(trading_days)} trading days")

    # Build daily score lookup with carry-forward
    print("  Building daily score lookup...")
    daily_scores = build_daily_scores(score_df, trading_days)
    print(f"  {len(daily_scores)} days with scores")

    # ──────────────────────────────────────────────────────
    # SECTION 1: Entry Threshold Comparison
    # ──────────────────────────────────────────────────────
    print("\n  Running Section 1: Entry Threshold Comparison...")
    thresholds = [9.5, 9.0, 8.5]
    results_no_filter = {}

    for t in thresholds:
        print(f"    Threshold \u2265 {t}...", end="", flush=True)
        r = run_simulation(daily_scores, price_data, trading_days,
                           threshold=t, persistence_filter=False,
                           rotation_strategy="none")
        results_no_filter[t] = r
        print(f" {r['total_return']:+.1f}% return, {r['total_trades']} trades")

    print_section_1(results_no_filter)

    # ──────────────────────────────────────────────────────
    # SECTION 2: Persistence Filter Impact
    # ──────────────────────────────────────────────────────
    print("\n  Running Section 2: Persistence Filter Impact...")
    results_with_filter = {}

    for t in thresholds:
        print(f"    Threshold \u2265 {t} (2-day confirm)...", end="", flush=True)
        r = run_simulation(daily_scores, price_data, trading_days,
                           threshold=t, persistence_filter=True,
                           rotation_strategy="none")
        results_with_filter[t] = r
        print(f" {r['total_return']:+.1f}% return, {r['total_trades']} trades")

    print_section_2(results_no_filter, results_with_filter)

    # ──────────────────────────────────────────────────────
    # SECTION 3: Capital Rotation (use best threshold)
    # ──────────────────────────────────────────────────────
    # Find best threshold by alpha
    best_threshold = max(results_no_filter, key=lambda t: results_no_filter[t]["alpha"])
    print(f"\n  Best threshold by alpha: \u2265 {best_threshold} ({results_no_filter[best_threshold]['alpha']:+.1f}%)")

    print(f"\n  Running Section 3: Capital Rotation Strategies (threshold \u2265 {best_threshold})...")
    results_by_rotation = {}

    for strategy in ["none", "trim_weakest", "trim_worst_pnl"]:
        print(f"    Strategy: {strategy}...", end="", flush=True)
        r = run_simulation(daily_scores, price_data, trading_days,
                           threshold=best_threshold, persistence_filter=False,
                           rotation_strategy=strategy)
        results_by_rotation[strategy] = r
        print(f" {r['total_return']:+.1f}% return, {r['rotation_events']} rotations")

    print_section_3(results_by_rotation, best_threshold)

    # ──────────────────────────────────────────────────────
    # SECTION 4: Trade Log (from best config)
    # ──────────────────────────────────────────────────────
    best_result = results_no_filter[best_threshold]
    print_section_4(best_result["trades"])

    # ──────────────────────────────────────────────────────
    # SECTION 5: Equity Curve CSV
    # ──────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SECTION 5: EQUITY CURVE")
    print("=" * 80)
    csv_path = str(Path(__file__).parent / "portfolio_equity_curve.csv")
    save_equity_curve(best_result["history"], price_data, csv_path)

    # Save full trade log CSV
    trade_log_path = str(Path(__file__).parent / "portfolio_trade_log.csv")
    save_trade_log(best_result["trades"], trade_log_path)

    # ──────────────────────────────────────────────────────
    # SECTION 6: Summary Statistics
    # ──────────────────────────────────────────────────────
    print_section_6(best_result)

    print("\n" + "=" * 80)
    print("  BACKTEST COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
