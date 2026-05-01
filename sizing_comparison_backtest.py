"""
sizing_comparison_backtest.py — Head-to-head comparison of four portfolio
sizing strategies on identical historical data.

Built from scratch (does NOT reuse portfolio_backtest.py simulation logic,
which was found to have static-sizing and no-position-cap bugs).

Strategies tested:
  A. Fixed / 5-Max      — 20% of current equity per position, max 5 slots
  B. Fixed / 10-Max     — 10% of current equity per position, max 10 slots
  C. Dynamic / Trim     — equity/(N+1) sizing with proportional trim, max 10
  D. Fixed/10 + Swap    — same as B, but swaps weakest position at cap

Usage:
    python sizing_comparison_backtest.py
    python sizing_comparison_backtest.py --no-path-test      # skip path dep (faster)
    python sizing_comparison_backtest.py --path-starts 20    # more start dates
    python sizing_comparison_backtest.py --start 2023-06-01 --end 2026-04-15
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config
from data_fetcher import fetch_all

DB_PATH = Path(os.environ.get("ALPHA_DB_PATH", Path(__file__).parent / "breakout_tracker.db"))

# Persistence-days override (env var) — for testing extended-persistence variants
# without rewriting all the hardcoded `persistence_days=3` literals.
_PERSISTENCE_OVERRIDE = os.environ.get("PERSISTENCE_DAYS_OVERRIDE")
PERSISTENCE_DAYS = int(_PERSISTENCE_OVERRIDE) if _PERSISTENCE_OVERRIDE else 3


# ============================================================
# STRATEGY CONFIG
# ============================================================

@dataclass
class StopLossConfig:
    type: str          # "none", "fixed", "trailing", "atr_fixed", "atr_trailing"
    value: float = 0.0 # percentage (0.15 = 15%) or ATR multiplier (2.0, 3.0)


STOP_FIXED_15 = StopLossConfig(type="fixed", value=0.15)


@dataclass
class StrategyConfig:
    name: str
    max_positions: int
    sizing_mode: str          # "fixed_pct" or "dynamic"
    fixed_position_pct: float # for fixed_pct mode
    min_entry_pct: float      # floor for dynamic sizing
    trim_enabled: bool
    entry_protection_days: int
    entry_threshold: float
    exit_threshold: float
    stop_loss: StopLossConfig
    persistence_days: int
    swap_at_cap: bool = False          # swap weakest position when at cap
    swap_score_threshold: float = 8.0  # victim must score below this OR have negative P&L


STRATEGY_A = StrategyConfig(
    name="Fixed/5-Max",
    max_positions=5,
    sizing_mode="fixed_pct",
    fixed_position_pct=0.20,
    min_entry_pct=0.05,
    trim_enabled=False,
    entry_protection_days=7,
    entry_threshold=8.5,
    exit_threshold=5.0,
    stop_loss=STOP_FIXED_15,
    persistence_days=PERSISTENCE_DAYS,
)

STRATEGY_B = StrategyConfig(
    name="Fixed/10-Max",
    max_positions=10,
    sizing_mode="fixed_pct",
    fixed_position_pct=0.10,
    min_entry_pct=0.05,
    trim_enabled=False,
    entry_protection_days=7,
    entry_threshold=8.5,
    exit_threshold=5.0,
    stop_loss=STOP_FIXED_15,
    persistence_days=PERSISTENCE_DAYS,
)

STRATEGY_C = StrategyConfig(
    name="Dynamic/Trim",
    max_positions=10,
    sizing_mode="dynamic",
    fixed_position_pct=0.10,
    min_entry_pct=0.05,
    trim_enabled=True,
    entry_protection_days=7,
    entry_threshold=8.5,
    exit_threshold=5.0,
    stop_loss=STOP_FIXED_15,
    persistence_days=PERSISTENCE_DAYS,
)

STRATEGY_D = StrategyConfig(
    name="Fixed/10+Swap",
    max_positions=10,
    sizing_mode="fixed_pct",
    fixed_position_pct=0.10,
    min_entry_pct=0.05,
    trim_enabled=False,
    entry_protection_days=7,
    entry_threshold=8.5,
    exit_threshold=5.0,
    stop_loss=STOP_FIXED_15,
    persistence_days=PERSISTENCE_DAYS,
    swap_at_cap=True,
    swap_score_threshold=8.0,
)


# ============================================================
# DATA LOADING
# ============================================================

def load_score_data() -> pd.DataFrame:
    """Load all ticker scores from SQLite."""
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT date, ticker, score FROM ticker_scores ORDER BY date, ticker",
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_score_lookup(score_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Build {date_str: {ticker: score}} from score dataframe."""
    lookup: dict[str, dict[str, float]] = {}
    for _, row in score_df.iterrows():
        d = row["date"].strftime("%Y-%m-%d")
        if d not in lookup:
            lookup[d] = {}
        lookup[d][row["ticker"]] = row["score"]
    return lookup


def build_daily_scores(
    score_df: pd.DataFrame,
    trading_days: list[pd.Timestamp],
) -> dict[str, dict[str, float]]:
    """Build daily score lookup with carry-forward for gaps."""
    score_lookup = build_score_lookup(score_df)
    daily: dict[str, dict[str, float]] = {}
    current_scores: dict[str, float] = {}

    for day in trading_days:
        day_str = day.strftime("%Y-%m-%d")
        snap = score_lookup.get(day_str)
        if snap:
            current_scores.update(snap)
        daily[day_str] = dict(current_scores)

    return daily


def get_trading_days(price_data: dict) -> list[pd.Timestamp]:
    """Sorted list of trading days from SPY (or first available ticker)."""
    spy_df = price_data.get("SPY")
    if spy_df is not None and not spy_df.empty:
        idx = spy_df.index.tz_localize(None) if spy_df.index.tz else spy_df.index
        return sorted(idx)
    for df in price_data.values():
        if df is not None and not df.empty:
            idx = df.index.tz_localize(None) if df.index.tz else df.index
            return sorted(idx)
    return []


def get_price(
    price_data: dict, ticker: str, date_val, col: str = "Open",
) -> float | None:
    """Price for ticker on date. Falls back to nearest within 3 days."""
    df = price_data.get(ticker)
    if df is None or df.empty:
        return None

    idx = df.index.tz_localize(None) if df.index.tz else df.index
    target = pd.Timestamp(date_val)

    matches = idx[idx == target]
    if len(matches) > 0:
        return float(df.loc[df.index[idx == target][0], col])

    diffs = abs(idx - target)
    min_i = diffs.argmin()
    if diffs[min_i] <= timedelta(days=3):
        return float(df.iloc[min_i][col])
    return None


def compute_atr(
    price_data: dict, ticker: str, date_val, period: int = 14,
) -> float | None:
    """Compute the Average True Range for a ticker up to a given date."""
    df = price_data.get(ticker)
    if df is None or df.empty:
        return None

    idx = df.index.tz_localize(None) if df.index.tz else df.index
    target = pd.Timestamp(date_val)

    mask = idx <= target
    if mask.sum() < period + 1:
        return None

    recent = df.loc[df.index[mask]].tail(period + 1)

    # Extract columns, handling potential multi-level from yfinance
    cols = {}
    for col_name in ("High", "Low", "Close"):
        c = recent[col_name]
        if hasattr(c, "columns"):
            c = c.iloc[:, 0]
        cols[col_name] = c

    tr_values = []
    for i in range(1, len(recent)):
        h = float(cols["High"].iloc[i])
        l = float(cols["Low"].iloc[i])
        pc = float(cols["Close"].iloc[i - 1])
        tr = max(h - l, abs(h - pc), abs(l - pc))
        tr_values.append(tr)

    if not tr_values:
        return None
    return float(np.mean(tr_values[-period:]))


def check_stop_triggered(
    pos: "Position", stop_cfg: StopLossConfig, day_low: float,
) -> bool:
    """Check if a stop loss has been triggered for a position."""
    if stop_cfg.type == "none":
        return False

    if stop_cfg.type == "fixed":
        stop_price = pos.entry_price * (1 - stop_cfg.value)
        return day_low <= stop_price

    if stop_cfg.type == "trailing":
        stop_price = pos.max_close_since_entry * (1 - stop_cfg.value)
        return day_low <= stop_price

    if stop_cfg.type == "atr_fixed":
        if pos.entry_atr <= 0:
            return False
        stop_price = pos.entry_price - (stop_cfg.value * pos.entry_atr)
        return day_low <= stop_price

    if stop_cfg.type == "atr_trailing":
        if pos.entry_atr <= 0:
            return False
        stop_price = pos.max_close_since_entry - (stop_cfg.value * pos.entry_atr)
        return day_low <= stop_price

    return False


def parse_stop_loss_config(config_str: str) -> StopLossConfig:
    """Parse a stop loss config string like 'fixed-15', 'trail-20', 'ATR-3x'."""
    s = config_str.strip().lower()
    if s == "none":
        return StopLossConfig(type="none")
    if s.startswith("fixed-"):
        pct = int(s.split("-")[1]) / 100
        return StopLossConfig(type="fixed", value=pct)
    if s.startswith("trail-"):
        pct = int(s.split("-")[1]) / 100
        return StopLossConfig(type="trailing", value=pct)
    if s.startswith("atr-trail-"):
        mult = int(s.replace("atr-trail-", "").replace("x", ""))
        return StopLossConfig(type="atr_trailing", value=float(mult))
    if s.startswith("atr-"):
        mult = int(s.replace("atr-", "").replace("x", ""))
        return StopLossConfig(type="atr_fixed", value=float(mult))
    raise ValueError(f"Unknown stop loss config: {config_str}")


def stop_loss_label(cfg: StopLossConfig) -> str:
    """Human-readable label for a StopLossConfig."""
    if cfg.type == "none":
        return "none"
    if cfg.type == "fixed":
        return f"fixed-{int(cfg.value * 100)}"
    if cfg.type == "trailing":
        return f"trail-{int(cfg.value * 100)}"
    if cfg.type == "atr_fixed":
        return f"ATR-{int(cfg.value)}x"
    if cfg.type == "atr_trailing":
        return f"ATR-trail-{int(cfg.value)}x"
    return str(cfg)


# ============================================================
# PORTFOLIO SIMULATOR
# ============================================================

@dataclass
class Position:
    ticker: str
    entry_date: str       # YYYY-MM-DD
    entry_price: float
    shares: float
    entry_score: float
    cost_basis: float = 0.0
    trimmed: bool = False           # was this position ever trimmed?
    trim_funded_entry: bool = False # was entry funded by trimming others?
    max_close_since_entry: float = 0.0  # trailing high-water mark
    entry_atr: float = 0.0              # 14-day ATR at entry (for ATR-based stops)

    def __post_init__(self):
        self.cost_basis = self.entry_price * self.shares
        if self.max_close_since_entry == 0.0:
            self.max_close_since_entry = self.entry_price

    def value(self, price: float) -> float:
        return self.shares * price

    def pnl(self, price: float) -> float:
        return self.value(price) - self.cost_basis

    def pnl_pct(self, price: float) -> float:
        return self.pnl(price) / self.cost_basis if self.cost_basis > 0 else 0.0

    def hold_days(self, current_date: str) -> int:
        return (pd.Timestamp(current_date) - pd.Timestamp(self.entry_date)).days


@dataclass
class Trade:
    ticker: str
    entry_date: str
    entry_price: float
    entry_score: float
    cost_basis: float
    exit_date: str
    exit_price: float
    exit_reason: str
    pnl: float
    pnl_pct: float
    hold_days: int
    trim_funded: bool = False


@dataclass
class TrimEvent:
    date: str
    ticker_trimmed: str
    shares_trimmed: float
    dollar_amount: float
    pnl_at_trim: float      # P&L % of position when trimmed
    new_entry_ticker: str    # what entry was funded


@dataclass
class SwapEvent:
    date: str
    swapped_out_ticker: str
    swapped_out_score: float
    swapped_out_pnl_pct: float
    swapped_out_days_held: int
    swapped_out_entry_price: float   # for "what would have happened" analysis
    swapped_in_ticker: str
    swapped_in_score: float


@dataclass
class SimResult:
    config: StrategyConfig
    trades: list[Trade]
    trims: list[TrimEvent]
    swaps: list[SwapEvent]
    history: list[dict]         # daily snapshots
    signals_skipped: int
    signals_generated: int
    signals_filtered: int
    trim_count: int
    total_trimmed_dollars: float
    swap_count: int
    swap_skipped_threshold: int     # swaps skipped because victim didn't meet threshold
    start_date: str
    end_date: str
    starting_capital: float
    # positions held on each day (for overlap analysis)
    daily_holdings: dict[str, set[str]]   # date -> set of tickers


class PortfolioSimulator:
    """Run a portfolio simulation with a given strategy config."""

    def __init__(
        self,
        config: StrategyConfig,
        daily_scores: dict[str, dict[str, float]],
        price_data: dict,
        trading_days: list[pd.Timestamp],
        starting_capital: float = 100_000,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        self.cfg = config
        self.daily_scores = daily_scores
        self.price_data = price_data
        self.starting_capital = starting_capital

        # Determine simulation window
        score_dates = sorted(daily_scores.keys())
        sim_start = start_date or score_dates[0]
        sim_end = end_date or score_dates[-1]

        self.trading_days = [
            d for d in trading_days
            if sim_start <= d.strftime("%Y-%m-%d") <= sim_end
        ]

    def run(self) -> SimResult:
        cash = self.starting_capital
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        trims: list[TrimEvent] = []
        swaps: list[SwapEvent] = []
        history: list[dict] = []
        daily_holdings: dict[str, set[str]] = {}

        signals_generated = 0
        signals_filtered = 0
        signals_skipped = 0
        swap_skipped_threshold = 0

        cfg = self.cfg
        days = self.trading_days

        for i, day in enumerate(days):
            day_str = day.strftime("%Y-%m-%d")
            scores = self.daily_scores.get(day_str, {})

            # ── 1. EVALUATE EXITS ──
            # Check exit conditions on today's prices.
            # Actual sell executes at next day's open.
            to_exit: list[tuple[str, str]] = []  # (ticker, reason)
            for ticker, pos in positions.items():
                close = get_price(self.price_data, ticker, day, "Close")
                if close is None:
                    continue
                # Update trailing high-water mark
                pos.max_close_since_entry = max(pos.max_close_since_entry, close)
                score = scores.get(ticker, 0)
                if score < cfg.exit_threshold:
                    to_exit.append((ticker, "score_exit"))
                else:
                    # Check stop loss using daily low (more realistic)
                    low = get_price(self.price_data, ticker, day, "Low")
                    if low is None:
                        low = close
                    if check_stop_triggered(pos, cfg.stop_loss, low):
                        to_exit.append((ticker, "stop_loss"))

            for ticker, reason in to_exit:
                pos = positions[ticker]
                if i + 1 < len(days):
                    sell_day = days[i + 1]
                    sell_price = get_price(self.price_data, ticker, sell_day, "Open")
                    sell_date = sell_day.strftime("%Y-%m-%d")
                else:
                    sell_price = get_price(self.price_data, ticker, day, "Close")
                    sell_date = day_str
                if sell_price is None:
                    sell_price = get_price(self.price_data, ticker, day, "Close") or pos.entry_price
                    sell_date = day_str

                pnl = pos.pnl(sell_price)
                pnl_pct = pos.pnl_pct(sell_price)
                hd = pos.hold_days(sell_date)

                trades.append(Trade(
                    ticker=ticker, entry_date=pos.entry_date,
                    entry_price=pos.entry_price, entry_score=pos.entry_score,
                    cost_basis=pos.cost_basis, exit_date=sell_date,
                    exit_price=sell_price, exit_reason=reason,
                    pnl=pnl, pnl_pct=pnl_pct, hold_days=hd,
                    trim_funded=pos.trim_funded_entry,
                ))
                cash += pos.value(sell_price)
                del positions[ticker]

            # ── 2. FIND NEW ENTRY SIGNALS ──
            candidates = []
            for ticker, score in scores.items():
                if score < cfg.entry_threshold:
                    continue
                if ticker in positions:
                    continue
                signals_generated += 1

                # Persistence filter
                if cfg.persistence_days > 0:
                    if i < cfg.persistence_days:
                        signals_filtered += 1
                        continue
                    ok = True
                    for back in range(1, cfg.persistence_days + 1):
                        prior_str = days[i - back].strftime("%Y-%m-%d")
                        prior_scores = self.daily_scores.get(prior_str, {})
                        if prior_scores.get(ticker, 0) < cfg.entry_threshold:
                            ok = False
                            break
                    if not ok:
                        signals_filtered += 1
                        continue

                candidates.append((ticker, score))

            candidates.sort(key=lambda x: -x[1])

            # ── 3. PROCESS ENTRIES ──
            for ticker, score in candidates:
                if i + 1 >= len(days):
                    break  # can't buy on last day

                buy_day = days[i + 1]
                buy_price = get_price(self.price_data, ticker, buy_day, "Open")
                if not buy_price or buy_price <= 0:
                    continue
                buy_date = buy_day.strftime("%Y-%m-%d")

                # Current equity
                equity = cash
                for t, p in positions.items():
                    cp = get_price(self.price_data, t, day, "Close")
                    equity += p.value(cp) if cp else p.cost_basis

                # Target position size
                if cfg.sizing_mode == "fixed_pct":
                    target_size = equity * cfg.fixed_position_pct
                else:
                    # dynamic: equity / (N+1), floor at min_entry_pct
                    n = len(positions)
                    target_size = max(
                        equity / (n + 1),
                        equity * cfg.min_entry_pct,
                    )

                min_entry = max(500, equity * 0.01)  # absolute floor

                # ── Position cap check ──
                if len(positions) >= cfg.max_positions:
                    if cfg.swap_at_cap:
                        # ── SWAP-AT-CAP (Strategy D) ──
                        # Combined score + P&L ranking to find victim
                        eligible = []
                        for t in positions:
                            if positions[t].hold_days(day_str) < cfg.entry_protection_days:
                                continue
                            t_score = scores.get(t, 0)
                            cp = get_price(self.price_data, t, day, "Close")
                            t_pnl = positions[t].pnl_pct(cp) if cp else 0.0
                            eligible.append((t, t_score, t_pnl))

                        if not eligible:
                            signals_skipped += 1
                            continue

                        # Rank by score (lowest = rank 1) and P&L % (lowest = rank 1)
                        by_score = sorted(eligible, key=lambda x: x[1])
                        by_pnl = sorted(eligible, key=lambda x: x[2])
                        score_rank = {t: i + 1 for i, (t, _, _) in enumerate(by_score)}
                        pnl_rank = {t: i + 1 for i, (t, _, _) in enumerate(by_pnl)}

                        # Combined rank (average); tie-break by lower score
                        ranked = [
                            (t, (score_rank[t] + pnl_rank[t]) / 2, t_sc)
                            for t, t_sc, _ in eligible
                        ]
                        ranked.sort(key=lambda x: (x[1], x[2]))  # lowest combined, then lowest score
                        victim_t, _, victim_sc = ranked[0]
                        victim_pos = positions[victim_t]

                        cp = get_price(self.price_data, victim_t, day, "Close")
                        victim_pnl = victim_pos.pnl_pct(cp) * 100 if cp else 0.0

                        # Threshold check: victim must have score < 8.0 OR P&L < 0%
                        if victim_sc >= cfg.swap_score_threshold and victim_pnl >= 0:
                            swap_skipped_threshold += 1
                            signals_skipped += 1
                            continue

                        # Execute the swap: sell victim
                        sell_price = get_price(self.price_data, victim_t, buy_day, "Open")
                        if sell_price is None:
                            signals_skipped += 1
                            continue

                        vpnl = victim_pos.pnl(sell_price)
                        vpnl_pct = victim_pos.pnl_pct(sell_price)
                        vhd = victim_pos.hold_days(buy_date)
                        trades.append(Trade(
                            ticker=victim_t, entry_date=victim_pos.entry_date,
                            entry_price=victim_pos.entry_price,
                            entry_score=victim_pos.entry_score,
                            cost_basis=victim_pos.cost_basis, exit_date=buy_date,
                            exit_price=sell_price, exit_reason="swap",
                            pnl=vpnl, pnl_pct=vpnl_pct, hold_days=vhd,
                            trim_funded=victim_pos.trim_funded_entry,
                        ))

                        swaps.append(SwapEvent(
                            date=buy_date,
                            swapped_out_ticker=victim_t,
                            swapped_out_score=victim_sc,
                            swapped_out_pnl_pct=victim_pnl,
                            swapped_out_days_held=vhd,
                            swapped_out_entry_price=victim_pos.entry_price,
                            swapped_in_ticker=ticker,
                            swapped_in_score=score,
                        ))

                        cash += victim_pos.value(sell_price)
                        del positions[victim_t]

                    elif cfg.trim_enabled:
                        # ── TRIM-AT-CAP (Strategy C) ──
                        # At cap: close lowest-scoring position if new candidate is higher
                        eligible = [
                            (t, scores.get(t, 0))
                            for t in positions
                            if positions[t].hold_days(day_str) >= cfg.entry_protection_days
                        ]
                        if not eligible:
                            signals_skipped += 1
                            continue
                        worst_t, worst_s = min(eligible, key=lambda x: x[1])
                        if score <= worst_s:
                            signals_skipped += 1
                            continue
                        # Close the worst position
                        wp = positions[worst_t]
                        sell_price = get_price(self.price_data, worst_t, buy_day, "Open")
                        if sell_price is None:
                            signals_skipped += 1
                            continue
                        pnl = wp.pnl(sell_price)
                        pnl_pct = wp.pnl_pct(sell_price)
                        hd = wp.hold_days(buy_date)
                        trades.append(Trade(
                            ticker=worst_t, entry_date=wp.entry_date,
                            entry_price=wp.entry_price, entry_score=wp.entry_score,
                            cost_basis=wp.cost_basis, exit_date=buy_date,
                            exit_price=sell_price, exit_reason="replaced",
                            pnl=pnl, pnl_pct=pnl_pct, hold_days=hd,
                            trim_funded=wp.trim_funded_entry,
                        ))
                        cash += wp.value(sell_price)
                        del positions[worst_t]
                    else:
                        signals_skipped += 1
                        continue

                # ── Cash check + trimming ──
                trim_funded = False
                if cash < target_size:
                    if cfg.trim_enabled:
                        needed = target_size - cash
                        # Find trim-eligible positions
                        eligible = {
                            t: p for t, p in positions.items()
                            if p.hold_days(day_str) >= cfg.entry_protection_days
                        }
                        if not eligible:
                            signals_skipped += 1
                            continue

                        # Inverse P&L weighting
                        pnl_pcts = {}
                        for t, p in eligible.items():
                            cp = get_price(self.price_data, t, day, "Close")
                            pnl_pcts[t] = p.pnl_pct(cp) if cp else 0.0

                        min_pnl = min(pnl_pcts.values())
                        raw_weights = {}
                        for t, pct in pnl_pcts.items():
                            shifted = pct - min_pnl + 1.0
                            raw_weights[t] = 1.0 / shifted

                        total_w = sum(raw_weights.values())
                        trim_weights = {t: w / total_w for t, w in raw_weights.items()}

                        # Execute trims
                        freed = 0.0
                        for t, weight in trim_weights.items():
                            trim_dollars = needed * weight
                            p = positions[t]
                            cp = get_price(self.price_data, t, buy_day, "Open")
                            if cp is None or cp <= 0:
                                continue
                            trim_shares = math.floor(trim_dollars / cp)
                            if trim_shares <= 0:
                                continue
                            if trim_shares >= p.shares:
                                trim_shares = math.floor(p.shares) - 1
                                if trim_shares <= 0:
                                    continue

                            actual_trim = trim_shares * cp
                            pnl_at_trim = p.pnl_pct(cp)

                            trims.append(TrimEvent(
                                date=buy_date,
                                ticker_trimmed=t,
                                shares_trimmed=trim_shares,
                                dollar_amount=actual_trim,
                                pnl_at_trim=pnl_at_trim,
                                new_entry_ticker=ticker,
                            ))

                            p.shares -= trim_shares
                            p.cost_basis = p.entry_price * p.shares
                            p.trimmed = True
                            cash += actual_trim
                            freed += actual_trim

                        if cash < min_entry:
                            signals_skipped += 1
                            continue
                        trim_funded = True
                        # Recalculate target with available cash
                        target_size = min(target_size, cash)
                    else:
                        if cash < min_entry:
                            signals_skipped += 1
                            continue
                        target_size = cash  # buy what we can

                target_size = min(target_size, cash)
                shares = target_size / buy_price
                if shares <= 0:
                    signals_skipped += 1
                    continue

                # Compute ATR for ATR-based stops
                entry_atr = 0.0
                if cfg.stop_loss.type in ("atr_fixed", "atr_trailing"):
                    atr_val = compute_atr(self.price_data, ticker, day)
                    entry_atr = atr_val if atr_val else 0.0

                positions[ticker] = Position(
                    ticker=ticker,
                    entry_date=buy_date,
                    entry_price=buy_price,
                    shares=shares,
                    entry_score=score,
                    trim_funded_entry=trim_funded,
                    entry_atr=entry_atr,
                )
                cash -= target_size

            # ── 4. DAILY SNAPSHOT ──
            pos_value = 0.0
            for t, p in positions.items():
                cp = get_price(self.price_data, t, day, "Close")
                pos_value += p.value(cp) if cp else p.cost_basis

            total_value = cash + pos_value
            history.append({
                "date": day_str,
                "total_value": total_value,
                "cash": cash,
                "num_positions": len(positions),
                "invested": pos_value,
            })
            daily_holdings[day_str] = set(positions.keys())

        # ── CLOSE REMAINING POSITIONS ──
        if days:
            final_day = days[-1]
            final_str = final_day.strftime("%Y-%m-%d")
            for ticker, pos in list(positions.items()):
                sp = get_price(self.price_data, ticker, final_day, "Close")
                if sp is None:
                    sp = pos.entry_price
                pnl = pos.pnl(sp)
                pnl_pct = pos.pnl_pct(sp)
                hd = pos.hold_days(final_str)
                trades.append(Trade(
                    ticker=ticker, entry_date=pos.entry_date,
                    entry_price=pos.entry_price, entry_score=pos.entry_score,
                    cost_basis=pos.cost_basis, exit_date=final_str,
                    exit_price=sp, exit_reason="end_of_backtest",
                    pnl=pnl, pnl_pct=pnl_pct, hold_days=hd,
                    trim_funded=pos.trim_funded_entry,
                ))
                cash += pos.value(sp)

        total_trimmed = sum(te.dollar_amount for te in trims)

        return SimResult(
            config=self.cfg,
            trades=trades,
            trims=trims,
            swaps=swaps,
            history=history,
            signals_skipped=signals_skipped,
            signals_generated=signals_generated,
            signals_filtered=signals_filtered,
            trim_count=len(trims),
            total_trimmed_dollars=total_trimmed,
            swap_count=len(swaps),
            swap_skipped_threshold=swap_skipped_threshold,
            start_date=self.trading_days[0].strftime("%Y-%m-%d") if self.trading_days else "",
            end_date=self.trading_days[-1].strftime("%Y-%m-%d") if self.trading_days else "",
            starting_capital=self.starting_capital,
            daily_holdings=daily_holdings,
        )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(result: SimResult) -> dict:
    """Compute all reporting metrics from a simulation result."""
    hist = result.history
    if not hist:
        return {}

    starting = result.starting_capital
    final_value = hist[-1]["total_value"]
    total_return = (final_value - starting) / starting * 100

    # Daily returns for Sharpe/Sortino
    values = [h["total_value"] for h in hist]
    daily_returns = np.diff(values) / values[:-1] if len(values) > 1 else np.array([])

    risk_free_daily = 0.04 / 252  # ~4% annual
    excess = daily_returns - risk_free_daily

    sharpe = (
        float(np.mean(excess) / np.std(excess) * np.sqrt(252))
        if len(excess) > 1 and np.std(excess) > 0 else 0.0
    )

    downside = excess[excess < 0]
    sortino = (
        float(np.mean(excess) / np.std(downside) * np.sqrt(252))
        if len(downside) > 1 and np.std(downside) > 0 else 0.0
    )

    # Max drawdown
    peak = starting
    max_dd = 0.0
    for h in hist:
        if h["total_value"] > peak:
            peak = h["total_value"]
        dd = (h["total_value"] - peak) / peak
        if dd < max_dd:
            max_dd = dd

    trades = result.trades
    # Exclude end_of_backtest trades from win/loss stats
    real_trades = [t for t in trades if t.exit_reason != "end_of_backtest"]

    winners = [t for t in real_trades if t.pnl > 0]
    losers = [t for t in real_trades if t.pnl <= 0]
    win_rate = len(winners) / len(real_trades) * 100 if real_trades else 0
    avg_hold = np.mean([t.hold_days for t in real_trades]) if real_trades else 0

    total_gains = sum(t.pnl for t in winners)
    total_losses = abs(sum(t.pnl for t in losers))
    profit_factor = total_gains / total_losses if total_losses > 0 else float("inf")

    concurrent = [h["num_positions"] for h in hist]
    max_conc = max(concurrent) if concurrent else 0
    avg_conc = float(np.mean(concurrent)) if concurrent else 0.0

    best = max(real_trades, key=lambda t: t.pnl_pct) if real_trades else None
    worst = min(real_trades, key=lambda t: t.pnl_pct) if real_trades else None

    return {
        "total_return": total_return,
        "final_value": final_value,
        "max_drawdown": max_dd * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "total_trades": len(real_trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_hold": avg_hold,
        "max_concurrent": max_conc,
        "avg_concurrent": avg_conc,
        "signals_skipped": result.signals_skipped,
        "signals_generated": result.signals_generated,
        "signals_filtered": result.signals_filtered,
        "trim_count": result.trim_count,
        "total_trimmed": result.total_trimmed_dollars,
        "swap_count": result.swap_count,
        "swap_skipped_threshold": result.swap_skipped_threshold,
        "best_trade": best,
        "worst_trade": worst,
    }


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def print_comparison(results: list[tuple[StrategyConfig, SimResult, dict]]):
    """Output 1: Head-to-head comparison table."""
    _, r0, _ = results[0]

    print("\n" + "=" * 90)
    print("  SIZING STRATEGY COMPARISON")
    print(f"  Period: {r0.start_date} to {r0.end_date}")
    print(f"  Starting Capital: ${r0.starting_capital:,.0f}")
    print("=" * 90)

    # Column headers
    names = [cfg.name for cfg, _, _ in results]
    col_w = 16
    header = f"\n  {'Metric':<34s}" + "".join(f"| {n:>{col_w-2}s} " for n in names)
    print(header)
    print("  " + "-" * 34 + ("+" + "-" * col_w) * len(names))

    def row(label: str, values: list[str]):
        cells = "".join(f"| {v:>{col_w-2}s} " for v in values)
        print(f"  {label:<34s}{cells}")

    metrics_list = [r[2] for r in results]

    row("Total Return",
        [f"{m['total_return']:+.1f}%" for m in metrics_list])
    row("Final Equity",
        [f"${m['final_value']:,.0f}" for m in metrics_list])
    row("Max Drawdown",
        [f"{m['max_drawdown']:+.1f}%" for m in metrics_list])
    row("Sharpe Ratio",
        [f"{m['sharpe']:.2f}" for m in metrics_list])
    row("Sortino Ratio",
        [f"{m['sortino']:.2f}" for m in metrics_list])
    row("Total Trades (entries)",
        [f"{m['total_trades']}" for m in metrics_list])
    row("Win Rate",
        [f"{m['win_rate']:.1f}%" for m in metrics_list])
    row("Profit Factor",
        [f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else "inf" for m in metrics_list])
    row("Avg Position Hold (days)",
        [f"{m['avg_hold']:.0f}" for m in metrics_list])
    row("Max Concurrent Positions",
        [f"{m['max_concurrent']}" for m in metrics_list])
    row("Avg Concurrent Positions",
        [f"{m['avg_concurrent']:.1f}" for m in metrics_list])
    print("  " + "-" * 34 + ("+" + "-" * col_w) * len(names))
    row("Signals Skipped (no cash/full)",
        [f"{m['signals_skipped']}" for m in metrics_list])
    row("Swap Events",
        [f"{m['swap_count']}" for m in metrics_list])
    row("Trim Events",
        [f"{m['trim_count']}" for m in metrics_list])
    row("Total $ Trimmed",
        [f"${m['total_trimmed']:,.0f}" for m in metrics_list])
    print("  " + "-" * 34 + ("+" + "-" * col_w) * len(names))

    # Largest winner/loser
    def trade_str(t):
        if t is None:
            return "N/A"
        return f"{t.ticker} {t.pnl_pct*100:+.1f}%"

    row("Largest Winner",
        [trade_str(m["best_trade"]) for m in metrics_list])
    row("Largest Loser",
        [trade_str(m["worst_trade"]) for m in metrics_list])

    print()


def print_path_dependency(
    all_runs: list[list[tuple[str, dict]]],   # per-strategy: [(start_date, metrics), ...]
    names: list[str],
):
    """Output 2: Path dependency analysis."""
    print("\n" + "=" * 110)
    print("  PATH DEPENDENCY ANALYSIS")
    print("=" * 110)

    # Table header — strategy names
    col = 22
    name_hdr = f"\n  {'':13s}"
    for name in names:
        name_hdr += f"| {name:^{col-2}s} "
    print(name_hdr)

    col_hdr = f"  {'Start Date':13s}"
    for _ in names:
        col_hdr += f"| {'Return':>9s}  {'Max DD':>8s} "
    print(col_hdr)
    print("  " + "-" * 13 + ("+" + "-" * col) * len(names))

    # Per-row data
    num_starts = len(all_runs[0])
    for j in range(num_starts):
        start_date = all_runs[0][j][0]
        line = f"  {start_date:13s}"
        for strat_runs in all_runs:
            _, m = strat_runs[j]
            line += f"| {m['total_return']:>+9.1f}%  {m['max_drawdown']:>+8.1f}% "
        print(line)

    # Summary stats
    print()
    print(f"  {'Summary Stats':16s}", end="")
    for name in names:
        print(f"| {name:>{col-2}s} ", end="")
    print()
    print("  " + "-" * 16 + ("+" + "-" * col) * len(names))

    for label, fn in [
        ("Mean Return", lambda runs: np.mean([m["total_return"] for _, m in runs])),
        ("Std Dev Return", lambda runs: np.std([m["total_return"] for _, m in runs])),
        ("Min Return", lambda runs: np.min([m["total_return"] for _, m in runs])),
        ("Max Return", lambda runs: np.max([m["total_return"] for _, m in runs])),
        ("Range", lambda runs: np.ptp([m["total_return"] for _, m in runs])),
        ("Mean Max DD", lambda runs: np.mean([m["max_drawdown"] for _, m in runs])),
        ("Std Dev Max DD", lambda runs: np.std([m["max_drawdown"] for _, m in runs])),
    ]:
        line = f"  {label:16s}"
        for strat_runs in all_runs:
            val = fn(strat_runs)
            if "Std" in label or "Range" in label:
                line += f"| {val:>{col-3}.1f}% "
            else:
                line += f"| {val:>+{col-3}.1f}% "
        print(line)

    print()


def print_trim_analysis(result: SimResult, metrics: dict):
    """Output 3: Trim impact analysis for Dynamic/Trim strategy."""
    print("\n" + "=" * 80)
    print("  TRIM IMPACT ANALYSIS (Dynamic/Trim strategy)")
    print("=" * 80)

    trims = result.trims
    trades = result.trades

    if not trims:
        print("\n  No trim events occurred.")
        return

    total_trim = sum(te.dollar_amount for te in trims)
    avg_trim = total_trim / len(trims) if trims else 0

    print(f"\n  Total trim events: {len(trims)}")
    print(f"  Total $ trimmed: ${total_trim:,.0f}")
    print(f"  Avg trim per event: ${avg_trim:,.0f}")

    # Trimmed positions: what happened next?
    # Track tickers that were trimmed and their subsequent exit trades
    trimmed_tickers = {te.ticker_trimmed for te in trims}
    trimmed_exits = [t for t in trades if t.ticker in trimmed_tickers and t.exit_reason != "end_of_backtest"]

    # Post-trim P&L: average of positions that were trimmed (from trim point to exit)
    trim_pnl_at_trim = [te.pnl_at_trim * 100 for te in trims]
    avg_pnl_at_trim = np.mean(trim_pnl_at_trim) if trim_pnl_at_trim else 0

    score_exits = [t for t in trimmed_exits if t.exit_reason == "score_exit"]
    stop_exits = [t for t in trimmed_exits if t.exit_reason == "stop_loss"]

    print(f"\n  --- Trimmed Positions: What Happened Next ---")
    print(f"    Avg P&L % when trimmed: {avg_pnl_at_trim:+.1f}%")
    if trimmed_exits:
        print(f"    Positions that subsequently exited (score < 5): {len(score_exits)} / {len(trimmed_exits)} ({len(score_exits)/len(trimmed_exits)*100:.0f}%)")
        print(f"    Positions that hit stop loss after trim: {len(stop_exits)} / {len(trimmed_exits)} ({len(stop_exits)/len(trimmed_exits)*100:.0f}%)")

    # Trim-funded vs cash-funded entries
    trim_funded = [t for t in trades if t.trim_funded and t.exit_reason != "end_of_backtest"]
    cash_funded = [t for t in trades if not t.trim_funded and t.exit_reason != "end_of_backtest"]

    print(f"\n  --- New Entries Funded by Trims ---")
    print(f"    Total entries that required trimming to fund: {len(trim_funded)}")
    if trim_funded:
        avg_tf = np.mean([t.pnl_pct * 100 for t in trim_funded])
        print(f"    Avg return of trim-funded entries: {avg_tf:+.1f}%")
    if cash_funded:
        avg_cf = np.mean([t.pnl_pct * 100 for t in cash_funded])
        print(f"    Avg return of cash-funded entries: {avg_cf:+.1f}%")

    print()


def print_swap_analysis(result: SimResult, price_data: dict, daily_scores: dict):
    """Output: Swap impact analysis for Fixed/10+Swap strategy."""
    print("\n" + "=" * 80)
    print("  SWAP ANALYSIS (Fixed/10+Swap strategy)")
    print("=" * 80)

    swap_events = result.swaps
    trades = result.trades

    if not swap_events:
        print("\n  No swap events occurred.")
        return

    print(f"\n  Total swap events: {len(swap_events)}")
    print(f"  Signals skipped despite swap available (victim healthy): "
          f"{result.swap_skipped_threshold}")

    # ── Swapped-out positions ──
    out_scores = [s.swapped_out_score for s in swap_events]
    out_pnls = [s.swapped_out_pnl_pct for s in swap_events]
    out_days = [s.swapped_out_days_held for s in swap_events]

    print(f"\n  --- Swapped-Out Positions ---")
    print(f"    Avg score at swap: {np.mean(out_scores):.1f}")
    print(f"    Avg P&L % at swap: {np.mean(out_pnls):+.1f}%")
    print(f"    Avg days held at swap: {np.mean(out_days):.0f}")

    # "What would have happened" — track swapped-out tickers' subsequent prices
    # For each swap, check if the swapped-out ticker would have:
    # 1. Hit exit threshold (score < 5.0)
    # 2. Recovered above entry price
    would_exit = 0
    would_recover = 0
    eventual_returns = []

    for se in swap_events:
        ticker = se.swapped_out_ticker
        swap_date = se.date
        entry_price = se.swapped_out_entry_price

        # Find all score dates after the swap
        score_dates_after = sorted(
            d for d in daily_scores.keys() if d > swap_date
        )

        hit_exit = False
        best_price = 0.0
        last_price = None

        for d in score_dates_after:
            d_score = daily_scores.get(d, {}).get(ticker, 0)
            if d_score < 5.0 and d_score > 0:
                hit_exit = True
                break

            cp = get_price(price_data, ticker, d, "Close")
            if cp:
                last_price = cp
                best_price = max(best_price, cp)

        if hit_exit:
            would_exit += 1
        if best_price > entry_price:
            would_recover += 1
        if last_price and entry_price > 0:
            eventual_returns.append((last_price - entry_price) / entry_price * 100)

    total = len(swap_events)
    print(f"\n    What would have happened if NOT swapped:")
    print(f"      Would have hit exit threshold (< 5.0): {would_exit} / {total} ({would_exit/total*100:.0f}%)")
    print(f"      Would have recovered above entry: {would_recover} / {total} ({would_recover/total*100:.0f}%)")
    if eventual_returns:
        print(f"      Avg eventual return if held: {np.mean(eventual_returns):+.1f}%")

    # ── Swapped-in positions ──
    swap_in_tickers = {se.swapped_in_ticker for se in swap_events}
    swap_trades = [t for t in trades if t.ticker in swap_in_tickers
                   and t.exit_reason != "end_of_backtest"]
    normal_trades = [t for t in trades if t.ticker not in swap_in_tickers
                     and t.exit_reason not in ("end_of_backtest", "swap")]

    in_scores = [s.swapped_in_score for s in swap_events]

    print(f"\n  --- Swapped-In Positions ---")
    print(f"    Avg score at entry: {np.mean(in_scores):.1f}")
    if swap_trades:
        avg_swap_ret = np.mean([t.pnl_pct * 100 for t in swap_trades])
        print(f"    Avg return of swapped-in entries: {avg_swap_ret:+.1f}%")
    if normal_trades:
        avg_normal_ret = np.mean([t.pnl_pct * 100 for t in normal_trades])
        print(f"    Avg return of normal entries (no swap needed): {avg_normal_ret:+.1f}%")

    # ── Net swap value ──
    swap_in_pnl = sum(t.pnl for t in swap_trades) if swap_trades else 0
    # Foregone = what we would have made on swapped-out positions
    # Use eventual_returns as proxy (relative to their cost bases)
    swap_out_cost_bases = []
    for se in swap_events:
        # Find the matching exit trade for the swapped-out ticker
        for t in trades:
            if t.ticker == se.swapped_out_ticker and t.exit_reason == "swap" and t.exit_date == se.date:
                swap_out_cost_bases.append(t.cost_basis)
                break

    foregone = 0.0
    if eventual_returns and swap_out_cost_bases:
        for ret_pct, cb in zip(eventual_returns, swap_out_cost_bases):
            foregone += cb * ret_pct / 100

    net = swap_in_pnl - foregone

    print(f"\n  --- Net Swap Value ---")
    print(f"    Total P&L from swapped-in positions: ${swap_in_pnl:+,.0f}")
    print(f"    Estimated foregone return from swapped-out: ${foregone:+,.0f}")
    print(f"    Net swap value: ${net:+,.0f}")
    if net > 0:
        print(f"    → Positive: swaps are adding value")
    else:
        print(f"    → Negative: better to skip when full")

    print()


def print_overlap(results: list[tuple[StrategyConfig, SimResult, dict]]):
    """Output 4: Position overlap analysis."""
    print("\n" + "=" * 80)
    print("  POSITION OVERLAP ANALYSIS")
    print("=" * 80)

    names = [cfg.name for cfg, _, _ in results]
    holdings = [r.daily_holdings for _, r, _ in results]

    # Find common dates
    common_dates = sorted(set.intersection(*[set(h.keys()) for h in holdings]))

    if not common_dates:
        print("\n  No overlapping dates found.")
        return

    # Pairwise overlap
    print(f"\n  Avg daily overlap (shared tickers):")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            overlaps = []
            for d in common_dates:
                s1 = holdings[i].get(d, set())
                s2 = holdings[j].get(d, set())
                if s1 or s2:
                    shared = len(s1 & s2)
                    total = len(s1 | s2)
                    overlaps.append((shared, total))
            if overlaps:
                avg_shared = np.mean([o[0] for o in overlaps])
                avg_total = np.mean([o[1] for o in overlaps])
                pct = avg_shared / avg_total * 100 if avg_total > 0 else 0
                print(f"    {names[i]} vs {names[j]}: {avg_shared:.1f} positions ({pct:.0f}%)")

    # Unique tickers per strategy
    print(f"\n  Unique tickers ever held:")
    all_tickers = []
    for cfg, r, _ in results:
        ever_held = set()
        for s in r.daily_holdings.values():
            ever_held.update(s)
        all_tickers.append(ever_held)
        print(f"    {cfg.name}: {len(ever_held)} tickers")

    # Exclusive tickers
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            only_i = all_tickers[i] - all_tickers[j]
            only_j = all_tickers[j] - all_tickers[i]
            print(f"\n    Held by {names[i]} but never by {names[j]}: {len(only_i)}")
            print(f"    Held by {names[j]} but never by {names[i]}: {len(only_j)}")

    print()


# ============================================================
# STOP LOSS SWEEP ANALYSIS
# ============================================================

def compute_whipsaw_stats(
    trades: list[Trade],
    price_data: dict,
    daily_scores: dict[str, dict[str, float]],
    trading_days: list[pd.Timestamp],
) -> dict:
    """Compute whipsaw statistics for stop-loss exits.

    A whipsaw is when a stop-out position recovers above entry price
    within 30 trading days.
    """
    stop_outs = [t for t in trades if t.exit_reason == "stop_loss"]

    if not stop_outs:
        return {
            "stop_outs": 0,
            "whipsawed": 0,
            "whipsaw_rate": 0.0,
            "avg_missed_return": 0.0,
            "avg_stop_pnl": 0.0,
        }

    td_strs = [d.strftime("%Y-%m-%d") for d in trading_days]

    whipsawed = 0
    missed_returns = []

    for t in stop_outs:
        # Find the index of the exit date
        try:
            exit_idx = td_strs.index(t.exit_date)
        except ValueError:
            continue

        # Look ahead 30 trading days
        lookahead_end = min(exit_idx + 31, len(td_strs))
        recovered = False
        last_price = t.exit_price
        score_exited = False

        for j in range(exit_idx + 1, lookahead_end):
            d_str = td_strs[j]

            # Check if score would have triggered exit anyway
            score = daily_scores.get(d_str, {}).get(t.ticker, 0)
            if 0 < score < 5.0:
                score_exited = True
                cp = get_price(price_data, t.ticker, trading_days[j], "Close")
                if cp:
                    last_price = cp
                break

            cp = get_price(price_data, t.ticker, trading_days[j], "Close")
            if cp:
                last_price = cp
                if cp > t.entry_price:
                    recovered = True

        if recovered:
            whipsawed += 1

        # Missed return: from stop-out price to last_price (or score-exit point)
        if t.exit_price > 0:
            missed_ret = (last_price - t.exit_price) / t.exit_price * 100
            missed_returns.append(missed_ret)

    avg_stop_pnl = float(np.mean([t.pnl_pct * 100 for t in stop_outs]))

    return {
        "stop_outs": len(stop_outs),
        "whipsawed": whipsawed,
        "whipsaw_rate": whipsawed / len(stop_outs) * 100 if stop_outs else 0.0,
        "avg_missed_return": float(np.mean(missed_returns)) if missed_returns else 0.0,
        "avg_stop_pnl": avg_stop_pnl,
    }


def compute_exit_breakdown(trades: list[Trade]) -> dict:
    """Break down exits by type with avg P&L for each."""
    real = [t for t in trades if t.exit_reason != "end_of_backtest"]
    score_exits = [t for t in real if t.exit_reason == "score_exit"]
    stop_exits = [t for t in real if t.exit_reason == "stop_loss"]

    return {
        "total_exits": len(real),
        "score_exits": len(score_exits),
        "stop_exits": len(stop_exits),
        "score_exit_avg_pnl": (
            float(np.mean([t.pnl_pct * 100 for t in score_exits]))
            if score_exits else 0.0
        ),
        "stop_exit_avg_pnl": (
            float(np.mean([t.pnl_pct * 100 for t in stop_exits]))
            if stop_exits else 0.0
        ),
    }


def print_stop_loss_sweep(sweep_results: list[dict]):
    """Output: Stop loss strategy sweep summary table."""
    print("\n" + "=" * 145)
    print("  STOP LOSS STRATEGY SWEEP (Fixed/10-Max base)")
    print("=" * 145)

    header = (
        f"\n  {'Config':<14s} | {'Return':>9s} | {'Max DD':>8s} | "
        f"{'Sharpe':>6s} | {'Sortino':>7s} | {'Win Rate':>8s} | "
        f"{'Trades':>6s} | {'Stop Outs':>9s} | {'Avg Stop P&L':>12s} | "
        f"{'Path Std Dev':>12s}"
    )
    print(header)
    print("  " + "-" * 143)

    best_sharpe_idx = max(range(len(sweep_results)),
                          key=lambda i: sweep_results[i]["sharpe"])

    for i, r in enumerate(sweep_results):
        marker = "  ◀ BEST" if i == best_sharpe_idx else ""
        current = " *" if r.get("is_current") else ""
        label = r["label"] + current

        stop_pnl_str = f"{r['avg_stop_pnl']:+.1f}%" if r["stop_outs"] > 0 else "—"

        print(
            f"  {label:<14s} | {r['return']:>+8.1f}% | {r['max_dd']:>+7.1f}% | "
            f"{r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | {r['win_rate']:>7.1f}% | "
            f"{r['total_trades']:>6d} | {r['stop_outs']:>9d} | "
            f"{stop_pnl_str:>12s} | {r['path_std']:>11.1f}%{marker}"
        )

    print()
    best = sweep_results[best_sharpe_idx]
    print(f"  Peak Sharpe: {best['sharpe']:.2f} with {best['label']} "
          f"(return={best['return']:+.1f}%, DD={best['max_dd']:+.1f}%, "
          f"stop outs={best['stop_outs']}, path std={best['path_std']:.1f}%)")
    print()


def print_whipsaw_analysis(sweep_results: list[dict]):
    """Output: Whipsaw analysis for stop loss sweep."""
    print("\n" + "=" * 100)
    print("  WHIPSAW ANALYSIS")
    print("=" * 100)

    header = (
        f"\n  {'Config':<14s} | {'Stop Outs':>9s} | {'Whipsawed':>9s} | "
        f"{'Whipsaw Rate':>12s} | {'Avg Missed Return':>17s}"
    )
    print(header)
    print("  " + "-" * 98)

    for r in sweep_results:
        current = " *" if r.get("is_current") else ""
        label = r["label"] + current
        ws = r["whipsaw"]

        if r["stop_outs"] == 0:
            print(
                f"  {label:<14s} | {r['stop_outs']:>9d} | {'—':>9s} | "
                f"{'—':>12s} | {'—':>17s}"
            )
        else:
            print(
                f"  {label:<14s} | {r['stop_outs']:>9d} | {ws['whipsawed']:>9d} | "
                f"{ws['whipsaw_rate']:>11.0f}% | {ws['avg_missed_return']:>+16.1f}%"
            )

    print()
    print("  Whipsaw = price recovers above entry within 30 trading days of stop-out")
    print("  Avg Missed Return = avg return from stop-out price to 30-day-later price")
    print("                      (or to score-exit point, whichever comes first)")
    print()


def print_exit_type_breakdown(sweep_results: list[dict]):
    """Output: Exit type breakdown for stop loss sweep."""
    print("\n" + "=" * 110)
    print("  EXIT TYPE BREAKDOWN")
    print("=" * 110)

    header = (
        f"\n  {'Config':<14s} | {'Total Exits':>11s} | {'Score Exits':>11s} | "
        f"{'Stop Exits':>10s} | {'Score Exit Avg P&L':>18s} | "
        f"{'Stop Exit Avg P&L':>17s}"
    )
    print(header)
    print("  " + "-" * 108)

    for r in sweep_results:
        current = " *" if r.get("is_current") else ""
        label = r["label"] + current
        eb = r["exit_breakdown"]

        score_pnl_str = f"{eb['score_exit_avg_pnl']:+.1f}%" if eb["score_exits"] > 0 else "—"
        stop_pnl_str = f"{eb['stop_exit_avg_pnl']:+.1f}%" if eb["stop_exits"] > 0 else "—"

        print(
            f"  {label:<14s} | {eb['total_exits']:>11d} | {eb['score_exits']:>11d} | "
            f"{eb['stop_exits']:>10d} | {score_pnl_str:>18s} | "
            f"{stop_pnl_str:>17s}"
        )

    print()


# ============================================================
# PATH DEPENDENCY HELPERS
# ============================================================

def compute_path_start_dates(
    score_df: pd.DataFrame,
    daily_scores: dict[str, dict[str, float]],
    num_starts: int,
    persistence_days: int = 3,
) -> list[str]:
    """
    Find `num_starts` staggered start dates within the actual score data range.
    Spaced 1 week apart, starting after enough warm-up for the persistence filter.
    """
    first_score = score_df["date"].min()
    last_score = score_df["date"].max()
    total_score_days = (last_score - first_score).days

    warmup = timedelta(days=max(14, persistence_days * 3))
    latest_start = first_score + timedelta(days=int(total_score_days * 0.4))

    start_candidates: list[str] = []
    d = first_score + warmup
    while len(start_candidates) < num_starts:
        if d > latest_start:
            break
        d_str = d.strftime("%Y-%m-%d")
        if d_str in daily_scores:
            start_candidates.append(d_str)
            d += timedelta(days=7)
        else:
            d += timedelta(days=1)

    return start_candidates


# ============================================================
# POSITION CAP SWEEP
# ============================================================

def print_cap_sweep(sweep_results: list[dict]):
    """Output: Position cap sweep summary table."""
    print("\n" + "=" * 130)
    print("  POSITION CAP SWEEP (Fixed / No-Rotation)")
    print("=" * 130)

    header = (
        f"\n  {'Max Pos':>7s} | {'Size %':>7s} | {'Return':>9s} | {'Max DD':>8s} | "
        f"{'Sharpe':>6s} | {'Sortino':>7s} | {'Win Rate':>8s} | "
        f"{'Signals Skip':>12s} | {'Path Std Dev':>12s} | {'Path Range':>10s}"
    )
    print(header)
    print("  " + "-" * 7 + "-+-" + "-" * 7 + "-+-" + "-" * 9 + "-+-" + "-" * 8 +
          "-+-" + "-" * 6 + "-+-" + "-" * 7 + "-+-" + "-" * 8 +
          "-+-" + "-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 10)

    # Track the best Sharpe for highlighting
    best_sharpe_idx = max(range(len(sweep_results)), key=lambda i: sweep_results[i]["sharpe"])

    for i, r in enumerate(sweep_results):
        marker = "  ◀ BEST" if i == best_sharpe_idx else ""
        print(
            f"  {r['max_pos']:>7d} | {r['size_pct']:>6.1f}% | {r['return']:>+8.1f}% | "
            f"{r['max_dd']:>+7.1f}% | {r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | "
            f"{r['win_rate']:>7.1f}% | {r['signals_skipped']:>12,d} | "
            f"{r['path_std']:>11.1f}% | {r['path_range']:>9.1f}%{marker}"
        )

    print()

    # Recommendation
    best = sweep_results[best_sharpe_idx]
    print(f"  Peak Sharpe: {best['sharpe']:.2f} at max_positions={best['max_pos']} "
          f"(size={best['size_pct']:.1f}%, return={best['return']:+.1f}%, "
          f"DD={best['max_dd']:+.1f}%, path std={best['path_std']:.1f}%)")
    print()


def print_entry_threshold_sweep(sweep_results: list[dict]):
    """Output: Entry threshold sweep summary table."""
    print("\n" + "=" * 140)
    print("  ENTRY THRESHOLD SWEEP (Fixed/12-Max, 8.3% sizing, -20% stop)")
    print("=" * 140)

    header = (
        f"\n  {'Threshold':>9s} | {'Return':>9s} | {'Max DD':>8s} | "
        f"{'Sharpe':>6s} | {'Sortino':>7s} | {'Win Rate':>8s} | "
        f"{'Trades':>6s} | {'Signals Skip':>12s} | {'Path Std Dev':>12s} | "
        f"{'Path Range':>10s}"
    )
    print(header)
    print("  " + "-" * 138)

    best_sharpe_idx = max(range(len(sweep_results)),
                          key=lambda i: sweep_results[i]["sharpe"])

    for i, r in enumerate(sweep_results):
        marker = "  ◀ BEST" if i == best_sharpe_idx else ""
        current = " *" if r.get("is_current") else ""
        label = f"{r['threshold']:.1f}{current}"

        print(
            f"  {label:>9s} | {r['return']:>+8.1f}% | {r['max_dd']:>+7.1f}% | "
            f"{r['sharpe']:>6.2f} | {r['sortino']:>7.2f} | {r['win_rate']:>7.1f}% | "
            f"{r['total_trades']:>6d} | {r['signals_skipped']:>12,d} | "
            f"{r['path_std']:>11.1f}% | {r['path_range']:>9.1f}%{marker}"
        )

    print()
    best = sweep_results[best_sharpe_idx]
    print(f"  Peak Sharpe: {best['sharpe']:.2f} at threshold={best['threshold']:.1f} "
          f"(return={best['return']:+.1f}%, DD={best['max_dd']:+.1f}%, "
          f"trades={best['total_trades']}, path std={best['path_std']:.1f}%)")
    print()
    print("  * = current live configuration")
    print("  All runs use 3-day persistence filter, 12-pos cap, 8.3% sizing, -20% fixed stop")
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare portfolio sizing strategies (Fixed/5, Fixed/10, Dynamic/Trim, Fixed/10+Swap)"
    )
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--path-starts", type=int, default=10,
                        help="Number of staggered start dates for path dependency test (default: 10)")
    parser.add_argument("--no-path-test", action="store_true",
                        help="Skip path dependency analysis (faster)")
    parser.add_argument("--sweep-max-positions", type=str, default=None,
                        help="Comma-separated cap values for position cap sweep "
                             "(e.g. 5,8,10,12,15). Runs Fixed/No-Rotation at each cap.")
    parser.add_argument("--sweep-stop-loss", type=str, default=None,
                        help="Comma-separated stop loss configs for stop loss sweep "
                             "(e.g. none,fixed-10,fixed-15,trail-15,ATR-3x)")
    parser.add_argument("--sweep-entry-threshold", type=str, default=None,
                        help="Comma-separated entry thresholds for threshold sweep "
                             "(e.g. 7.5,8.0,8.5,9.0,9.5). Uses current live config as base.")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  SIZING COMPARISON BACKTEST")
    print("  Alpha Scanner — Built from scratch (no portfolio_backtest.py reuse)")
    print("=" * 80)

    # ── Load data ──
    print("\n  Loading score data from database...")
    score_df = load_score_data()
    print(f"  {len(score_df)} score records")
    print(f"  Date range: {score_df['date'].min().strftime('%Y-%m-%d')} to "
          f"{score_df['date'].max().strftime('%Y-%m-%d')}")

    print("\n  Fetching price data from yfinance (2y)...")
    cfg = load_config()
    price_data = fetch_all(cfg, period="2y", verbose=False)
    print(f"  {len(price_data)} tickers fetched")

    trading_days = get_trading_days(price_data)
    print(f"  {len(trading_days)} trading days")

    print("\n  Building daily score lookup with carry-forward...")
    daily_scores = build_daily_scores(score_df, trading_days)
    print(f"  {len(daily_scores)} days with scores")

    # ── Run primary comparison ──
    strategies = [STRATEGY_A, STRATEGY_B, STRATEGY_C, STRATEGY_D]
    primary_results = []

    print("\n  Running primary simulations...")
    for strat in strategies:
        print(f"    {strat.name}...", end="", flush=True)
        sim = PortfolioSimulator(
            config=strat,
            daily_scores=daily_scores,
            price_data=price_data,
            trading_days=trading_days,
            start_date=args.start,
            end_date=args.end,
        )
        result = sim.run()
        metrics = compute_metrics(result)
        primary_results.append((strat, result, metrics))
        print(f" {metrics['total_return']:+.1f}% return, {metrics['total_trades']} trades, "
              f"max {metrics['max_concurrent']} concurrent")

    # Output 1: Comparison table
    print_comparison(primary_results)

    # Output 3: Trim analysis (Strategy C)
    for strat, result, metrics in primary_results:
        if strat.trim_enabled:
            print_trim_analysis(result, metrics)

    # Output 3b: Swap analysis (Strategy D)
    for strat, result, metrics in primary_results:
        if strat.swap_at_cap:
            print_swap_analysis(result, price_data, daily_scores)

    # Output 4: Position overlap
    print_overlap(primary_results)

    # ── Path dependency test ──
    if not args.no_path_test:
        print("\n" + "=" * 80)
        print(f"  RUNNING PATH DEPENDENCY TEST ({args.path_starts} start dates)")
        print("=" * 80)

        start_candidates = compute_path_start_dates(
            score_df, daily_scores, args.path_starts,
            persistence_days=strategies[0].persistence_days,
        )

        if len(start_candidates) < 3:
            first_score = score_df["date"].min()
            last_score = score_df["date"].max()
            total_score_days = (last_score - first_score).days
            print(f"\n  [WARN] Only {len(start_candidates)} valid start dates found "
                  f"(score data spans {total_score_days} days). "
                  f"Path dependency results may be limited.")
            if not start_candidates:
                print("  Skipping path dependency test — insufficient data.")

        if start_candidates:
            print(f"  Start dates: {start_candidates[0]} to {start_candidates[-1]}")

            all_path_runs: list[list[tuple[str, dict]]] = [[] for _ in strategies]

            for start in start_candidates:
                for si, strat in enumerate(strategies):
                    sim = PortfolioSimulator(
                        config=strat,
                        daily_scores=daily_scores,
                        price_data=price_data,
                        trading_days=trading_days,
                        start_date=start,
                        end_date=args.end,
                    )
                    result = sim.run()
                    metrics = compute_metrics(result)
                    all_path_runs[si].append((start, metrics))
                print(f"    {start}: done", flush=True)

            names = [s.name for s in strategies]
            print_path_dependency(all_path_runs, names)

    # ── Position cap sweep ──
    if args.sweep_max_positions:
        cap_values = [int(v.strip()) for v in args.sweep_max_positions.split(",")]
        cap_values.sort()

        print("\n" + "=" * 130)
        print(f"  RUNNING POSITION CAP SWEEP: {cap_values}")
        print("=" * 130)

        # Compute path dependency start dates (reuse same logic)
        sweep_start_candidates = compute_path_start_dates(
            score_df, daily_scores, args.path_starts,
            persistence_days=STRATEGY_B.persistence_days,
        )

        sweep_results: list[dict] = []

        for cap in cap_values:
            size_pct = 1.0 / cap
            strat = StrategyConfig(
                name=f"Fixed/{cap}-Max",
                max_positions=cap,
                sizing_mode="fixed_pct",
                fixed_position_pct=size_pct,
                min_entry_pct=0.05,
                trim_enabled=False,
                entry_protection_days=7,
                entry_threshold=8.5,
                exit_threshold=5.0,
                stop_loss=STOP_FIXED_15,
                persistence_days=PERSISTENCE_DAYS,
            )

            # Primary run (default start)
            print(f"    cap={cap:>2d} (size={size_pct*100:.1f}%)...", end="", flush=True)
            sim = PortfolioSimulator(
                config=strat,
                daily_scores=daily_scores,
                price_data=price_data,
                trading_days=trading_days,
                start_date=args.start,
                end_date=args.end,
            )
            result = sim.run()
            metrics = compute_metrics(result)
            print(f" {metrics['total_return']:+.1f}%", end="", flush=True)

            # Path dependency runs
            path_returns: list[float] = []
            if sweep_start_candidates:
                for start in sweep_start_candidates:
                    psim = PortfolioSimulator(
                        config=strat,
                        daily_scores=daily_scores,
                        price_data=price_data,
                        trading_days=trading_days,
                        start_date=start,
                        end_date=args.end,
                    )
                    presult = psim.run()
                    pmetrics = compute_metrics(presult)
                    path_returns.append(pmetrics["total_return"])

            path_std = float(np.std(path_returns)) if len(path_returns) > 1 else 0.0
            path_range = float(np.ptp(path_returns)) if path_returns else 0.0

            sweep_results.append({
                "max_pos": cap,
                "size_pct": size_pct * 100,
                "return": metrics["total_return"],
                "max_dd": metrics["max_drawdown"],
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "win_rate": metrics["win_rate"],
                "signals_skipped": metrics["signals_skipped"],
                "path_std": path_std,
                "path_range": path_range,
                "total_trades": metrics["total_trades"],
            })

            print(f", path std={path_std:.1f}%, done", flush=True)

        print_cap_sweep(sweep_results)

    # ── Stop loss sweep ──
    if args.sweep_stop_loss:
        stop_configs_str = [s.strip() for s in args.sweep_stop_loss.split(",")]

        print("\n" + "=" * 145)
        print(f"  RUNNING STOP LOSS SWEEP: {stop_configs_str}")
        print("=" * 145)

        # Compute path dependency start dates
        sl_start_candidates = compute_path_start_dates(
            score_df, daily_scores, args.path_starts,
            persistence_days=STRATEGY_B.persistence_days,
        )

        sl_sweep_results: list[dict] = []

        for config_str in stop_configs_str:
            sl_cfg = parse_stop_loss_config(config_str)
            label = stop_loss_label(sl_cfg)
            is_current = (sl_cfg.type == "fixed" and abs(sl_cfg.value - 0.15) < 0.001)

            strat = StrategyConfig(
                name=f"Fixed/10-{label}",
                max_positions=10,
                sizing_mode="fixed_pct",
                fixed_position_pct=0.10,
                min_entry_pct=0.05,
                trim_enabled=False,
                entry_protection_days=7,
                entry_threshold=8.5,
                exit_threshold=5.0,
                stop_loss=sl_cfg,
                persistence_days=PERSISTENCE_DAYS,
            )

            print(f"    {label:<14s}...", end="", flush=True)

            # Primary run
            sim = PortfolioSimulator(
                config=strat,
                daily_scores=daily_scores,
                price_data=price_data,
                trading_days=trading_days,
                start_date=args.start,
                end_date=args.end,
            )
            result = sim.run()
            metrics = compute_metrics(result)
            print(f" {metrics['total_return']:+.1f}%", end="", flush=True)

            # Whipsaw analysis
            ws_stats = compute_whipsaw_stats(
                result.trades, price_data, daily_scores, trading_days,
            )

            # Exit type breakdown
            eb_stats = compute_exit_breakdown(result.trades)

            # Path dependency runs
            path_returns: list[float] = []
            if sl_start_candidates:
                for start in sl_start_candidates:
                    psim = PortfolioSimulator(
                        config=strat,
                        daily_scores=daily_scores,
                        price_data=price_data,
                        trading_days=trading_days,
                        start_date=start,
                        end_date=args.end,
                    )
                    presult = psim.run()
                    pmetrics = compute_metrics(presult)
                    path_returns.append(pmetrics["total_return"])

            path_std = float(np.std(path_returns)) if len(path_returns) > 1 else 0.0

            sl_sweep_results.append({
                "label": label,
                "is_current": is_current,
                "return": metrics["total_return"],
                "max_dd": metrics["max_drawdown"],
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "win_rate": metrics["win_rate"],
                "total_trades": metrics["total_trades"],
                "stop_outs": ws_stats["stop_outs"],
                "avg_stop_pnl": ws_stats["avg_stop_pnl"],
                "path_std": path_std,
                "whipsaw": ws_stats,
                "exit_breakdown": eb_stats,
            })

            print(f", stop outs={ws_stats['stop_outs']}, "
                  f"whipsawed={ws_stats['whipsawed']}, done", flush=True)

        # Print all three output tables
        print_stop_loss_sweep(sl_sweep_results)
        print_whipsaw_analysis(sl_sweep_results)
        print_exit_type_breakdown(sl_sweep_results)

    # ── Entry threshold sweep ──
    if args.sweep_entry_threshold:
        thresholds = [float(v.strip()) for v in args.sweep_entry_threshold.split(",")]
        thresholds.sort()

        print("\n" + "=" * 140)
        print(f"  RUNNING ENTRY THRESHOLD SWEEP: {thresholds}")
        print("=" * 140)

        # Compute path dependency start dates
        et_start_candidates = compute_path_start_dates(
            score_df, daily_scores, args.path_starts,
            persistence_days=PERSISTENCE_DAYS,
        )

        et_sweep_results: list[dict] = []

        for threshold in thresholds:
            is_current = abs(threshold - 8.5) < 0.01

            strat = StrategyConfig(
                name=f"Entry-{threshold:.1f}",
                max_positions=12,
                sizing_mode="fixed_pct",
                fixed_position_pct=0.083,
                min_entry_pct=0.05,
                trim_enabled=False,
                entry_protection_days=7,
                entry_threshold=threshold,
                exit_threshold=5.0,
                stop_loss=StopLossConfig(type="fixed", value=0.20),
                persistence_days=PERSISTENCE_DAYS,
            )

            print(f"    threshold={threshold:.1f}...", end="", flush=True)

            # Primary run
            sim = PortfolioSimulator(
                config=strat,
                daily_scores=daily_scores,
                price_data=price_data,
                trading_days=trading_days,
                start_date=args.start,
                end_date=args.end,
            )
            result = sim.run()
            metrics = compute_metrics(result)
            print(f" {metrics['total_return']:+.1f}%", end="", flush=True)

            # Path dependency runs
            path_returns: list[float] = []
            if not args.no_path_test and et_start_candidates:
                for start in et_start_candidates:
                    psim = PortfolioSimulator(
                        config=strat,
                        daily_scores=daily_scores,
                        price_data=price_data,
                        trading_days=trading_days,
                        start_date=start,
                        end_date=args.end,
                    )
                    presult = psim.run()
                    pmetrics = compute_metrics(presult)
                    path_returns.append(pmetrics["total_return"])

            path_std = float(np.std(path_returns)) if len(path_returns) > 1 else 0.0
            path_range = float(np.ptp(path_returns)) if path_returns else 0.0

            et_sweep_results.append({
                "threshold": threshold,
                "is_current": is_current,
                "return": metrics["total_return"],
                "max_dd": metrics["max_drawdown"],
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "win_rate": metrics["win_rate"],
                "total_trades": metrics["total_trades"],
                "signals_skipped": metrics["signals_skipped"],
                "path_std": path_std,
                "path_range": path_range,
            })

            print(f", trades={metrics['total_trades']}, "
                  f"path std={path_std:.1f}%, done", flush=True)

        print_entry_threshold_sweep(et_sweep_results)

    # ── Done ──
    print("\n" + "=" * 80)
    print("  BACKTEST COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
