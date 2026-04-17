"""
entry_mode_backtest.py — Compare entry-order-type variants on the full
portfolio simulation.

Configurations (all share: 8.5 entry / 3d persist / <5 exit / 12 cap /
8.3% dynamic sizing / -20% fixed stop / skip-when-full):

  baseline    — market order, no cash floor (current live behavior)
  defensive   — market order + 5% cash floor
  limit-2     — 2% limit order + 5% cash floor + deferred re-qualification
  limit-3     — 3% limit order + 5% cash floor + deferred re-qualification

Realism model (matches trade_executor.py at 5:30 PM ET):
  • Sizing reference = prior-day Close (what the live code uses:
    yfinance close is latest cash-session bar available at 5:30 PM).
    In backtest we cannot simulate Alpaca's extended-hours quote from
    daily OHLC, so baseline vs defensive differ by cash-floor only.
  • Fill determined at next-day Open:
      - market: always fills at next-day Open
      - limit:  fills iff next-day Open ≤ sizing × (1 + limit_pct)
  • Whole-share qty = int(target_size / sizing_price)
  • Actual cash deducted = qty × fill_price (can differ from target_size
    → this is what causes the real-world negative cash bug on gap days)

Deferred re-qualification: a ticker that fails its limit on day N is
retried every subsequent day while it still scores ≥ entry_threshold
(persistence auto-holds once a signal is established). If the score
drops below threshold before a fill lands, the signal is invalidated.

Usage:
    python entry_mode_backtest.py
    python entry_mode_backtest.py --no-path-test
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from sizing_comparison_backtest import (
    build_daily_scores,
    compute_path_start_dates,
    get_price,
    get_trading_days,
    load_score_data,
)
from config import load_config
from data_fetcher import fetch_all


# ============================================================
# CONFIG
# ============================================================

@dataclass
class EntryModeConfig:
    name: str
    use_limit_orders: bool = False
    limit_pct: float = 0.02           # only used if use_limit_orders
    cash_floor_pct: float = 0.0       # 0.05 = reserve 5% of equity uncommitted
    # shared
    max_positions: int = 12
    fixed_position_pct: float = 0.083
    entry_threshold: float = 8.5
    exit_threshold: float = 5.0
    persistence_days: int = 3
    stop_loss_pct: float = 0.20
    min_position_size: float = 500.0


CONFIGS = [
    EntryModeConfig(
        name="baseline",
        use_limit_orders=False,
        cash_floor_pct=0.0,
    ),
    EntryModeConfig(
        name="defensive",
        use_limit_orders=False,
        cash_floor_pct=0.05,
    ),
    EntryModeConfig(
        name="limit-2%",
        use_limit_orders=True,
        limit_pct=0.02,
        cash_floor_pct=0.05,
    ),
    EntryModeConfig(
        name="limit-3%",
        use_limit_orders=True,
        limit_pct=0.03,
        cash_floor_pct=0.05,
    ),
]


# ============================================================
# SIMULATION STATE
# ============================================================

@dataclass
class Position:
    ticker: str
    entry_date: str
    sizing_price: float    # today's close (what we sized off of)
    entry_price: float     # next-day open (actual fill)
    shares: int
    entry_score: float
    cost_basis: float = 0.0
    slippage_pct: float = 0.0

    def __post_init__(self):
        self.cost_basis = self.entry_price * self.shares
        if self.sizing_price > 0:
            self.slippage_pct = (self.entry_price / self.sizing_price - 1.0) * 100

    def value(self, price: float) -> float:
        return self.shares * price


@dataclass
class Trade:
    ticker: str
    entry_date: str
    entry_price: float
    entry_score: float
    sizing_price: float
    slippage_pct: float
    cost_basis: float
    exit_date: str
    exit_price: float
    exit_reason: str
    pnl: float           # dollars
    pnl_pct: float       # %
    hold_days: int


@dataclass
class SimResult:
    config_name: str
    trades: list[Trade]
    history: list[dict]         # daily snapshots
    filled_entries: int
    deferred_fills: int          # entries that required ≥1 retry
    invalidated_signals: int     # limit missed and score dropped
    signals_skipped_nocash: int  # ran out of cash / floor
    signals_skipped_cap: int     # position cap reached
    starting_capital: float


# ============================================================
# SIMULATOR
# ============================================================

class Simulator:
    def __init__(
        self,
        cfg: EntryModeConfig,
        daily_scores: dict,
        price_data: dict,
        trading_days: list[pd.Timestamp],
        starting_capital: float = 100_000.0,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        self.cfg = cfg
        self.daily_scores = daily_scores
        self.price_data = price_data
        self.starting_capital = starting_capital

        score_dates = sorted(daily_scores.keys())
        sim_start = start_date or score_dates[0]
        sim_end = end_date or score_dates[-1]
        self.trading_days = [
            d for d in trading_days
            if sim_start <= d.strftime("%Y-%m-%d") <= sim_end
        ]

    def run(self) -> SimResult:
        cfg = self.cfg
        days = self.trading_days
        cash = self.starting_capital
        positions: dict[str, Position] = {}
        trades: list[Trade] = []
        history: list[dict] = []

        # Deferred re-qualification tracking
        # ticker -> first_miss_date (for reporting deferred fills)
        pending_limit: dict[str, str] = {}

        filled_entries = 0
        deferred_fills = 0
        invalidated_signals = 0
        signals_skipped_nocash = 0
        signals_skipped_cap = 0

        for i, day in enumerate(days):
            day_str = day.strftime("%Y-%m-%d")
            scores_today = self.daily_scores.get(day_str, {})

            # ── 0. Invalidate pending limit signals whose score dropped ──
            for ticker in list(pending_limit):
                if scores_today.get(ticker, 0) < cfg.entry_threshold:
                    invalidated_signals += 1
                    del pending_limit[ticker]

            # ── 1. Evaluate exits (on today's close & low) ──
            to_exit: list[tuple[str, str]] = []
            for ticker, pos in positions.items():
                close = get_price(self.price_data, ticker, day, "Close")
                if close is None:
                    continue
                score = scores_today.get(ticker, 0)
                if score < cfg.exit_threshold:
                    to_exit.append((ticker, "score_exit"))
                    continue
                low = get_price(self.price_data, ticker, day, "Low") or close
                stop_price = pos.entry_price * (1 - cfg.stop_loss_pct)
                if low <= stop_price:
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
                    sell_price = pos.entry_price
                    sell_date = day_str
                pnl = (sell_price - pos.entry_price) * pos.shares
                pnl_pct = (sell_price / pos.entry_price - 1) * 100
                hd = (pd.Timestamp(sell_date) - pd.Timestamp(pos.entry_date)).days
                trades.append(Trade(
                    ticker=ticker,
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    entry_score=pos.entry_score,
                    sizing_price=pos.sizing_price,
                    slippage_pct=pos.slippage_pct,
                    cost_basis=pos.cost_basis,
                    exit_date=sell_date,
                    exit_price=sell_price,
                    exit_reason=reason,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    hold_days=hd,
                ))
                cash += sell_price * pos.shares
                del positions[ticker]

            # ── 2. Build today's entry candidates ──
            # A candidate = score ≥ threshold + persistence OK + not held
            candidates = []
            for ticker, score in scores_today.items():
                if score < cfg.entry_threshold:
                    continue
                if ticker in positions:
                    continue
                # Persistence
                if i < cfg.persistence_days:
                    continue
                ok = True
                for back in range(1, cfg.persistence_days + 1):
                    prior_str = days[i - back].strftime("%Y-%m-%d")
                    if self.daily_scores.get(prior_str, {}).get(ticker, 0) < cfg.entry_threshold:
                        ok = False
                        break
                if not ok:
                    continue
                candidates.append((ticker, score))

            candidates.sort(key=lambda x: -x[1])

            # ── 3. Process entries ──
            for ticker, score in candidates:
                if i + 1 >= len(days):
                    break  # no day to fill on
                next_day = days[i + 1]

                sizing_price = get_price(self.price_data, ticker, day, "Close")
                next_open = get_price(self.price_data, ticker, next_day, "Open")
                if not sizing_price or not next_open or sizing_price <= 0 or next_open <= 0:
                    continue

                # Current equity for sizing
                equity = cash
                for t, p in positions.items():
                    cp = get_price(self.price_data, t, day, "Close")
                    equity += p.value(cp) if cp else p.cost_basis

                # Position cap
                if len(positions) >= cfg.max_positions:
                    signals_skipped_cap += 1
                    # If on pending list, don't invalidate (still qualifies, just full)
                    continue

                # Cash floor
                cash_floor = equity * cfg.cash_floor_pct
                available_cash = max(0.0, cash - cash_floor)

                max_per_pos = equity * cfg.fixed_position_pct
                target_size = min(max_per_pos, available_cash)

                if target_size < cfg.min_position_size:
                    signals_skipped_nocash += 1
                    continue

                qty = int(target_size // sizing_price)
                if qty <= 0:
                    signals_skipped_nocash += 1
                    continue

                # ── Limit check (if applicable) ──
                if cfg.use_limit_orders:
                    limit_price = sizing_price * (1 + cfg.limit_pct)
                    if next_open > limit_price:
                        # Missed — register for deferred retry if not already
                        if ticker not in pending_limit:
                            pending_limit[ticker] = day_str
                        continue

                # ── FILL ──
                fill_price = next_open
                actual_cost = qty * fill_price
                cash -= actual_cost

                if ticker in pending_limit:
                    deferred_fills += 1
                    del pending_limit[ticker]
                filled_entries += 1

                positions[ticker] = Position(
                    ticker=ticker,
                    entry_date=next_day.strftime("%Y-%m-%d"),
                    sizing_price=sizing_price,
                    entry_price=fill_price,
                    shares=qty,
                    entry_score=score,
                )

            # ── 4. Daily snapshot ──
            pos_value = 0.0
            for t, p in positions.items():
                cp = get_price(self.price_data, t, day, "Close")
                pos_value += p.value(cp) if cp else p.cost_basis

            total_value = cash + pos_value
            invested_pct = (pos_value / total_value * 100) if total_value > 0 else 0.0
            history.append({
                "date": day_str,
                "total_value": total_value,
                "cash": cash,
                "pos_value": pos_value,
                "num_positions": len(positions),
                "invested_pct": invested_pct,
            })

        # ── Close any remaining positions at final close ──
        if days:
            final = days[-1]
            final_str = final.strftime("%Y-%m-%d")
            for ticker, pos in list(positions.items()):
                sp = get_price(self.price_data, ticker, final, "Close") or pos.entry_price
                pnl = (sp - pos.entry_price) * pos.shares
                pnl_pct = (sp / pos.entry_price - 1) * 100
                hd = (pd.Timestamp(final_str) - pd.Timestamp(pos.entry_date)).days
                trades.append(Trade(
                    ticker=ticker, entry_date=pos.entry_date,
                    entry_price=pos.entry_price, entry_score=pos.entry_score,
                    sizing_price=pos.sizing_price, slippage_pct=pos.slippage_pct,
                    cost_basis=pos.cost_basis,
                    exit_date=final_str, exit_price=sp, exit_reason="end_of_backtest",
                    pnl=pnl, pnl_pct=pnl_pct, hold_days=hd,
                ))

        # Count remaining pending as invalidated (never resolved)
        invalidated_signals += len(pending_limit)

        return SimResult(
            config_name=cfg.name,
            trades=trades,
            history=history,
            filled_entries=filled_entries,
            deferred_fills=deferred_fills,
            invalidated_signals=invalidated_signals,
            signals_skipped_nocash=signals_skipped_nocash,
            signals_skipped_cap=signals_skipped_cap,
            starting_capital=self.starting_capital,
        )


# ============================================================
# METRICS
# ============================================================

def compute_metrics(res: SimResult) -> dict:
    hist = res.history
    if not hist:
        return {}
    starting = res.starting_capital
    final = hist[-1]["total_value"]
    total_return = (final - starting) / starting * 100

    # Approximate CAGR
    n_days = len(hist)
    years = max(n_days / 252.0, 1e-9)
    cagr = ((final / starting) ** (1.0 / years) - 1.0) * 100 if starting > 0 else 0.0

    values = [h["total_value"] for h in hist]
    daily = np.diff(values) / values[:-1] if len(values) > 1 else np.array([])
    rf_daily = 0.04 / 252
    excess = daily - rf_daily

    sharpe = (float(np.mean(excess) / np.std(excess) * np.sqrt(252))
              if len(excess) > 1 and np.std(excess) > 0 else 0.0)
    downside = excess[excess < 0]
    sortino = (float(np.mean(excess) / np.std(downside) * np.sqrt(252))
               if len(downside) > 1 and np.std(downside) > 0 else 0.0)

    peak = starting
    max_dd = 0.0
    for h in hist:
        if h["total_value"] > peak:
            peak = h["total_value"]
        dd = (h["total_value"] - peak) / peak
        if dd < max_dd:
            max_dd = dd

    real = [t for t in res.trades if t.exit_reason != "end_of_backtest"]
    winners = [t for t in real if t.pnl > 0]
    win_rate = len(winners) / len(real) * 100 if real else 0.0

    neg_cash_days = sum(1 for h in hist if h["cash"] < 0)
    avg_invested = float(np.mean([h["invested_pct"] for h in hist])) if hist else 0.0

    max_slip = max((t.slippage_pct for t in res.trades), default=0.0)
    min_slip = min((t.slippage_pct for t in res.trades), default=0.0)

    return {
        "total_return": total_return,
        "cagr": cagr,
        "final_value": final,
        "max_drawdown": max_dd * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "total_trades": len(real),
        "win_rate": win_rate,
        "filled_entries": res.filled_entries,
        "deferred_fills": res.deferred_fills,
        "invalidated_signals": res.invalidated_signals,
        "signals_skipped_nocash": res.signals_skipped_nocash,
        "signals_skipped_cap": res.signals_skipped_cap,
        "neg_cash_days": neg_cash_days,
        "avg_invested_pct": avg_invested,
        "max_slippage_pct": max_slip,
        "min_slippage_pct": min_slip,
    }


# ============================================================
# REPORTING
# ============================================================

def print_primary_comparison(results: list[tuple[EntryModeConfig, SimResult, dict]]):
    print("\n" + "=" * 110)
    print("  PRIMARY COMPARISON  (starting capital $100,000)")
    print("=" * 110)
    header = f"  {'Metric':<36s}" + "".join(f" | {c.name:>13s}" for c, _, _ in results)
    print(header)
    print("  " + "-" * (36 + 16 * len(results)))

    def row(label, formatter):
        parts = []
        for _, _, m in results:
            parts.append(f" | {formatter(m):>13s}")
        print(f"  {label:<36s}" + "".join(parts))

    row("Total Return",        lambda m: f"{m['total_return']:+.1f}%")
    row("CAGR",                lambda m: f"{m['cagr']:+.1f}%")
    row("Final Value",         lambda m: f"${m['final_value']:,.0f}")
    row("Max Drawdown",        lambda m: f"{m['max_drawdown']:+.1f}%")
    row("Sharpe",              lambda m: f"{m['sharpe']:.2f}")
    row("Sortino",             lambda m: f"{m['sortino']:.2f}")
    row("Win Rate",            lambda m: f"{m['win_rate']:.1f}%")
    row("Total Trades",        lambda m: f"{m['total_trades']}")
    print("  " + "-" * (36 + 16 * len(results)))
    row("Filled Entries",      lambda m: f"{m['filled_entries']}")
    row("Deferred Fills",      lambda m: f"{m['deferred_fills']}")
    row("Invalidated Signals", lambda m: f"{m['invalidated_signals']}")
    row("Skip: Position Cap",  lambda m: f"{m['signals_skipped_cap']}")
    row("Skip: Cash/Floor",    lambda m: f"{m['signals_skipped_nocash']}")
    print("  " + "-" * (36 + 16 * len(results)))
    row("Avg Invested %",      lambda m: f"{m['avg_invested_pct']:.1f}%")
    row("Neg-Cash Days",       lambda m: f"{m['neg_cash_days']}")
    row("Max Entry Slippage",  lambda m: f"{m['max_slippage_pct']:+.2f}%")
    row("Min Entry Slippage",  lambda m: f"{m['min_slippage_pct']:+.2f}%")


def print_path_dependency(path_runs: dict[str, list[dict]]):
    print("\n" + "=" * 110)
    print("  PATH DEPENDENCY  (staggered start dates)")
    print("=" * 110)
    names = list(path_runs.keys())
    print(f"\n  {'Config':<14s}  {'Mean Rtn':>10s}  {'Std Dev':>9s}  {'Min':>9s}  {'Max':>9s}  "
          f"{'Mean DD':>8s}  {'Mean Sharpe':>12s}")
    print("  " + "-" * 80)
    for name in names:
        runs = path_runs[name]
        rets = [r["total_return"] for r in runs]
        dds = [r["max_drawdown"] for r in runs]
        shp = [r["sharpe"] for r in runs]
        print(f"  {name:<14s}  {np.mean(rets):>+9.1f}%  {np.std(rets):>8.1f}%  "
              f"{min(rets):>+8.1f}%  {max(rets):>+8.1f}%  {np.mean(dds):>+7.1f}%  "
              f"{np.mean(shp):>12.2f}")


def print_tail_analysis(results: list[tuple[EntryModeConfig, SimResult, dict]]):
    print("\n" + "=" * 110)
    print("  TAIL ANALYSIS — Top/Bottom 10 trades by $ P&L")
    print("=" * 110)

    for cfg, res, _ in results:
        trades = [t for t in res.trades if t.exit_reason != "end_of_backtest"]
        if not trades:
            print(f"\n  {cfg.name}: no completed trades")
            continue
        top10 = sorted(trades, key=lambda t: -t.pnl)[:10]
        bot10 = sorted(trades, key=lambda t: t.pnl)[:10]

        print(f"\n  {cfg.name.upper()}  (n={len(trades)})")
        print(f"    {'Top 10 winners':<16s}                          {'Bottom 10 losers':<16s}")
        print(f"    {'Ticker':<8s}{'$P&L':>10s}{'%':>8s}{'Slip':>7s}    "
              f"{'Ticker':<8s}{'$P&L':>10s}{'%':>8s}{'Slip':>7s}")
        for w, l in zip(top10, bot10):
            print(f"    {w.ticker:<8s}${w.pnl:>+8,.0f}{w.pnl_pct:>+7.1f}%{w.slippage_pct:>+6.2f}%    "
                  f"{l.ticker:<8s}${l.pnl:>+8,.0f}{l.pnl_pct:>+7.1f}%{l.slippage_pct:>+6.2f}%")

        top_sum = sum(t.pnl for t in top10)
        bot_sum = sum(t.pnl for t in bot10)
        print(f"    Top-10 sum:    ${top_sum:+,.0f}           "
              f"Bottom-10 sum: ${bot_sum:+,.0f}")


# ============================================================
# MAIN
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--no-path-test", action="store_true")
    p.add_argument("--path-starts", type=int, default=10)
    args = p.parse_args()

    print("=" * 80)
    print("  ENTRY MODE BACKTEST — Market vs Limit Orders")
    print("=" * 80)
    print("\n  Loading score data...")
    score_df = load_score_data()
    print(f"  {len(score_df)} score records")

    print("\n  Fetching price data (1y)...")
    cfg = load_config()
    price_data = fetch_all(cfg, period="1y", verbose=False)
    print(f"  {len(price_data)} tickers fetched")

    trading_days = get_trading_days(price_data)
    print(f"  {len(trading_days)} trading days")

    print("\n  Building daily score lookup...")
    daily_scores = build_daily_scores(score_df, trading_days)

    # ── Primary run ──
    print("\n  Running primary simulations...")
    results: list[tuple[EntryModeConfig, SimResult, dict]] = []
    for emc in CONFIGS:
        print(f"    {emc.name}...", end="", flush=True)
        sim = Simulator(
            cfg=emc, daily_scores=daily_scores, price_data=price_data,
            trading_days=trading_days, start_date=args.start, end_date=args.end,
        )
        res = sim.run()
        metrics = compute_metrics(res)
        results.append((emc, res, metrics))
        print(f" {metrics['total_return']:+.1f}%, "
              f"{metrics['filled_entries']} fills, "
              f"{metrics['neg_cash_days']} neg-cash days", flush=True)

    print_primary_comparison(results)

    # ── Path dependency ──
    if not args.no_path_test:
        print("\n  Running path dependency test...")
        path_starts = compute_path_start_dates(
            score_df, daily_scores, args.path_starts,
            persistence_days=CONFIGS[0].persistence_days,
        )
        if path_starts:
            print(f"    Start dates: {path_starts[0]} → {path_starts[-1]}")
            path_runs: dict[str, list[dict]] = {c.name: [] for c in CONFIGS}
            for start in path_starts:
                for emc in CONFIGS:
                    sim = Simulator(
                        cfg=emc, daily_scores=daily_scores, price_data=price_data,
                        trading_days=trading_days, start_date=start, end_date=args.end,
                    )
                    res = sim.run()
                    path_runs[emc.name].append(compute_metrics(res))
                print(f"    {start}: done", flush=True)
            print_path_dependency(path_runs)

    # ── Tail analysis ──
    print_tail_analysis(results)

    # ── Sanity check note ──
    print("\n" + "=" * 110)
    print("  SANITY CHECK")
    print("=" * 110)
    print("""
  • Executor timing: trade_executor.py runs at 5:30 PM ET (after close).
    Sizing reference = today's Close. Fill reference = next-day Open.
    This backtest uses the same convention: on day D, candidates are
    scored/sized using D's Close, then filled (or limit-checked) against
    D+1's Open. Verified in Simulator.run() at the entry loop.

  • "Alpaca-first pricing" (live-code fix to prefer Alpaca's latest-trade
    over yfinance close) cannot be meaningfully backtested: both sources
    resolve to the same daily Close in historical OHLC data. So baseline
    and defensive differ by cash-floor only in this backtest. In live
    trading, Alpaca-first pricing would add additional protection by
    catching after-hours gap-ups that yfinance close misses (e.g. AEHR
    2026-04-16: yfinance close $73.22 → Alpaca after-hours likely ~$80+).

  • Whole-share qty math: qty = int(target_size / sizing_price). Actual
    cash deduction = qty × fill_price. When fill_price > sizing_price
    (gap-up), actual cost exceeds target → cash can go negative. This
    is the AEHR scenario, faithfully modeled in baseline.
""")

    print("=" * 80)
    print("  DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
