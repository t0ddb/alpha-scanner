"""
threshold_optimizer.py — Grid search over entry/exit thresholds.

Reuses the existing simulation engine from portfolio_backtest.py and
runs a 3x3 grid:
    Entry thresholds: 9.5, 9.0, 8.5
    Exit thresholds:  5.0, 6.0, 7.0

All other parameters stay locked (starting capital $100k, 20% cap,
15% stop loss, no rotation, no persistence filter, buy/sell at next
open) so the only variables are entry and exit thresholds.

Usage:
    python3 threshold_optimizer.py
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from config import load_config
from data_fetcher import fetch_all
from portfolio_backtest import (
    load_score_data,
    build_daily_scores,
    get_trading_days,
    run_simulation,
)


ENTRY_THRESHOLDS = [9.5, 9.0, 8.5]
EXIT_THRESHOLDS = [5.0, 6.0, 7.0]

BASELINE_ENTRY = 9.5
BASELINE_EXIT = 5.0


# ─────────────────────────────────────────────────────────────
# Grid runner
# ─────────────────────────────────────────────────────────────

def run_grid(daily_scores: dict, price_data: dict, trading_days: list) -> dict:
    """Run all 9 simulations and return a nested dict keyed by (entry, exit)."""
    grid: dict[tuple[float, float], dict] = {}

    for entry_thresh in ENTRY_THRESHOLDS:
        for exit_thresh in EXIT_THRESHOLDS:
            label = f"Entry >={entry_thresh} / Exit <{int(exit_thresh)}"
            print(f"    {label}...", end=" ", flush=True)
            result = run_simulation(
                daily_scores=daily_scores,
                price_data=price_data,
                trading_days=trading_days,
                threshold=entry_thresh,
                persistence_filter=False,
                rotation_strategy="none",
                starting_capital=100_000,
                max_position_pct=0.20,
                stop_loss_pct=0.15,
                score_exit_threshold=exit_thresh,
                min_buy=1000,
                verbose=False,
            )
            grid[(entry_thresh, exit_thresh)] = result
            print(f"+{result['total_return']:.1f}% return, "
                  f"{result['total_trades']} trades, "
                  f"{result['win_rate']:.0f}% WR, "
                  f"{result['max_drawdown']:.1f}% DD")

    return grid


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────

def _fmt_pct(v: float, signed: bool = True) -> str:
    sign = "+" if signed and v >= 0 else ""
    return f"{sign}{v:.1f}%"


def print_section_1_grid(grid: dict) -> None:
    print("\n" + "=" * 100)
    print("  SECTION 1: ENTRY/EXIT THRESHOLD GRID — 9 COMBINATIONS")
    print("=" * 100)
    print()
    # Header
    header = (f"  {'Entry':<8s} {'Exit':<6s} "
              f"{'Total Ret':>11s} {'SPY':>9s} {'Alpha':>9s} "
              f"{'Max DD':>9s} {'Win %':>8s} {'Trades':>8s} "
              f"{'Hold':>7s} {'Avg Gain':>10s} {'PF':>8s}")
    print(header)
    print("  " + "-" * 98)

    for entry_thresh in ENTRY_THRESHOLDS:
        for exit_thresh in EXIT_THRESHOLDS:
            r = grid[(entry_thresh, exit_thresh)]
            pf = r["profit_factor"]
            pf_str = f"{pf:.2f}x" if pf != float("inf") else "inf"
            print(f"  >={entry_thresh:<4.1f}  <{int(exit_thresh):<4d}  "
                  f"{_fmt_pct(r['total_return']):>11s} "
                  f"{_fmt_pct(r['spy_return']):>9s} "
                  f"{_fmt_pct(r['alpha']):>9s} "
                  f"{_fmt_pct(r['max_drawdown'], signed=False):>9s} "
                  f"{r['win_rate']:>7.1f}% "
                  f"{r['total_trades']:>8d} "
                  f"{r['avg_hold_days']:>5.0f}d  "
                  f"{_fmt_pct(r['avg_gain_per_trade']):>10s} "
                  f"{pf_str:>8s}")
    print("  " + "-" * 98)
    print()

    # Find best by total return
    best_key, best = max(grid.items(), key=lambda kv: kv[1]["total_return"])
    baseline = grid[(BASELINE_ENTRY, BASELINE_EXIT)]

    print(f"  BEST COMBINATION:  Entry >={best_key[0]} / Exit <{int(best_key[1])}  ->  "
          f"{_fmt_pct(best['total_return'])} return, "
          f"{best['win_rate']:.1f}% win rate, "
          f"{_fmt_pct(best['max_drawdown'], signed=False)} max DD")
    print(f"  BASELINE:          Entry >={BASELINE_ENTRY} / Exit <{int(BASELINE_EXIT)}  ->  "
          f"{_fmt_pct(baseline['total_return'])} return, "
          f"{baseline['win_rate']:.1f}% win rate, "
          f"{_fmt_pct(baseline['max_drawdown'], signed=False)} max DD")


def print_section_2_dimensions(grid: dict) -> None:
    print("\n" + "=" * 100)
    print("  SECTION 2: DIMENSION ANALYSIS")
    print("=" * 100)

    # --- Entry effect (exit held at < 5) ---
    print("\n  ENTRY THRESHOLD EFFECT (holding exit constant at < 5)")
    print("  " + "-" * 80)
    print(f"  {'':<12s} {'Return':>12s} {'Trades':>9s} {'Win %':>9s} {'Max DD':>10s}")
    for entry_thresh in ENTRY_THRESHOLDS:
        r = grid[(entry_thresh, BASELINE_EXIT)]
        print(f"  Entry >={entry_thresh:<3.1f}  "
              f"{_fmt_pct(r['total_return']):>12s} "
              f"{r['total_trades']:>9d} "
              f"{r['win_rate']:>8.1f}% "
              f"{_fmt_pct(r['max_drawdown'], signed=False):>10s}")

    baseline = grid[(BASELINE_ENTRY, BASELINE_EXIT)]["total_return"]
    print()
    for entry_thresh in [9.0, 8.5]:
        delta = grid[(entry_thresh, BASELINE_EXIT)]["total_return"] - baseline
        direction = "better" if delta > 0 else "worse"
        print(f"  -> Lowering entry from 9.5 to {entry_thresh}: "
              f"{direction} by {abs(delta):.1f}% (absolute return)")

    # --- Exit effect (entry held at ≥ 9.5) ---
    print("\n  EXIT THRESHOLD EFFECT (holding entry constant at >= 9.5)")
    print("  " + "-" * 80)
    print(f"  {'':<12s} {'Return':>12s} {'Avg Hold':>11s} {'Win %':>9s} {'Max DD':>10s}")
    for exit_thresh in EXIT_THRESHOLDS:
        r = grid[(BASELINE_ENTRY, exit_thresh)]
        print(f"  Exit <{int(exit_thresh):<3d}  "
              f"{_fmt_pct(r['total_return']):>12s} "
              f"{r['avg_hold_days']:>9.0f}d  "
              f"{r['win_rate']:>8.1f}% "
              f"{_fmt_pct(r['max_drawdown'], signed=False):>10s}")

    print()
    for exit_thresh in [6.0, 7.0]:
        delta = grid[(BASELINE_ENTRY, exit_thresh)]["total_return"] - baseline
        direction = "better" if delta > 0 else "worse"
        print(f"  -> Tightening exit from 5 to {int(exit_thresh)}: "
              f"{direction} by {abs(delta):.1f}% (absolute return)")


def _positions_and_cash_stats(result: dict) -> tuple[float, float, int]:
    """Compute avg positions held, avg cash idle, and signals skipped for a result."""
    history = result.get("history", [])
    if not history:
        return 0.0, 0.0, 0

    avg_positions = float(np.mean([h["num_positions"] for h in history]))
    avg_cash = float(np.mean([h["cash"] for h in history]))
    signals_missed = result.get("signals_missed", 0)
    return avg_positions, avg_cash, signals_missed


def _turnover_per_year(result: dict) -> float:
    history = result.get("history", [])
    if len(history) < 2:
        return 0.0
    from pandas import Timestamp
    start = Timestamp(history[0]["date"])
    end = Timestamp(history[-1]["date"])
    years = max((end - start).days / 365.25, 0.01)
    return result["total_trades"] / years


def print_section_3_capital(grid: dict) -> None:
    print("\n" + "=" * 100)
    print("  SECTION 3: CAPITAL UTILIZATION")
    print("=" * 100)
    print()
    header = (f"  {'Entry':<8s} {'Exit':<6s} "
              f"{'Avg Pos':>9s} {'Avg Cash':>13s} "
              f"{'Util %':>9s} {'Missed':>9s} {'Turnover':>12s}")
    print(header)
    print("  " + "-" * 78)

    for entry_thresh in ENTRY_THRESHOLDS:
        for exit_thresh in EXIT_THRESHOLDS:
            r = grid[(entry_thresh, exit_thresh)]
            avg_pos, avg_cash, missed = _positions_and_cash_stats(r)
            turnover = _turnover_per_year(r)
            print(f"  >={entry_thresh:<4.1f}  <{int(exit_thresh):<4d}  "
                  f"{avg_pos:>8.1f}  "
                  f"${avg_cash:>11,.0f}  "
                  f"{r['capital_utilization']:>7.1f}%  "
                  f"{missed:>9d}  "
                  f"{turnover:>8.1f}x/yr")


def print_section_4_risk(grid: dict) -> None:
    print("\n" + "=" * 100)
    print("  SECTION 4: RISK-ADJUSTED METRICS")
    print("=" * 100)
    print()
    header = (f"  {'Entry':<8s} {'Exit':<6s} "
              f"{'Return':>11s} {'Max DD':>10s} {'Ret/DD':>10s} "
              f"{'Calmar':>10s} {'Worst':>10s}")
    print(header)
    print("  " + "-" * 72)

    for entry_thresh in ENTRY_THRESHOLDS:
        for exit_thresh in EXIT_THRESHOLDS:
            r = grid[(entry_thresh, exit_thresh)]
            total_ret = r["total_return"]
            max_dd = r["max_drawdown"]
            ret_dd = abs(total_ret / max_dd) if max_dd != 0 else float("inf")

            # Calmar = annualized return / |max DD|
            history = r.get("history", [])
            if len(history) >= 2:
                from pandas import Timestamp
                years = max((Timestamp(history[-1]["date"]) -
                             Timestamp(history[0]["date"])).days / 365.25, 0.01)
                ann_return = (1 + total_ret / 100) ** (1 / years) - 1
                calmar = abs(ann_return * 100 / max_dd) if max_dd != 0 else float("inf")
            else:
                calmar = 0.0

            worst = r.get("worst_trade")
            worst_pct = worst.pnl_pct * 100 if worst else 0.0

            ret_dd_str = f"{ret_dd:.2f}x" if ret_dd != float("inf") else "inf"
            calmar_str = f"{calmar:.2f}x" if calmar != float("inf") else "inf"

            print(f"  >={entry_thresh:<4.1f}  <{int(exit_thresh):<4d}  "
                  f"{_fmt_pct(total_ret):>11s} "
                  f"{_fmt_pct(max_dd, signed=False):>10s} "
                  f"{ret_dd_str:>10s} "
                  f"{calmar_str:>10s} "
                  f"{_fmt_pct(worst_pct, signed=False):>10s}")


def print_section_5_recommendation(grid: dict) -> None:
    # Classify winners on each dimension
    best_overall = max(grid.items(), key=lambda kv: kv[1]["total_return"])

    def ret_dd(r):
        return abs(r["total_return"] / r["max_drawdown"]) if r["max_drawdown"] != 0 else float("inf")

    best_risk = max(grid.items(), key=lambda kv: ret_dd(kv[1]))

    best_active = max(grid.items(),
                      key=lambda kv: kv[1]["total_trades"] if kv[1]["alpha"] > 0 else -1)

    safest = min(grid.items(), key=lambda kv: abs(kv[1]["max_drawdown"]))

    baseline = grid[(BASELINE_ENTRY, BASELINE_EXIT)]

    print("\n" + "=" * 100)
    print("  SECTION 5: RECOMMENDATION")
    print("=" * 100)
    print()
    print(f"  Best overall (max return):     Entry >={best_overall[0][0]} / Exit <{int(best_overall[0][1])}  "
          f"-> {_fmt_pct(best_overall[1]['total_return'])}")
    print(f"  Best risk-adjusted (Ret/DD):   Entry >={best_risk[0][0]} / Exit <{int(best_risk[0][1])}  "
          f"-> {_fmt_pct(best_risk[1]['total_return'])} return, "
          f"{_fmt_pct(best_risk[1]['max_drawdown'], signed=False)} DD, "
          f"{ret_dd(best_risk[1]):.2f}x ratio")
    print(f"  Best for active trader:        Entry >={best_active[0][0]} / Exit <{int(best_active[0][1])}  "
          f"-> {best_active[1]['total_trades']} trades, "
          f"{_fmt_pct(best_active[1]['alpha'])} alpha")
    print(f"  Safest (lowest max DD):        Entry >={safest[0][0]} / Exit <{int(safest[0][1])}  "
          f"-> {_fmt_pct(safest[1]['max_drawdown'], signed=False)} DD, "
          f"{_fmt_pct(safest[1]['total_return'])} return")
    print()
    print(f"  Current baseline:              Entry >=9.5 / Exit <5  "
          f"-> {_fmt_pct(baseline['total_return'])} return, "
          f"{_fmt_pct(baseline['max_drawdown'], signed=False)} DD")
    print()

    # Rationale
    overall_key = best_overall[0]
    overall_better_than_baseline = best_overall[1]["total_return"] - baseline["total_return"]
    risk_key = best_risk[0]
    risk_ratio = ret_dd(best_risk[1])
    baseline_ratio = ret_dd(baseline)

    print("  Rationale:")
    if overall_key == (BASELINE_ENTRY, BASELINE_EXIT):
        print("    The baseline (Entry >=9.5, Exit <5) is already the highest-return configuration.")
        print("    Lowering the entry bar adds more trades but dilutes average trade quality,")
        print("    and tightening the exit cuts winners short before they fully run.")
    else:
        print(f"    {_fmt_pct(overall_better_than_baseline)} absolute return improvement is achievable")
        print(f"    by moving from the baseline to Entry >={overall_key[0]} / Exit <{int(overall_key[1])}.")
        if risk_key != overall_key:
            print(f"    However, on a risk-adjusted basis Entry >={risk_key[0]} / Exit <{int(risk_key[1])}")
            print(f"    is stronger ({risk_ratio:.2f}x Return/DD vs the baseline's {baseline_ratio:.2f}x),")
            print("    trading some upside for significantly lower drawdown.")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 100)
    print("  ENTRY/EXIT THRESHOLD OPTIMIZATION")
    print("  Alpha Scanner — 3x3 Grid Search (Entry 9.5/9.0/8.5 x Exit 5/6/7)")
    print("=" * 100)

    # Load data once (reused across all 9 simulations)
    print("\n  Loading score data from database...")
    score_df = load_score_data()
    print(f"  {len(score_df):,} score records loaded")
    print(f"  Date range: {score_df['date'].min().strftime('%Y-%m-%d')} -> "
          f"{score_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"  Unique tickers: {score_df['ticker'].nunique()}")

    print("\n  Fetching 5 years of price data (covers full score history)...")
    cfg = load_config()
    price_data = fetch_all(cfg, period="5y", verbose=False)
    print(f"  {len(price_data)} tickers fetched")

    trading_days = get_trading_days(price_data)
    print(f"  {len(trading_days)} trading days")

    print("\n  Building daily score lookup...")
    daily_scores = build_daily_scores(score_df, trading_days)
    print(f"  {len(daily_scores)} days with scores")

    print("\n  Running grid (9 simulations)...")
    grid = run_grid(daily_scores, price_data, trading_days)

    print_section_1_grid(grid)
    print_section_2_dimensions(grid)
    print_section_3_capital(grid)
    print_section_4_risk(grid)
    print_section_5_recommendation(grid)

    print("\n" + "=" * 100)
    print("  OPTIMIZATION COMPLETE")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
