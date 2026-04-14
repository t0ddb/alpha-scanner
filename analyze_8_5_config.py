"""
analyze_8_5_config.py — Deep analysis of the Entry >=8.5 / Exit <5 configuration.

Answers four questions before changing live thresholds:
  1. Win rate by entry score bucket (8.5-9.0, 9.0-9.5, 9.5+)
  2. Average return by bucket
  3. Is alpha from a few monster trades or broad improvement?
  4. How does 8.5/<5 perform during the 2023 drawdown vs the baseline 9.5/<5?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtest import (
    load_score_data,
    build_daily_scores,
    get_trading_days,
    get_price,
    get_spy_return,
    run_simulation,
)
from data_fetcher import fetch_all
from config import load_config


def bucket_of(score: float) -> str:
    if score >= 9.5:
        return "9.5+"
    if score >= 9.0:
        return "9.0-9.5"
    return "8.5-9.0"


def bucket_stats(trades: list) -> dict:
    buckets: dict[str, list] = {"8.5-9.0": [], "9.0-9.5": [], "9.5+": []}
    for t in trades:
        buckets[bucket_of(t.entry_score)].append(t)

    out = {}
    for name, ts in buckets.items():
        if not ts:
            out[name] = None
            continue
        winners = [t for t in ts if t.pnl > 0]
        losers = [t for t in ts if t.pnl <= 0]
        out[name] = {
            "count": len(ts),
            "pct_of_trades": len(ts) / len(trades) * 100,
            "win_rate": len(winners) / len(ts) * 100,
            "avg_pnl_pct": np.mean([t.pnl_pct for t in ts]) * 100,
            "median_pnl_pct": np.median([t.pnl_pct for t in ts]) * 100,
            "total_pnl": sum(t.pnl for t in ts),
            "avg_pnl_dollar": np.mean([t.pnl for t in ts]),
            "avg_winner_pct": np.mean([t.pnl_pct for t in winners]) * 100 if winners else 0,
            "avg_loser_pct": np.mean([t.pnl_pct for t in losers]) * 100 if losers else 0,
            "avg_hold": np.mean([t.hold_days for t in ts]),
            "best": max(ts, key=lambda t: t.pnl_pct),
            "worst": min(ts, key=lambda t: t.pnl_pct),
        }
    return out


def print_section_1_and_2(bs_8: dict):
    print("\n" + "=" * 100)
    print("  SECTION 1+2: ENTRY SCORE BUCKET ANALYSIS  (Entry >=8.5 / Exit <5)")
    print("=" * 100)
    print()
    print(f"  {'Bucket':<12s}{'Trades':>8s}{'% of':>8s}{'Win %':>8s}{'Avg P&L':>12s}"
          f"{'Median':>10s}{'Total $':>14s}{'Avg $':>12s}{'Avg Hold':>12s}")
    print(f"  {'':<12s}{'':>8s}{'total':>8s}{'':>8s}{'(%)':>12s}{'(%)':>10s}{'':>14s}{'':>12s}{'(days)':>12s}")
    print("  " + "-" * 98)

    for name in ["8.5-9.0", "9.0-9.5", "9.5+"]:
        s = bs_8[name]
        if s is None:
            print(f"  {name:<12s} {'(no trades)':>8s}")
            continue
        print(
            f"  {name:<12s}"
            f"{s['count']:>8d}"
            f"{s['pct_of_trades']:>7.1f}%"
            f"{s['win_rate']:>7.1f}%"
            f"{s['avg_pnl_pct']:>+11.1f}%"
            f"{s['median_pnl_pct']:>+9.1f}%"
            f"${s['total_pnl']:>12,.0f}"
            f"${s['avg_pnl_dollar']:>10,.0f}"
            f"{s['avg_hold']:>11.0f}d"
        )
    print()

    # Separate winners vs losers within each bucket
    print("  WINNER vs LOSER DYNAMICS BY BUCKET")
    print("  " + "-" * 70)
    print(f"  {'Bucket':<12s}{'Avg Winner %':>16s}{'Avg Loser %':>16s}{'Win/Loss Ratio':>18s}")
    print("  " + "-" * 70)
    for name in ["8.5-9.0", "9.0-9.5", "9.5+"]:
        s = bs_8[name]
        if s is None:
            continue
        ratio = abs(s["avg_winner_pct"] / s["avg_loser_pct"]) if s["avg_loser_pct"] else float("inf")
        print(
            f"  {name:<12s}"
            f"{s['avg_winner_pct']:>+15.1f}%"
            f"{s['avg_loser_pct']:>+15.1f}%"
            f"{ratio:>17.2f}x"
        )


def print_section_3_monsters(trades_8: list, trades_9: list):
    """Is the 8.5 alpha from monster trades or broad improvement?"""
    print("\n" + "=" * 100)
    print("  SECTION 3: MONSTER TRADE ATTRIBUTION")
    print("=" * 100)

    # Sort by dollar P&L, show top contributors
    sorted_8 = sorted(trades_8, key=lambda t: -t.pnl)
    sorted_9 = sorted(trades_9, key=lambda t: -t.pnl)

    total_8 = sum(t.pnl for t in trades_8)
    total_9 = sum(t.pnl for t in trades_9)

    print(f"\n  Total P&L across all trades:")
    print(f"    8.5/<5 config:  ${total_8:>14,.0f}  ({len(trades_8)} trades)")
    print(f"    9.5/<5 config:  ${total_9:>14,.0f}  ({len(trades_9)} trades)")
    print(f"    Delta:          ${total_8 - total_9:>+14,.0f}")

    # Top-10 contribution
    print(f"\n  Concentration — how much of total P&L is from the top N trades?")
    print(f"  {'Config':<14s}{'Top 1':>10s}{'Top 5':>10s}{'Top 10':>10s}{'Top 20':>10s}")
    print("  " + "-" * 54)
    for label, sorted_t, total in [("8.5/<5", sorted_8, total_8), ("9.5/<5", sorted_9, total_9)]:
        t1 = sum(t.pnl for t in sorted_t[:1])
        t5 = sum(t.pnl for t in sorted_t[:5])
        t10 = sum(t.pnl for t in sorted_t[:10])
        t20 = sum(t.pnl for t in sorted_t[:20])
        print(
            f"  {label:<14s}"
            f"{t1/total*100:>9.1f}%"
            f"{t5/total*100:>9.1f}%"
            f"{t10/total*100:>9.1f}%"
            f"{t20/total*100:>9.1f}%"
        )

    # Top 10 trades in 8.5 config
    print(f"\n  TOP 10 CONTRIBUTORS (8.5/<5 config)")
    print("  " + "-" * 90)
    print(f"  {'#':<4s}{'Ticker':<10s}{'Entry Date':<14s}{'Exit Date':<14s}"
          f"{'Score':>8s}{'P&L $':>14s}{'P&L %':>10s}{'Hold':>8s}")
    print("  " + "-" * 90)
    for i, t in enumerate(sorted_8[:10], 1):
        print(
            f"  {i:<4d}{t.ticker:<10s}{t.entry_date:<14s}{t.exit_date:<14s}"
            f"{t.entry_score:>8.2f}"
            f"${t.pnl:>12,.0f}"
            f"{t.pnl_pct*100:>+9.1f}%"
            f"{t.hold_days:>6d}d"
        )

    # Median trade comparison
    print(f"\n  MEDIAN TRADE COMPARISON — is the typical trade better at 8.5?")
    median_pnl_8 = np.median([t.pnl for t in trades_8])
    median_pnl_9 = np.median([t.pnl for t in trades_9])
    median_pct_8 = np.median([t.pnl_pct for t in trades_8]) * 100
    median_pct_9 = np.median([t.pnl_pct for t in trades_9]) * 100
    print(f"    Median $ P&L per trade:  8.5/<5 ${median_pnl_8:>+10,.0f}   vs   9.5/<5 ${median_pnl_9:>+10,.0f}")
    print(f"    Median % P&L per trade:  8.5/<5 {median_pct_8:>+10.2f}%   vs   9.5/<5 {median_pct_9:>+10.2f}%")

    # Trade count difference — what proportion of 8.5 extra P&L comes from trades that wouldn't exist at 9.5
    extra_trades = [t for t in trades_8 if t.entry_score < 9.5]
    extra_pnl = sum(t.pnl for t in extra_trades)
    print(f"\n  Trades unique to 8.5 config (entry score 8.5-9.49):")
    print(f"    Count: {len(extra_trades)} ({len(extra_trades)/len(trades_8)*100:.1f}% of total trades)")
    print(f"    Total P&L contribution: ${extra_pnl:>+14,.0f}")
    print(f"    Share of total P&L:     {extra_pnl/total_8*100:>+13.1f}%")

    extra_winners = [t for t in extra_trades if t.pnl > 0]
    if extra_trades:
        print(f"    Win rate of extra trades: {len(extra_winners)/len(extra_trades)*100:.1f}%")


def compute_period_stats(history: list, price_data: dict, start: str, end: str, label: str) -> dict:
    sub = [h for h in history if start <= h["date"] <= end]
    if not sub:
        return None
    start_val = sub[0]["total_value"]
    end_val = sub[-1]["total_value"]
    ret = (end_val - start_val) / start_val * 100

    # Drawdown within period
    peak = sub[0]["total_value"]
    max_dd = 0
    for h in sub:
        if h["total_value"] > peak:
            peak = h["total_value"]
        dd = (h["total_value"] - peak) / peak * 100
        if dd < max_dd:
            max_dd = dd

    spy_ret = get_spy_return(price_data,
                             pd.Timestamp(sub[0]["date"]),
                             pd.Timestamp(sub[-1]["date"])) * 100

    return {
        "label": label,
        "start": sub[0]["date"],
        "end": sub[-1]["date"],
        "start_value": start_val,
        "end_value": end_val,
        "return_pct": ret,
        "max_dd_pct": max_dd,
        "spy_return_pct": spy_ret,
        "alpha_pct": ret - spy_ret,
    }


def print_section_4_bear(trades_8, history_8, trades_9, history_9, price_data):
    print("\n" + "=" * 100)
    print("  SECTION 4: BEAR / DRAWDOWN REGIME PERFORMANCE")
    print("=" * 100)

    # Define regime periods we care about.
    # 2023 late correction (Aug 1 -> Oct 27 is SPY's 2023 drawdown).
    # 2024 summer dip (Jul 16 -> Aug 5).
    # April 2025 tariff selloff (varies by data — use Apr 2025).
    periods = [
        ("2023 correction (Aug–Oct 2023)",      "2023-08-01", "2023-11-01"),
        ("2024 summer dip (Jul–Aug 2024)",      "2024-07-16", "2024-08-20"),
        ("2025 tariff selloff (Apr 2025)",      "2025-04-01", "2025-05-15"),
        ("Full backtest period",                "2023-04-14", "2026-04-06"),
    ]

    print(f"\n  {'Period':<35s}{'Config':<12s}{'Return':>10s}{'SPY':>10s}{'Alpha':>10s}{'Max DD':>10s}")
    print("  " + "-" * 90)
    for label, start, end in periods:
        s8 = compute_period_stats(history_8, price_data, start, end, label)
        s9 = compute_period_stats(history_9, price_data, start, end, label)
        if s8 is None or s9 is None:
            print(f"  {label:<35s}(no data in range)")
            continue
        print(
            f"  {label:<35s}"
            f"{'8.5/<5':<12s}"
            f"{s8['return_pct']:>+9.1f}%"
            f"{s8['spy_return_pct']:>+9.1f}%"
            f"{s8['alpha_pct']:>+9.1f}%"
            f"{s8['max_dd_pct']:>+9.1f}%"
        )
        print(
            f"  {'':<35s}"
            f"{'9.5/<5':<12s}"
            f"{s9['return_pct']:>+9.1f}%"
            f"{s9['spy_return_pct']:>+9.1f}%"
            f"{s9['alpha_pct']:>+9.1f}%"
            f"{s9['max_dd_pct']:>+9.1f}%"
        )
        print()

    # During the 2023 correction, where were the losing trades concentrated by bucket?
    print("\n  TRADES CLOSED DURING 2023 CORRECTION (2023-08-01 → 2023-11-01)")
    print("  " + "-" * 80)
    for label, config_trades in [("8.5/<5", trades_8), ("9.5/<5", trades_9)]:
        bear_trades = [t for t in config_trades if "2023-08-01" <= t.exit_date <= "2023-11-01"]
        if not bear_trades:
            print(f"  {label}: no trades closed in this window")
            continue
        winners = [t for t in bear_trades if t.pnl > 0]
        avg_pct = np.mean([t.pnl_pct for t in bear_trades]) * 100
        total_pnl = sum(t.pnl for t in bear_trades)
        print(
            f"  {label:<10s}"
            f"trades={len(bear_trades):>4d}   "
            f"win={len(winners)/len(bear_trades)*100:>5.1f}%   "
            f"avg={avg_pct:>+6.1f}%   "
            f"total=${total_pnl:>+10,.0f}"
        )

        # Bucket by entry score for the 8.5 config
        if label == "8.5/<5":
            buckets = {"8.5-9.0": [], "9.0-9.5": [], "9.5+": []}
            for t in bear_trades:
                buckets[bucket_of(t.entry_score)].append(t)
            for bname, bts in buckets.items():
                if not bts:
                    continue
                bw = [t for t in bts if t.pnl > 0]
                print(
                    f"    bucket {bname:<10s}"
                    f"trades={len(bts):>3d}   "
                    f"win={len(bw)/len(bts)*100:>5.1f}%   "
                    f"avg={np.mean([t.pnl_pct for t in bts])*100:>+6.1f}%   "
                    f"total=${sum(t.pnl for t in bts):>+10,.0f}"
                )


def main():
    print("=" * 100)
    print("  DEEP ANALYSIS — ENTRY >=8.5 / EXIT <5  (vs baseline 9.5/<5)")
    print("=" * 100)

    print("\n  Loading score data...")
    score_df = load_score_data()
    print(f"  {len(score_df):,} score records loaded")
    print(f"  Range: {score_df['date'].min()} -> {score_df['date'].max()}")

    print("\n  Fetching 5 years of price data...")
    cfg = load_config()
    price_data = fetch_all(cfg, period="5y", verbose=False)
    print(f"  {len(price_data)} tickers fetched")

    trading_days = get_trading_days(price_data)
    daily_scores = build_daily_scores(score_df, trading_days)

    print("\n  Running simulation: Entry >=8.5 / Exit <5...")
    res_8 = run_simulation(
        daily_scores, price_data, trading_days,
        threshold=8.5, score_exit_threshold=5.0,
        rotation_strategy="none", persistence_filter=False,
        starting_capital=100_000, max_position_pct=0.20, stop_loss_pct=0.15,
    )
    print(f"  Done. {res_8['total_trades']} trades, {res_8['total_return']:+.1f}% return")

    print("\n  Running simulation: Entry >=9.5 / Exit <5 (baseline)...")
    res_9 = run_simulation(
        daily_scores, price_data, trading_days,
        threshold=9.5, score_exit_threshold=5.0,
        rotation_strategy="none", persistence_filter=False,
        starting_capital=100_000, max_position_pct=0.20, stop_loss_pct=0.15,
    )
    print(f"  Done. {res_9['total_trades']} trades, {res_9['total_return']:+.1f}% return")

    trades_8 = res_8["trades"]
    trades_9 = res_9["trades"]

    # Sections 1 & 2 — bucket analysis
    bs_8 = bucket_stats(trades_8)
    print_section_1_and_2(bs_8)

    # Section 3 — monster attribution
    print_section_3_monsters(trades_8, trades_9)

    # Section 4 — bear regime
    print_section_4_bear(trades_8, res_8["history"], trades_9, res_9["history"], price_data)

    print("\n" + "=" * 100)
    print("  ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()
