"""
axti_breakdown.py — Decompose the 8.5/<5 vs 9.5/<5 alpha into:
  - "Same ticker, earlier entry" P&L (tickers traded by BOTH configs)
  - "Unique to 8.5" P&L (tickers whose score never crossed 9.5)
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from portfolio_backtest import (
    load_score_data, build_daily_scores, get_trading_days, run_simulation,
)
from data_fetcher import fetch_all
from config import load_config


def run(threshold: float):
    score_df = load_score_data()
    cfg = load_config()
    price_data = fetch_all(cfg, period="5y", verbose=False)
    trading_days = get_trading_days(price_data)
    daily_scores = build_daily_scores(score_df, trading_days)
    return run_simulation(
        daily_scores, price_data, trading_days,
        threshold=threshold, score_exit_threshold=5.0,
        rotation_strategy="none", persistence_filter=False,
        starting_capital=100_000, max_position_pct=0.20, stop_loss_pct=0.15,
    )


def group_by_ticker(trades):
    """Group trades by ticker (a ticker may have multiple round-trips)."""
    by_ticker = defaultdict(list)
    for t in trades:
        by_ticker[t.ticker].append(t)
    return by_ticker


def main():
    print("=" * 100)
    print("  ALPHA DECOMPOSITION:  Earlier-Entry vs Unique-Trades")
    print("=" * 100)

    print("\n  Running 8.5/<5...")
    res_8 = run(8.5)
    print(f"  Done. {res_8['total_trades']} trades, {res_8['total_return']:+.1f}% return")

    print("\n  Running 9.5/<5...")
    res_9 = run(9.5)
    print(f"  Done. {res_9['total_trades']} trades, {res_9['total_return']:+.1f}% return")

    trades_8 = res_8["trades"]
    trades_9 = res_9["trades"]
    total_8 = sum(t.pnl for t in trades_8)
    total_9 = sum(t.pnl for t in trades_9)

    by_ticker_8 = group_by_ticker(trades_8)
    by_ticker_9 = group_by_ticker(trades_9)

    tickers_8 = set(by_ticker_8.keys())
    tickers_9 = set(by_ticker_9.keys())

    shared = tickers_8 & tickers_9
    unique_to_8 = tickers_8 - tickers_9
    unique_to_9 = tickers_9 - tickers_8

    print("\n" + "=" * 100)
    print("  BUCKET A: TICKERS TRADED BY BOTH CONFIGS")
    print("  (Same name, 8.5 typically enters earlier)")
    print("=" * 100)
    print(f"\n  {len(shared)} tickers in this bucket.\n")

    # For each shared ticker, compare total P&L, average entry date, avg entry price
    rows = []
    shared_total_8 = 0
    shared_total_9 = 0
    for ticker in shared:
        ts8 = by_ticker_8[ticker]
        ts9 = by_ticker_9[ticker]
        p8 = sum(t.pnl for t in ts8)
        p9 = sum(t.pnl for t in ts9)
        shared_total_8 += p8
        shared_total_9 += p9

        # Earliest entry for each config
        first_8 = min(ts8, key=lambda t: t.entry_date)
        first_9 = min(ts9, key=lambda t: t.entry_date)
        days_earlier = (np.datetime64(first_9.entry_date) - np.datetime64(first_8.entry_date)).astype(int)
        price_delta_pct = (first_9.entry_price - first_8.entry_price) / first_8.entry_price * 100 if first_8.entry_price > 0 else 0

        rows.append({
            "ticker": ticker,
            "pnl_8": p8,
            "pnl_9": p9,
            "delta": p8 - p9,
            "trades_8": len(ts8),
            "trades_9": len(ts9),
            "first_entry_8": first_8.entry_date,
            "first_entry_9": first_9.entry_date,
            "days_earlier": days_earlier,
            "entry_price_8": first_8.entry_price,
            "entry_price_9": first_9.entry_price,
            "price_delta_pct": price_delta_pct,
        })

    # Sort by delta (where 8.5 beats 9.5 most)
    rows.sort(key=lambda r: -r["delta"])

    print(f"  Totals from shared tickers:")
    print(f"    8.5/<5 P&L from shared:  ${shared_total_8:>+14,.0f}")
    print(f"    9.5/<5 P&L from shared:  ${shared_total_9:>+14,.0f}")
    print(f"    Delta:                    ${shared_total_8 - shared_total_9:>+14,.0f}")
    print(f"    Avg days earlier (8.5):   {np.mean([r['days_earlier'] for r in rows]):.1f}")
    print(f"    Avg entry price advantage (8.5): {np.mean([r['price_delta_pct'] for r in rows]):+.1f}%")

    print("\n  TOP 15 SHARED TICKERS WHERE 8.5 BEAT 9.5 MOST (by $ delta):")
    print("  " + "-" * 98)
    print(f"  {'Ticker':<10s}{'8.5 P&L':>14s}{'9.5 P&L':>14s}{'Delta':>14s}"
          f"{'Days early':>12s}{'Price adv':>12s}{'8.5 entry':>13s}{'9.5 entry':>13s}")
    print("  " + "-" * 98)
    for r in rows[:15]:
        print(
            f"  {r['ticker']:<10s}"
            f"${r['pnl_8']:>+12,.0f}"
            f"${r['pnl_9']:>+12,.0f}"
            f"${r['delta']:>+12,.0f}"
            f"{r['days_earlier']:>11}d"
            f"{r['price_delta_pct']:>+11.1f}%"
            f"  {r['first_entry_8']:<11s}"
            f"  {r['first_entry_9']:<11s}"
        )

    print("\n  TOP 5 SHARED TICKERS WHERE 9.5 BEAT 8.5 (negative delta — 8.5 stopped out / got worse fill):")
    print("  " + "-" * 98)
    negatives = [r for r in rows if r["delta"] < 0]
    for r in sorted(negatives, key=lambda x: x["delta"])[:5]:
        print(
            f"  {r['ticker']:<10s}"
            f"${r['pnl_8']:>+12,.0f}"
            f"${r['pnl_9']:>+12,.0f}"
            f"${r['delta']:>+12,.0f}"
            f"{r['days_earlier']:>11}d"
            f"{r['price_delta_pct']:>+11.1f}%"
            f"  {r['first_entry_8']:<11s}"
            f"  {r['first_entry_9']:<11s}"
        )
    print(f"\n  Shared tickers where 8.5 beat 9.5: {len([r for r in rows if r['delta'] > 0])}")
    print(f"  Shared tickers where 9.5 beat 8.5: {len(negatives)}")
    print(f"  Shared tickers with zero delta:    {len([r for r in rows if r['delta'] == 0])}")

    print("\n" + "=" * 100)
    print("  BUCKET B: TICKERS UNIQUE TO 8.5/<5")
    print("  (Their peak score never reached 9.5 during an entry opportunity)")
    print("=" * 100)
    print(f"\n  {len(unique_to_8)} tickers in this bucket.")

    unique_total = 0
    unique_rows = []
    for ticker in unique_to_8:
        ts = by_ticker_8[ticker]
        p = sum(t.pnl for t in ts)
        unique_total += p
        winners = [t for t in ts if t.pnl > 0]
        unique_rows.append({
            "ticker": ticker,
            "trades": len(ts),
            "pnl": p,
            "win_rate": len(winners) / len(ts) * 100,
            "avg_pnl_pct": np.mean([t.pnl_pct for t in ts]) * 100,
            "max_entry_score": max(t.entry_score for t in ts),
        })

    unique_rows.sort(key=lambda r: -r["pnl"])
    winners_in_bucket = [r for r in unique_rows if r["pnl"] > 0]
    losers_in_bucket = [r for r in unique_rows if r["pnl"] <= 0]

    print(f"  Total P&L contribution:     ${unique_total:>+14,.0f}")
    print(f"  Winners: {len(winners_in_bucket)} tickers  |  Losers: {len(losers_in_bucket)} tickers")
    if unique_rows:
        print(f"  Ticker win rate (majority-winning): "
              f"{len(winners_in_bucket)/len(unique_rows)*100:.1f}%")

    print("\n  TOP 10 UNIQUE-TO-8.5 CONTRIBUTORS:")
    print("  " + "-" * 80)
    print(f"  {'Ticker':<10s}{'Trades':>9s}{'P&L $':>14s}{'Win %':>9s}"
          f"{'Avg %':>10s}{'Max Entry':>13s}")
    print("  " + "-" * 80)
    for r in unique_rows[:10]:
        print(
            f"  {r['ticker']:<10s}"
            f"{r['trades']:>9d}"
            f"${r['pnl']:>+12,.0f}"
            f"{r['win_rate']:>8.1f}%"
            f"{r['avg_pnl_pct']:>+9.1f}%"
            f"{r['max_entry_score']:>12.2f}"
        )

    print("\n  BOTTOM 5 UNIQUE-TO-8.5 (worst P&L):")
    print("  " + "-" * 80)
    for r in unique_rows[-5:][::-1]:
        print(
            f"  {r['ticker']:<10s}"
            f"{r['trades']:>9d}"
            f"${r['pnl']:>+12,.0f}"
            f"{r['win_rate']:>8.1f}%"
            f"{r['avg_pnl_pct']:>+9.1f}%"
            f"{r['max_entry_score']:>12.2f}"
        )

    # Unique to 9.5 (edge case)
    if unique_to_9:
        unique_9_total = sum(sum(t.pnl for t in by_ticker_9[tk]) for tk in unique_to_9)
        print(f"\n  (Edge case: {len(unique_to_9)} tickers unique to 9.5/<5, total P&L ${unique_9_total:+,.0f})")
        print(f"  These are tickers the 8.5 config couldn't enter, likely because it was fully capital-"
              f"constrained when the signal fired.")
        for tk in unique_to_9:
            p = sum(t.pnl for t in by_ticker_9[tk])
            print(f"    {tk}: ${p:+,.0f}")

    print("\n" + "=" * 100)
    print("  FINAL DECOMPOSITION")
    print("=" * 100)
    print(f"\n  8.5/<5 total P&L:           ${total_8:>+14,.0f}")
    print(f"  9.5/<5 total P&L:           ${total_9:>+14,.0f}")
    print(f"  Delta (8.5 advantage):      ${total_8 - total_9:>+14,.0f}")
    print()
    print(f"  Shared tickers, timing/sizing edge to 8.5:  ${shared_total_8 - shared_total_9:>+14,.0f}")
    print(f"  Tickers unique to 8.5 (never hit 9.5):      ${unique_total:>+14,.0f}")
    if unique_to_9:
        unique_9_total = sum(sum(t.pnl for t in by_ticker_9[tk]) for tk in unique_to_9)
        print(f"  Tickers unique to 9.5 (capital conflict):   ${-unique_9_total:>+14,.0f}  (subtracted from 8.5)")
    print()
    pct_shared = (shared_total_8 - shared_total_9) / (total_8 - total_9) * 100 if (total_8 - total_9) else 0
    pct_unique = unique_total / (total_8 - total_9) * 100 if (total_8 - total_9) else 0
    print(f"  Share of total delta from shared-ticker edge:  {pct_shared:>6.1f}%")
    print(f"  Share of total delta from unique-to-8.5:       {pct_unique:>6.1f}%")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
