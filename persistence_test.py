"""
persistence_test.py — Test N-day persistence filters at entry thresholds 8.5, 9.0, 9.5.
Compare against the baseline 9.5/<5 (no persistence).

Specifically tracks:
  - Whether 'false start' tickers (IREN, AAOI, BTBT, RCAT, RGTI) get filtered out
  - Whether 'early catch' tickers (AXTI, WDC, QUBT, PLTR, KTOS) are retained
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_backtest import (
    load_score_data, build_daily_scores, get_trading_days, run_simulation,
)
from data_fetcher import fetch_all
from config import load_config


FALSE_START_TICKERS = ["IREN", "AAOI", "BTBT", "RCAT", "RGTI"]
EARLY_CATCH_TICKERS = ["AXTI", "WDC", "QUBT", "PLTR", "KTOS"]


def run_cfg(daily_scores, price_data, trading_days, entry, persistence):
    return run_simulation(
        daily_scores, price_data, trading_days,
        threshold=entry,
        score_exit_threshold=5.0,
        persistence_days=persistence,
        rotation_strategy="none",
        persistence_filter=False,
        starting_capital=100_000,
        max_position_pct=0.20,
        stop_loss_pct=0.15,
    )


def summarize_ticker(trades, ticker):
    ts = [t for t in trades if t.ticker == ticker]
    if not ts:
        return {"count": 0, "pnl": 0.0, "first_entry": None, "first_price": None}
    first = min(ts, key=lambda t: t.entry_date)
    return {
        "count": len(ts),
        "pnl": sum(t.pnl for t in ts),
        "first_entry": first.entry_date,
        "first_price": first.entry_price,
    }


def main():
    print("=" * 120)
    print("  PERSISTENCE FILTER TEST — Entry 8.5 & 9.0 with N-day persistence vs baseline 9.5/<5")
    print("=" * 120)

    print("\n  Loading data...")
    score_df = load_score_data()
    cfg = load_config()
    price_data = fetch_all(cfg, period="5y", verbose=False)
    trading_days = get_trading_days(price_data)
    daily_scores = build_daily_scores(score_df, trading_days)
    print(f"  {len(score_df):,} scores loaded ({score_df['date'].min()} -> {score_df['date'].max()})")

    configs = []
    # Baseline
    configs.append(("9.5", 0, "BASELINE"))
    # Entry 9.0 grid
    for p in [0, 1, 2, 3, 5]:
        configs.append(("9.0", p, ""))
    # Entry 8.5 grid
    for p in [0, 1, 2, 3, 5]:
        configs.append(("8.5", p, ""))

    results = []
    for entry_str, p, tag in configs:
        entry = float(entry_str)
        label = f"{entry_str}/p={p}"
        print(f"\n  Running {label}{' (baseline)' if tag else ''}...")
        res = run_cfg(daily_scores, price_data, trading_days, entry, p)
        print(f"    {res['total_trades']} trades, {res['total_return']:+.1f}% return, "
              f"{res['win_rate']:.1f}% WR, {res['max_drawdown']:.1f}% DD")
        results.append({
            "label": label,
            "entry": entry,
            "persistence": p,
            "tag": tag,
            "res": res,
        })

    # ============ SECTION 1: HEADLINE GRID ============
    print("\n" + "=" * 120)
    print("  SECTION 1: HEADLINE METRICS")
    print("=" * 120)
    print(f"\n  {'Config':<14s}{'Total Ret':>12s}{'SPY':>10s}{'Alpha':>10s}"
          f"{'Max DD':>10s}{'Win %':>9s}{'Trades':>9s}{'Avg Hold':>11s}"
          f"{'Signals Gen':>14s}{'Filtered':>11s}")
    print("  " + "-" * 116)
    for r in results:
        res = r["res"]
        print(
            f"  {r['label']:<14s}"
            f"{res['total_return']:>+11.1f}%"
            f"{res['spy_return']:>+9.1f}%"
            f"{res['alpha']:>+9.1f}%"
            f"{res['max_drawdown']:>+9.1f}%"
            f"{res['win_rate']:>8.1f}%"
            f"{res['total_trades']:>9d}"
            f"{res['avg_hold_days']:>9.0f}d"
            f"{res['signals_generated']:>14d}"
            f"{res['signals_filtered']:>11d}"
        )

    # ============ SECTION 2: FALSE-START FILTERING ============
    print("\n" + "=" * 120)
    print("  SECTION 2: FALSE-START TICKER TRACKING (IREN, AAOI, BTBT, RCAT, RGTI)")
    print("  (we WANT to filter these out — they hurt the 8.5 config)")
    print("=" * 120)

    print(f"\n  {'Config':<14s}", end="")
    for tk in FALSE_START_TICKERS:
        print(f"{tk:>16s}", end="")
    print(f"{'Traded':>10s}{'Total P&L':>14s}")
    print("  " + "-" * (14 + 16 * 5 + 10 + 14))
    for r in results:
        trades = r["res"]["trades"]
        print(f"  {r['label']:<14s}", end="")
        traded_count = 0
        total_pnl = 0
        for tk in FALSE_START_TICKERS:
            s = summarize_ticker(trades, tk)
            if s["count"] > 0:
                traded_count += 1
                total_pnl += s["pnl"]
                print(f"{s['pnl']:>+15,.0f}", end="")
            else:
                print(f"{'—':>16s}", end="")
        print(f"{traded_count:>9d}/5${total_pnl:>+12,.0f}")

    # ============ SECTION 3: EARLY-CATCH RETENTION ============
    print("\n" + "=" * 120)
    print("  SECTION 3: EARLY-CATCH TICKER TRACKING (AXTI, WDC, QUBT, PLTR, KTOS)")
    print("  (we WANT to retain these — they drove the 8.5 alpha)")
    print("=" * 120)

    print(f"\n  {'Config':<14s}", end="")
    for tk in EARLY_CATCH_TICKERS:
        print(f"{tk:>16s}", end="")
    print(f"{'Kept':>10s}{'Total P&L':>14s}")
    print("  " + "-" * (14 + 16 * 5 + 10 + 14))
    for r in results:
        trades = r["res"]["trades"]
        print(f"  {r['label']:<14s}", end="")
        kept = 0
        total_pnl = 0
        for tk in EARLY_CATCH_TICKERS:
            s = summarize_ticker(trades, tk)
            if s["count"] > 0:
                kept += 1
                total_pnl += s["pnl"]
                print(f"{s['pnl']:>+15,.0f}", end="")
            else:
                print(f"{'—':>16s}", end="")
        print(f"{kept:>9d}/5${total_pnl:>+12,.0f}")

    # Also show entry date drift for early-catch tickers (did persistence delay them?)
    print("\n  ENTRY DATE DRIFT FOR EARLY-CATCH TICKERS (date of first entry)")
    print("  " + "-" * 110)
    print(f"  {'Config':<14s}", end="")
    for tk in EARLY_CATCH_TICKERS:
        print(f"{tk:>14s}", end="")
    print()
    print("  " + "-" * 110)
    for r in results:
        trades = r["res"]["trades"]
        print(f"  {r['label']:<14s}", end="")
        for tk in EARLY_CATCH_TICKERS:
            s = summarize_ticker(trades, tk)
            print(f"{(s['first_entry'] or '—'):>14s}", end="")
        print()

    # ============ SECTION 4: RECOMMENDATION ============
    print("\n" + "=" * 120)
    print("  SECTION 4: ALPHA vs FALSE-START TRADE-OFF")
    print("=" * 120)

    baseline_res = next(r for r in results if r["label"] == "9.5/p=0")["res"]
    baseline_pnl = baseline_res["final_value"] - baseline_res["starting_capital"]

    print(f"\n  Baseline (9.5/p=0):  ${baseline_pnl:+,.0f}  ({baseline_res['total_return']:+.1f}%)")
    print()
    print(f"  {'Config':<14s}{'Total P&L':>14s}{'vs Base':>14s}{'False-Start':>14s}{'Early-Catch':>14s}"
          f"{'Max DD':>10s}{'Trades':>9s}")
    print("  " + "-" * 91)
    for r in results:
        res = r["res"]
        pnl = res["final_value"] - res["starting_capital"]
        vs_base = pnl - baseline_pnl
        trades = res["trades"]
        fs_pnl = sum(summarize_ticker(trades, tk)["pnl"] for tk in FALSE_START_TICKERS)
        ec_pnl = sum(summarize_ticker(trades, tk)["pnl"] for tk in EARLY_CATCH_TICKERS)
        print(
            f"  {r['label']:<14s}"
            f"${pnl:>+12,.0f}"
            f"${vs_base:>+12,.0f}"
            f"${fs_pnl:>+12,.0f}"
            f"${ec_pnl:>+12,.0f}"
            f"{res['max_drawdown']:>+9.1f}%"
            f"{res['total_trades']:>9d}"
        )

    print("\n" + "=" * 120)
    print("  DONE")
    print("=" * 120)


if __name__ == "__main__":
    main()
