"""Three sweep tests for Path C with exit=5.5 (newly validated):
  1. Persistence days (1, 2, 3, 4, 5)
  2. Stop loss percentage (0.10, 0.15, 0.20, 0.25, 0.30)
  3. Entry threshold finer (7.0, 7.25, 7.5, 7.75, 8.0)

Each sweep holds others fixed at validated defaults:
  Entry: 7.5  (will sweep this in #3)
  Exit: 5.5
  Persistence: 3 (will sweep in #1)
  Stop: 0.20 (will sweep in #2)
  Cap: 12, sizing: 8.3%

10-start within-regime path-dep validation per config.
"""
import sys
sys.path.insert(0, ".")
from sizing_comparison_backtest import (
    load_score_data, build_daily_scores, get_trading_days,
    PortfolioSimulator, StrategyConfig, StopLossConfig, compute_metrics,
)
from config import load_config
from data_fetcher import fetch_all
import numpy as np
import os

print("loading...")
cfg_yaml = load_config()
price_data = fetch_all(cfg_yaml, period="2y", verbose=False)
trading_days = get_trading_days(price_data)
score_df = load_score_data("parquet:backtest_results/scheme_i_plus_v2_pathC.parquet")
daily_scores = build_daily_scores(score_df, trading_days)

PATH_STARTS = [
    "2025-04-08", "2025-04-15", "2025-04-22", "2025-04-29",
    "2025-05-06", "2025-05-13", "2025-05-20", "2025-05-27",
    "2025-06-03", "2025-06-10",
]

# Validated defaults
DEFAULT_ENTRY = 7.5
DEFAULT_EXIT = 5.5
DEFAULT_PERSIST = 3
DEFAULT_STOP = 0.20


def per_trade_metrics(trades):
    if not trades:
        return dict(n=0, mean=0, median=0, win_count=0, mean_hold=0,
                    n_score_exits=0, n_stop_exits=0)
    pnls = [t.pnl_pct for t in trades]
    holds = [t.hold_days for t in trades]
    return dict(
        n=len(trades),
        mean=float(np.mean(pnls)),
        median=float(np.median(pnls)),
        win_count=sum(1 for p in pnls if p > 0),
        mean_hold=float(np.mean(holds)),
        n_score_exits=sum(1 for t in trades if t.exit_reason == "score_exit"),
        n_stop_exits=sum(1 for t in trades if t.exit_reason == "stop_loss"),
    )


def run(label: str, entry: float, exit_thresh: float, persist: int,
        stop_pct: float):
    # Need to override PERSISTENCE_DAYS via env var before loading the strategy
    # — it's read at import time. Easier: set on StrategyConfig directly.
    strat = StrategyConfig(
        name=label, max_positions=12, sizing_mode="fixed_pct",
        fixed_position_pct=0.0833, min_entry_pct=0.05, trim_enabled=False,
        entry_protection_days=7,
        entry_threshold=entry,
        exit_threshold=exit_thresh,
        stop_loss=StopLossConfig(type="fixed", value=stop_pct),
        persistence_days=persist,
    )
    sim = PortfolioSimulator(config=strat, daily_scores=daily_scores,
                             price_data=price_data, trading_days=trading_days,
                             start_date="2025-05-01")
    pres = sim.run()
    pmetrics = compute_metrics(pres)
    pt = per_trade_metrics(pres.trades)
    path_returns = []
    for s in PATH_STARTS:
        psim = PortfolioSimulator(config=strat, daily_scores=daily_scores,
                                  price_data=price_data, trading_days=trading_days,
                                  start_date=s)
        path_returns.append(compute_metrics(psim.run())["total_return"])
    return dict(
        label=label,
        primary_return=pmetrics["total_return"],
        primary_sharpe=pmetrics["sharpe"],
        primary_dd=pmetrics["max_drawdown"],
        path_mean=float(np.mean(path_returns)),
        path_std=float(np.std(path_returns)),
        path_min=float(min(path_returns)), path_max=float(max(path_returns)),
        pt_n=pt["n"], pt_mean=pt["mean"], pt_median=pt["median"],
        pt_winners=pt["win_count"], pt_hold=pt["mean_hold"],
        n_score_exits=pt["n_score_exits"], n_stop_exits=pt["n_stop_exits"],
    )


def print_sweep(title: str, sweep_var: str, results: list[dict]):
    print("\n" + "=" * 145)
    print(title)
    print("=" * 145)
    print(f"{sweep_var:>12}  {'cum mean':>9} {'std':>5} "
          f"{'min':>9} {'max':>9}  {'Sharpe':>6} {'DD':>6}   "
          f"{'trades':>6} {'hold':>5} {'PT mean':>8} {'PT med':>8} {'win%':>5}   "
          f"{'score-X':>7} {'stops':>5}")
    print("-" * 145)
    for r in results:
        win_pct = (r["pt_winners"] / r["pt_n"] * 100) if r["pt_n"] else 0
        print(f"{r['label']:>12}  "
              f"{r['path_mean']:>+8.1f}% {r['path_std']:>4.1f}% "
              f"{r['path_min']:>+8.1f}% {r['path_max']:>+8.1f}%  "
              f"{r['primary_sharpe']:>6.2f} {r['primary_dd']:>+5.1f}%  "
              f"{r['pt_n']:>6} {r['pt_hold']:>5.0f} "
              f"{r['pt_mean']*100:>+7.1f}% {r['pt_median']*100:>+7.1f}% {win_pct:>4.0f}%   "
              f"{r['n_score_exits']:>7} {r['n_stop_exits']:>5}")


# ─── SWEEP 1: PERSISTENCE DAYS ──
print("\n>>> Sweep 1: Persistence days <<<")
persist_results = []
for p in [1, 2, 3, 4, 5]:
    label = f"persist={p}"
    print(f"  {label}...", end="", flush=True)
    r = run(label, DEFAULT_ENTRY, DEFAULT_EXIT, p, DEFAULT_STOP)
    print(f" cum={r['path_mean']:+.0f}% std={r['path_std']:.0f}% trades={r['pt_n']}")
    persist_results.append(r)

# ─── SWEEP 2: STOP LOSS ──
print("\n>>> Sweep 2: Stop loss % <<<")
stop_results = []
for s in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
    label = f"stop={s:.2f}"
    print(f"  {label}...", end="", flush=True)
    r = run(label, DEFAULT_ENTRY, DEFAULT_EXIT, DEFAULT_PERSIST, s)
    print(f" cum={r['path_mean']:+.0f}% std={r['path_std']:.0f}% stops-fired={r['n_stop_exits']}")
    stop_results.append(r)

# ─── SWEEP 3: ENTRY THRESHOLD (finer) ──
print("\n>>> Sweep 3: Entry threshold (finer) <<<")
entry_results = []
for e in [7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5]:
    label = f"entry={e}"
    print(f"  {label}...", end="", flush=True)
    r = run(label, e, DEFAULT_EXIT, DEFAULT_PERSIST, DEFAULT_STOP)
    print(f" cum={r['path_mean']:+.0f}% std={r['path_std']:.0f}% trades={r['pt_n']}")
    entry_results.append(r)


print_sweep("SWEEP 1 — PERSISTENCE DAYS", "persist", persist_results)
print_sweep("SWEEP 2 — STOP LOSS %", "stop_pct", stop_results)
print_sweep("SWEEP 3 — ENTRY THRESHOLD (finer)", "entry", entry_results)

# ─── Best per metric per sweep ──
def best_per_metric(name: str, results: list):
    print(f"\nBest {name}:")
    for metric, key, fmt, comparator in [
        ("Cumulative path mean", "path_mean", lambda x: f"{x:+.1f}%", max),
        ("Sharpe", "primary_sharpe", lambda x: f"{x:.2f}", max),
        ("Per-trade median", "pt_median", lambda x: f"{x*100:+.1f}%", max),
        ("Lowest path std", "path_std", lambda x: f"{x:.1f}%", min),
        ("Lowest max DD (least negative)", "primary_dd", lambda x: f"{x:+.1f}%", max),
    ]:
        best = comparator(results, key=lambda r: r[key])
        print(f"  {metric:<35} {best['label']}: {fmt(best[key])}")


best_per_metric("PERSISTENCE", persist_results)
best_per_metric("STOP LOSS", stop_results)
best_per_metric("ENTRY", entry_results)
