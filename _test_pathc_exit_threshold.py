"""Test Path C with varying EXIT thresholds (entry fixed at 7.5).

Scheme C uses exit < 5.0, but Path C's scoring distribution is different
(Layer 1 0-10 plus Layer 2 ±2-3). Testing whether 5.0 is still optimal
or if a different exit threshold improves outcomes.

Hypothesis: Path C may benefit from a different exit because:
  - Layer 2 sequence overlay can change over time as patterns shift
  - A 5.0 threshold against a max score ~13 = 38% of max (vs Scheme C's 50%)
  - Or 5.0 may still be right because Layer 1 is the dominant component
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

ENTRY_THRESHOLD = 7.5
EXIT_THRESHOLDS = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]


def per_trade_metrics(trades):
    if not trades:
        return dict(n=0, mean=0, median=0, win_count=0, mean_hold_days=0,
                    n_score_exits=0, n_stop_exits=0, n_eob=0)
    pnls = [t.pnl_pct for t in trades]
    holds = [t.hold_days for t in trades]
    n_score_exits = sum(1 for t in trades if t.exit_reason == "score_exit")
    n_stop_exits = sum(1 for t in trades if t.exit_reason == "stop_loss")
    n_eob = sum(1 for t in trades if t.exit_reason == "end_of_backtest")
    return dict(
        n=len(trades),
        mean=float(np.mean(pnls)),
        median=float(np.median(pnls)),
        win_count=sum(1 for p in pnls if p > 0),
        mean_hold_days=float(np.mean(holds)),
        n_score_exits=n_score_exits,
        n_stop_exits=n_stop_exits,
        n_eob=n_eob,
    )


def run_exit_test(exit_thresh: float):
    strat = StrategyConfig(
        name=f"PathC-E{ENTRY_THRESHOLD}-X{exit_thresh}",
        max_positions=12, sizing_mode="fixed_pct",
        fixed_position_pct=0.0833, min_entry_pct=0.05, trim_enabled=False,
        entry_protection_days=7,
        entry_threshold=ENTRY_THRESHOLD,
        exit_threshold=exit_thresh,
        stop_loss=StopLossConfig(type="fixed", value=0.20),
        persistence_days=3,
    )

    # Primary
    sim = PortfolioSimulator(config=strat, daily_scores=daily_scores,
                             price_data=price_data, trading_days=trading_days,
                             start_date="2025-05-01")
    pres = sim.run()
    pmetrics = compute_metrics(pres)
    pt = per_trade_metrics(pres.trades)

    # Path-dep
    path_returns = []
    for s in PATH_STARTS:
        psim = PortfolioSimulator(config=strat, daily_scores=daily_scores,
                                  price_data=price_data, trading_days=trading_days,
                                  start_date=s)
        path_returns.append(compute_metrics(psim.run())["total_return"])

    return {
        "exit_thresh": exit_thresh,
        "primary_return": pmetrics["total_return"],
        "primary_sharpe": pmetrics["sharpe"],
        "primary_dd": pmetrics["max_drawdown"],
        "path_mean": float(np.mean(path_returns)),
        "path_std": float(np.std(path_returns)),
        "path_min": float(min(path_returns)),
        "path_max": float(max(path_returns)),
        "pt_n": pt["n"], "pt_mean": pt["mean"], "pt_median": pt["median"],
        "pt_winners": pt["win_count"],
        "pt_hold": pt["mean_hold_days"],
        "n_score_exits": pt["n_score_exits"],
        "n_stop_exits": pt["n_stop_exits"],
        "n_eob": pt["n_eob"],
    }


results = []
for thr in EXIT_THRESHOLDS:
    print(f"running exit_threshold={thr}...", end="", flush=True)
    r = run_exit_test(thr)
    print(f" cum={r['path_mean']:+.0f}% std={r['path_std']:.0f}% trades={r['pt_n']} "
          f"hold={r['pt_hold']:.0f}d score-exits={r['n_score_exits']} stops={r['n_stop_exits']}")
    results.append(r)

# ─── Summary ──
print("\n" + "=" * 145)
print(f"PATH C EXIT THRESHOLD SWEEP — entry fixed at {ENTRY_THRESHOLD}")
print("=" * 145)
print(f"{'exit':>5}  {'cum mean':>9} {'std':>5} {'min':>9} {'max':>9}  "
      f"{'Sharpe':>6} {'DD':>6}   "
      f"{'trades':>6} {'hold(d)':>7} {'PT mean':>8} {'PT med':>8} {'win%':>5}   "
      f"{'score-X':>7} {'stops':>5} {'EOB':>5}")
print("-" * 145)
for r in results:
    win_pct = (r["pt_winners"] / r["pt_n"] * 100) if r["pt_n"] else 0
    print(f"{r['exit_thresh']:>5.1f}  "
          f"{r['path_mean']:>+8.1f}% {r['path_std']:>4.1f}% "
          f"{r['path_min']:>+8.1f}% {r['path_max']:>+8.1f}%  "
          f"{r['primary_sharpe']:>6.2f} {r['primary_dd']:>+5.1f}%  "
          f"{r['pt_n']:>6} {r['pt_hold']:>7.0f} "
          f"{r['pt_mean']*100:>+7.1f}% {r['pt_median']*100:>+7.1f}% {win_pct:>4.0f}%   "
          f"{r['n_score_exits']:>7} {r['n_stop_exits']:>5} {r['n_eob']:>5}")

# ─── Best per metric ──
print("\n" + "=" * 145)
print("BEST EXIT THRESHOLD PER METRIC")
print("=" * 145)
for metric, key, fmt in [
    ("Cumulative path mean", "path_mean", lambda x: f"{x:+.1f}%"),
    ("Sharpe", "primary_sharpe", lambda x: f"{x:.2f}"),
    ("Per-trade mean", "pt_mean", lambda x: f"{x*100:+.1f}%"),
    ("Per-trade median", "pt_median", lambda x: f"{x*100:+.1f}%"),
    ("Lowest path std", "path_std", lambda x: f"{x:.1f}%"),
    ("Lowest max DD", "primary_dd", lambda x: f"{x:+.1f}%"),
]:
    if metric == "Lowest path std":
        best = min(results, key=lambda r: r[key])
    elif metric == "Lowest max DD":
        best = max(results, key=lambda r: r[key])  # max DD is negative; max value = least bad
    else:
        best = max(results, key=lambda r: r[key])
    print(f"  {metric:<25} exit={best['exit_thresh']:.1f}: {fmt(best[key])}")
