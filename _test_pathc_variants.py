"""Test Path C scoring variants (Layer 1 alone, Layer 1 + various Layer 2)
against the same per-trade metric framework as the Scheme C variants.

Comparison set:
  - Scheme C @ 9.0 (anchor)
  - Scheme I+ Layer 1 only (no Layer 2 sequence overlay)
  - Scheme I+ v1.1 (Layer 1 + win-rate-based Layer 2; original)
  - Scheme I+ Path C (Layer 1 + mean-return-based Layer 2)
  - Scheme I+ Path C coarse (Path C with 0.5 rounding)
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

PATH_STARTS = [
    "2025-04-08", "2025-04-15", "2025-04-22", "2025-04-29",
    "2025-05-06", "2025-05-13", "2025-05-20", "2025-05-27",
    "2025-06-03", "2025-06-10",
]


def per_trade_metrics(trades):
    if not trades:
        return dict(n=0, mean=0, median=0, p25=0, p75=0, win_count=0)
    pnls = [t.pnl_pct for t in trades]
    return dict(
        n=len(trades),
        mean=float(np.mean(pnls)),
        median=float(np.median(pnls)),
        p25=float(np.quantile(pnls, 0.25)),
        p75=float(np.quantile(pnls, 0.75)),
        win_count=sum(1 for p in pnls if p > 0),
    )


def run_variant(label: str, source: str, threshold: float):
    score_df = load_score_data(source)
    daily_scores = build_daily_scores(score_df, trading_days)
    strat = StrategyConfig(
        name=f"{label}-T{threshold}",
        max_positions=12, sizing_mode="fixed_pct",
        fixed_position_pct=0.0833, min_entry_pct=0.05, trim_enabled=False,
        entry_protection_days=7, entry_threshold=threshold, exit_threshold=5.0,
        stop_loss=StopLossConfig(type="fixed", value=0.20), persistence_days=3,
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
    return {
        "label": label, "threshold": threshold,
        "primary_return": pmetrics["total_return"],
        "primary_sharpe": pmetrics["sharpe"],
        "primary_dd": pmetrics["max_drawdown"],
        "path_mean": float(np.mean(path_returns)),
        "path_std": float(np.std(path_returns)),
        "pt_n": pt["n"], "pt_mean": pt["mean"], "pt_median": pt["median"],
        "pt_p25": pt["p25"], "pt_p75": pt["p75"], "pt_win_count": pt["win_count"],
    }


# Path C variants — test each at multiple thresholds to find peak
# (Layer 1 max ~10, with Layer 2 bonuses can exceed; coarse rounded to 0.5)
TESTS = [
    # Anchor
    ("Scheme C (live)",        "sqlite",                                                          9.0),
    # Layer 1 only — no sequence overlay
    ("L1 only",                "parquet:backtest_results/scheme_i_plus_layer1_only.parquet",      6.5),
    ("L1 only",                "parquet:backtest_results/scheme_i_plus_layer1_only.parquet",      7.0),
    ("L1 only",                "parquet:backtest_results/scheme_i_plus_layer1_only.parquet",      7.5),
    ("L1 only",                "parquet:backtest_results/scheme_i_plus_layer1_only.parquet",      8.0),
    # v1.1 — original Layer 2 (win-rate based, with -3.0 catastrophic penalties)
    ("L1+L2 v1.1 (winrate)",   "parquet:backtest_results/scheme_i_plus_scores_v11.parquet",       7.0),
    ("L1+L2 v1.1 (winrate)",   "parquet:backtest_results/scheme_i_plus_scores_v11.parquet",       7.5),
    ("L1+L2 v1.1 (winrate)",   "parquet:backtest_results/scheme_i_plus_scores_v11.parquet",       8.0),
    ("L1+L2 v1.1 (winrate)",   "parquet:backtest_results/scheme_i_plus_scores_v11.parquet",       9.0),
    # Path C — mean-return-based Layer 2
    ("Path C (mean-return)",   "parquet:backtest_results/scheme_i_plus_v2_pathC.parquet",         7.0),
    ("Path C (mean-return)",   "parquet:backtest_results/scheme_i_plus_v2_pathC.parquet",         7.5),
    ("Path C (mean-return)",   "parquet:backtest_results/scheme_i_plus_v2_pathC.parquet",         8.0),
    ("Path C (mean-return)",   "parquet:backtest_results/scheme_i_plus_v2_pathC.parquet",         9.0),
    # Path C coarse (rounded to 0.5)
    ("Path C coarse",          "parquet:backtest_results/scheme_i_plus_v2_pathC_coarse05.parquet", 6.5),
    ("Path C coarse",          "parquet:backtest_results/scheme_i_plus_v2_pathC_coarse05.parquet", 7.0),
    ("Path C coarse",          "parquet:backtest_results/scheme_i_plus_v2_pathC_coarse05.parquet", 7.5),
    ("Path C coarse",          "parquet:backtest_results/scheme_i_plus_v2_pathC_coarse05.parquet", 8.0),
]

results = []
for label, source, threshold in TESTS:
    print(f"running {label} @ T={threshold}...", end="", flush=True)
    r = run_variant(label, source, threshold)
    print(f" cum={r['path_mean']:+.0f}% std={r['path_std']:.0f}% "
          f"trades={r['pt_n']} PT-med={r['pt_median']*100:+.1f}%")
    results.append(r)

print("\n" + "=" * 145)
print("PATH C LAYER VARIANTS — same metrics as Scheme C variants")
print("=" * 145)
print(f"{'variant':<24} {'thr':>5}  {'cum mean':>9} {'std':>5}  "
      f"{'Sharpe':>6} {'DD':>6}   "
      f"{'trades':>6}  {'PT mean':>8} {'PT med':>8} {'PT p25':>8} {'PT p75':>8} {'win%':>5}")
print("-" * 145)
for r in results:
    win_pct = (r["pt_win_count"] / r["pt_n"] * 100) if r["pt_n"] else 0
    print(f"{r['label']:<24} {r['threshold']:>5.1f}  "
          f"{r['path_mean']:>+8.1f}% {r['path_std']:>4.1f}%  {r['primary_sharpe']:>6.2f} "
          f"{r['primary_dd']:>+5.1f}%  "
          f"{r['pt_n']:>6}  {r['pt_mean']*100:>+7.1f}% {r['pt_median']*100:>+7.1f}% "
          f"{r['pt_p25']*100:>+7.1f}% {r['pt_p75']*100:>+7.1f}% {win_pct:>4.0f}%")

# ─── Best per metric ──
print("\n" + "=" * 145)
print("BEST CONFIG PER METRIC")
print("=" * 145)
for metric, key, fmt in [
    ("Cumulative path mean", "path_mean", lambda x: f"{x:+.1f}%"),
    ("Sharpe", "primary_sharpe", lambda x: f"{x:.2f}"),
    ("Per-trade mean", "pt_mean", lambda x: f"{x*100:+.1f}%"),
    ("Per-trade median", "pt_median", lambda x: f"{x*100:+.1f}%"),
    ("Lowest path std", "path_std", lambda x: f"{x:.1f}%"),
]:
    if metric == "Lowest path std":
        best = min(results, key=lambda r: r[key])
    else:
        best = max(results, key=lambda r: r[key])
    print(f"  {metric:<25} {best['label']:<24} @ T={best['threshold']:.1f}: {fmt(best[key])}")
