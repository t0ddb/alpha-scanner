"""Test the most promising variants at LOWER thresholds. Hypothesis:
high per-trade variants (C+all, C+DTF-split, C+TYPE-bonus) might match
Scheme C's cumulative if they take more trades.
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
        return dict(n=0, mean=0, median=0, p25=0, p75=0)
    pnls = [t.pnl_pct for t in trades]
    return dict(
        n=len(trades),
        mean=float(np.mean(pnls)),
        median=float(np.median(pnls)),
        p25=float(np.quantile(pnls, 0.25)),
        p75=float(np.quantile(pnls, 0.75)),
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
        "path_mean": float(np.mean(path_returns)),
        "path_std": float(np.std(path_returns)),
        "pt_n": pt["n"], "pt_mean": pt["mean"], "pt_median": pt["median"],
        "pt_p25": pt["p25"], "pt_p75": pt["p75"],
    }


# Test promising variants at SWEEPS of lower thresholds
TESTS = [
    # Scheme C anchor
    ("Scheme C (live)",  "sqlite",                                                       9.0),
    # C+all — best per-trade — try lower thresholds
    ("C+all",            "parquet:backtest_results/scheme_c_variant_all.parquet",        8.0),
    ("C+all",            "parquet:backtest_results/scheme_c_variant_all.parquet",        8.5),
    ("C+all",            "parquet:backtest_results/scheme_c_variant_all.parquet",        9.0),
    ("C+all",            "parquet:backtest_results/scheme_c_variant_all.parquet",        9.5),
    # C+DTF-split — second-best per-trade
    ("C+DTF-split",      "parquet:backtest_results/scheme_c_variant_dtf_split.parquet",  7.0),
    ("C+DTF-split",      "parquet:backtest_results/scheme_c_variant_dtf_split.parquet",  7.5),
    ("C+DTF-split",      "parquet:backtest_results/scheme_c_variant_dtf_split.parquet",  8.0),
    # C+TYPE-bonus — competitive cum + good per-trade
    ("C+TYPE-bonus",     "parquet:backtest_results/scheme_c_variant_type_bonus.parquet", 8.0),
    ("C+TYPE-bonus",     "parquet:backtest_results/scheme_c_variant_type_bonus.parquet", 8.5),
    ("C+TYPE-bonus",     "parquet:backtest_results/scheme_c_variant_type_bonus.parquet", 9.0),
    # C+MOM-heavy — best Sharpe
    ("C+MOM-heavy",      "parquet:backtest_results/scheme_c_variant_mom_heavy.parquet",  8.0),
    ("C+MOM-heavy",      "parquet:backtest_results/scheme_c_variant_mom_heavy.parquet",  8.5),
    ("C+MOM-heavy",      "parquet:backtest_results/scheme_c_variant_mom_heavy.parquet",  9.0),
]

results = []
for label, source, threshold in TESTS:
    print(f"running {label} @ T={threshold}...", end="", flush=True)
    r = run_variant(label, source, threshold)
    print(f" cum={r['path_mean']:+.0f}% std={r['path_std']:.0f}% "
          f"trades={r['pt_n']} PT-med={r['pt_median']*100:+.1f}% Sharpe={r['primary_sharpe']:.2f}")
    results.append(r)

print("\n" + "=" * 145)
print("LOWER-THRESHOLD SWEEP — promising variants vs Scheme C")
print("=" * 145)
print(f"{'variant':<20} {'thr':>5}  {'cum mean':>9} {'std':>5}  "
      f"{'Sharpe':>6}   "
      f"{'trades':>6}  {'PT mean':>8} {'PT med':>8} {'PT p25':>8} {'PT p75':>8}")
print("-" * 145)
for r in results:
    print(f"{r['label']:<20} {r['threshold']:>5.1f}  "
          f"{r['path_mean']:>+8.1f}% {r['path_std']:>4.1f}%  {r['primary_sharpe']:>6.2f}   "
          f"{r['pt_n']:>6}  {r['pt_mean']*100:>+7.1f}% {r['pt_median']*100:>+7.1f}% "
          f"{r['pt_p25']*100:>+7.1f}% {r['pt_p75']*100:>+7.1f}%")
