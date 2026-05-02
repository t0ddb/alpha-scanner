"""Full variant test — Scheme C + 5 single-fix variants + 4 type-grouping
variants, with per-trade metrics added alongside path-dep cumulative.

Per-trade metrics give a different lens: compares signal/trade QUALITY
not just total compounded return. This tells us whether a scheme picks
worse stocks on average vs misses outliers.
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
    """Compute per-trade aggregates from a list of Trade objects."""
    if not trades:
        return dict(n=0, mean_pnl_pct=0, median_pnl_pct=0,
                    mean_winner=0, mean_loser=0, win_count=0)
    pnls = [t.pnl_pct for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    return dict(
        n=len(trades),
        mean_pnl_pct=float(np.mean(pnls)),
        median_pnl_pct=float(np.median(pnls)),
        mean_winner=float(np.mean(winners)) if winners else 0.0,
        mean_loser=float(np.mean(losers)) if losers else 0.0,
        win_count=len(winners),
    )


def run_variant(label: str, score_source: str, threshold: float):
    score_df = load_score_data(score_source)
    daily_scores = build_daily_scores(score_df, trading_days)

    strat = StrategyConfig(
        name=f"{label}-T{threshold}",
        max_positions=12, sizing_mode="fixed_pct",
        fixed_position_pct=0.0833, min_entry_pct=0.05, trim_enabled=False,
        entry_protection_days=7, entry_threshold=threshold, exit_threshold=5.0,
        stop_loss=StopLossConfig(type="fixed", value=0.20), persistence_days=3,
    )

    # Primary
    sim = PortfolioSimulator(config=strat, daily_scores=daily_scores,
                             price_data=price_data, trading_days=trading_days,
                             start_date="2025-05-01")
    primary_result = sim.run()
    primary = compute_metrics(primary_result)
    pt = per_trade_metrics(primary_result.trades)

    # Path-dep cumulative returns
    path_returns = []
    for s in PATH_STARTS:
        psim = PortfolioSimulator(config=strat, daily_scores=daily_scores,
                                  price_data=price_data, trading_days=trading_days,
                                  start_date=s)
        path_returns.append(compute_metrics(psim.run())["total_return"])

    return {
        "label": label,
        "threshold": threshold,
        "primary_return": primary["total_return"],
        "primary_sharpe": primary["sharpe"],
        "primary_dd": primary["max_drawdown"],
        "primary_trades": primary["total_trades"],
        "path_mean": float(np.mean(path_returns)),
        "path_std": float(np.std(path_returns)),
        # Per-trade
        "pt_n": pt["n"],
        "pt_mean": pt["mean_pnl_pct"],
        "pt_median": pt["median_pnl_pct"],
        "pt_winners": pt["mean_winner"],
        "pt_losers": pt["mean_loser"],
        "pt_win_count": pt["win_count"],
    }


# Test grid: Scheme C base + each variant at peak threshold from prior test
TESTS = [
    ("Scheme C (live)",   "sqlite",                                                       9.0),
    ("Scheme C (live)",   "sqlite",                                                       8.5),
    # Existing single-fix variants at their peak thresholds
    ("C+ICH-tier",        "parquet:backtest_results/scheme_c_variant_ich_tier.parquet",   9.0),
    ("C+CMF-add",         "parquet:backtest_results/scheme_c_variant_cmf_add.parquet",    9.0),
    ("C+ROC-cap",         "parquet:backtest_results/scheme_c_variant_roc_cap.parquet",    8.5),
    ("C+DTF-split",       "parquet:backtest_results/scheme_c_variant_dtf_split.parquet",  8.5),
    ("C+RS-dip",          "parquet:backtest_results/scheme_c_variant_rs_dip.parquet",     9.0),
    ("C+all",             "parquet:backtest_results/scheme_c_variant_all.parquet",       10.0),
    # NEW: type-grouping variants
    ("C+TYPE-required",   "parquet:backtest_results/scheme_c_variant_type_required.parquet", 9.0),
    ("C+TYPE-required",   "parquet:backtest_results/scheme_c_variant_type_required.parquet", 8.5),
    ("C+TYPE-bonus",      "parquet:backtest_results/scheme_c_variant_type_bonus.parquet",  9.0),
    ("C+TYPE-bonus",      "parquet:backtest_results/scheme_c_variant_type_bonus.parquet",  9.5),
    ("C+MOM-heavy",       "parquet:backtest_results/scheme_c_variant_mom_heavy.parquet",   9.0),
    ("C+MOM-heavy",       "parquet:backtest_results/scheme_c_variant_mom_heavy.parquet",   9.5),
    ("C+MOM-heavy",       "parquet:backtest_results/scheme_c_variant_mom_heavy.parquet",  10.0),
    ("C+TREND-only",      "parquet:backtest_results/scheme_c_variant_trend_only.parquet",  4.5),
    ("C+TREND-only",      "parquet:backtest_results/scheme_c_variant_trend_only.parquet",  5.0),
]

results = []
for label, source, threshold in TESTS:
    print(f"running {label} @ T={threshold}...", end="", flush=True)
    r = run_variant(label, source, threshold)
    print(f" cum={r['path_mean']:+.0f}% std={r['path_std']:.0f}% "
          f"per-trade mean={r['pt_mean']*100:+.1f}% median={r['pt_median']*100:+.1f}% n={r['pt_n']}")
    results.append(r)

# ─── Summary table — cumulative + per-trade side-by-side ──────
print("\n" + "=" * 145)
print("FULL VARIANT TEST — cumulative (path-dep) + per-trade quality metrics")
print("=" * 145)
print(f"{'variant':<22} {'thr':>5}   "
      f"{'cum mean':>9} {'std':>5} {'Sharpe':>6} {'DD':>6}   "
      f"{'trades':>6} {'PT mean':>8} {'PT med':>8} {'PT wins':>8} {'PT loss':>8} {'win%':>5}")
print("-" * 145)
for r in results:
    win_pct = (r["pt_win_count"] / r["pt_n"] * 100) if r["pt_n"] else 0
    print(f"{r['label']:<22} {r['threshold']:>5.1f}   "
          f"{r['path_mean']:>+8.1f}% {r['path_std']:>4.1f}% {r['primary_sharpe']:>6.2f} "
          f"{r['primary_dd']:>+5.1f}%  "
          f"{r['pt_n']:>6} {r['pt_mean']*100:>+7.1f}% {r['pt_median']*100:>+7.1f}% "
          f"{r['pt_winners']*100:>+7.1f}% {r['pt_losers']*100:>+7.1f}% {win_pct:>4.0f}%")
