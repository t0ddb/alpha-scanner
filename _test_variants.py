"""Test each Scheme C variant: each variant changes ONE thing from Scheme C.
Goal: identify which single fixes actually help, vs which hurt or are neutral."""
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
    primary = compute_metrics(sim.run())

    # Path-dep
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
        "path_min": float(min(path_returns)),
        "path_max": float(max(path_returns)),
    }


# Variants to test, paired with appropriate threshold range.
# Variants with max=10.0: test at 8.5, 9.0, 9.5
# Variants with max=10.5: test at 9.0, 9.5, 10.0 (same selectivity range)
# Variants with max=10.88 or 11.5: test at 9.5, 10.0, 10.5
TESTS = [
    # Scheme C base (control)
    ("Scheme C (live)",    "sqlite",                                                       8.5),
    ("Scheme C (live)",    "sqlite",                                                       9.0),
    ("Scheme C (live)",    "sqlite",                                                       9.5),
    # Single-fix variants — max=10.0 or 10.5
    ("C+ICH-tier",         "parquet:backtest_results/scheme_c_variant_ich_tier.parquet",   9.0),
    ("C+ICH-tier",         "parquet:backtest_results/scheme_c_variant_ich_tier.parquet",   9.5),
    ("C+ICH-tier",         "parquet:backtest_results/scheme_c_variant_ich_tier.parquet",  10.0),
    ("C+CMF-add",          "parquet:backtest_results/scheme_c_variant_cmf_add.parquet",    9.0),
    ("C+CMF-add",          "parquet:backtest_results/scheme_c_variant_cmf_add.parquet",    9.5),
    ("C+CMF-add",          "parquet:backtest_results/scheme_c_variant_cmf_add.parquet",   10.0),
    ("C+ROC-cap",          "parquet:backtest_results/scheme_c_variant_roc_cap.parquet",    8.5),
    ("C+ROC-cap",          "parquet:backtest_results/scheme_c_variant_roc_cap.parquet",    9.0),
    ("C+ROC-cap",          "parquet:backtest_results/scheme_c_variant_roc_cap.parquet",    9.5),
    ("C+DTF-split",        "parquet:backtest_results/scheme_c_variant_dtf_split.parquet",  8.5),
    ("C+DTF-split",        "parquet:backtest_results/scheme_c_variant_dtf_split.parquet",  9.0),
    ("C+DTF-split",        "parquet:backtest_results/scheme_c_variant_dtf_split.parquet",  9.5),
    ("C+RS-dip",           "parquet:backtest_results/scheme_c_variant_rs_dip.parquet",     8.5),
    ("C+RS-dip",           "parquet:backtest_results/scheme_c_variant_rs_dip.parquet",     9.0),
    ("C+RS-dip",           "parquet:backtest_results/scheme_c_variant_rs_dip.parquet",     9.5),
    # All-fixes combined — max=10.88
    ("C+all",              "parquet:backtest_results/scheme_c_variant_all.parquet",        9.5),
    ("C+all",              "parquet:backtest_results/scheme_c_variant_all.parquet",       10.0),
    ("C+all",              "parquet:backtest_results/scheme_c_variant_all.parquet",       10.5),
]

results = []
for label, source, threshold in TESTS:
    print(f"running {label} @ T={threshold}...", end="", flush=True)
    r = run_variant(label, source, threshold)
    print(f" mean={r['path_mean']:+.1f}% std={r['path_std']:.1f}% Sharpe={r['primary_sharpe']:.2f}")
    results.append(r)

# ─── Print summary ────────────────────────────────────────────────
print("\n" + "=" * 130)
print("SCHEME C SINGLE-FIX VARIANTS — within-regime path-dep test")
print("=" * 130)
print(f"{'variant':<22} {'thr':>5}   "
      f"{'primary':>9} {'sharpe':>6} {'DD':>7} {'trades':>6}   "
      f"{'path mean':>10} {'path std':>9} {'path min':>9} {'path max':>9}")
print("-" * 130)
for r in results:
    print(f"{r['label']:<22} {r['threshold']:>5.1f}   "
          f"{r['primary_return']:>+8.1f}% {r['primary_sharpe']:>6.2f} "
          f"{r['primary_dd']:>+6.1f}% {r['primary_trades']:>6}   "
          f"{r['path_mean']:>+9.1f}% {r['path_std']:>+8.1f}% "
          f"{r['path_min']:>+8.1f}% {r['path_max']:>+8.1f}%")

# ─── Find peaks per variant ────────────────────────────────
print("\n" + "=" * 130)
print("BEST PATH-MEAN per variant (across threshold sweep)")
print("=" * 130)
by_label = {}
for r in results:
    by_label.setdefault(r["label"], []).append(r)
for label in by_label:
    best = max(by_label[label], key=lambda r: r["path_mean"])
    print(f"  {label:<22} best @ T={best['threshold']:.1f}: "
          f"path mean={best['path_mean']:+.1f}%  std={best['path_std']:.1f}%  "
          f"Sharpe={best['primary_sharpe']:.2f}")
