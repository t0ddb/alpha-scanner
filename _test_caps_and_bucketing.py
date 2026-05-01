"""Custom test grid: Scheme C and Path C at multiple position caps,
plus a coarse-bucketed Path C variant. All tested with 10-start path-dep.

Goal: determine whether (a) raising the position cap improves Path C,
(b) Scheme C also benefits from higher caps, and (c) coarser bucketing
on Path C reduces path-dep without sacrificing return.
"""
import sys
sys.path.insert(0, ".")
from sizing_comparison_backtest import (
    load_score_data, build_daily_scores, get_trading_days,
    PortfolioSimulator, StrategyConfig, StopLossConfig, compute_metrics,
    compute_path_start_dates,
)
from config import load_config
from data_fetcher import fetch_all
import pandas as pd
import numpy as np

print("loading...")
cfg_yaml = load_config()
price_data = fetch_all(cfg_yaml, period="2y", verbose=False)
trading_days = get_trading_days(price_data)


def run_path_dep(score_source: str, label: str, cap: int, threshold: float,
                 start_filter: str = "2025-05-01", n_paths: int = 10):
    """Run primary + n_paths-start path-dep test for one config.
    Returns dict with mean/std/range across paths and primary metrics."""
    score_df = load_score_data(score_source)
    daily_scores = build_daily_scores(score_df, trading_days)

    sizing_pct = round(1.0 / cap, 4)
    strat = StrategyConfig(
        name=f"{label}-cap{cap}-T{threshold}",
        max_positions=cap,
        sizing_mode="fixed_pct",
        fixed_position_pct=sizing_pct,
        min_entry_pct=0.05,
        trim_enabled=False,
        entry_protection_days=7,
        entry_threshold=threshold,
        exit_threshold=5.0,
        stop_loss=StopLossConfig(type="fixed", value=0.20),
        persistence_days=3,
    )

    # Primary run starting at start_filter
    sim = PortfolioSimulator(
        config=strat, daily_scores=daily_scores,
        price_data=price_data, trading_days=trading_days,
        start_date=start_filter,
    )
    primary = compute_metrics(sim.run())

    # Path dep — use 10 fixed dates spanning ~9 weeks within regime
    candidates = [
        "2025-04-08", "2025-04-15", "2025-04-22", "2025-04-29",
        "2025-05-06", "2025-05-13", "2025-05-20", "2025-05-27",
        "2025-06-03", "2025-06-10",
    ][:n_paths]
    path_returns = []
    for start in candidates:
        psim = PortfolioSimulator(
            config=strat, daily_scores=daily_scores,
            price_data=price_data, trading_days=trading_days,
            start_date=start,
        )
        pm = compute_metrics(psim.run())
        path_returns.append(pm["total_return"])

    return {
        "label": label,
        "cap": cap,
        "threshold": threshold,
        "primary_return": primary["total_return"],
        "primary_sharpe": primary["sharpe"],
        "primary_dd": primary["max_drawdown"],
        "primary_winrate": primary["win_rate"],
        "primary_trades": primary["total_trades"],
        "path_n": len(path_returns),
        "path_mean": float(np.mean(path_returns)) if path_returns else 0.0,
        "path_std": float(np.std(path_returns)) if len(path_returns) > 1 else 0.0,
        "path_min": float(min(path_returns)) if path_returns else 0.0,
        "path_max": float(max(path_returns)) if path_returns else 0.0,
    }


# ─── Step 1: produce coarse-bucketed Path C scores ────────────────
def make_coarse_pathc():
    print("creating coarse-bucketed Path C parquet...")
    sip = pd.read_parquet("backtest_results/scheme_i_plus_v2_pathC.parquet")
    # Round final score to 0.5 increments to restore ties
    sip["score"] = (sip["score"] / 0.5).round() * 0.5
    out_path = "backtest_results/scheme_i_plus_v2_pathC_coarse05.parquet"
    sip[["date", "ticker", "score"]].to_parquet(out_path, index=False)
    print(f"  wrote {out_path}")
    print(f"  unique scores: {sip['score'].nunique()}")
    return out_path


coarse_path = make_coarse_pathc()

# ─── Step 2: run grid ─────────────────────────────────────────────
configs = [
    # Scheme C across caps
    ("sqlite",                                                    "Scheme C",   12, 9.0),
    ("sqlite",                                                    "Scheme C",   15, 9.0),
    ("sqlite",                                                    "Scheme C",   20, 9.0),
    ("sqlite",                                                    "Scheme C",   12, 8.0),
    ("sqlite",                                                    "Scheme C",   15, 8.0),
    ("sqlite",                                                    "Scheme C",   20, 8.0),

    # Path C across caps
    ("parquet:backtest_results/scheme_i_plus_v2_pathC.parquet",   "Path C",     12, 7.5),
    ("parquet:backtest_results/scheme_i_plus_v2_pathC.parquet",   "Path C",     15, 7.5),
    ("parquet:backtest_results/scheme_i_plus_v2_pathC.parquet",   "Path C",     20, 7.5),

    # Coarse-bucketed Path C at cap 12 (test if rounding restores stability)
    ("parquet:" + coarse_path,                                    "Path C coarse",  12, 7.5),
    ("parquet:" + coarse_path,                                    "Path C coarse",  15, 7.5),
    ("parquet:" + coarse_path,                                    "Path C coarse",  20, 7.5),
]

results = []
for score_source, label, cap, thr in configs:
    print(f"\nrunning {label} cap={cap} threshold={thr}...", end="", flush=True)
    r = run_path_dep(score_source, label, cap, thr)
    print(f" primary={r['primary_return']:+.1f}% path-mean={r['path_mean']:+.1f}% std={r['path_std']:.1f}%")
    results.append(r)

# ─── Step 3: print summary table ───────────────────────────────
print("\n" + "=" * 130)
print("SUMMARY TABLE")
print("=" * 130)
print(f"{'config':<22} {'cap':>4} {'thr':>5}   "
      f"{'primary':>9} {'sharpe':>6} {'DD':>7} {'trades':>6}   "
      f"{'path mean':>10} {'path std':>9} {'path min':>9} {'path max':>9}")
print("-" * 130)
for r in results:
    print(f"{r['label']:<22} {r['cap']:>4} {r['threshold']:>5.1f}   "
          f"{r['primary_return']:>+8.1f}% {r['primary_sharpe']:>6.2f} "
          f"{r['primary_dd']:>+6.1f}% {r['primary_trades']:>6}   "
          f"{r['path_mean']:>+9.1f}% {r['path_std']:>+8.1f}% "
          f"{r['path_min']:>+8.1f}% {r['path_max']:>+8.1f}%")
