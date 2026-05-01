"""Quick diagnostic: run Scheme C @ threshold 9.0 at two different start dates,
log all entries with timing, and compare to verify the path-dep finding."""
import sys
sys.path.insert(0, ".")
from sizing_comparison_backtest import (
    load_score_data, build_daily_scores, get_trading_days,
    PortfolioSimulator, StrategyConfig, StopLossConfig, compute_metrics,
    PERSISTENCE_DAYS,
)
from config import load_config
from data_fetcher import fetch_all
import pandas as pd

cfg_yaml = load_config()
print("loading data...")
score_df = load_score_data("sqlite")
price_data = fetch_all(cfg_yaml, period="2y", verbose=False)
trading_days = get_trading_days(price_data)
daily_scores = build_daily_scores(score_df, trading_days)

strat = StrategyConfig(
    name="SchemeC-9.0-skipwhenfull",
    max_positions=12,
    sizing_mode="fixed_pct",
    fixed_position_pct=0.083,
    min_entry_pct=0.05,
    trim_enabled=False,
    entry_protection_days=7,
    entry_threshold=9.0,
    exit_threshold=5.0,
    stop_loss=StopLossConfig(type="fixed", value=0.20),
    persistence_days=3,
)

# Two test start dates ~2 months apart
for start in ["2025-04-08", "2025-06-10"]:
    sim = PortfolioSimulator(
        config=strat,
        daily_scores=daily_scores,
        price_data=price_data,
        trading_days=trading_days,
        start_date=start,
    )
    res = sim.run()
    metrics = compute_metrics(res)
    trades = res.trades
    print(f"\n========== START {start} ==========")
    print(f"Total return: {metrics['total_return']:+.1f}%")
    print(f"Total trades: {len(trades)}")
    print(f"Signals skipped (cap or other): {metrics['signals_skipped']}")
    print(f"\nFirst 20 entries by date:")
    for t in trades[:20]:
        print(f"  {t.entry_date}: BUY  {t.ticker:<6} score={t.entry_score:.1f}")
    print(f"\nLast 5 trades:")
    for t in trades[-5:]:
        print(f"  {t.entry_date} → {t.exit_date}: {t.ticker:<6} "
              f"P&L {t.pnl_pct*100:+.1f}%  reason={t.exit_reason}")
    # Tally tickers ever held
    tickers_held = set(t.ticker for t in trades)
    print(f"\nUnique tickers traded: {len(tickers_held)}")
    print(f"  list: {sorted(tickers_held)}")
