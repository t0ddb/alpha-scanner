"""Check AXTI's treatment in both configs."""
from portfolio_backtest import (
    load_score_data, build_daily_scores, get_trading_days, run_simulation,
)
from data_fetcher import fetch_all
from config import load_config

score_df = load_score_data()
cfg = load_config()
price_data = fetch_all(cfg, period="5y", verbose=False)
trading_days = get_trading_days(price_data)
daily_scores = build_daily_scores(score_df, trading_days)

for label, thresh in [("8.5/<5", 8.5), ("9.5/<5", 9.5)]:
    res = run_simulation(
        daily_scores, price_data, trading_days,
        threshold=thresh, score_exit_threshold=5.0,
        rotation_strategy="none", persistence_filter=False,
        starting_capital=100_000, max_position_pct=0.20, stop_loss_pct=0.15,
    )
    axti_trades = [t for t in res["trades"] if t.ticker == "AXTI"]
    print(f"\n{label}: {res['total_trades']} total trades, {res['total_return']:+.1f}% return")
    print(f"  AXTI trades: {len(axti_trades)}")
    for t in axti_trades:
        print(f"    entry {t.entry_date} @ ${t.entry_price:.2f} (score {t.entry_score}) -> "
              f"exit {t.exit_date} @ ${t.exit_price:.2f} [{t.exit_reason}] "
              f"P&L ${t.pnl:+,.0f} ({t.pnl_pct*100:+.1f}%) hold {t.hold_days}d")

    # Show total excluding AXTI
    non_axti = [t for t in res["trades"] if t.ticker != "AXTI"]
    axti_pnl = sum(t.pnl for t in axti_trades)
    non_axti_pnl = sum(t.pnl for t in non_axti)
    total_pnl = sum(t.pnl for t in res["trades"])
    print(f"  Total P&L: ${total_pnl:+,.0f}")
    print(f"    AXTI contribution:    ${axti_pnl:+,.0f}  ({axti_pnl/total_pnl*100:+.1f}%)")
    print(f"    Non-AXTI contribution: ${non_axti_pnl:+,.0f}  ({non_axti_pnl/total_pnl*100:+.1f}%)")
