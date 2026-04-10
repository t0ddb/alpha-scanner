"""
portfolio_analysis.py — Follow-up analysis on portfolio backtest results.

Reads portfolio_trade_log.csv and portfolio_equity_curve.csv.
Runs 5 analyses:
  1. Realized vs Unrealized P&L
  2. AXTI Concentration Risk
  3. Monthly Return Breakdown
  4. Wash Sale Flag Audit
  5. Drawdown Analysis
"""

from __future__ import annotations
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

BASE = Path(__file__).parent
TRADES_CSV = BASE / "portfolio_trade_log.csv"
EQUITY_CSV = BASE / "portfolio_equity_curve.csv"


def load_data():
    trades = pd.read_csv(TRADES_CSV)
    trades["Entry Date"] = pd.to_datetime(trades["Entry Date"])
    trades["Exit Date"] = pd.to_datetime(trades["Exit Date"])
    equity = pd.read_csv(EQUITY_CSV)
    equity["DATE"] = pd.to_datetime(equity["DATE"])
    return trades, equity


# ─────────────────────────────────────────────────────────────
# Task 1: Realized vs Unrealized P&L
# ─────────────────────────────────────────────────────────────

def task1_realized_vs_unrealized(trades: pd.DataFrame):
    closed = trades[trades["Exit Reason"] != "end_of_backtest"]
    open_t = trades[trades["Exit Reason"] == "end_of_backtest"]

    total_pnl = trades["P&L ($)"].sum()
    closed_pnl = closed["P&L ($)"].sum()
    open_pnl = open_t["P&L ($)"].sum()

    total_cost = trades["Cost Basis"].sum()
    closed_cost = closed["Cost Basis"].sum()
    open_cost = open_t["Cost Basis"].sum()

    closed_pct = (closed_pnl / closed_cost * 100) if closed_cost else 0
    open_pct = (open_pnl / open_cost * 100) if open_cost else 0
    total_pct = (total_pnl / total_cost * 100) if total_cost else 0

    closed_wr = (closed["P&L ($)"] > 0).mean() * 100 if len(closed) else 0
    open_wr = (open_t["P&L ($)"] > 0).mean() * 100 if len(open_t) else 0
    total_wr = (trades["P&L ($)"] > 0).mean() * 100

    closed_avg = closed["P&L ($)"].mean() if len(closed) else 0
    open_avg = open_t["P&L ($)"].mean() if len(open_t) else 0

    print("\n" + "=" * 80)
    print("  TASK 1: REALIZED vs UNREALIZED P&L")
    print("=" * 80)
    print()
    hdr = f"  {'':24s} {'Closed Trades':>14s} {'Open Trades':>14s} {'Total':>14s}"
    print(hdr)
    print("  " + "─" * 70)
    print(f"  {'Trades':<24s} {len(closed):>14d} {len(open_t):>14d} {len(trades):>14d}")
    print(f"  {'Total P&L ($)':<24s} {'${:>,.0f}'.format(closed_pnl):>14s} {'${:>,.0f}'.format(open_pnl):>14s} {'${:>,.0f}'.format(total_pnl):>14s}")
    print(f"  {'Total P&L (%)':<24s} {'+{:.1f}%'.format(closed_pct):>14s} {'+{:.1f}%'.format(open_pct):>14s} {'+{:.1f}%'.format(total_pct):>14s}")
    print(f"  {'Win Rate':<24s} {'{:.1f}%'.format(closed_wr):>14s} {'{:.1f}%'.format(open_wr):>14s} {'{:.1f}%'.format(total_wr):>14s}")
    print(f"  {'Avg P&L / Trade':<24s} {'${:>,.0f}'.format(closed_avg):>14s} {'${:>,.0f}'.format(open_avg):>14s} {'—':>14s}")
    print("  " + "─" * 70)
    print()
    banked_pct = closed_pnl / total_pnl * 100 if total_pnl else 0
    at_risk_pct = open_pnl / total_pnl * 100 if total_pnl else 0
    print(f"  Banked (realized):   {banked_pct:.1f}% of total gains")
    print(f"  At risk (unrealized): {at_risk_pct:.1f}% of total gains")


# ─────────────────────────────────────────────────────────────
# Task 2: AXTI Concentration Risk
# ─────────────────────────────────────────────────────────────

def task2_concentration_risk(trades: pd.DataFrame):
    axti_trades = trades[trades["Ticker"] == "AXTI"]
    no_axti = trades[trades["Ticker"] != "AXTI"]

    total_pnl = trades["P&L ($)"].sum()
    no_axti_pnl = no_axti["P&L ($)"].sum()
    axti_pnl = axti_trades["P&L ($)"].sum()

    total_cost = trades["Cost Basis"].sum()
    no_axti_cost = no_axti["Cost Basis"].sum()

    total_ret = total_pnl / 100000 * 100  # vs starting capital
    no_axti_ret = no_axti_pnl / 100000 * 100

    total_wr = (trades["P&L ($)"] > 0).mean() * 100
    no_axti_wr = (no_axti["P&L ($)"] > 0).mean() * 100

    total_avg = trades["P&L ($)"].mean()
    no_axti_avg = no_axti["P&L ($)"].mean()

    # Profit factor
    total_gains = trades[trades["P&L ($)"] > 0]["P&L ($)"].sum()
    total_losses = abs(trades[trades["P&L ($)"] < 0]["P&L ($)"].sum())
    total_pf = total_gains / total_losses if total_losses else float("inf")

    no_axti_gains = no_axti[no_axti["P&L ($)"] > 0]["P&L ($)"].sum()
    no_axti_losses = abs(no_axti[no_axti["P&L ($)"] < 0]["P&L ($)"].sum())
    no_axti_pf = no_axti_gains / no_axti_losses if no_axti_losses else float("inf")

    print("\n" + "=" * 80)
    print("  TASK 2: CONCENTRATION RISK — AXTI IMPACT")
    print("=" * 80)
    print()
    hdr = f"  {'':24s} {'With AXTI':>14s} {'Without AXTI':>14s} {'Difference':>14s}"
    print(hdr)
    print("  " + "─" * 70)
    print(f"  {'Total Return':<24s} {'+{:.1f}%'.format(total_ret):>14s} {'+{:.1f}%'.format(no_axti_ret):>14s} {'-{:.1f}%'.format(total_ret - no_axti_ret):>14s}")
    print(f"  {'Total P&L ($)':<24s} {'${:>,.0f}'.format(total_pnl):>14s} {'${:>,.0f}'.format(no_axti_pnl):>14s} {'-${:>,.0f}'.format(axti_pnl):>14s}")
    print(f"  {'Win Rate':<24s} {'{:.1f}%'.format(total_wr):>14s} {'{:.1f}%'.format(no_axti_wr):>14s} {'—':>14s}")
    print(f"  {'Avg Gain / Trade':<24s} {'${:>,.0f}'.format(total_avg):>14s} {'${:>,.0f}'.format(no_axti_avg):>14s} {'—':>14s}")
    print(f"  {'Profit Factor':<24s} {'{:.2f}x'.format(total_pf):>14s} {'{:.2f}x'.format(no_axti_pf):>14s} {'—':>14s}")
    print("  " + "─" * 70)

    # Top 5 trades by P&L contribution
    print()
    print("  Top 5 Trades by P&L Contribution:")
    print("  " + "─" * 70)
    print(f"  {'Ticker':<10s} {'Entry Date':<12s} {'P&L ($)':>12s} {'P&L (%)':>10s} {'% of Gains':>12s}")
    print("  " + "─" * 70)

    winners = trades[trades["P&L ($)"] > 0].sort_values("P&L ($)", ascending=False)
    for _, row in winners.head(5).iterrows():
        pct_of_gains = row["P&L ($)"] / total_gains * 100
        print(f"  {row['Ticker']:<10s} {row['Entry Date'].strftime('%Y-%m-%d'):<12s} "
              f"{'${:>,.0f}'.format(row['P&L ($)']):>12s} "
              f"{'+{:.1f}%'.format(row['P&L ($)'] / row['Cost Basis'] * 100):>10s} "
              f"{'{:.1f}%'.format(pct_of_gains):>12s}")

    top5_total = winners.head(5)["P&L ($)"].sum()
    print("  " + "─" * 70)
    print(f"  {'Top 5 total':<22s} {'${:>,.0f}'.format(top5_total):>12s} {'':>10s} "
          f"{'{:.1f}%'.format(top5_total / total_gains * 100):>12s}")


# ─────────────────────────────────────────────────────────────
# Task 3: Monthly Return Breakdown
# ─────────────────────────────────────────────────────────────

def task3_monthly_returns(trades: pd.DataFrame, equity: pd.DataFrame):
    equity["MONTH"] = equity["DATE"].dt.to_period("M")

    # Get first and last value per month
    monthly = equity.groupby("MONTH").agg(
        start_val=("PORTFOLIO_VALUE", "first"),
        end_val=("PORTFOLIO_VALUE", "last"),
        spy_start=("SPY_VALUE", "first"),
        spy_end=("SPY_VALUE", "last"),
    )
    monthly["port_ret"] = (monthly["end_val"] / monthly["start_val"] - 1) * 100
    monthly["spy_ret"] = (monthly["spy_end"] / monthly["spy_start"] - 1) * 100
    monthly["alpha"] = monthly["port_ret"] - monthly["spy_ret"]

    # Count trades opened/closed per month
    trades["entry_month"] = trades["Entry Date"].dt.to_period("M")
    trades["exit_month"] = trades["Exit Date"].dt.to_period("M")
    opened_per_month = trades.groupby("entry_month").size()
    closed_per_month = trades[trades["Exit Reason"] != "end_of_backtest"].groupby("exit_month").size()

    # Only show months where portfolio was active (had positions or trades)
    first_trade_month = trades["entry_month"].min()
    monthly = monthly[monthly.index >= first_trade_month]

    print("\n" + "=" * 80)
    print("  TASK 3: MONTHLY RETURNS")
    print("=" * 80)
    print()
    print(f"  {'Month':<12s} {'Portfolio':>10s} {'SPY':>10s} {'Alpha':>10s} {'Opened':>10s} {'Closed':>10s}")
    print("  " + "─" * 65)

    profitable_months = 0
    best_month = (None, -999)
    worst_month = (None, 999)

    for period, row in monthly.iterrows():
        mo_str = str(period)
        opened = opened_per_month.get(period, 0)
        closed = closed_per_month.get(period, 0)

        p_sign = "+" if row["port_ret"] >= 0 else ""
        s_sign = "+" if row["spy_ret"] >= 0 else ""
        a_sign = "+" if row["alpha"] >= 0 else ""

        print(f"  {mo_str:<12s} {p_sign}{row['port_ret']:.1f}%{'':<4s} "
              f"{s_sign}{row['spy_ret']:.1f}%{'':<4s} "
              f"{a_sign}{row['alpha']:.1f}%{'':<4s} "
              f"{opened:>6d}       {closed:>5d}")

        if row["port_ret"] > 0:
            profitable_months += 1
        if row["port_ret"] > best_month[1]:
            best_month = (mo_str, row["port_ret"])
        if row["port_ret"] < worst_month[1]:
            worst_month = (mo_str, row["port_ret"])

    total_months = len(monthly)
    print("  " + "─" * 65)
    print(f"  Best Month:        {best_month[0]} (+{best_month[1]:.1f}%)")
    print(f"  Worst Month:       {worst_month[0]} ({worst_month[1]:.1f}%)")
    print(f"  Profitable Months: {profitable_months} of {total_months} ({profitable_months/total_months*100:.1f}%)")


# ─────────────────────────────────────────────────────────────
# Task 4: Wash Sale Flag Audit
# ─────────────────────────────────────────────────────────────

def task4_wash_sale_audit(trades: pd.DataFrame):
    loss_trades = trades[trades["P&L ($)"] < 0].copy()

    flagged = []
    for _, loss in loss_trades.iterrows():
        ticker = loss["Ticker"]
        exit_date = loss["Exit Date"]

        # Look for re-entries of same ticker within 30 calendar days
        reentries = trades[
            (trades["Ticker"] == ticker)
            & (trades["Entry Date"] > exit_date)
            & (trades["Entry Date"] <= exit_date + timedelta(days=30))
        ]

        # Also check for same-day re-entry (entry date == exit date)
        same_day = trades[
            (trades["Ticker"] == ticker)
            & (trades["Entry Date"] == exit_date)
            & (trades.index != loss.name)  # not the same trade
        ]
        reentries = pd.concat([reentries, same_day]).drop_duplicates()

        if len(reentries) > 0:
            for _, re in reentries.iterrows():
                days_between = (re["Entry Date"] - exit_date).days
                flagged.append({
                    "Ticker": ticker,
                    "Loss Exit Date": exit_date.strftime("%Y-%m-%d"),
                    "Loss Amount": loss["P&L ($)"],
                    "Re-Entry Date": re["Entry Date"].strftime("%Y-%m-%d"),
                    "Days Between": days_between,
                })

    print("\n" + "=" * 80)
    print("  TASK 4: WASH SALE AUDIT")
    print("=" * 80)
    print()
    print(f"  Total loss exits:         {len(loss_trades)}")
    print(f"  Wash sale violations:     {len(flagged)}")
    print()

    if flagged:
        print("  Flagged Trades:")
        print("  " + "─" * 75)
        print(f"  {'Ticker':<10s} {'Loss Exit Date':<16s} {'Loss Amount':>12s} "
              f"{'Re-Entry Date':<16s} {'Days Between':>14s}")
        print("  " + "─" * 75)

        total_disallowed = 0
        for f in flagged:
            total_disallowed += abs(f["Loss Amount"])
            print(f"  {f['Ticker']:<10s} {f['Loss Exit Date']:<16s} "
                  f"{'${:>,.0f}'.format(abs(f['Loss Amount'])):>12s} "
                  f"{f['Re-Entry Date']:<16s} "
                  f"{f['Days Between']:>10d} days")

        print("  " + "─" * 75)
        print(f"\n  Total disallowed losses:  ${total_disallowed:,.0f}")
        print(f"\n  Tax impact: These losses cannot be claimed on taxes in the year")
        print(f"  they occurred. Instead, the disallowed loss is added to the cost")
        print(f"  basis of the replacement shares, deferring (not eliminating) the")
        print(f"  tax benefit until the replacement position is eventually sold.")
    else:
        print("  No wash sale violations found.")


# ─────────────────────────────────────────────────────────────
# Task 5: Drawdown Analysis
# ─────────────────────────────────────────────────────────────

def task5_drawdown_analysis(equity: pd.DataFrame):
    vals = equity[["DATE", "PORTFOLIO_VALUE"]].copy()
    vals["peak"] = vals["PORTFOLIO_VALUE"].cummax()
    vals["drawdown"] = (vals["PORTFOLIO_VALUE"] / vals["peak"] - 1) * 100

    # Max drawdown
    max_dd_idx = vals["drawdown"].idxmin()
    max_dd = vals.loc[max_dd_idx, "drawdown"]
    trough_date = vals.loc[max_dd_idx, "DATE"]
    trough_val = vals.loc[max_dd_idx, "PORTFOLIO_VALUE"]

    # Find peak before the trough
    peak_before = vals.loc[:max_dd_idx]
    peak_idx = peak_before["PORTFOLIO_VALUE"].idxmax()
    peak_date = vals.loc[peak_idx, "DATE"]
    peak_val = vals.loc[peak_idx, "PORTFOLIO_VALUE"]

    # Find recovery after trough
    after_trough = vals.loc[max_dd_idx:]
    recovered = after_trough[after_trough["PORTFOLIO_VALUE"] >= peak_val]
    if len(recovered) > 0:
        recovery_date = recovered.iloc[0]["DATE"]
        recovery_str = recovery_date.strftime("%Y-%m-%d")
        recovery_days = (recovery_date - trough_date).days
    else:
        recovery_str = "not yet recovered"
        recovery_days = None

    peak_to_trough_days = (trough_date - peak_date).days

    # Find top 3 drawdowns (separate episodes)
    # An episode: drawdown starts when we go below 0, ends when we recover to 0
    episodes = []
    in_dd = False
    ep_start = None
    ep_trough = 0
    ep_trough_date = None
    ep_peak_val = None
    ep_peak_date = None

    for i, row in vals.iterrows():
        dd = row["drawdown"]
        if dd < -1 and not in_dd:  # >1% drawdown to filter noise
            in_dd = True
            ep_start = i
            ep_peak_val = row["peak"]
            # Find the date of this peak
            peak_rows = vals[vals["PORTFOLIO_VALUE"] == ep_peak_val]
            ep_peak_date = peak_rows.iloc[0]["DATE"] if len(peak_rows) > 0 else row["DATE"]
            ep_trough = dd
            ep_trough_date = row["DATE"]
        elif in_dd:
            if dd < ep_trough:
                ep_trough = dd
                ep_trough_date = row["DATE"]
            if dd >= -0.5:  # recovered (within 0.5%)
                in_dd = False
                episodes.append({
                    "magnitude": ep_trough,
                    "peak_date": ep_peak_date,
                    "trough_date": ep_trough_date,
                    "peak_to_trough": (ep_trough_date - ep_peak_date).days,
                    "recovery_date": row["DATE"],
                    "trough_to_recovery": (row["DATE"] - ep_trough_date).days,
                })

    # If still in drawdown at end, capture it
    if in_dd:
        episodes.append({
            "magnitude": ep_trough,
            "peak_date": ep_peak_date,
            "trough_date": ep_trough_date,
            "peak_to_trough": (ep_trough_date - ep_peak_date).days,
            "recovery_date": None,
            "trough_to_recovery": None,
        })

    episodes.sort(key=lambda x: x["magnitude"])
    top3 = episodes[:3]

    # Time spent in drawdown
    total_days = len(vals)
    dd_gt5 = (vals["drawdown"] < -5).sum()
    dd_gt10 = (vals["drawdown"] < -10).sum()
    dd_gt20 = (vals["drawdown"] < -20).sum()

    print("\n" + "=" * 80)
    print("  TASK 5: DRAWDOWN ANALYSIS")
    print("=" * 80)
    print()
    print(f"  Max Drawdown:     {max_dd:.1f}%")
    print(f"  Peak Date:        {peak_date.strftime('%Y-%m-%d')} (${peak_val:,.0f})")
    print(f"  Trough Date:      {trough_date.strftime('%Y-%m-%d')} (${trough_val:,.0f})")
    print(f"  Recovery Date:    {recovery_str}")
    print(f"  Duration:         {peak_to_trough_days} days peak-to-trough", end="")
    if recovery_days is not None:
        print(f", {recovery_days} days trough-to-recovery")
    else:
        print()

    print()
    print("  Top 3 Drawdowns:")
    print("  " + "─" * 70)
    print(f"  {'#':<4s} {'Magnitude':>10s} {'Peak Date':>12s} {'Trough Date':>13s} {'Duration':>10s} {'Recovery':>12s}")
    print("  " + "─" * 70)
    for i, ep in enumerate(top3, 1):
        p2t = f"{ep['peak_to_trough']}d"
        rec = f"{ep['trough_to_recovery']}d" if ep["trough_to_recovery"] is not None else "ongoing"
        print(f"  {i:<4d} {ep['magnitude']:>9.1f}%  {ep['peak_date'].strftime('%Y-%m-%d'):>12s} "
              f"{ep['trough_date'].strftime('%Y-%m-%d'):>13s} {p2t:>10s} {rec:>12s}")

    print()
    print("  Time Spent in Drawdown:")
    print("  " + "─" * 40)
    print(f"  > -5%:   {dd_gt5} days ({dd_gt5/total_days*100:.1f}% of backtest)")
    print(f"  > -10%:  {dd_gt10} days ({dd_gt10/total_days*100:.1f}% of backtest)")
    print(f"  > -20%:  {dd_gt20} days ({dd_gt20/total_days*100:.1f}% of backtest)")


# ─────────────────────────────────────────────────────────────

def main():
    print()
    print("=" * 80)
    print("  PORTFOLIO BACKTEST — FOLLOW-UP ANALYSIS")
    print("  Alpha Scanner")
    print("=" * 80)

    trades, equity = load_data()

    task1_realized_vs_unrealized(trades)
    task2_concentration_risk(trades)
    task3_monthly_returns(trades, equity)
    task4_wash_sale_audit(trades)
    task5_drawdown_analysis(equity)

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
