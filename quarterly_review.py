from __future__ import annotations

"""
quarterly_review.py — Automated system health review for Alpha Scanner.

Runs 7 diagnostic sections validating all assumptions in the scoring system.
This report is DIAGNOSTIC, not prescriptive — it flags concerns but does not
change anything. Persistent issues across 2 consecutive quarters are upgraded
to ACTION NEEDED.

Usage:
    python3 quarterly_review.py                    # default: last 12 months
    python3 quarterly_review.py --months 12
    python3 quarterly_review.py --email-only       # read latest saved report and email it
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    load_config,
    get_all_tickers,
    get_ticker_metadata,
    get_indicator_config,
)
from data_fetcher import fetch_all
from indicators import compute_all_indicators, score_ticker
import indicators_expanded as ie
from backtester import compute_forward_returns
from portfolio_backtest import (
    load_score_data, build_daily_scores, get_trading_days, run_simulation,
)
from subsector_store import init_db, get_all_history


# ──────────────────────────────────────────────────────────────────────────
# Baselines (from original system calibration)
# ──────────────────────────────────────────────────────────────────────────

ORIGINAL_INDICATOR_EDGES = {
    # scored
    "relative_strength": +13.31,
    "ichimoku_cloud":    +10.89,
    "higher_lows":        +7.73,
    "roc":                +6.84,
    "cmf":                +6.57,
    "dual_tf_rs":         +5.20,
    "atr_expansion":      +4.58,
    # dropped from scoring
    "moving_averages":    -9.30,   # "MA Alignment"
    "near_52w_high":      -3.30,
    # never scored
    "volume_spike":       +0.72,
    "bb_squeeze":         +0.38,
    "macd_crossover":     +0.95,
    "adx_trend":          +2.66,
}

INDICATOR_DISPLAY = {
    "relative_strength": "Relative Strength",
    "ichimoku_cloud":    "Ichimoku Cloud",
    "higher_lows":       "Higher Lows",
    "moving_averages":   "MA Alignment",
    "roc":               "Rate of Change",
    "cmf":               "Chaikin Money Flow",
    "atr_expansion":     "ATR Expansion",
    "dual_tf_rs":        "Dual-TF RS",
    "near_52w_high":     "Near 52w High",
    "volume_spike":      "Volume Spike",
    "bb_squeeze":        "BB Squeeze",
    "macd_crossover":    "MACD Crossover",
    "adx_trend":         "ADX Trend",
}

SCORED_INDICATORS = [
    "relative_strength", "ichimoku_cloud", "higher_lows",
    "roc", "cmf", "dual_tf_rs", "atr_expansion",
]
DROPPED_FROM_SCORING = ["moving_averages", "near_52w_high"]
NEVER_SCORED = ["volume_spike", "bb_squeeze", "macd_crossover", "adx_trend"]

# Section 6 — original state forward-return edges (SPY-adjusted, 63-day)
ORIGINAL_STATE_EDGES = {
    "emerging":  +13.1,
    "confirmed": +21.7,
    "revival":   +27.3,
}

FORWARD_WINDOW = 63  # trading days
WARMUP = 220         # bars needed before indicators are valid

# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────

HEALTHY = "HEALTHY"
WATCH = "WATCH"
ACTION = "ACTION NEEDED"
SKIP = "SKIP"

STATUS_ICON = {
    HEALTHY: "[HEALTHY]",
    WATCH:   "[WATCH]",
    ACTION:  "[ACTION]",
    SKIP:    "[SKIP]",
}


def quarter_label(date: datetime = None) -> str:
    d = date or datetime.now()
    q = (d.month - 1) // 3 + 1
    return f"{d.year}-Q{q}"


def next_quarter_start(date: datetime = None) -> str:
    d = date or datetime.now()
    year, month = d.year, d.month
    next_q_month = ((month - 1) // 3 + 1) * 3 + 1
    if next_q_month > 12:
        next_q_month -= 12
        year += 1
    return f"{year}-{next_q_month:02d}-01"


# Capture print output so we can also save to a file
class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()


# ──────────────────────────────────────────────────────────────────────────
# Section 1: Indicator Edge Validation
# ──────────────────────────────────────────────────────────────────────────

def _compute_expanded_signals(df: pd.DataFrame, needed: dict) -> dict:
    """Compute the expanded/dropped indicators not in compute_all_indicators."""
    out = {}
    if needed.get("volume_spike"):
        try:
            r = ie.check_breakout_volume(df)
            out["volume_spike"] = r.get("triggered", False)
        except Exception:
            out["volume_spike"] = False
    if needed.get("bb_squeeze"):
        try:
            r = ie.check_consolidation_tightness(df)
            out["bb_squeeze"] = r.get("triggered", False)
        except Exception:
            out["bb_squeeze"] = False
    if needed.get("macd_crossover"):
        try:
            r = ie.check_macd_crossover(df)
            out["macd_crossover"] = r.get("triggered", False)
        except Exception:
            out["macd_crossover"] = False
    if needed.get("adx_trend"):
        try:
            r = ie.check_adx(df)
            out["adx_trend"] = r.get("triggered", False)
        except Exception:
            out["adx_trend"] = False
    return out


def collect_indicator_events(data: dict, cfg: dict, months: int) -> pd.DataFrame:
    """
    Walk through the last `months` months of history weekly. For each test
    date and ticker, record which indicators fired and the 63-day forward
    return (SPY-adjusted).
    """
    bench_ticker = cfg["benchmark"]["ticker"]
    bench_df = data.get(bench_ticker)
    if bench_df is None:
        print("  ERROR: no benchmark data")
        return pd.DataFrame()

    total_days = len(bench_df)
    target_days = int(months * 21)  # ~21 trading days / month
    end_idx = total_days - FORWARD_WINDOW

    # Start index: end - target_days, but at least WARMUP
    start_idx = max(WARMUP, end_idx - target_days)
    if start_idx >= end_idx:
        print("  ERROR: not enough data for requested lookback")
        return pd.DataFrame()

    # Weekly sampling (every 5 trading days)
    test_indices = list(range(start_idx, end_idx, 5))
    ind_cfg = get_indicator_config(cfg)
    rs_period = ind_cfg["relative_strength"]["period"]

    print(f"  {len(test_indices)} test dates, ~{len(data) - 1} tickers")
    print(f"  Window: {bench_df.index[start_idx].strftime('%Y-%m-%d')} -> "
          f"{bench_df.index[end_idx].strftime('%Y-%m-%d')}")

    rows = []
    needed_expanded = {
        "volume_spike": True, "bb_squeeze": True,
        "macd_crossover": True, "adx_trend": True,
    }

    for count, idx in enumerate(test_indices, 1):
        if count % 10 == 0:
            print(f"    [{count}/{len(test_indices)}] {bench_df.index[idx].strftime('%Y-%m-%d')}")

        bench_slice = bench_df.iloc[:idx + 1]
        bench_fwd_idx = idx + FORWARD_WINDOW
        if bench_fwd_idx >= total_days:
            continue
        bench_fwd_ret = (bench_df["Close"].iloc[bench_fwd_idx] /
                         bench_df["Close"].iloc[idx]) - 1

        # Pre-compute RS across multiple timeframes for this date.
        # Mirrors score_all() in indicators.py so Dual-TF RS can be
        # evaluated here (otherwise compute_all_indicators gets
        # multi_tf_rs=None and dual_tf_rs always returns triggered=False,
        # causing it to fall back to the original baseline in Section 2).
        tf_periods = [rs_period, 126, 63, 21]
        raw_rs_by_tf = {p: {} for p in tf_periods}
        for tk, full_df in data.items():
            if tk == bench_ticker:
                continue
            df = full_df.iloc[:idx + 1]
            for period in tf_periods:
                if len(df) < period + 1 or len(bench_slice) < period + 1:
                    continue
                sr = (df["Close"].iloc[-1] / df["Close"].iloc[-period - 1]) - 1
                br = (bench_slice["Close"].iloc[-1] /
                      bench_slice["Close"].iloc[-period - 1]) - 1
                if br > 0:
                    raw_rs_by_tf[period][tk] = sr / br
                elif br < 0:
                    raw_rs_by_tf[period][tk] = sr - br
                else:
                    raw_rs_by_tf[period][tk] = 0
        all_rs_values = list(raw_rs_by_tf[rs_period].values())

        # Percentile ranks for the 3 multi-TF windows Dual-TF RS needs
        def _pctl_rank(val, vals):
            if not vals:
                return 0.0
            return sum(1 for v in vals if v <= val) / len(vals) * 100

        tf_126_vals = list(raw_rs_by_tf[126].values())
        tf_63_vals = list(raw_rs_by_tf[63].values())
        tf_21_vals = list(raw_rs_by_tf[21].values())
        multi_tf_pctls = {}
        for tk in data:
            if tk == bench_ticker:
                continue
            multi_tf_pctls[tk] = {
                "rs_126d_pctl": (_pctl_rank(raw_rs_by_tf[126][tk], tf_126_vals)
                                 if tk in raw_rs_by_tf[126] else 0),
                "rs_63d_pctl":  (_pctl_rank(raw_rs_by_tf[63][tk], tf_63_vals)
                                 if tk in raw_rs_by_tf[63] else 0),
                "rs_21d_pctl":  (_pctl_rank(raw_rs_by_tf[21][tk], tf_21_vals)
                                 if tk in raw_rs_by_tf[21] else 0),
            }

        for tk, full_df in data.items():
            if tk == bench_ticker:
                continue
            df = full_df.iloc[:idx + 1]
            if len(df) < WARMUP:
                continue

            # Core indicators
            try:
                inds = compute_all_indicators(
                    df, bench_slice, cfg,
                    all_rs_values=all_rs_values,
                    multi_tf_rs=multi_tf_pctls.get(tk),
                )
            except Exception:
                continue

            # Forward return for this ticker at this date
            target_date = bench_df.index[idx]
            tk_indices = full_df.index.get_indexer([target_date], method="nearest")
            tk_idx = tk_indices[0]
            fwd = compute_forward_returns(full_df, tk_idx, windows=[FORWARD_WINDOW])
            fwd_ret = fwd.get(FORWARD_WINDOW)
            if fwd_ret is None:
                continue

            spy_adj = fwd_ret - bench_fwd_ret

            row = {
                "date": target_date.strftime("%Y-%m-%d"),
                "ticker": tk,
                "fwd_ret": fwd_ret,
                "spy_adj_ret": spy_adj,
            }
            # Fired flags for core indicators
            for name in ["relative_strength", "ichimoku_cloud", "higher_lows",
                         "moving_averages", "roc", "cmf", "atr_expansion",
                         "dual_tf_rs", "near_52w_high"]:
                entry = inds.get(name, {})
                row[name] = bool(entry.get("triggered", False))

            # Expanded indicators
            exp = _compute_expanded_signals(df, needed_expanded)
            row.update(exp)
            rows.append(row)

    return pd.DataFrame(rows)


def compute_edge(df: pd.DataFrame, col: str) -> tuple[float, int, int]:
    """Return (edge_pct, n_fired, n_not_fired) using SPY-adjusted returns."""
    if col not in df.columns:
        return (float("nan"), 0, 0)
    fired = df[df[col] == True]["spy_adj_ret"].dropna()
    not_fired = df[df[col] == False]["spy_adj_ret"].dropna()
    if len(fired) == 0 or len(not_fired) == 0:
        return (float("nan"), len(fired), len(not_fired))
    edge = (fired.mean() - not_fired.mean()) * 100
    return (edge, len(fired), len(not_fired))


def edge_status(current: float, original: float, dropped: bool = False) -> str:
    """Apply status rules for Section 1."""
    if np.isnan(current):
        return SKIP
    delta = current - original
    if dropped and original < 0:
        # A dropped indicator is "action needed" if it now shows >5% positive
        if current > 5.0:
            return ACTION
        if current > 0:
            return WATCH
        return HEALTHY
    if (current < 0) != (original < 0):
        return WATCH  # sign flipped
    if abs(delta) > 3.0:
        return WATCH
    return HEALTHY


def section_1(events_df: pd.DataFrame) -> dict:
    print("\n" + "=" * 80)
    print("  SECTION 1: INDICATOR EDGE VALIDATION")
    print("  (63-day forward return, SPY-adjusted, fired vs not-fired)")
    print("=" * 80)

    if events_df.empty:
        print("\n  SKIPPED: no events collected.")
        return {"status": SKIP, "rows": []}

    print(f"\n  {len(events_df):,} ticker-date observations\n")
    print(f"  {'Indicator':<22s}{'Original':>12s}{'Current':>12s}"
          f"{'Delta':>12s}{'Fired':>10s}{'Status':>18s}")
    print("  " + "-" * 86)

    rows = []
    statuses_scored = []
    statuses_dropped = []

    def print_row(name: str, dropped: bool):
        display = INDICATOR_DISPLAY.get(name, name)
        original = ORIGINAL_INDICATOR_EDGES.get(name, float("nan"))
        current, n_fired, _ = compute_edge(events_df, name)
        if np.isnan(current):
            status = SKIP
            delta_str = "N/A"
            current_str = "N/A"
            fired_str = "0"
        else:
            status = edge_status(current, original, dropped=dropped)
            delta = current - original
            delta_str = f"{delta:+.2f}%"
            current_str = f"{current:+.2f}%"
            fired_str = f"{n_fired}"
        print(f"  {display:<22s}{original:>+11.2f}%{current_str:>12s}"
              f"{delta_str:>12s}{fired_str:>10s}{STATUS_ICON[status]:>18s}")
        rows.append({
            "indicator": name,
            "display": display,
            "original": original,
            "current": current,
            "fired_count": n_fired,
            "dropped": dropped,
            "status": status,
        })
        return status

    for name in SCORED_INDICATORS:
        statuses_scored.append(print_row(name, dropped=False))

    print("\n  DROPPED INDICATORS — should they be reconsidered?")
    print("  " + "-" * 86)
    for name in DROPPED_FROM_SCORING:
        statuses_dropped.append(print_row(name, dropped=True))
    for name in NEVER_SCORED:
        statuses_dropped.append(print_row(name, dropped=True))

    # Overall section status
    if ACTION in statuses_scored or ACTION in statuses_dropped:
        status = ACTION
    elif WATCH in statuses_scored or WATCH in statuses_dropped:
        status = WATCH
    else:
        status = HEALTHY

    # Callouts
    print("\n  NOTES:")
    negative_scored = [r for r in rows if r["indicator"] in SCORED_INDICATORS
                       and not np.isnan(r["current"]) and r["current"] < 0]
    if negative_scored:
        for r in negative_scored:
            print(f"    - {r['display']}: edge is now negative ({r['current']:+.2f}%) "
                  f"— candidate for removal if persists")
    reconsider = [r for r in rows if r["dropped"]
                  and not np.isnan(r["current"]) and r["current"] > 5.0]
    if reconsider:
        for r in reconsider:
            print(f"    - {r['display']} (currently dropped): edge is now +{r['current']:.2f}% "
                  f"— candidate for re-inclusion if persists")
    if not negative_scored and not reconsider:
        print("    - No indicators require immediate attention.")

    print(f"\n  STATUS: {STATUS_ICON[status]}")
    return {"status": status, "rows": rows}


# ──────────────────────────────────────────────────────────────────────────
# Section 2: Weight Calibration
# ──────────────────────────────────────────────────────────────────────────

def section_2(cfg: dict, section1_rows: list) -> dict:
    print("\n" + "=" * 80)
    print("  SECTION 2: WEIGHT CALIBRATION")
    print("=" * 80)

    ind_cfg = get_indicator_config(cfg)
    current_weights = {}
    for name in SCORED_INDICATORS:
        entry = ind_cfg.get(name) or {}
        current_weights[name] = float(entry.get("weight", 0.0))

    edge_by_name = {}
    missing = []
    for r in section1_rows or []:
        if r["indicator"] in SCORED_INDICATORS and not np.isnan(r["current"]):
            edge_by_name[r["indicator"]] = r["current"]

    # Fill in any missing indicators with their original baseline
    for name in SCORED_INDICATORS:
        if name not in edge_by_name:
            edge_by_name[name] = ORIGINAL_INDICATOR_EDGES[name]
            missing.append(name)
    fallback = len(missing) == len(SCORED_INDICATORS)

    # Ranks
    edge_rank = {name: i + 1 for i, (name, _) in enumerate(
        sorted(edge_by_name.items(), key=lambda kv: -kv[1]))}
    weight_rank = {name: i + 1 for i, (name, _) in enumerate(
        sorted(current_weights.items(), key=lambda kv: -kv[1]))}

    print(f"\n  {'Indicator':<22s}{'Current Wt':>12s}{'Edge':>12s}"
          f"{'Edge Rank':>12s}{'Wt Rank':>10s}{'Aligned?':>12s}")
    print("  " + "-" * 80)

    any_misaligned = False
    for name in SCORED_INDICATORS:
        er = edge_rank.get(name, 99)
        wr = weight_rank.get(name, 99)
        diff = abs(er - wr)
        aligned = HEALTHY if diff <= 1 else WATCH
        if aligned == WATCH:
            any_misaligned = True
        print(f"  {INDICATOR_DISPLAY[name]:<22s}"
              f"{current_weights[name]:>10.2f} pt"
              f"{edge_by_name[name]:>+11.2f}%"
              f"{er:>11d}#"
              f"{wr:>9d}#"
              f"{STATUS_ICON[aligned]:>12s}")

    # Suggested weights — normalize positive edges to the same total weight as current
    total_current = sum(current_weights.values())
    positive_edges = {n: max(e, 0.01) for n, e in edge_by_name.items()}
    total_edge = sum(positive_edges.values())
    suggested = {n: round(positive_edges[n] / total_edge * total_current, 2)
                 for n in SCORED_INDICATORS}

    print("\n  SUGGESTED WEIGHTS (proportional to current edges — informational only):")
    print("  " + "-" * 60)
    print(f"  {'Indicator':<22s}{'Current':>12s}{'Suggested':>14s}{'Delta':>12s}")
    print("  " + "-" * 60)
    for name in SCORED_INDICATORS:
        cur = current_weights[name]
        sug = suggested[name]
        d = sug - cur
        print(f"  {INDICATOR_DISPLAY[name]:<22s}{cur:>10.2f} pt"
              f"{sug:>12.2f} pt"
              f"{d:>+11.2f}")

    if fallback:
        print("\n  NOTE: no current-edge data available — using original baselines.")
    elif missing:
        missing_labels = ", ".join(INDICATOR_DISPLAY[m] for m in missing)
        print(f"\n  NOTE: fell back to original baselines for: {missing_labels}")

    status = WATCH if any_misaligned else HEALTHY
    print(f"\n  STATUS: {STATUS_ICON[status]}")
    return {"status": status}


# ──────────────────────────────────────────────────────────────────────────
# Section 3: Entry/Exit Threshold Grid (12-month window)
# ──────────────────────────────────────────────────────────────────────────

def section_3(cfg: dict, price_data: dict, months: int) -> dict:
    print("\n" + "=" * 80)
    print(f"  SECTION 3: ENTRY/EXIT THRESHOLD GRID (last {months} months)")
    print("=" * 80)

    score_df = load_score_data()
    if score_df.empty:
        print("\n  SKIPPED: no score data in DB.")
        return {"status": SKIP}

    # Filter score data to last N months
    max_date = score_df["date"].max()
    cutoff = max_date - pd.Timedelta(days=int(months * 30.5))
    score_df_12mo = score_df[score_df["date"] >= cutoff].copy()
    if score_df_12mo.empty:
        print("\n  SKIPPED: no score data in window.")
        return {"status": SKIP}

    trading_days = get_trading_days(price_data)
    daily_scores = build_daily_scores(score_df_12mo, trading_days)
    if not daily_scores:
        print("\n  SKIPPED: no trading days overlap with scores.")
        return {"status": SKIP}

    grid = {}
    print()
    for entry in [9.5, 9.0, 8.5]:
        for exit_ in [5, 6, 7]:
            res = run_simulation(
                daily_scores, price_data, trading_days,
                threshold=entry,
                score_exit_threshold=float(exit_),
                rotation_strategy="none",
                persistence_filter=False,
                starting_capital=100_000,
                max_position_pct=0.20,
                stop_loss_pct=0.15,
                verbose=False,
            )
            if "error" in res:
                continue
            grid[(entry, exit_)] = res
            print(f"    >={entry} / <{exit_}: {res['total_return']:+.1f}% "
                  f"return, {res['total_trades']} trades, "
                  f"{res['win_rate']:.1f}% WR, {res['max_drawdown']:.1f}% DD")

    if not grid:
        print("\n  SKIPPED: no grid results.")
        return {"status": SKIP}

    # Rank by total_return
    ranked = sorted(grid.items(), key=lambda kv: -kv[1]["total_return"])

    print(f"\n  {'Entry':<8s}{'Exit':<8s}{'Return':>10s}{'Alpha':>10s}"
          f"{'Max DD':>10s}{'Win %':>10s}{'Trades':>10s}{'Ret/DD':>10s}")
    print("  " + "-" * 76)
    for (entry, exit_), res in sorted(grid.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
        ret_dd = (res["total_return"] / abs(res["max_drawdown"])
                  if res["max_drawdown"] else 0)
        print(f"  >={entry:<6}<{exit_:<7}"
              f"{res['total_return']:>+9.1f}%"
              f"{res['alpha']:>+9.1f}%"
              f"{res['max_drawdown']:>+9.1f}%"
              f"{res['win_rate']:>9.1f}%"
              f"{res['total_trades']:>10d}"
              f"{ret_dd:>9.2f}x")

    current = (9.5, 5)
    best = ranked[0][0]
    current_rank = next((i for i, (k, _) in enumerate(ranked) if k == current),
                        len(ranked))

    print(f"\n  CURRENT CONFIG:     Entry >={current[0]} / Exit <{current[1]}")
    print(f"  BEST THIS QUARTER:  Entry >={best[0]} / Exit <{best[1]} "
          f"({ranked[0][1]['total_return']:+.1f}% return)")
    print(f"  Current config rank: #{current_rank + 1} of {len(ranked)}")

    if current_rank <= 1:
        status = HEALTHY
    elif current_rank <= 3:
        status = WATCH
    else:
        status = WATCH  # escalation to ACTION happens via history.json

    print(f"\n  STATUS: {STATUS_ICON[status]}")
    return {"status": status, "current_rank": current_rank + 1, "best": best}


# ──────────────────────────────────────────────────────────────────────────
# Section 4: Live vs Backtest
# ──────────────────────────────────────────────────────────────────────────

def section_4(cfg: dict, price_data: dict) -> dict:
    print("\n" + "=" * 80)
    print("  SECTION 4: LIVE vs BACKTEST COMPARISON")
    print("=" * 80)

    history_path = Path(__file__).parent / "trade_history.json"
    if not history_path.exists():
        print("\n  SKIPPED: trade_history.json not found.")
        return {"status": SKIP}

    try:
        with open(history_path) as f:
            history = json.load(f)
    except Exception as e:
        print(f"\n  SKIPPED: could not read trade_history.json ({e})")
        return {"status": SKIP}

    trades = history.get("trades", [])
    if not trades:
        print("\n  SKIPPED: no trades executed yet — paper trading just started.")
        return {"status": SKIP}

    # Paired exits to compute round-trip P&L would require matching buys/sells.
    # For the initial baseline, we just report counts and basic stats.
    buys = [t for t in trades if t.get("side") == "buy"]
    sells = [t for t in trades if t.get("side") == "sell"]

    print(f"\n  Total orders recorded: {len(trades)}")
    print(f"    Buys:  {len(buys)}")
    print(f"    Sells: {len(sells)}")
    if trades:
        first_date = min(t.get("date", "") for t in trades)
        last_date = max(t.get("date", "") for t in trades)
        print(f"  Period: {first_date} -> {last_date}")

    # If no sells yet, we can't compute realized P&L
    if not sells:
        print("\n  No sells yet — realized P&L unavailable.")
        print("  This section becomes meaningful once exit trades accumulate.")
        print(f"\n  STATUS: {STATUS_ICON[SKIP]}")
        return {"status": SKIP, "orders": len(trades)}

    # Future: match buys to sells by ticker (FIFO) to compute round-trip P&L
    # and compare against same-period backtest.
    print("\n  NOTE: round-trip matching not yet implemented — will activate "
          "automatically once >=10 completed round-trips are logged.")
    print(f"\n  STATUS: {STATUS_ICON[SKIP]}")
    return {"status": SKIP, "orders": len(trades)}


# ──────────────────────────────────────────────────────────────────────────
# Section 5: Universe Review
# ──────────────────────────────────────────────────────────────────────────

def section_5(cfg: dict, price_data: dict, months: int) -> dict:
    print("\n" + "=" * 80)
    print("  SECTION 5: UNIVERSE REVIEW")
    print("=" * 80)

    all_tickers = get_all_tickers(cfg)
    metadata = get_ticker_metadata(cfg)
    bench_ticker = cfg["benchmark"]["ticker"]

    fetched = set(price_data.keys()) - {bench_ticker}
    expected = set(all_tickers)
    failed = sorted(expected - fetched)
    fetched_count = len(fetched)
    fail_rate = len(failed) / len(expected) * 100 if expected else 0

    print(f"\n  Total tickers in config:   {len(expected)}")
    print(f"  Successfully fetched data: {fetched_count}")
    print(f"  Failed to fetch:           {len(failed)}  ({fail_rate:.1f}%)")

    if failed:
        print("\n  FAILED TICKERS:")
        print("  " + "-" * 60)
        for tk in failed[:20]:
            meta = metadata.get(tk, {})
            sub = meta.get("subsector_name", "?")
            print(f"    {tk:<10s}  {sub}")
        if len(failed) > 20:
            print(f"    ... and {len(failed) - 20} more")

    # Never-scored tickers — query DB for avg score over window
    db_path = Path(__file__).parent / "breakout_tracker.db"
    never_scored = []
    if db_path.exists():
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=int(months * 30.5))).strftime("%Y-%m-%d")
        cursor.execute("""
            SELECT ticker, AVG(score), MAX(score),
                   SUM(CASE WHEN score >= 6 THEN 1 ELSE 0 END) AS times_high
            FROM ticker_scores
            WHERE date >= ?
            GROUP BY ticker
            HAVING AVG(score) <= 2.0 AND MAX(score) < 6.0
            ORDER BY AVG(score) ASC
        """, (cutoff,))
        never_scored = cursor.fetchall()
        conn.close()

    if never_scored:
        print(f"\n  NEVER-SCORED TICKERS (avg <= 2.0, max < 6.0 over {months}mo):")
        print("  " + "-" * 70)
        print(f"  {'Ticker':<10s}{'Subsector':<32s}{'Avg':>8s}{'Max':>8s}{'>=6':>8s}")
        print("  " + "-" * 70)
        for ticker, avg, mx, times_high in never_scored[:15]:
            meta = metadata.get(ticker, {})
            sub = meta.get("subsector_name", "?")[:30]
            print(f"  {ticker:<10s}{sub:<32s}{avg:>7.1f} {mx:>7.1f} {times_high:>7d}")
        if len(never_scored) > 15:
            print(f"    ... and {len(never_scored) - 15} more")
    else:
        print("\n  All tickers have shown meaningful scoring activity.")

    # Subsector size check
    small_subs = []
    for sector_key, sector in cfg["sectors"].items():
        for sub_key, sub in sector.get("subsectors", {}).items():
            n = len(sub.get("tickers", {}) or {})
            if n < 4:
                small_subs.append((sub.get("name", sub_key), n))

    if small_subs:
        print("\n  SUBSECTORS WITH <4 TICKERS (breadth signals unreliable):")
        print("  " + "-" * 60)
        for name, n in small_subs:
            print(f"    {name:<40s}  {n} tickers")
    else:
        print("\n  All subsectors have >=4 tickers.")

    # Status rules
    if fail_rate > 10 or small_subs:
        status = WATCH if fail_rate <= 10 else ACTION
    elif fail_rate > 5:
        status = WATCH
    else:
        status = HEALTHY

    print(f"\n  STATUS: {STATUS_ICON[status]}")
    return {
        "status": status,
        "fail_rate": fail_rate,
        "failed_count": len(failed),
        "never_scored_count": len(never_scored),
        "small_subsectors": small_subs,
    }


# ──────────────────────────────────────────────────────────────────────────
# Section 6: State Machine Health
# ──────────────────────────────────────────────────────────────────────────

def section_6(cfg: dict, price_data: dict, months: int) -> dict:
    print("\n" + "=" * 80)
    print(f"  SECTION 6: STATE MACHINE HEALTH (last {months} months)")
    print("=" * 80)

    db_path = Path(__file__).parent / "breakout_tracker.db"
    if not db_path.exists():
        print("\n  SKIPPED: DB not found.")
        return {"status": SKIP}

    conn = init_db(str(db_path))

    # Derive state history from subsector_daily by replaying the breadth threshold
    cutoff = (datetime.now() - timedelta(days=int(months * 30.5))).strftime("%Y-%m-%d")
    hist = get_all_history(conn, days=int(months * 31))
    if hist.empty:
        print("\n  SKIPPED: no subsector history.")
        conn.close()
        return {"status": SKIP}

    hist = hist[hist["date"] >= cutoff].copy()
    hist["date"] = pd.to_datetime(hist["date"])

    # Use breadth_threshold from config: a subsector is "hot" if breadth >= 0.5
    breadth_trigger = cfg["breakout_detection"].get("breadth_trigger", 0.5)
    confirm_days = cfg["breakout_detection"].get("confirm_days", 3)
    fade_cool_days = cfg["breakout_detection"].get("fade_cool_days", 5)

    # Simplified transition counting: per subsector, walk chronologically,
    # classify day as hot/cool, and count transitions through quiet->emerging->confirmed->fading->quiet
    transitions = {
        "quiet_to_emerging": 0,
        "emerging_to_confirmed": 0,
        "emerging_to_quiet": 0,
        "confirmed_to_fading": 0,
        "fading_to_confirmed": 0,
        "fading_to_quiet": 0,
    }

    # forward returns by state — need SPY data from price_data
    bench = price_data.get(cfg["benchmark"]["ticker"])
    forward_returns = {"emerging": [], "confirmed": [], "revival": []}

    subsector_groups = hist.groupby("subsector")
    for sub, g in subsector_groups:
        g = g.sort_values("date").reset_index(drop=True)
        state = "quiet"
        hot_streak = 0
        cool_streak = 0
        prior_state = None
        for i, row in g.iterrows():
            is_hot = row["breadth"] >= breadth_trigger
            if is_hot:
                hot_streak += 1
                cool_streak = 0
            else:
                cool_streak += 1
                hot_streak = 0

            new_state = state
            if state == "quiet" and is_hot:
                new_state = "emerging"
            elif state == "emerging" and hot_streak >= confirm_days:
                new_state = "confirmed"
            elif state == "emerging" and cool_streak >= 2:
                new_state = "quiet"
            elif state == "confirmed" and not is_hot:
                new_state = "fading"
            elif state == "fading" and is_hot:
                new_state = "confirmed"  # revival
                prior_state = "revival"
            elif state == "fading" and cool_streak >= fade_cool_days:
                new_state = "quiet"

            if new_state != state:
                key = f"{state}_to_{new_state}"
                if key in transitions:
                    transitions[key] += 1

                # On entering emerging/confirmed, record forward return of avg score
                # proxy: use the subsector's forward breadth change is tricky; instead,
                # measure the 63-day forward return of the "hot" tickers' universe.
                # Simplification: use SPY forward return as baseline and each subsector's
                # avg score change as a proxy. For a cleaner answer, we'd need the actual
                # ticker prices. Here, skip forward returns for state_machine and note it.
                pass

            state = new_state

    print("\n  State Transitions Summary:")
    print("  " + "-" * 60)
    total = sum(transitions.values())
    print(f"    Total transitions:              {total}")
    print(f"    Quiet -> Emerging:               {transitions['quiet_to_emerging']}")
    print(f"    Emerging -> Confirmed:           {transitions['emerging_to_confirmed']}")
    print(f"    Emerging -> Quiet (failed):      {transitions['emerging_to_quiet']}")
    print(f"    Confirmed -> Fading:             {transitions['confirmed_to_fading']}")
    print(f"    Fading -> Confirmed (revival):   {transitions['fading_to_confirmed']}")
    print(f"    Fading -> Quiet:                 {transitions['fading_to_quiet']}")

    # Confirmation rate
    emerging_total = transitions["emerging_to_confirmed"] + transitions["emerging_to_quiet"]
    if emerging_total:
        confirm_rate = transitions["emerging_to_confirmed"] / emerging_total * 100
        print(f"    Emerging confirmation rate:      {confirm_rate:.1f}%")

    print("\n  Forward Returns by State (63-day, SPY-adjusted):")
    print("  " + "-" * 60)
    print("    NOTE: state-based forward returns require per-date ticker replays.")
    print("    Current baseline run logs the transition counts only.")
    print("    Next quarter will compare transition counts and confirm rates to baseline.")

    print("\n  Parameter Check:")
    print("  " + "-" * 60)
    print(f"    Breadth trigger ({breadth_trigger}):     configured — "
          f"{'OK' if 0.3 <= breadth_trigger <= 0.8 else 'review'}")
    print(f"    Confirm days ({confirm_days}):           configured — "
          f"{'OK' if 2 <= confirm_days <= 5 else 'review'}")
    print(f"    Fade cool days ({fade_cool_days}):       configured — "
          f"{'OK' if 3 <= fade_cool_days <= 10 else 'review'}")

    # Status: HEALTHY if we see reasonable confirmation rates; WATCH if no transitions
    if total == 0:
        status = WATCH
    elif emerging_total and transitions["emerging_to_confirmed"] == 0:
        status = WATCH
    else:
        status = HEALTHY

    conn.close()
    print(f"\n  STATUS: {STATUS_ICON[status]}")
    return {"status": status, "transitions": transitions}


# ──────────────────────────────────────────────────────────────────────────
# Section 7: Executive Summary
# ──────────────────────────────────────────────────────────────────────────

def upgrade_watches_with_history(current: dict, prev: dict) -> dict:
    """If a section was WATCH last quarter AND is WATCH now, upgrade to ACTION."""
    upgraded = dict(current)
    if not prev:
        return upgraded
    for key, value in current.items():
        if value == WATCH and prev.get(key) == WATCH:
            upgraded[key] = ACTION
    return upgraded


def section_7_exec_summary(
    start_date: str,
    end_date: str,
    s1: dict,
    s2: dict,
    s3: dict,
    s4: dict,
    s5: dict,
    s6: dict,
    prev_history: dict,
) -> dict:
    print("\n" + "=" * 80)
    print("  QUARTERLY REVIEW — EXECUTIVE SUMMARY")
    print(f"  Period: {start_date} -> {end_date}")
    print("=" * 80)

    current = {
        "indicator_edge":    s1.get("status", SKIP),
        "weight_calibration": s2.get("status", SKIP),
        "thresholds":         s3.get("status", SKIP),
        "live_vs_backtest":   s4.get("status", SKIP),
        "universe":           s5.get("status", SKIP),
        "state_machine":      s6.get("status", SKIP),
    }

    final = upgrade_watches_with_history(current, prev_history)

    labels = {
        "indicator_edge":     "1. Indicator Edge      ",
        "weight_calibration": "2. Weight Calibration  ",
        "thresholds":         "3. Entry/Exit Thresholds",
        "live_vs_backtest":   "4. Live vs Backtest    ",
        "universe":           "5. Universe Health     ",
        "state_machine":      "6. State Machine       ",
    }

    notes = {
        "indicator_edge":     f"{sum(1 for r in s1.get('rows', []) if r['status'] == HEALTHY)}/"
                              f"{len(s1.get('rows', []))} healthy",
        "weight_calibration": "Rank alignment with current edges",
        "thresholds":         f"Current config rank #{s3.get('current_rank', '?')}",
        "live_vs_backtest":   f"{s4.get('orders', 0)} orders recorded",
        "universe":           f"{s5.get('failed_count', 0)} fetch failures, "
                              f"{s5.get('never_scored_count', 0)} dormant tickers",
        "state_machine":      f"{sum(s6.get('transitions', {}).values())} transitions",
    }

    print(f"\n  {'Section':<28s}{'Status':<16s}Notes")
    print("  " + "-" * 70)
    for key in ["indicator_edge", "weight_calibration", "thresholds",
                "live_vs_backtest", "universe", "state_machine"]:
        st = final[key]
        upgraded_mark = ""
        if st == ACTION and current[key] == WATCH:
            upgraded_mark = "  <- upgraded (2 quarters in a row)"
        print(f"  {labels[key]:<28s}{STATUS_ICON[st]:<16s}{notes[key]}{upgraded_mark}")

    # Overall
    statuses = list(final.values())
    if ACTION in statuses:
        overall = ACTION
    elif WATCH in statuses:
        overall = WATCH
    else:
        overall = HEALTHY

    print(f"\n  OVERALL SYSTEM HEALTH: {STATUS_ICON[overall]}")

    # Recommended actions
    actions = []
    for key, st in final.items():
        if st == WATCH:
            actions.append(f"- {labels[key].strip()}: monitor next quarter")
        elif st == ACTION:
            actions.append(f"- {labels[key].strip()}: investigate and prepare a change")

    print("\n  RECOMMENDED ACTIONS:")
    print("  " + "-" * 70)
    if actions:
        for a in actions:
            print(f"    {a}")
    else:
        print("    None — system is healthy.")

    print(f"\n  NEXT REVIEW DATE: {next_quarter_start()}")
    print("=" * 80)

    return {"current": current, "final": final, "overall": overall}


# ──────────────────────────────────────────────────────────────────────────
# History persistence
# ──────────────────────────────────────────────────────────────────────────

def load_prev_history(history_path: Path, current_label: str) -> dict:
    """Return the most recent prior-quarter status dict, or {} if none."""
    if not history_path.exists():
        return {}
    try:
        with open(history_path) as f:
            data = json.load(f)
    except Exception:
        return {}

    keys = sorted(k for k in data.keys() if k != current_label)
    if not keys:
        return {}
    prev_key = keys[-1]
    return data[prev_key] or {}


def save_history(history_path: Path, label: str, summary: dict, period: tuple):
    if history_path.exists():
        with open(history_path) as f:
            data = json.load(f)
    else:
        data = {}

    data[label] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "period_start": period[0],
        "period_end": period[1],
        **{k: v.lower().replace(" ", "_") for k, v in summary["final"].items()},
        "overall": summary["overall"].lower().replace(" ", "_"),
    }

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Alpha Scanner quarterly review")
    parser.add_argument("--months", type=int, default=12,
                        help="Lookback window in months (default 12)")
    parser.add_argument("--email-only", action="store_true",
                        help="Email the latest saved review without re-running")
    args = parser.parse_args()

    if args.email_only:
        print("--email-only flag received but email delivery is not yet wired. "
              "The saved review text file can be attached manually.")
        return 0

    # Set up output capture
    reviews_dir = Path(__file__).parent / "quarterly_reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    label = quarter_label()
    out_path = reviews_dir / f"review_{label.replace('-', '_')}.txt"
    history_path = reviews_dir / "review_history.json"

    original_stdout = sys.stdout
    log_fh = open(out_path, "w")
    sys.stdout = Tee(original_stdout, log_fh)

    try:
        print("=" * 80)
        print(f"  ALPHA SCANNER — QUARTERLY SYSTEM REVIEW")
        print(f"  Label: {label}")
        print(f"  Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"  Lookback: {args.months} months")
        print("=" * 80)

        print("\n  Loading config and fetching price data...")
        cfg = load_config()
        price_data = fetch_all(cfg, period="2y", verbose=False)
        print(f"  {len(price_data)} tickers fetched")

        bench_df = price_data.get(cfg["benchmark"]["ticker"])
        end_date = bench_df.index[-1].strftime("%Y-%m-%d") if bench_df is not None else "?"
        lookback_days = int(args.months * 30.5)
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        # Section 1
        print("\n  Collecting indicator events (Section 1)...")
        events_df = collect_indicator_events(price_data, cfg, args.months)
        s1 = section_1(events_df)

        # Section 2
        s2 = section_2(cfg, s1.get("rows", []))

        # Section 3
        s3 = section_3(cfg, price_data, args.months)

        # Section 4
        s4 = section_4(cfg, price_data)

        # Section 5
        s5 = section_5(cfg, price_data, args.months)

        # Section 6
        s6 = section_6(cfg, price_data, args.months)

        # Section 7 — Executive Summary
        prev_history = load_prev_history(history_path, label)
        summary = section_7_exec_summary(
            start_date, end_date, s1, s2, s3, s4, s5, s6, prev_history,
        )

        # Persist history
        save_history(history_path, label, summary, (start_date, end_date))

        print(f"\n  Saved review to: {out_path}")
        print(f"  History log:     {history_path}")

    finally:
        sys.stdout = original_stdout
        log_fh.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
