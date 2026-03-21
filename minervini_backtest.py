from __future__ import annotations

"""
minervini_backtest.py — Backtest Mark Minervini's SEPA system.

Level 1: Trend Template (8 criteria)
Level 2: Trend Template + VCP Breakout + Volume Confirmation

Usage:
    python3 minervini_backtest.py
"""

import pandas as pd
import numpy as np
from config import load_config, get_all_tickers, get_ticker_metadata
from data_fetcher import fetch_batch

FORWARD_WINDOWS = [10, 21, 42, 63]


# =============================================================
# LEVEL 1: TREND TEMPLATE (8 criteria)
# =============================================================
def check_minervini_trend_template(df: pd.DataFrame) -> dict:
    """
    Check all 8 Minervini Trend Template criteria.
    Requires ~274 rows minimum (200-day SMA + 52-week range + slope).
    """
    result = {
        "triggered": False,
        "criteria_met": 0,
        "criteria_detail": {},
        "values": {},
    }

    if len(df) < 274:
        result["criteria_detail"] = {k: False for k in [
            "price_above_150sma", "price_above_200sma", "sma150_above_sma200",
            "sma200_rising", "sma50_above_150_and_200", "price_above_50sma",
            "above_25pct_from_52w_low", "within_25pct_of_52w_high",
        ]}
        return result

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    price = close.iloc[-1]
    sma_50 = close.rolling(50).mean().iloc[-1]
    sma_150 = close.rolling(150).mean().iloc[-1]
    sma_200 = close.rolling(200).mean().iloc[-1]

    # 200-day SMA slope over last 22 trading days
    sma_200_series = close.rolling(200).mean()
    sma_200_ago = sma_200_series.iloc[-23]
    if pd.isna(sma_200_ago) or sma_200_ago == 0:
        sma_200_slope = 0
    else:
        sma_200_slope = (sma_200 - sma_200_ago) / sma_200_ago

    # 52-week high/low (252 trading days)
    lookback_252 = min(252, len(df))
    high_52w = high.iloc[-lookback_252:].max()
    low_52w = low.iloc[-lookback_252:].min()

    pct_above_52w_low = (price - low_52w) / low_52w if low_52w > 0 else 0
    pct_below_52w_high = (price - high_52w) / high_52w if high_52w > 0 else 0

    if pd.isna(sma_50) or pd.isna(sma_150) or pd.isna(sma_200):
        result["criteria_detail"] = {k: False for k in [
            "price_above_150sma", "price_above_200sma", "sma150_above_sma200",
            "sma200_rising", "sma50_above_150_and_200", "price_above_50sma",
            "above_25pct_from_52w_low", "within_25pct_of_52w_high",
        ]}
        return result

    # The 8 criteria
    criteria = {
        "price_above_150sma": price > sma_150,
        "price_above_200sma": price > sma_200,
        "sma150_above_sma200": sma_150 > sma_200,
        "sma200_rising": sma_200_slope > 0,
        "sma50_above_150_and_200": sma_50 > sma_150 and sma_50 > sma_200,
        "price_above_50sma": price > sma_50,
        "above_25pct_from_52w_low": pct_above_52w_low >= 0.25,
        "within_25pct_of_52w_high": pct_below_52w_high >= -0.25,
    }

    criteria_met = sum(criteria.values())

    return {
        "triggered": criteria_met == 8,
        "criteria_met": criteria_met,
        "criteria_detail": criteria,
        "values": {
            "price": round(price, 2),
            "sma_50": round(sma_50, 2),
            "sma_150": round(sma_150, 2),
            "sma_200": round(sma_200, 2),
            "sma_200_slope": round(sma_200_slope * 100, 4),
            "pct_above_52w_low": round(pct_above_52w_low, 4),
            "pct_below_52w_high": round(pct_below_52w_high, 4),
        },
    }


# =============================================================
# VCP DETECTION (swing-based pullback analysis)
# =============================================================
def _find_swing_points(prices: pd.Series, window: int = 5):
    """Find local highs and lows using a rolling window approach."""
    highs = []
    lows = []
    arr = prices.values

    for i in range(window, len(arr) - window):
        # Local high: higher than all neighbors within window
        if arr[i] == max(arr[i - window:i + window + 1]):
            highs.append((i, arr[i]))
        # Local low: lower than all neighbors within window
        if arr[i] == min(arr[i - window:i + window + 1]):
            lows.append((i, arr[i]))

    return highs, lows


def detect_vcp(
    df: pd.DataFrame,
    base_period: int = 90,
    min_contractions: int = 2,
    max_last_pullback: float = 0.15,
    max_base_depth: float = 0.35,
) -> dict:
    """
    Detect Volatility Contraction Pattern using swing-based pullback analysis.
    Looks for progressively shallower pullbacks within a base.
    """
    result = {
        "detected": False,
        "num_contractions": 0,
        "pullback_depths": [],
        "base_depth": 0,
        "base_length_days": 0,
        "pivot_price": 0,
    }

    if len(df) < base_period + 10:
        return result

    recent = df.iloc[-base_period:]
    close = recent["Close"]
    high = recent["High"]

    # Overall base metrics
    base_high = high.max()
    base_low = recent["Low"].min()
    base_depth = (base_low - base_high) / base_high if base_high > 0 else 0

    result["base_depth"] = round(abs(base_depth), 4)
    result["base_length_days"] = base_period
    result["pivot_price"] = round(base_high, 2)

    if abs(base_depth) > max_base_depth:
        return result

    # Find swing highs and lows in the base
    swing_highs, swing_lows = _find_swing_points(close, window=5)

    if len(swing_highs) < 2 or len(swing_lows) < 1:
        return result

    # Measure pullback depths: from each swing high to the next swing low
    pullback_depths = []
    for i in range(len(swing_highs)):
        sh_idx, sh_price = swing_highs[i]

        # Find the next swing low after this high
        next_low = None
        for sl_idx, sl_price in swing_lows:
            if sl_idx > sh_idx:
                next_low = (sl_idx, sl_price)
                break

        if next_low is not None:
            depth = (next_low[1] - sh_price) / sh_price
            pullback_depths.append(round(abs(depth), 4))

    if len(pullback_depths) < 2:
        return result

    result["pullback_depths"] = pullback_depths

    # Count successive contractions (each pullback shallower than the last)
    contractions = 0
    for i in range(1, len(pullback_depths)):
        if pullback_depths[i] < pullback_depths[i - 1]:
            contractions += 1
        else:
            contractions = 0  # reset on expansion

    result["num_contractions"] = contractions

    # Check trigger conditions
    last_pullback = pullback_depths[-1] if pullback_depths else 1.0
    detected = (
        contractions >= min_contractions
        and last_pullback <= max_last_pullback
        and abs(base_depth) <= max_base_depth
        and base_period >= 20
    )

    result["detected"] = detected
    return result


# =============================================================
# LEVEL 2: FULL SEPA (Template + VCP + Breakout + Volume)
# =============================================================
def check_minervini_full(
    df: pd.DataFrame,
    volume_lookback: int = 50,
    volume_threshold: float = 1.5,
) -> dict:
    """
    Full Minervini SEPA check:
    1. All 8 Trend Template criteria must be TRUE
    2. VCP must be detected
    3. Price must be breaking out of the VCP base TODAY
    4. Volume must confirm the breakout
    """
    template = check_minervini_trend_template(df)
    vcp = detect_vcp(df)

    # Breakout detection: is today's close above the VCP pivot (base high)?
    pivot_price = vcp["pivot_price"]
    current_close = df["Close"].iloc[-1] if len(df) > 0 else 0
    is_breakout = current_close > pivot_price and pivot_price > 0
    breakout_pct = (current_close - pivot_price) / pivot_price if pivot_price > 0 else 0

    # Volume confirmation
    if len(df) >= volume_lookback + 1:
        today_volume = df["Volume"].iloc[-1]
        avg_volume = df["Volume"].iloc[-(volume_lookback + 1):-1].mean()
        volume_ratio = today_volume / avg_volume if avg_volume > 0 else 0
        volume_confirmed = volume_ratio >= volume_threshold
    else:
        today_volume = 0
        avg_volume = 0
        volume_ratio = 0
        volume_confirmed = False

    triggered = (
        template["triggered"]
        and vcp["detected"]
        and is_breakout
        and volume_confirmed
    )

    return {
        "triggered": triggered,
        "trend_template": template,
        "vcp": vcp,
        "breakout": {
            "is_breakout_day": is_breakout,
            "pivot_price": pivot_price,
            "breakout_pct": round(breakout_pct, 4),
        },
        "volume": {
            "confirmed": volume_confirmed,
            "today_volume": int(today_volume),
            "avg_volume": round(avg_volume, 0),
            "volume_ratio": round(volume_ratio, 2),
        },
    }


# =============================================================
# BACKTEST ENGINE
# =============================================================
def run_minervini_backtest(
    data: dict,
    cfg: dict,
    test_frequency: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """Walk through historical dates and collect Minervini signals + forward returns."""

    benchmark_ticker = cfg["benchmark"]["ticker"]
    benchmark_df = data.get(benchmark_ticker)
    if benchmark_df is None:
        print("  [ERROR] No benchmark data.")
        return pd.DataFrame()

    max_forward = max(FORWARD_WINDOWS)
    warmup = 280  # 200 SMA + 52-week range + slope buffer
    total_days = len(benchmark_df)

    if total_days < warmup + max_forward:
        print(f"  [ERROR] Not enough data. Have {total_days}, need {warmup + max_forward}+.")
        return pd.DataFrame()

    start_idx = warmup
    end_idx = total_days - max_forward
    test_indices = list(range(start_idx, end_idx, test_frequency))

    metadata = get_ticker_metadata(cfg)

    if verbose:
        start_date = benchmark_df.index[start_idx].strftime("%Y-%m-%d")
        end_date = benchmark_df.index[end_idx].strftime("%Y-%m-%d")
        print(f"  Backtest window: {start_date} -> {end_date}")
        print(f"  Testing {len(test_indices)} dates across {len(data) - 1} tickers")
        print()

    all_rows = []

    for count, idx in enumerate(test_indices, 1):
        if verbose and count % 10 == 0:
            print(f"  [{count}/{len(test_indices)}] {benchmark_df.index[idx].strftime('%Y-%m-%d')}...")

        for ticker, full_df in data.items():
            if ticker == benchmark_ticker:
                continue

            df = full_df.iloc[:idx + 1]
            if len(df) < warmup:
                continue

            # Run both levels
            template = check_minervini_trend_template(df)
            full_sepa = check_minervini_full(df)

            # Forward returns
            target_date = benchmark_df.index[idx]
            ticker_indices = full_df.index.get_indexer([target_date], method="nearest")
            ticker_idx = ticker_indices[0]
            entry_price = full_df["Close"].iloc[ticker_idx]

            fwd = {}
            for w in FORWARD_WINDOWS:
                future_idx = ticker_idx + w
                if future_idx < len(full_df):
                    fwd[w] = round((full_df["Close"].iloc[future_idx] - entry_price) / entry_price, 4)
                else:
                    fwd[w] = None

            meta = metadata.get(ticker, {})
            row = {
                "date": benchmark_df.index[idx].strftime("%Y-%m-%d"),
                "ticker": ticker,
                "name": meta.get("name", ""),
                "sector": meta.get("sector_name", ""),
                # Level 1
                "template_triggered": template["triggered"],
                "criteria_met": template["criteria_met"],
                # Individual criteria
                **{f"c_{k}": v for k, v in template["criteria_detail"].items()},
                # Level 2 components
                "vcp_detected": full_sepa["vcp"]["detected"],
                "vcp_contractions": full_sepa["vcp"]["num_contractions"],
                "breakout_day": full_sepa["breakout"]["is_breakout_day"],
                "volume_confirmed": full_sepa["volume"]["confirmed"],
                "volume_ratio": full_sepa["volume"]["volume_ratio"],
                "full_sepa": full_sepa["triggered"],
                # Intermediate: template + VCP (without breakout/volume)
                "template_plus_vcp": template["triggered"] and full_sepa["vcp"]["detected"],
            }

            for w in FORWARD_WINDOWS:
                row[f"fwd_{w}d"] = fwd.get(w)

            all_rows.append(row)

    return pd.DataFrame(all_rows)


# =============================================================
# REPORTING
# =============================================================
def _return_stats(series: pd.Series):
    """Compute win rate and average return from a series of returns."""
    valid = series.dropna()
    if len(valid) == 0:
        return 0, 0, 0
    wr = (valid > 0).mean() * 100
    avg = valid.mean() * 100
    return wr, avg, len(valid)


def print_section1(df: pd.DataFrame):
    """Section 1: Level 1 Trend Template results."""
    total = len(df)
    met = df[df["template_triggered"] == True]
    not_met = df[df["template_triggered"] == False]

    print(f"\n{'=' * 90}")
    print(f"  SECTION 1: MINERVINI TREND TEMPLATE — BACKTEST RESULTS")
    print(f"{'=' * 90}")
    print(f"\n  Total observations: {total}")
    print(f"  Times template fully met (8/8): {len(met)} ({len(met)/total*100:.1f}%)")

    # Forward returns when met vs not met
    print(f"\n  {'─' * 70}")
    print(f"  Forward Returns When All 8 Criteria Met:")
    for w in FORWARD_WINDOWS:
        wr, avg, n = _return_stats(met[f"fwd_{w}d"])
        print(f"    {w:>2}-day:  Win rate: {wr:5.1f}%  |  Avg: {avg:+7.2f}%  |  N={n}")

    print(f"\n  Forward Returns When Template NOT Met:")
    for w in FORWARD_WINDOWS:
        wr, avg, n = _return_stats(not_met[f"fwd_{w}d"])
        print(f"    {w:>2}-day:  Win rate: {wr:5.1f}%  |  Avg: {avg:+7.2f}%  |  N={n}")

    print(f"\n  Edge (Met vs Not Met):")
    for w in FORWARD_WINDOWS:
        _, avg_met, _ = _return_stats(met[f"fwd_{w}d"])
        _, avg_not, _ = _return_stats(not_met[f"fwd_{w}d"])
        edge = avg_met - avg_not
        emoji = "✅" if edge > 3 else "🔶" if edge > 0 else "❌"
        print(f"    {w:>2}-day: {edge:+7.2f}% {emoji}")

    # Criteria breakdown
    criteria_cols = [c for c in df.columns if c.startswith("c_")]
    print(f"\n  {'─' * 70}")
    print(f"  Criteria Breakdown — How Often Each Passes:")
    for col in criteria_cols:
        label = col[2:].replace("_", " ").title()
        pct = df[col].mean() * 100
        print(f"    {label:35s} {pct:5.1f}%")

    # Partial template results
    print(f"\n  {'─' * 70}")
    print(f"  Partial Template Results (how many criteria matter?):")
    for n_criteria in range(5, 9):
        subset = df[df["criteria_met"] >= n_criteria]
        valid = subset["fwd_63d"].dropna()
        if len(valid) < 10:
            continue
        wr = (valid > 0).mean() * 100
        avg = valid.mean() * 100
        print(f"    {n_criteria}/8+ criteria met:  Win rate: {wr:5.1f}%  |  Avg 63d: {avg:+7.2f}%  |  N={len(valid)}")


def print_section2(df: pd.DataFrame):
    """Section 2: Level 2 Full SEPA results."""
    template_met = df[df["template_triggered"] == True]
    template_vcp = df[df["template_plus_vcp"] == True]
    full_sepa = df[df["full_sepa"] == True]

    print(f"\n\n{'=' * 90}")
    print(f"  SECTION 2: MINERVINI FULL SEPA — BACKTEST RESULTS")
    print(f"{'=' * 90}")
    print(f"\n  Template met:                    {len(template_met)} times")
    print(f"  VCP detected within template:    {len(template_vcp)} times")
    print(f"  Full SEPA (+ breakout + volume):  {len(full_sepa)} times")

    if len(full_sepa) >= 5:
        print(f"\n  Forward Returns (Full SEPA Signal):")
        for w in FORWARD_WINDOWS:
            wr, avg, n = _return_stats(full_sepa[f"fwd_{w}d"])
            print(f"    {w:>2}-day:  Win rate: {wr:5.1f}%  |  Avg: {avg:+7.2f}%  |  N={n}")
    else:
        print(f"\n  [Too few Full SEPA signals ({len(full_sepa)}) for meaningful statistics]")

    # Comparison table
    print(f"\n  {'─' * 70}")
    print(f"  Comparison:")
    print(f"  {'':30s} {'Events':>8} {'63d Win%':>10} {'63d Avg':>10}")
    print(f"  {'─' * 60}")

    for label, subset in [
        ("Template only (8/8)", template_met),
        ("Template + VCP", template_vcp),
        ("Full SEPA", full_sepa),
    ]:
        wr, avg, n = _return_stats(subset["fwd_63d"])
        if n >= 5:
            print(f"  {label:30s} {n:>8} {wr:>9.1f}% {avg:>+9.2f}%")
        else:
            print(f"  {label:30s} {n:>8}       —         —")


def print_section3(df: pd.DataFrame):
    """Section 3: Comparison to our indicator stack."""
    total = len(df)
    template_met = df[df["template_triggered"] == True]
    full_sepa = df[df["full_sepa"] == True]

    print(f"\n\n{'=' * 90}")
    print(f"  SECTION 3: MINERVINI vs OUR SCORING SYSTEM — COMPARISON")
    print(f"{'=' * 90}")

    print(f"\n  {'':35s} {'Events':>8} {'63d Win%':>10} {'63d Avg':>10} {'Selectivity':>12}")
    print(f"  {'─' * 77}")

    for label, subset in [
        ("Minervini Template (8/8)", template_met),
        ("Minervini Full SEPA", full_sepa),
    ]:
        wr, avg, n = _return_stats(subset["fwd_63d"])
        sel = n / total * 100 if total > 0 else 0
        if n >= 5:
            print(f"  {label:35s} {n:>8} {wr:>9.1f}% {avg:>+9.2f}% {sel:>10.1f}%")
        else:
            print(f"  {label:35s} {n:>8}       —         —          —")

    print(f"\n  Note: Compare to our indicator stack results from indicator_analysis_full.py")
    print(f"  (Run that separately to get our stack's numbers for direct comparison)")


def print_section4(df: pd.DataFrame):
    """Section 4: Most frequently flagged tickers."""
    template_met = df[df["template_triggered"] == True]

    if len(template_met) == 0:
        print(f"\n  No tickers met the template.")
        return

    print(f"\n\n{'=' * 90}")
    print(f"  SECTION 4: MINERVINI TOP PICKS — MOST FREQUENTLY IN TEMPLATE")
    print(f"{'=' * 90}\n")

    top = template_met.groupby(["ticker", "name"]).agg(
        times_in_template=("date", "count"),
        avg_63d=("fwd_63d", lambda x: x.dropna().mean() * 100 if len(x.dropna()) > 0 else 0),
        wr_63d=("fwd_63d", lambda x: (x.dropna() > 0).mean() * 100 if len(x.dropna()) > 0 else 0),
    ).sort_values("times_in_template", ascending=False).head(20)

    print(f"  {'Ticker':8s} {'Name':30s} {'In Template':>12} {'63d Avg':>10} {'63d Win%':>10}")
    print(f"  {'─' * 72}")

    for (ticker, name), row in top.iterrows():
        print(f"  {ticker:8s} {name[:28]:30s} {row['times_in_template']:>12.0f} "
              f"{row['avg_63d']:>+9.2f}% {row['wr_63d']:>8.1f}%")

    # By sector
    print(f"\n  {'─' * 72}")
    print(f"  By Sector:")

    sector_stats = template_met.groupby("sector").agg(
        events=("date", "count"),
        avg_63d=("fwd_63d", lambda x: x.dropna().mean() * 100 if len(x.dropna()) > 0 else 0),
    ).sort_values("events", ascending=False)

    for sector, row in sector_stats.iterrows():
        if not sector:
            continue
        print(f"    {sector[:40]:40s}  Events: {row['events']:>5.0f}  |  Avg 63d: {row['avg_63d']:+7.2f}%")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    cfg = load_config()

    print("=" * 90)
    print("  MINERVINI SEPA BACKTEST")
    print("  Level 1: Trend Template (8 criteria)")
    print("  Level 2: Trend Template + VCP + Breakout + Volume")
    print("=" * 90)
    print()

    # Fetch data — use 3yr for consistency with our other analyses
    print("  Fetching 3 years of data...\n")
    all_tickers = ["SPY"] + get_all_tickers(cfg)
    data = fetch_batch(all_tickers, period="3y", verbose=True)

    if not data:
        print("  [ERROR] No data fetched.")
        exit(1)

    print(f"\n  Successfully fetched {len(data)} tickers.\n")

    # Run backtest
    print("  Running Minervini backtest...\n")
    events_df = run_minervini_backtest(data, cfg, test_frequency=5, verbose=True)

    if events_df.empty:
        print("  [ERROR] No events collected.")
        exit(1)

    print(f"\n  Collected {len(events_df)} observations.\n")

    # Print all sections
    print_section1(events_df)
    print_section2(events_df)
    print_section3(events_df)
    print_section4(events_df)

    print(f"\n{'=' * 90}")
    print(f"  BACKTEST COMPLETE")
    print(f"{'=' * 90}\n")
