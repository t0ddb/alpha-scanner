from __future__ import annotations

"""
indicators.py -- Breakout scoring engine.

Computes 7 scored indicators per ticker (+ display-only MA Alignment and
Near 52w High for the dashboard). All regime-independent. Weights updated
2026-04-24 to Scheme C after full audit — see backtest_results/audit_*.

    GRADIENT (signal strength → proportional points):
        Relative Strength vs SPY     0-3.0 pts  (top signal, gradient by pctl)
        Higher Lows                  0-0.5 pts  (halved from 1.0 — near-zero
                                                 incremental edge in 3yr + 12mo)

    BINARY (fire if triggered, regardless of RS regime):
        Ichimoku Cloud               2.0 pts    (+11.9% incremental edge)
        Rate of Change               1.5 pts    (+7.5% incremental edge)
        Dual-TF RS Acceleration      2.5 pts    (was 0.5; +5.5% incremental
                                                 edge 3yr, +9.6% 12mo — biggest
                                                 under-weighted signal)
        ATR Expansion                0.5 pts    (+5.2% incremental edge)
        Chaikin Money Flow           0.0 pts    (was 1.5; NEGATIVE incremental
                                                 edge in both 3yr and 12mo —
                                                 audit 2026-04-24; still
                                                 computed for dashboard)

    Max weighted score: 10.0

Scheme C portfolio backtest (13mo, production-equivalent config, 5 path
starts) vs baseline:
    Return:    +438%  →  +598%   (+160 pp, +37% relative)
    Sharpe:     1.97  →   2.21   (+12%)
    Max DD:    -21.2% →  -20.3%  (slightly better)
    Win rate:  50.7%  →   59.0%  (+8.3 pp)
    Entry threshold: 8.5 → 9.0

Dropped: MA Alignment (-9.3% incremental edge, harmful when RS strong)
Dropped: Near 52w High (-3.3% incremental edge, redundant with RS)
Dropped earlier: Volume Spike, BB Squeeze, MACD Crossover, ADX Trend

Usage:
    from indicators import score_all, compute_all_indicators, score_ticker

    results = score_all(data, cfg)
"""

import pandas as pd
import numpy as np
from config import load_config, get_indicator_config, get_scoring_config, get_ticker_metadata


# =============================================================
# INDICATOR MAX WEIGHTS (from 3-year conditional edge analysis)
# =============================================================
# All regime-independent. Weights reflect incremental edge after
# controlling for RS strength (conditional edge analysis, 3yr data).
# =============================================================
INDICATOR_WEIGHTS = {
    # Gradient — max achievable weight
    "relative_strength": 3.0,
    "higher_lows":       0.5,   # was 1.0; halved — near-zero incremental edge
    # Binary — full weight if triggered (weight 0 skips; indicator still computed)
    "ichimoku_cloud":    2.0,
    "roc":               1.5,
    "dual_tf_rs":        2.5,   # was 0.5; biggest under-weighted signal per audit
    "atr_expansion":     0.5,
    "cmf":               0.0,   # was 1.5; dropped — NEGATIVE incremental edge
}

MAX_SCORE = 10.0

INDICATOR_LABELS = {
    "relative_strength": "Relative Strength",
    "higher_lows":       "Higher Lows",
    "ichimoku_cloud":    "Ichimoku Cloud",
    "cmf":               "Chaikin Money Flow",
    "roc":               "Rate of Change",
    "dual_tf_rs":        "Dual-TF RS",
    "atr_expansion":     "ATR Expansion",
}

# Gradient scoring breakpoints (from bucket analysis)
RS_GRADIENT = [
    # (min_percentile, points)
    (90, 3.0),
    (80, 2.4),
    (70, 1.8),
    (60, 1.2),
    (50, 0.6),
]

HIGHER_LOWS_GRADIENT = [
    # (min_count, points) — halved from prior (1.0 max → 0.5 max)
    # per Scheme C audit: HL has near-zero incremental edge after RS
    (5, 0.5),
    (4, 0.375),
    (3, 0.25),
    (2, 0.125),
]


# =============================================================
# INDICATOR 1: Relative Strength vs. SPY  (weight: 0-2.5, gradient)
# =============================================================
def check_relative_strength(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    period: int = 63,
    all_rs_values: list[float] = None,
    percentile: float = 75,
) -> dict:
    """
    Compare stock's performance over `period` days to SPY.
    Flag if stock's RS is in the top quartile across all tickers.
    """
    if len(df) < period + 1 or len(benchmark_df) < period + 1:
        return {
            "triggered": False, "rs_value": 0,
            "stock_return": 0, "benchmark_return": 0, "rs_percentile": 0,
        }

    stock_return = (df["Close"].iloc[-1] / df["Close"].iloc[-period - 1]) - 1
    bench_return = (benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[-period - 1]) - 1

    if bench_return == 0:
        rs_value = 0
    else:
        rs_value = stock_return / bench_return if bench_return > 0 else stock_return - bench_return

    rs_percentile = 0
    triggered = False
    if all_rs_values and len(all_rs_values) > 1:
        rs_percentile = (sum(1 for v in all_rs_values if v <= rs_value) / len(all_rs_values)) * 100
        triggered = rs_percentile >= percentile
    else:
        triggered = False

    return {
        "triggered": triggered,
        "rs_value": round(rs_value, 4),
        "stock_return": round(stock_return, 4),
        "benchmark_return": round(bench_return, 4),
        "rs_percentile": round(rs_percentile, 1),
    }


# =============================================================
# INDICATOR 2: Ichimoku Cloud  (weight: 1.5/0.5, confirmation)
# =============================================================
def check_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> dict:
    """
    Flag if price is above the Ichimoku Cloud and the cloud is bullish
    (Senkou Span A > Senkou Span B).
    """
    if len(df) < senkou_b + kijun:
        return {
            "triggered": False, "above_cloud": False,
            "cloud_bullish": False, "tenkan_above_kijun": False,
            "cloud_top": None, "senkou_a": None, "senkou_b": None,
        }

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
    kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_line = ((high.rolling(window=senkou_b).max() + low.rolling(window=senkou_b).min()) / 2).shift(kijun)

    current_close = close.iloc[-1]
    current_senkou_a = senkou_a.iloc[-1]
    current_senkou_b = senkou_b_line.iloc[-1]
    current_tenkan = tenkan_sen.iloc[-1]
    current_kijun = kijun_sen.iloc[-1]

    if pd.isna(current_senkou_a) or pd.isna(current_senkou_b):
        return {
            "triggered": False, "above_cloud": False,
            "cloud_bullish": False, "tenkan_above_kijun": False,
            "cloud_top": None, "senkou_a": None, "senkou_b": None,
        }

    cloud_top = max(current_senkou_a, current_senkou_b)
    above_cloud = current_close > cloud_top
    cloud_bullish = current_senkou_a > current_senkou_b
    tenkan_above_kijun = current_tenkan > current_kijun

    return {
        "triggered": above_cloud and cloud_bullish,
        "above_cloud": bool(above_cloud),
        "cloud_bullish": bool(cloud_bullish),
        "tenkan_above_kijun": bool(tenkan_above_kijun),
        "cloud_top": round(cloud_top, 2),
        "senkou_a": round(float(current_senkou_a), 2),
        "senkou_b": round(float(current_senkou_b), 2),
    }


# =============================================================
# INDICATOR 3: Higher Lows  (weight: 0-1.5, gradient)
# =============================================================
def check_higher_lows(df: pd.DataFrame, num_periods: int = 4, period_length: int = 5) -> dict:
    """
    Flag if the stock has made consecutive higher lows over the last
    N periods (each period_length days long). Detects staircase uptrends.
    """
    total_needed = num_periods * period_length
    if len(df) < total_needed + period_length:
        return {"triggered": False, "consecutive_higher_lows": 0}

    lows = []
    for i in range(num_periods + 1):
        end_idx = len(df) - (i * period_length)
        start_idx = end_idx - period_length
        if start_idx < 0:
            break
        period_low = df["Low"].iloc[start_idx:end_idx].min()
        lows.append(period_low)

    lows.reverse()

    consecutive = 0
    for i in range(1, len(lows)):
        if lows[i] > lows[i - 1]:
            consecutive += 1
        else:
            consecutive = 0

    return {
        "triggered": consecutive >= num_periods,
        "consecutive_higher_lows": consecutive,
    }


# =============================================================
# INDICATOR 4: Moving Average Alignment  (weight: 1.0/0.0, confirmation)
# =============================================================
def check_ma_alignment(
    df: pd.DataFrame,
    short_period: int = 50,
    long_period: int = 200,
    golden_cross_lookback: int = 20,
) -> dict:
    """
    Check if price is above both 50-day and 200-day SMA.
    Bonus detection: recent golden cross (50 crossing above 200).
    """
    if len(df) < long_period + golden_cross_lookback:
        return {
            "triggered": False, "price_above_50": False, "price_above_200": False,
            "golden_cross_recent": False, "current_close": 0, "sma_50": 0, "sma_200": 0,
        }

    close = df["Close"]
    sma_short = close.rolling(window=short_period).mean()
    sma_long = close.rolling(window=long_period).mean()

    current_close = close.iloc[-1]
    current_sma_short = sma_short.iloc[-1]
    current_sma_long = sma_long.iloc[-1]

    price_above_50 = current_close > current_sma_short
    price_above_200 = current_close > current_sma_long

    golden_cross_recent = False
    for i in range(-golden_cross_lookback, 0):
        if i - 1 < -len(sma_short):
            continue
        prev_short = sma_short.iloc[i - 1]
        curr_short = sma_short.iloc[i]
        prev_long = sma_long.iloc[i - 1]
        curr_long = sma_long.iloc[i]

        if prev_short <= prev_long and curr_short > curr_long:
            golden_cross_recent = True
            break

    return {
        "triggered": price_above_50 and price_above_200,
        "price_above_50": bool(price_above_50),
        "price_above_200": bool(price_above_200),
        "golden_cross_recent": golden_cross_recent,
        "current_close": round(current_close, 2),
        "sma_50": round(current_sma_short, 2),
        "sma_200": round(current_sma_long, 2),
    }


# =============================================================
# INDICATOR 5: Rate of Change  (weight: 1.0, independent)
# =============================================================
def check_roc(df: pd.DataFrame, period: int = 21, threshold: float = 5.0) -> dict:
    """
    Flag if price rate of change over N days exceeds threshold %.
    """
    if len(df) < period + 1:
        return {"triggered": False, "roc": 0}

    current = df["Close"].iloc[-1]
    past = df["Close"].iloc[-period - 1]
    roc = ((current - past) / past) * 100

    return {
        "triggered": roc > threshold,
        "roc": round(roc, 2),
    }


# =============================================================
# INDICATOR 6: Chaikin Money Flow  (weight: 1.5/0.5, rescue)
# =============================================================
def check_cmf(df: pd.DataFrame, period: int = 20, threshold: float = 0.05) -> dict:
    """
    Flag if CMF is above threshold, indicating institutional buying pressure.
    """
    if len(df) < period + 1:
        return {"triggered": False, "cmf": 0}

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    hl_range = high - low
    mf_multiplier = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    mf_multiplier = mf_multiplier.fillna(0)

    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum().replace(0, np.nan)

    current_cmf = cmf.iloc[-1]

    if pd.isna(current_cmf):
        return {"triggered": False, "cmf": 0}

    return {
        "triggered": current_cmf > threshold,
        "cmf": round(current_cmf, 4),
    }


# =============================================================
# INDICATOR 7: Near 52-Week High  (weight: 1.0)
# =============================================================
def check_near_52w_high(df: pd.DataFrame, threshold_pct: float = 0.02) -> dict:
    """
    Flag if current close is within threshold_pct of the 52-week high.
    """
    lookback = min(252, len(df))
    if lookback < 20:
        return {"triggered": False, "current_close": 0, "high_52w": 0, "pct_from_high": 0}

    high_52w = df["High"].iloc[-lookback:].max()
    current_close = df["Close"].iloc[-1]
    pct_from_high = (current_close - high_52w) / high_52w

    return {
        "triggered": pct_from_high >= -threshold_pct,
        "current_close": round(current_close, 2),
        "high_52w": round(high_52w, 2),
        "pct_from_high": round(pct_from_high, 4),
    }


# =============================================================
# INDICATOR 8: ATR Expansion  (weight: 1.0, independent)
# =============================================================
def check_atr_expansion(
    df: pd.DataFrame,
    period: int = 14,
    lookback: int = 50,
    threshold_percentile: float = 80,
) -> dict:
    """
    Flag if current ATR is in the top percentile of its recent range,
    indicating expanding volatility (often accompanies breakout moves).
    """
    if len(df) < period + lookback:
        return {"triggered": False, "atr": 0, "atr_percentile": 0}

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    current_atr = atr.iloc[-1]
    recent_atr = atr.iloc[-lookback:]

    percentile = (recent_atr <= current_atr).mean() * 100

    return {
        "triggered": percentile >= threshold_percentile,
        "atr": round(current_atr, 4),
        "atr_percentile": round(percentile, 1),
    }


# =============================================================
# INDICATOR 8: Dual-TF RS Acceleration  (weight: 0.5, binary)
# =============================================================
def check_dual_tf_rs(
    rs_126d_pctl: float = 0,
    rs_63d_pctl: float = 0,
    rs_21d_pctl: float = 0,
    long_min: float = 70,
    short_min: float = 80,
) -> dict:
    """
    Flag if RS is strong across multiple timeframes AND accelerating.
    Requires pre-computed percentile values at 3 timeframes.

    Triggers if:
      - 126d RS >= 70th AND 63d RS > 126d RS (accelerating), OR
      - 63d RS >= 80th AND 21d RS >= 80th (strong and getting stronger)
    """
    accelerating = rs_63d_pctl > rs_126d_pctl

    cond_a = rs_126d_pctl >= long_min and accelerating
    cond_b = rs_63d_pctl >= short_min and rs_21d_pctl >= short_min

    return {
        "triggered": cond_a or cond_b,
        "rs_126d_percentile": round(rs_126d_pctl, 1),
        "rs_63d_percentile": round(rs_63d_pctl, 1),
        "rs_21d_percentile": round(rs_21d_pctl, 1),
        "accelerating": accelerating,
    }


# =============================================================
# AGGREGATE: Compute all indicators for one ticker
# =============================================================
def compute_all_indicators(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    cfg: dict,
    all_rs_values: list[float] = None,
    multi_tf_rs: dict = None,
) -> dict:
    """
    Run all 8 scored indicators on a single ticker's data.
    MA Alignment still computed for dashboard display but NOT scored.
    """
    ind_cfg = get_indicator_config(cfg)

    # ── Scored indicators ──
    results = {
        "relative_strength": check_relative_strength(
            df,
            benchmark_df,
            period=ind_cfg["relative_strength"]["period"],
            all_rs_values=all_rs_values,
            percentile=ind_cfg["relative_strength"]["percentile"],
        ),
        "ichimoku_cloud": check_ichimoku(
            df,
            tenkan=ind_cfg["ichimoku_cloud"]["tenkan"],
            kijun=ind_cfg["ichimoku_cloud"]["kijun"],
            senkou_b=ind_cfg["ichimoku_cloud"]["senkou_b"],
        ),
        "higher_lows": check_higher_lows(
            df,
            num_periods=ind_cfg["higher_lows"]["num_periods"],
            period_length=ind_cfg["higher_lows"]["period_length"],
        ),
        "roc": check_roc(
            df,
            period=ind_cfg["roc"]["period"],
            threshold=ind_cfg["roc"]["threshold"],
        ),
        "cmf": check_cmf(
            df,
            period=ind_cfg["cmf"]["period"],
            threshold=ind_cfg["cmf"]["threshold"],
        ),
        "atr_expansion": check_atr_expansion(
            df,
            period=ind_cfg["atr_expansion"]["period"],
            lookback=ind_cfg["atr_expansion"]["lookback"],
            threshold_percentile=ind_cfg["atr_expansion"]["threshold_percentile"],
        ),
    }

    # Dual-TF RS — requires pre-computed multi-timeframe percentiles
    if multi_tf_rs is not None:
        results["dual_tf_rs"] = check_dual_tf_rs(
            rs_126d_pctl=multi_tf_rs.get("rs_126d_pctl", 0),
            rs_63d_pctl=multi_tf_rs.get("rs_63d_pctl", 0),
            rs_21d_pctl=multi_tf_rs.get("rs_21d_pctl", 0),
        )
    else:
        results["dual_tf_rs"] = {"triggered": False, "rs_126d_percentile": 0,
                                  "rs_63d_percentile": 0, "rs_21d_percentile": 0,
                                  "accelerating": False}

    # ── Display-only indicators (computed but NOT scored) ──
    # MA Alignment: -9.3% incremental edge, harmful when RS strong
    results["moving_averages"] = check_ma_alignment(
        df,
        short_period=ind_cfg["moving_averages"]["short_period"],
        long_period=ind_cfg["moving_averages"]["long_period"],
        golden_cross_lookback=ind_cfg["moving_averages"]["golden_cross_lookback"],
    )

    # Near 52w High: -3.3% incremental edge, redundant with RS
    near_52w_cfg = ind_cfg.get("near_52w_high", {})
    results["near_52w_high"] = check_near_52w_high(
        df,
        threshold_pct=near_52w_cfg.get("threshold_pct", 0.02),
    )

    return results


# =============================================================
# SCORING: Weighted score for a single ticker
# =============================================================
def score_ticker(indicators: dict) -> dict:
    """
    Compute breakout score from indicator results.

    All indicators are regime-independent (no RS-based weight switching).
    Gradient indicators (RS, Higher Lows) scale by signal strength.
    Binary indicators (Ichimoku, CMF, ROC, Dual-TF RS, ATR) award full weight.

    Returns:
        {
            "score": float,        # weighted score (0 to 10.0)
            "max_score": float,    # 10.0
            "signals": [str, ...], # which indicators contributed >0 points
            "signal_weights": {str: float, ...},  # actual weight contributed
        }
    """
    signals = []
    signal_weights = {}
    total = 0.0

    # ── Gradient — Relative Strength ──
    rs = indicators.get("relative_strength", {})
    rs_pctl = rs.get("rs_percentile", 0)
    rs_pts = 0.0
    for min_pctl, pts in RS_GRADIENT:
        if rs_pctl >= min_pctl:
            rs_pts = pts
            break
    if rs_pts > 0:
        signals.append("relative_strength")
        signal_weights["relative_strength"] = rs_pts
        total += rs_pts

    # ── Gradient — Higher Lows ──
    hl = indicators.get("higher_lows", {})
    hl_count = hl.get("consecutive_higher_lows", 0)
    hl_pts = 0.0
    for min_count, pts in HIGHER_LOWS_GRADIENT:
        if hl_count >= min_count:
            hl_pts = pts
            break
    if hl_pts > 0:
        signals.append("higher_lows")
        signal_weights["higher_lows"] = hl_pts
        total += hl_pts

    # ── Binary — all regime-independent ──
    # Skip indicators with weight 0 (computed but not scored; e.g. CMF per Scheme C).
    for name in ("ichimoku_cloud", "cmf", "roc", "dual_tf_rs", "atr_expansion"):
        weight = INDICATOR_WEIGHTS.get(name, 0.0)
        if weight <= 0:
            continue
        result = indicators.get(name, {})
        if result.get("triggered", False):
            signals.append(name)
            signal_weights[name] = weight
            total += weight

    return {
        "score": round(total, 1),
        "max_score": MAX_SCORE,
        "signals": signals,
        "signal_weights": signal_weights,
    }


# =============================================================
# SCORE ALL: Run the full pipeline across all tickers
# =============================================================
def score_all(data: dict[str, pd.DataFrame], cfg: dict) -> list[dict]:
    """
    Run indicators and scoring on every ticker in the data dict.

    1. Computes raw RS values for all tickers first (for percentile ranking)
    2. Runs all 8 indicators per ticker
    3. Computes weighted score
    4. Returns sorted results (highest score first)
    """
    benchmark_ticker = cfg["benchmark"]["ticker"]
    benchmark_df = data.get(benchmark_ticker)
    metadata = get_ticker_metadata(cfg)
    ind_cfg = get_indicator_config(cfg)

    if benchmark_df is None:
        # Surface as an exception rather than returning [] silently —
        # an empty results list cascades into empty subsector metrics,
        # empty dashboard, and zero observable failure mode.
        raise RuntimeError(
            f"Benchmark {benchmark_ticker} not found in fetched data. "
            f"Cannot compute relative strength. This usually means the "
            f"benchmark fetch failed; check yfinance connectivity."
        )

    # ---- Pass 1: Compute raw RS values at multiple timeframes ----
    rs_period = ind_cfg["relative_strength"]["period"]
    tf_periods = [rs_period, 126, 63, 21]  # primary + multi-TF

    raw_rs_by_tf = {p: {} for p in tf_periods}
    for ticker, df in data.items():
        if ticker == benchmark_ticker:
            continue
        for period in tf_periods:
            if len(df) < period + 1 or len(benchmark_df) < period + 1:
                continue
            stock_ret = (df["Close"].iloc[-1] / df["Close"].iloc[-period - 1]) - 1
            bench_ret = (benchmark_df["Close"].iloc[-1] / benchmark_df["Close"].iloc[-period - 1]) - 1
            if bench_ret != 0:
                raw_rs_by_tf[period][ticker] = stock_ret / bench_ret if bench_ret > 0 else stock_ret - bench_ret
            else:
                raw_rs_by_tf[period][ticker] = 0

    all_rs_values = list(raw_rs_by_tf[rs_period].values())

    # Pre-compute percentile ranks for multi-TF RS
    def _percentile_rank(val, all_vals):
        if not all_vals:
            return 0.0
        return sum(1 for v in all_vals if v <= val) / len(all_vals) * 100

    multi_tf_pctls = {}
    for ticker in data:
        if ticker == benchmark_ticker:
            continue
        multi_tf_pctls[ticker] = {
            "rs_126d_pctl": _percentile_rank(
                raw_rs_by_tf[126].get(ticker, 0), list(raw_rs_by_tf[126].values())
            ) if ticker in raw_rs_by_tf[126] else 0,
            "rs_63d_pctl": _percentile_rank(
                raw_rs_by_tf[63].get(ticker, 0), list(raw_rs_by_tf[63].values())
            ) if ticker in raw_rs_by_tf[63] else 0,
            "rs_21d_pctl": _percentile_rank(
                raw_rs_by_tf[21].get(ticker, 0), list(raw_rs_by_tf[21].values())
            ) if ticker in raw_rs_by_tf[21] else 0,
        }

    # ---- Pass 2: Compute all indicators and score ----
    results = []
    for ticker, df in data.items():
        if ticker == benchmark_ticker:
            continue

        meta = metadata.get(ticker, {})
        indicators = compute_all_indicators(
            df, benchmark_df, cfg,
            all_rs_values=all_rs_values,
            multi_tf_rs=multi_tf_pctls.get(ticker),
        )
        scoring = score_ticker(indicators)

        results.append({
            "ticker": ticker,
            "name": meta.get("name", ""),
            "sector": meta.get("sector_name", ""),
            "subsector": meta.get("subsector_name", ""),
            "score": scoring["score"],
            "max_score": scoring["max_score"],
            "signals": scoring["signals"],
            "signal_weights": scoring["signal_weights"],
            "indicators": indicators,
        })

    results.sort(key=lambda x: (-x["score"], x["ticker"]))

    return results


# =============================================================
# DISPLAY: Print a readable scorecard
# =============================================================
def print_scorecard(results: list[dict], min_score: float = 0) -> None:
    """Print a formatted scorecard of all scored tickers."""
    filtered = [r for r in results if r["score"] >= min_score]

    print(f"\n{'='*80}")
    print(f"  BREAKOUT SCORECARD -- {len(filtered)} tickers (min score: {min_score})")
    print(f"{'='*80}\n")

    if not filtered:
        print("  No tickers meet the minimum score threshold.\n")
        return

    for r in filtered:
        signals_str = ", ".join(
            f"{INDICATOR_LABELS.get(s, s)} ({r['signal_weights'].get(s, 0)})"
            for s in r["signals"]
        ) if r["signals"] else "none"

        print(f"  [{r['score']:5.1f}/{r['max_score']}]  {r['ticker']:8s}  {r['name'][:30]:30s}  {r['subsector'][:30]:30s}")
        print(f"          Signals: {signals_str}")

        # Show key details for triggered indicators
        ind = r["indicators"]
        details = []
        if ind["relative_strength"]["triggered"]:
            details.append(f"RS: {ind['relative_strength']['rs_percentile']:.0f}th pctl")
        if ind["ichimoku_cloud"]["triggered"]:
            extra = " + TK>KJ" if ind["ichimoku_cloud"]["tenkan_above_kijun"] else ""
            details.append(f"Above cloud{extra}")
        if ind["higher_lows"]["triggered"]:
            details.append(f"{ind['higher_lows']['consecutive_higher_lows']} higher lows")
        if ind["roc"]["triggered"]:
            details.append(f"ROC: {ind['roc']['roc']:+.1f}%")
        if ind["cmf"]["triggered"]:
            details.append(f"CMF: {ind['cmf']['cmf']:+.4f}")
        if ind["dual_tf_rs"]["triggered"]:
            accel = " accel" if ind["dual_tf_rs"]["accelerating"] else ""
            details.append(f"Dual-TF RS{accel}")
        if ind["atr_expansion"]["triggered"]:
            details.append(f"ATR: {ind['atr_expansion']['atr_percentile']:.0f}th pctl")
        # Display-only (not scored)
        if ind["moving_averages"]["triggered"]:
            gc = " + golden cross" if ind["moving_averages"]["golden_cross_recent"] else ""
            details.append(f"[MA: above 50/200{gc}]")
        if ind["near_52w_high"]["triggered"]:
            pct = ind["near_52w_high"]["pct_from_high"] * 100
            details.append(f"[{pct:+.1f}% from 52w high]")

        if details:
            print(f"          Detail:  {' | '.join(details)}")
        print()

    print(f"{'='*80}\n")


# =============================================================
# Quick test
# =============================================================
if __name__ == "__main__":
    from data_fetcher import fetch_batch

    cfg = load_config()

    test_tickers = ["SPY", "NVDA", "AMD", "GLD", "NEM", "BTC-USD"]
    print("=" * 80)
    print("  WEIGHTED INDICATOR ENGINE -- SMOKE TEST")
    print("=" * 80)
    print("  Fetching 1 year of data for test tickers...\n")

    data = fetch_batch(test_tickers, period="1y", verbose=True)

    if data:
        print("\n  Running weighted indicator engine...")
        results = score_all(data, cfg)
        print_scorecard(results, min_score=0)
