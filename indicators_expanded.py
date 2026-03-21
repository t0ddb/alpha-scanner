from __future__ import annotations

"""
indicators_expanded.py — Extended indicator library for testing.

Adds 11 new indicators to the original 5 for a total of 16.
Each returns a dict with "triggered": bool plus supporting values.

Original 5:
    1. Volume Spike
    2. Near 52w High
    3. Bollinger Band Squeeze → Expansion
    4. Relative Strength vs SPY
    5. Moving Average Alignment

New — Momentum:
    6. RSI Momentum (RSI trending above 50-60)
    7. MACD Crossover
    8. ADX Trend Strength
    9. Rate of Change (ROC)

New — Volume Flow:
    10. OBV Trend (On-Balance Volume)
    11. Chaikin Money Flow (CMF)

New — Volatility / Structure:
    12. ATR Expansion
    13. Consolidation Tightness

New — Price Pattern:
    14. Higher Lows
    15. Donchian Channel Breakout
    16. Ichimoku Cloud
"""

import pandas as pd
import numpy as np


# =============================================================
# 6. RSI MOMENTUM
# =============================================================
def check_rsi_momentum(df: pd.DataFrame, period: int = 14, threshold: float = 55) -> dict:
    """
    Flag if RSI is above threshold, indicating sustained bullish momentum.
    Not overbought/oversold — just confirming momentum direction.

    Returns:
        triggered: True if RSI > threshold
        rsi: current RSI value
    """
    if len(df) < period + 1:
        return {"triggered": False, "rsi": 0}

    close = df["Close"]
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    current_rsi = rsi.iloc[-1]

    if pd.isna(current_rsi):
        return {"triggered": False, "rsi": 0}

    return {
        "triggered": current_rsi > threshold,
        "rsi": round(current_rsi, 2),
    }


# =============================================================
# 7. MACD CROSSOVER
# =============================================================
def check_macd_crossover(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    lookback: int = 5,
) -> dict:
    """
    Flag if MACD line recently crossed above the signal line (bullish crossover)
    or if MACD histogram is positive and growing.

    Returns:
        triggered: True if bullish crossover in last `lookback` days
        macd: current MACD value
        signal_line: current signal value
        histogram: current histogram value
        crossover_recent: True if crossover happened recently
    """
    if len(df) < slow + signal + lookback:
        return {"triggered": False, "macd": 0, "signal_line": 0, "histogram": 0, "crossover_recent": False}

    close = df["Close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]
    current_hist = histogram.iloc[-1]

    # Check for recent bullish crossover
    crossover_recent = False
    for i in range(-lookback, 0):
        if i - 1 < -len(macd_line):
            continue
        prev_macd = macd_line.iloc[i - 1]
        curr_macd = macd_line.iloc[i]
        prev_sig = signal_line.iloc[i - 1]
        curr_sig = signal_line.iloc[i]

        if prev_macd <= prev_sig and curr_macd > curr_sig:
            crossover_recent = True
            break

    # Triggered if MACD above signal and histogram positive
    triggered = current_macd > current_signal and current_hist > 0

    return {
        "triggered": triggered,
        "macd": round(current_macd, 4),
        "signal_line": round(current_signal, 4),
        "histogram": round(current_hist, 4),
        "crossover_recent": crossover_recent,
    }


# =============================================================
# 8. ADX TREND STRENGTH
# =============================================================
def check_adx(df: pd.DataFrame, period: int = 14, threshold: float = 25) -> dict:
    """
    Flag if ADX is above threshold, indicating a strong trend.
    Also checks if +DI > -DI (bullish direction).

    Returns:
        triggered: True if ADX > threshold AND +DI > -DI
        adx: current ADX value
        plus_di: current +DI
        minus_di: current -DI
    """
    if len(df) < period * 3:
        return {"triggered": False, "adx": 0, "plus_di": 0, "minus_di": 0}

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=df.index)

    # Smoothed averages
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.rolling(window=period).mean()

    current_adx = adx.iloc[-1]
    current_plus = plus_di.iloc[-1]
    current_minus = minus_di.iloc[-1]

    if pd.isna(current_adx):
        return {"triggered": False, "adx": 0, "plus_di": 0, "minus_di": 0}

    return {
        "triggered": current_adx > threshold and current_plus > current_minus,
        "adx": round(current_adx, 2),
        "plus_di": round(current_plus, 2),
        "minus_di": round(current_minus, 2),
    }


# =============================================================
# 9. RATE OF CHANGE (ROC)
# =============================================================
def check_roc(df: pd.DataFrame, period: int = 21, threshold: float = 5.0) -> dict:
    """
    Flag if price rate of change over N days exceeds threshold %.

    Returns:
        triggered: True if ROC > threshold
        roc: current ROC percentage
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
# 10. ON-BALANCE VOLUME (OBV) TREND
# =============================================================
def check_obv_trend(df: pd.DataFrame, short_period: int = 20, long_period: int = 50) -> dict:
    """
    Flag if OBV's short-term moving average is above its long-term MA,
    indicating sustained accumulation.

    Returns:
        triggered: True if OBV short MA > OBV long MA and OBV is rising
        obv_trend: "bullish" or "bearish"
        obv_slope: recent slope of OBV (positive = accumulation)
    """
    if len(df) < long_period + 10:
        return {"triggered": False, "obv_trend": "neutral", "obv_slope": 0}

    close = df["Close"]
    volume = df["Volume"]

    # Compute OBV
    direction = np.sign(close.diff())
    obv = (direction * volume).cumsum()

    obv_short_ma = obv.rolling(window=short_period).mean()
    obv_long_ma = obv.rolling(window=long_period).mean()

    # OBV slope over last 10 days (normalized)
    recent_obv = obv.iloc[-10:]
    if len(recent_obv) >= 2:
        obv_slope = (recent_obv.iloc[-1] - recent_obv.iloc[0]) / abs(recent_obv.iloc[0]) if recent_obv.iloc[0] != 0 else 0
    else:
        obv_slope = 0

    current_short = obv_short_ma.iloc[-1]
    current_long = obv_long_ma.iloc[-1]

    bullish = current_short > current_long and obv_slope > 0

    return {
        "triggered": bullish,
        "obv_trend": "bullish" if bullish else "bearish",
        "obv_slope": round(obv_slope, 4),
    }


# =============================================================
# 11. CHAIKIN MONEY FLOW (CMF)
# =============================================================
def check_cmf(df: pd.DataFrame, period: int = 20, threshold: float = 0.05) -> dict:
    """
    Flag if CMF is above threshold, indicating institutional buying pressure.

    Returns:
        triggered: True if CMF > threshold
        cmf: current CMF value (-1 to +1)
    """
    if len(df) < period + 1:
        return {"triggered": False, "cmf": 0}

    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    volume = df["Volume"]

    # Money Flow Multiplier
    hl_range = high - low
    mf_multiplier = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
    mf_multiplier = mf_multiplier.fillna(0)

    # Money Flow Volume
    mf_volume = mf_multiplier * volume

    # CMF
    cmf = mf_volume.rolling(window=period).sum() / volume.rolling(window=period).sum().replace(0, np.nan)

    current_cmf = cmf.iloc[-1]

    if pd.isna(current_cmf):
        return {"triggered": False, "cmf": 0}

    return {
        "triggered": current_cmf > threshold,
        "cmf": round(current_cmf, 4),
    }


# =============================================================
# 12. ATR EXPANSION
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

    Returns:
        triggered: True if ATR > threshold_percentile of recent ATR
        atr: current ATR
        atr_percentile: where current ATR ranks (0-100)
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
    recent_atr = atr.iloc[-(lookback):]

    percentile = (recent_atr <= current_atr).mean() * 100

    return {
        "triggered": percentile >= threshold_percentile,
        "atr": round(current_atr, 4),
        "atr_percentile": round(percentile, 1),
    }


# =============================================================
# 13. CONSOLIDATION TIGHTNESS
# =============================================================
def check_consolidation_tightness(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 50,
    threshold: float = 0.5,
) -> dict:
    """
    Flag if the recent trading range is much tighter than the longer-term range.
    Ratio < threshold means recent action is compressed = potential breakout setup.

    Note: This fires when consolidation IS happening (before the breakout).
    Pair with expansion indicators for "tight then breakout" signals.

    Returns:
        triggered: True if range ratio < threshold
        range_ratio: short range / long range (lower = tighter)
    """
    if len(df) < long_window + 1:
        return {"triggered": False, "range_ratio": 0}

    close = df["Close"]

    short_range = close.iloc[-short_window:].max() - close.iloc[-short_window:].min()
    long_range = close.iloc[-long_window:].max() - close.iloc[-long_window:].min()

    if long_range == 0:
        return {"triggered": False, "range_ratio": 0}

    ratio = short_range / long_range

    return {
        "triggered": ratio < threshold,
        "range_ratio": round(ratio, 4),
    }


# =============================================================
# 14. HIGHER LOWS
# =============================================================
def check_higher_lows(df: pd.DataFrame, num_periods: int = 4, period_length: int = 5) -> dict:
    """
    Flag if the stock has made consecutive higher lows over the last
    N periods (each period_length days long). This detects staircase
    uptrend patterns.

    Returns:
        triggered: True if N consecutive higher lows detected
        consecutive_higher_lows: count of consecutive higher low periods
    """
    total_needed = num_periods * period_length
    if len(df) < total_needed + period_length:
        return {"triggered": False, "consecutive_higher_lows": 0}

    # Split recent data into periods and find each period's low
    lows = []
    for i in range(num_periods + 1):
        end_idx = len(df) - (i * period_length)
        start_idx = end_idx - period_length
        if start_idx < 0:
            break
        period_low = df["Low"].iloc[start_idx:end_idx].min()
        lows.append(period_low)

    # Reverse so oldest is first
    lows.reverse()

    # Count consecutive higher lows
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
# 15. DONCHIAN CHANNEL BREAKOUT
# =============================================================
def check_donchian_breakout(df: pd.DataFrame, period: int = 55) -> dict:
    """
    Flag if price is breaking above the N-day Donchian Channel
    (highest high of last N days). Simpler and more direct than
    Bollinger Bands for pure breakout detection.

    Returns:
        triggered: True if close > Donchian upper
        current_close: current price
        donchian_high: N-day high
        pct_above: how far above the channel (negative = below)
    """
    if len(df) < period + 1:
        return {"triggered": False, "current_close": 0, "donchian_high": 0, "pct_above": 0}

    # Donchian high = highest high of last N days (excluding today)
    donchian_high = df["High"].iloc[-(period + 1):-1].max()
    current_close = df["Close"].iloc[-1]

    pct_above = (current_close - donchian_high) / donchian_high

    return {
        "triggered": current_close > donchian_high,
        "current_close": round(current_close, 2),
        "donchian_high": round(donchian_high, 2),
        "pct_above": round(pct_above, 4),
    }


# =============================================================
# 16. ICHIMOKU CLOUD
# =============================================================
def check_ichimoku(df: pd.DataFrame, tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> dict:
    """
    Flag if price is above the Ichimoku Cloud and the cloud is bullish
    (Senkou Span A > Senkou Span B).

    Returns:
        triggered: True if price above cloud and cloud is bullish
        above_cloud: True if price is above both Senkou spans
        cloud_bullish: True if Span A > Span B
        tenkan_above_kijun: True if Tenkan > Kijun (short-term momentum)
    """
    if len(df) < senkou_b + kijun:
        return {
            "triggered": False, "above_cloud": False,
            "cloud_bullish": False, "tenkan_above_kijun": False,
        }

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # Tenkan-sen (Conversion Line)
    tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2

    # Kijun-sen (Base Line)
    kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2

    # Senkou Span A (Leading Span A) — shifted forward 26 periods
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)

    # Senkou Span B (Leading Span B) — shifted forward 26 periods
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
    }


# =============================================================
# 17. RISING MOVING AVERAGE SLOPE
# =============================================================
def check_rising_ma_slope(
    df: pd.DataFrame,
    long_ma: int = 200,
    short_ma: int = 50,
    slope_lookback: int = 30,
) -> dict:
    """
    Flag if both the 50-day and 200-day SMA slopes are positive (rising).
    Minervini SEPA requires the 200-day MA itself to be rising.

    Returns:
        triggered: True if both MAs are rising
        sma_50_slope: 50-day MA slope (% change over lookback)
        sma_200_slope: 200-day MA slope (% change over lookback)
    """
    if len(df) < long_ma + slope_lookback:
        return {"triggered": False, "sma_50_slope": 0, "sma_200_slope": 0}

    close = df["Close"]
    sma_50 = close.rolling(window=short_ma).mean()
    sma_200 = close.rolling(window=long_ma).mean()

    sma_50_now = sma_50.iloc[-1]
    sma_50_ago = sma_50.iloc[-slope_lookback - 1]
    sma_200_now = sma_200.iloc[-1]
    sma_200_ago = sma_200.iloc[-slope_lookback - 1]

    if pd.isna(sma_200_ago) or sma_200_ago == 0 or sma_50_ago == 0:
        return {"triggered": False, "sma_50_slope": 0, "sma_200_slope": 0}

    slope_50 = (sma_50_now - sma_50_ago) / sma_50_ago
    slope_200 = (sma_200_now - sma_200_ago) / sma_200_ago

    return {
        "triggered": slope_50 > 0 and slope_200 > 0,
        "sma_50_slope": round(slope_50 * 100, 4),
        "sma_200_slope": round(slope_200 * 100, 4),
    }


# =============================================================
# 18. VOLATILITY CONTRACTION PATTERN (VCP)
# =============================================================
def check_vcp(
    df: pd.DataFrame,
    base_period: int = 60,
    num_segments: int = 3,
    max_base_depth: float = 0.30,
) -> dict:
    """
    Minervini's VCP — detect decreasing volatility (range contraction)
    across segments of a base, indicating a coiling pattern before breakout.

    Returns:
        triggered: True if VCP detected
        contractions: number of successive contractions (0 to num_segments-1)
        range_segments: range % for each segment
        base_depth: max drawdown from high (e.g., -0.15 = 15%)
    """
    if len(df) < base_period + 1:
        return {"triggered": False, "contractions": 0, "range_segments": [], "base_depth": 0}

    recent = df.iloc[-base_period:]
    high_price = recent["High"].max()
    low_price = recent["Low"].min()
    current_close = df["Close"].iloc[-1]

    # Base depth — max drawdown from the high
    base_depth = (low_price - high_price) / high_price

    # Split into segments and measure each segment's range
    seg_len = base_period // num_segments
    ranges = []
    for i in range(num_segments):
        start = i * seg_len
        end = start + seg_len
        seg = recent.iloc[start:end]
        seg_high = seg["High"].max()
        seg_low = seg["Low"].min()
        seg_range = (seg_high - seg_low) / seg_high if seg_high > 0 else 0
        ranges.append(round(seg_range * 100, 2))

    # Count successive contractions
    contractions = 0
    for i in range(1, len(ranges)):
        if ranges[i] < ranges[i - 1]:
            contractions += 1

    # Trigger if all segments contract AND base isn't too deep
    triggered = contractions == num_segments - 1 and abs(base_depth) <= max_base_depth

    return {
        "triggered": triggered,
        "contractions": contractions,
        "range_segments": ranges,
        "base_depth": round(base_depth, 4),
    }


# =============================================================
# 19. BREAKOUT DAY VOLUME CONFIRMATION
# =============================================================
def check_breakout_volume(
    df: pd.DataFrame,
    price_breakout_period: int = 20,
    volume_lookback: int = 20,
    volume_threshold: float = 1.5,
) -> dict:
    """
    Flag if price is breaking out above a recent high AND volume confirms
    the move. Volume alone is noise; volume confirming a price breakout
    is signal (O'Neil / Minervini).

    Returns:
        triggered: True if price breakout + volume confirm
        price_breakout: True if making new 20-day high
        volume_confirmed: True if volume > threshold × average
        volume_ratio: today's volume / 20-day avg
        breakout_pct: how far above prior 20-day high
    """
    if len(df) < price_breakout_period + 2:
        return {
            "triggered": False, "price_breakout": False,
            "volume_confirmed": False, "volume_ratio": 0, "breakout_pct": 0,
        }

    current_close = df["Close"].iloc[-1]
    current_volume = df["Volume"].iloc[-1]

    # Prior 20-day high (excluding today)
    prior_high = df["High"].iloc[-(price_breakout_period + 1):-1].max()
    price_breakout = current_close > prior_high
    breakout_pct = (current_close - prior_high) / prior_high if prior_high > 0 else 0

    # Volume check
    avg_volume = df["Volume"].iloc[-(volume_lookback + 1):-1].mean()
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
    volume_confirmed = volume_ratio >= volume_threshold

    return {
        "triggered": price_breakout and volume_confirmed,
        "price_breakout": bool(price_breakout),
        "volume_confirmed": bool(volume_confirmed),
        "volume_ratio": round(volume_ratio, 2),
        "breakout_pct": round(breakout_pct, 4),
    }


# =============================================================
# 20. DUAL-TIMEFRAME RELATIVE STRENGTH
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
      - 126d RS ≥ 70th AND 63d RS > 126d RS (accelerating), OR
      - 63d RS ≥ 80th AND 21d RS ≥ 80th (strong and getting stronger)

    Returns:
        triggered: bool
        rs_126d_percentile: float
        rs_63d_percentile: float
        rs_21d_percentile: float
        accelerating: bool
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
# 21. WEINSTEIN STAGE 2 DETECTION
# =============================================================
def check_weinstein_stage2(
    df: pd.DataFrame,
    ma_period: int = 150,
    crossover_lookback: int = 10,
    slope_lookback: int = 20,
    min_base_weeks: int = 4,
) -> dict:
    """
    Detect transition from Stage 1 (basing) to Stage 2 (advancing)
    per Weinstein's system. Looks for price crossing above the 150-day
    SMA (30-week proxy) with the MA slope turning positive.

    Returns:
        triggered: bool
        price_above_150sma: bool
        recent_crossover: bool
        sma_150_slope: float (% change over slope_lookback)
        base_detected: bool
    """
    if len(df) < ma_period + slope_lookback + (min_base_weeks * 5):
        return {
            "triggered": False, "price_above_150sma": False,
            "recent_crossover": False, "sma_150_slope": 0, "base_detected": False,
        }

    close = df["Close"]
    sma_150 = close.rolling(window=ma_period).mean()

    current_close = close.iloc[-1]
    current_sma = sma_150.iloc[-1]

    if pd.isna(current_sma):
        return {
            "triggered": False, "price_above_150sma": False,
            "recent_crossover": False, "sma_150_slope": 0, "base_detected": False,
        }

    price_above = current_close > current_sma

    # Check for recent crossover (was below within lookback, now above)
    recent_crossover = False
    if price_above:
        for i in range(2, crossover_lookback + 2):
            if -i < -len(close):
                break
            past_close = close.iloc[-i]
            past_sma = sma_150.iloc[-i]
            if not pd.isna(past_sma) and past_close <= past_sma:
                recent_crossover = True
                break

    # SMA slope
    sma_ago = sma_150.iloc[-slope_lookback - 1]
    if pd.isna(sma_ago) or sma_ago == 0:
        sma_slope = 0
    else:
        sma_slope = (current_sma - sma_ago) / sma_ago

    # Base detection: was price range-bound in the period before the crossover?
    base_period = min_base_weeks * 5  # convert weeks to trading days
    pre_breakout = close.iloc[-(base_period + crossover_lookback):-(crossover_lookback)]
    if len(pre_breakout) >= base_period:
        base_range = (pre_breakout.max() - pre_breakout.min()) / pre_breakout.mean()
        base_detected = base_range < 0.30  # range < 30% of mean price = basing
    else:
        base_detected = False

    triggered = price_above and recent_crossover and sma_slope >= 0

    return {
        "triggered": triggered,
        "price_above_150sma": bool(price_above),
        "recent_crossover": recent_crossover,
        "sma_150_slope": round(sma_slope * 100, 4),
        "base_detected": base_detected,
    }


# =============================================================
# 22. TURTLE / ATR POSITION CONTEXT
# =============================================================
def check_turtle_atr_context(
    df: pd.DataFrame,
    atr_period: int = 14,
    move_threshold: float = 1.5,
    trend_period: int = 10,
    trend_threshold: float = 3.0,
) -> dict:
    """
    Assess whether today's move or recent trend is significant relative
    to the stock's normal volatility (ATR). Normalizes breakout significance
    across different-volatility assets (stocks vs crypto vs metals).

    Triggers if:
      - Today's move > move_threshold × ATR, OR
      - N-day price change > trend_threshold × ATR

    Returns:
        triggered: bool
        atr: float
        daily_move_atr: float (today's move / ATR)
        trend_move_atr: float (N-day move / ATR)
    """
    if len(df) < atr_period + trend_period + 1:
        return {"triggered": False, "atr": 0, "daily_move_atr": 0, "trend_move_atr": 0}

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    # ATR
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean().iloc[-1]

    if pd.isna(atr) or atr == 0:
        return {"triggered": False, "atr": 0, "daily_move_atr": 0, "trend_move_atr": 0}

    # Daily move normalized by ATR
    daily_move = abs(close.iloc[-1] - close.iloc[-2])
    daily_move_atr = daily_move / atr

    # N-day trend move normalized by ATR
    trend_move = close.iloc[-1] - close.iloc[-trend_period - 1]
    trend_move_atr = trend_move / atr

    triggered = daily_move_atr >= move_threshold or trend_move_atr >= trend_threshold

    return {
        "triggered": triggered,
        "atr": round(atr, 4),
        "daily_move_atr": round(daily_move_atr, 2),
        "trend_move_atr": round(trend_move_atr, 2),
    }


# =============================================================
# REGISTRY & MASTER FUNCTION
# =============================================================
ALL_EXPANDED_INDICATORS = {
    # Original 5 — imported from indicators.py at runtime
    "volume_spike": None,
    "near_52w_high": None,
    "bollinger_bands": None,
    "relative_strength": None,
    "moving_averages": None,
    # Expanded 11
    "rsi_momentum": check_rsi_momentum,
    "macd_crossover": check_macd_crossover,
    "adx_trend": check_adx,
    "roc": check_roc,
    "obv_trend": check_obv_trend,
    "cmf": check_cmf,
    "atr_expansion": check_atr_expansion,
    "consolidation": check_consolidation_tightness,
    "higher_lows": check_higher_lows,
    "donchian_breakout": check_donchian_breakout,
    "ichimoku_cloud": check_ichimoku,
    # New 6 (classic breakout methodology)
    "rising_ma_slope": check_rising_ma_slope,
    "vcp": check_vcp,
    "breakout_volume": check_breakout_volume,
    "dual_tf_rs": None,  # requires multi-timeframe RS data
    "weinstein_stage2": check_weinstein_stage2,
    "turtle_atr": check_turtle_atr_context,
}

INDICATOR_LABELS = {
    "volume_spike": "Volume Spike",
    "near_52w_high": "Near 52w High",
    "bollinger_bands": "BB Squeeze",
    "relative_strength": "Relative Strength",
    "moving_averages": "MA Alignment",
    "rsi_momentum": "RSI Momentum",
    "macd_crossover": "MACD Crossover",
    "adx_trend": "ADX Trend",
    "roc": "Rate of Change",
    "obv_trend": "OBV Trend",
    "cmf": "Chaikin Money Flow",
    "atr_expansion": "ATR Expansion",
    "consolidation": "Consolidation Tight",
    "higher_lows": "Higher Lows",
    "donchian_breakout": "Donchian Breakout",
    "ichimoku_cloud": "Ichimoku Cloud",
    "rising_ma_slope": "Rising MA Slope",
    "vcp": "Volatility Contraction",
    "breakout_volume": "Breakout Volume",
    "dual_tf_rs": "Dual-TF Rel Strength",
    "weinstein_stage2": "Weinstein Stage 2",
    "turtle_atr": "Turtle ATR Context",
}


def compute_all_expanded(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    cfg: dict,
    all_rs_values: list[float] = None,
    multi_tf_rs: dict = None,
) -> dict:
    """
    Compute all 22 indicators for a single ticker.

    Args:
        df:             Ticker price DataFrame
        benchmark_df:   Benchmark price DataFrame
        cfg:            Config dict
        all_rs_values:  List of RS values for percentile ranking (63d)
        multi_tf_rs:    Dict with keys 'rs_21d_pctl', 'rs_63d_pctl', 'rs_126d_pctl'
                        for the Dual-Timeframe RS indicator

    Returns dict of {indicator_name: {triggered: bool, ...}}
    """
    from indicators import compute_all_indicators

    # Get original 5
    original = compute_all_indicators(df, benchmark_df, cfg, all_rs_values=all_rs_values)

    # Compute expanded 11
    expanded = {
        "rsi_momentum": check_rsi_momentum(df),
        "macd_crossover": check_macd_crossover(df),
        "adx_trend": check_adx(df),
        "roc": check_roc(df),
        "obv_trend": check_obv_trend(df),
        "cmf": check_cmf(df),
        "atr_expansion": check_atr_expansion(df),
        "consolidation": check_consolidation_tightness(df),
        "higher_lows": check_higher_lows(df),
        "donchian_breakout": check_donchian_breakout(df),
        "ichimoku_cloud": check_ichimoku(df),
    }

    # Compute new 6 (classic breakout methodology)
    new_indicators = {
        "rising_ma_slope": check_rising_ma_slope(df),
        "vcp": check_vcp(df),
        "breakout_volume": check_breakout_volume(df),
        "weinstein_stage2": check_weinstein_stage2(df),
        "turtle_atr": check_turtle_atr_context(df),
    }

    # Dual-Timeframe RS requires pre-computed multi-timeframe percentiles
    if multi_tf_rs:
        new_indicators["dual_tf_rs"] = check_dual_tf_rs(
            rs_126d_pctl=multi_tf_rs.get("rs_126d_pctl", 0),
            rs_63d_pctl=multi_tf_rs.get("rs_63d_pctl", 0),
            rs_21d_pctl=multi_tf_rs.get("rs_21d_pctl", 0),
        )
    else:
        new_indicators["dual_tf_rs"] = check_dual_tf_rs()

    return {**original, **expanded, **new_indicators}
