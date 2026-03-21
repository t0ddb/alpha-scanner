# New Indicator Proposals: Classic Breakout Methodology Additions
## For Claude Code Implementation

---

## Summary

Based on analysis of proven breakout systems (O'Neil CAN SLIM, Minervini SEPA, Weinstein Stage Analysis, Turtle Trading), we identified 6 new indicators that address gaps in our current 16-indicator library. These should be added to `indicators_expanded.py`, then run through the same backtesting framework (`indicator_analysis_full.py`) to measure their predictive edge before deciding on final weights.

---

## New Indicator #17: Rising Moving Average Slope

### Rationale
Our current MA Alignment checks if price is above the 50 and 200 SMA, but Minervini's SEPA system requires that the 200-day MA itself must be *rising* for at least 1 month. A stock above a flat or declining 200-day MA is range-bound, not trending. A rising MA confirms the underlying trend is healthy.

### Computation
```
1. Compute 200-day SMA
2. Measure the slope over the last 20-30 trading days:
   slope = (SMA_today - SMA_30_days_ago) / SMA_30_days_ago
3. Also check 50-day SMA slope the same way
4. Trigger if BOTH the 50-day and 200-day SMA slopes are positive
```

### Suggested Parameters
- `long_ma_period`: 200
- `short_ma_period`: 50
- `slope_lookback`: 30 (trading days to measure slope over)
- `min_slope`: 0.0 (any positive slope = rising)

### Returns
```python
{
    "triggered": bool,       # True if both MAs are rising
    "sma_50_slope": float,   # 50-day MA slope (% change over lookback)
    "sma_200_slope": float,  # 200-day MA slope (% change over lookback)
}
```

---

## New Indicator #18: Volatility Contraction Pattern (VCP)

### Rationale
Minervini's signature pattern. The best breakouts come from stocks that have been consolidating with *decreasing* volatility — each pullback within the base is shallower than the last. This creates a "coiling" effect before an explosive move. Our Consolidation Tightness indicator is a crude version of this; the VCP is more specific and was the weakest point in our current library.

### Computation
```
1. Divide the last 60 trading days into 3 equal segments of 20 days
2. Measure the high-low range (as % of price) for each segment
3. Check if ranges are contracting: range_1 > range_2 > range_3
4. Also require that the overall base is not too deep
   (max drawdown from the 60-day high should be < 30-35%)
5. Trigger if ranges are contracting AND base is not too deep
```

### Suggested Parameters
- `base_period`: 60 (total days to analyze)
- `num_segments`: 3
- `max_base_depth`: 0.30 (max 30% drawdown from high)

### Returns
```python
{
    "triggered": bool,           # True if VCP detected
    "contractions": int,         # number of successive contractions (0-2)
    "range_segments": [float],   # range % for each segment
    "base_depth": float,         # max drawdown from high (e.g., -0.15 = 15%)
}
```

---

## New Indicator #19: Breakout Day Volume Confirmation

### Rationale
Our generic Volume Spike indicator tested as near-useless (+0.72% edge). But O'Neil and Minervini both insist that volume matters specifically *on the breakout day* — when price is simultaneously making a new high or breaking above a resistance level. The insight is that volume alone is noise; volume *confirming a price breakout* is signal.

### Computation
```
1. Check if today's close is above the 20-day high (a breakout)
   OR if today's close is within 2% of the 52-week high
2. If a price breakout is happening, THEN check volume:
   - Is today's volume > 1.5x the 20-day average volume?
3. Trigger only if BOTH conditions are true simultaneously
   (price breakout + volume confirmation)
```

### Suggested Parameters
- `price_breakout_period`: 20 (days for local high)
- `volume_lookback`: 20
- `volume_threshold`: 1.5 (1.5x average)

### Returns
```python
{
    "triggered": bool,           # True if price breakout + volume confirm
    "price_breakout": bool,      # True if making new 20-day high
    "volume_confirmed": bool,    # True if volume > threshold
    "volume_ratio": float,       # today's volume / 20-day avg
    "breakout_pct": float,       # how far above prior 20-day high
}
```

---

## New Indicator #20: Dual-Timeframe Relative Strength

### Rationale
Our current RS indicator measures 63-day (3-month) relative performance vs. SPY. But stocks that are both strong over 6 months AND accelerating over the last 1-3 months are in a much more powerful position than stocks that were strong 6 months ago but are now decelerating. This dual-timeframe check captures momentum *acceleration*.

### Computation
```
1. Compute 126-day (6-month) RS value: stock_return_126d / spy_return_126d
2. Compute 63-day (3-month) RS value: stock_return_63d / spy_return_63d
3. Compute 21-day (1-month) RS value: stock_return_21d / spy_return_21d
4. Rank each across the full ticker universe (percentile)
5. Trigger if:
   - 126-day RS percentile >= 70 (strong over 6 months)
   - AND 63-day RS percentile > 126-day RS percentile (accelerating)
   OR
   - 63-day RS percentile >= 80 AND 21-day RS percentile >= 80
     (strong and getting stronger in short term)
```

### Suggested Parameters
- `long_period`: 126 (6 months)
- `mid_period`: 63 (3 months)
- `short_period`: 21 (1 month)
- `long_min_percentile`: 70
- `short_min_percentile`: 80

### Returns
```python
{
    "triggered": bool,
    "rs_126d_percentile": float,
    "rs_63d_percentile": float,
    "rs_21d_percentile": float,
    "accelerating": bool,        # True if shorter-term RS > longer-term RS
}
```

### Note
This indicator requires `all_rs_values` at multiple timeframes, similar to how the current Relative Strength indicator works. The `compute_all_expanded()` function will need to pass multi-timeframe RS universe data.

---

## New Indicator #21: Weinstein Stage 2 Detection

### Rationale
Stan Weinstein's system identifies stocks entering Stage 2 (the advancing phase) by looking for price breaking above the 30-week moving average after a basing period. The 30-week MA (~150-day) is a distinct timeframe from our existing 50/200-day checks. The key insight is that the *transition* from Stage 1 (basing) to Stage 2 (advancing) is the highest-probability entry point.

### Computation
```
1. Compute 150-day SMA (proxy for 30-week MA)
2. Check if price recently crossed above 150-day SMA
   (was below within the last 10 trading days, now above)
3. Check if 150-day SMA slope is turning positive
   (slope over last 20 days >= 0)
4. Check if price has been in a base (trading within a range)
   for at least 4 weeks prior to the breakout
5. Trigger if price crossed above 150 SMA AND slope is turning up
```

### Suggested Parameters
- `ma_period`: 150
- `crossover_lookback`: 10 (days to check for recent crossover)
- `slope_lookback`: 20
- `min_base_weeks`: 4 (must have been basing before breakout)

### Returns
```python
{
    "triggered": bool,
    "price_above_150sma": bool,
    "recent_crossover": bool,    # crossed above within lookback
    "sma_150_slope": float,      # slope of the 150 SMA
    "base_detected": bool,       # was in a base before breakout
}
```

---

## New Indicator #22: Turtle/ATR Position Context

### Rationale
The Turtle Trading system's key insight wasn't the entry signal (Donchian channel breakout, which we already have) — it was using ATR to assess whether a stock's current move is *significant relative to its normal volatility*. A stock moving 5% means nothing if it routinely moves 5%; but a stock moving 5% when its ATR implies 2% moves is a 2.5-sigma event. This normalizes breakout significance across different-volatility assets (especially useful in our cross-asset system with stocks, metals, and crypto).

### Computation
```
1. Compute 14-day ATR
2. Compute today's move: abs(close - prior_close)
3. Compute the ATR-normalized move: today_move / ATR
4. Also compute the N-day price change normalized by ATR:
   (close - close_N_days_ago) / ATR
5. Trigger if the normalized move exceeds a threshold
   (today's move is outsized relative to normal volatility)
```

### Suggested Parameters
- `atr_period`: 14
- `move_threshold`: 1.5 (today's move > 1.5x ATR = significant)
- `trend_period`: 10 (days for trend-normalized calculation)
- `trend_threshold`: 3.0 (10-day move > 3x ATR = strong trend)

### Returns
```python
{
    "triggered": bool,
    "atr": float,
    "daily_move_atr": float,     # today's move / ATR
    "trend_move_atr": float,     # N-day move / ATR
}
```

---

## Implementation Plan

### Step 1: Add to indicators_expanded.py
Add all 6 new indicator functions following the same pattern as existing indicators:
- Each returns a dict with `"triggered": bool` plus supporting values
- Update `ALL_EXPANDED_INDICATORS` dict and `INDICATOR_LABELS` dict
- Update `compute_all_expanded()` to include all 6

For indicator #20 (Dual-Timeframe RS), `compute_all_expanded()` needs to accept
and pass multi-timeframe RS percentile data, similar to how `all_rs_values` is
currently passed for the existing RS indicator.

### Step 2: Update indicator_analysis_full.py
- Add the new indicator names to `ALL_INDICATOR_NAMES`
- The `collect_all_indicator_events()` function needs to:
  - Compute RS values at 3 timeframes (21d, 63d, 126d) in the first pass
  - Pass all 3 sets of RS values to `compute_all_expanded()`

### Step 3: Run the analysis
```bash
python3 indicator_analysis_full.py
```
This will produce the same output as before but with 22 indicators instead of 16,
giving us edge/win-rate/fire-rate for each new indicator.

### Step 4: Update weighted scoring
Based on the results, assign weights to any new indicators that show meaningful
predictive edge (> 3-4%), and update the scoring engine accordingly.

---

## Expected Outcomes

Based on the trading literature and what we've already seen in our data:

- **Rising MA Slope** — likely to test well since it refines our already-strong MA Alignment signal
- **VCP** — should be highly selective (low fire rate) but very predictive when it does fire
- **Breakout Day Volume** — the composite version should test MUCH better than our raw Volume Spike (+0.72%) since it only fires when volume confirms a price breakout
- **Dual-Timeframe RS** — should outperform our current single-timeframe RS since it adds the acceleration dimension
- **Weinstein Stage 2** — captures a specific regime transition; should be selective and predictive
- **Turtle ATR Context** — particularly valuable for cross-asset comparison (normalizing crypto volatility vs. gold vs. stocks)
