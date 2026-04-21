# Minervini SEPA Backtest Specification
## For Claude Code Implementation

---

## Overview

Implement and backtest Mark Minervini's SEPA (Specific Entry Point Analysis) system as a standalone strategy. Test it at two levels of specificity against our full ticker universe using the same backtesting framework we've already built.

This is separate from (but complementary to) our main indicator-based scoring system. The goal is to see whether Minervini's complete methodology, applied systematically, has predictive edge — and how it compares to our data-driven indicator stack.

---

## Reference: Who is Minervini?

Mark Minervini is a U.S. stock trader who won the U.S. Investing Championship in 1997 (155% return) and again in 2021. His system is detailed in "Trade Like a Stock Market Wizard" and "Think & Trade Like a Champion." The core thesis: buy leading stocks (strong fundamentals + relative strength) as they break out of a proper base pattern (Volatility Contraction Pattern) during confirmed uptrends.

---

## Level 1: Trend Template Filter

### The 8 Criteria (ALL must be true simultaneously)

```
1. price > 150-day SMA
2. price > 200-day SMA
3. 150-day SMA > 200-day SMA
4. 200-day SMA is rising (slope positive over last 22 trading days, ~1 month)
5. 50-day SMA > 150-day SMA
6. 50-day SMA > 200-day SMA
7. price >= 1.25 * 52-week low  (at least 25% above the yearly low)
8. price >= 0.75 * 52-week high (within 25% of the yearly high)
```

### Implementation

Create a function `check_minervini_trend_template(df)` that returns:

```python
{
    "triggered": bool,               # True only if ALL 8 criteria pass
    "criteria_met": int,             # count of criteria met (0-8)
    "criteria_detail": {
        "price_above_150sma": bool,
        "price_above_200sma": bool,
        "sma150_above_sma200": bool,
        "sma200_rising": bool,
        "sma50_above_150_and_200": bool,
        "price_above_50sma": bool,
        "above_25pct_from_52w_low": bool,
        "within_25pct_of_52w_high": bool,
    },
    "values": {
        "price": float,
        "sma_50": float,
        "sma_150": float,
        "sma_200": float,
        "sma_200_slope": float,      # % change over 22 days
        "pct_above_52w_low": float,  # e.g., 0.45 = 45% above low
        "pct_below_52w_high": float, # e.g., -0.10 = 10% below high
    }
}
```

### Data Requirements
- Minimum 252 trading days (1 year) for 200-day SMA + 52-week range
- Plus 22 days for slope calculation
- Total: ~274 rows minimum

---

## Level 2: Trend Template + VCP Breakout + Volume Confirmation

### Additional Criteria (on top of all 8 Trend Template criteria)

#### Volatility Contraction Pattern (VCP)

The VCP detects a stock building a base with progressively tighter price contractions. Minervini describes it as a series of pullbacks where each one is shallower than the last (e.g., 20% → 12% → 6% → 3%).

```
Implementation:
1. Look at the last 60-90 trading days
2. Identify pullback troughs within this window:
   - A pullback = price decline from a local high to a local low
   - Use a simple swing detection: a local high is a point higher than
     the N days before and after it (N=5 works well)
3. Measure each pullback depth as % from the preceding local high
4. Count how many successive contractions there are:
   - If pullback depths go 15% → 8% → 4%, that's 2 contractions
5. VCP is detected if:
   - At least 2 successive contractions are found
   - The most recent pullback is < 15% (not too deep)
   - The overall base depth from the highest high is < 35%
   - The base has been forming for at least 20 trading days (4 weeks)
```

```python
def detect_vcp(df, base_period=90, min_contractions=2, max_last_pullback=0.15, max_base_depth=0.35):
    """
    Returns:
    {
        "detected": bool,
        "num_contractions": int,
        "pullback_depths": [float],     # e.g., [0.18, 0.10, 0.05]
        "base_depth": float,            # max drawdown from high
        "base_length_days": int,        # how long the base has been forming
    }
    """
```

#### Breakout Detection

```
A breakout occurs when:
1. Price closes above the highest point in the VCP base
   (the "pivot point" — the left side high of the base)
2. This is today's close — we're checking if today is the breakout day
```

#### Volume Confirmation

```
On the breakout day:
1. Today's volume must be >= 1.5x the 50-day average volume
   (Minervini often wants to see 40-50%+ above average)
```

### Combined Level 2 Function

```python
def check_minervini_full(df, volume_lookback=50, volume_threshold=1.5):
    """
    Full Minervini SEPA check:
    1. All 8 Trend Template criteria must be TRUE
    2. VCP must be detected
    3. Price must be breaking out of the VCP base TODAY
    4. Volume must confirm the breakout

    Returns:
    {
        "triggered": bool,                 # True if ALL conditions met
        "trend_template": { ... },         # Level 1 results
        "vcp": { ... },                    # VCP detection results
        "breakout": {
            "is_breakout_day": bool,       # price above pivot today
            "pivot_price": float,          # the breakout level
            "breakout_pct": float,         # how far above pivot
        },
        "volume": {
            "confirmed": bool,
            "today_volume": int,
            "avg_volume": float,
            "volume_ratio": float,
        },
    }
    """
```

---

## Backtesting Approach

### Create: `minervini_backtest.py`

Use the same framework as our existing `backtester.py`:

```
1. Fetch 2 years of data for all tickers (reuse data_fetcher.py)
2. Walk through every 5th trading day (weekly frequency)
3. On each date, for each ticker:
   a. Run Level 1 check (Trend Template)
   b. Run Level 2 check (Trend Template + VCP + Volume)
4. For each signal, measure forward returns at 10, 21, 42, 63 days
5. Report results separately for Level 1 and Level 2
```

### Output Structure

The script should print three sections:

#### Section 1: Level 1 Results (Trend Template Only)

```
MINERVINI TREND TEMPLATE — BACKTEST RESULTS
─────────────────────────────────────────────
Total observations: [N]
Times template was fully met: [N] ([X]%)

Forward Returns When All 8 Criteria Met:
  10-day:  Win rate: X%  |  Avg: +X.XX%
  21-day:  Win rate: X%  |  Avg: +X.XX%
  42-day:  Win rate: X%  |  Avg: +X.XX%
  63-day:  Win rate: X%  |  Avg: +X.XX%

Forward Returns When Template NOT Met:
  10-day:  Win rate: X%  |  Avg: +X.XX%
  ...
  
Edge (Met vs Not Met):
  10-day: +X.XX%
  ...

Criteria Breakdown — How Often Each Passes:
  price > 150 SMA:       XX.X%
  price > 200 SMA:       XX.X%
  150 SMA > 200 SMA:     XX.X%
  200 SMA rising:        XX.X%
  50 SMA > 150 & 200:    XX.X%
  price > 50 SMA:        XX.X%
  25%+ above 52w low:    XX.X%
  within 25% of 52w high: XX.X%

Partial Template Results (how many criteria matter?):
  6/8 criteria met: Win rate: X% | Avg: +X.XX% | N=XXX
  7/8 criteria met: Win rate: X% | Avg: +X.XX% | N=XXX
  8/8 criteria met: Win rate: X% | Avg: +X.XX% | N=XXX
```

#### Section 2: Level 2 Results (Full SEPA System)

```
MINERVINI FULL SEPA — BACKTEST RESULTS
─────────────────────────────────────────────
Template met: [N] times
VCP detected within template: [N] times
Breakout + volume confirmed: [N] times

Forward Returns (Full SEPA Signal):
  10-day:  Win rate: X%  |  Avg: +X.XX%
  ...

Comparison:
                    Events    63d Win%    63d Avg Return
  Template only:     XXX       XX.X%       +XX.XX%
  Template + VCP:    XXX       XX.X%       +XX.XX%
  Full SEPA:         XXX       XX.X%       +XX.XX%
```

#### Section 3: Comparison to Our Indicator Stack

```
MINERVINI vs OUR SCORING SYSTEM — COMPARISON
─────────────────────────────────────────────
                          Events    63d Win%    63d Avg    Selectivity
Minervini Template (8/8):  XXX       XX.X%     +XX.XX%      XX.X%
Minervini Full SEPA:       XXX       XX.X%     +XX.XX%      XX.X%
Our Top 3 (2/3 req):       965       80.1%     +28.73%      23.8%
Our Top 5 (4/5 req):       735       82.3%     +31.08%      18.2%
Our Top 5 (5/5 req):       198       84.3%     +37.33%       4.9%

Note: "Selectivity" = % of total observations that triggered (lower = more selective)
```

#### Section 4: Most Frequently Flagged Tickers

```
MINERVINI TOP PICKS — MOST FREQUENTLY IN TEMPLATE
─────────────────────────────────────────────
Ticker    Name                     Times in Template    Avg 63d Return
XXXX      XXXXXXXXXXXXX            XX                   +XX.XX%
...
```

---

## Comparison Metrics We Care About

For each level, we want to answer:

1. **Edge** — What's the average forward return when the signal fires vs. when it doesn't?
2. **Win rate** — What % of signals result in a positive return at 63 days?
3. **Selectivity** — How often does it fire? (Too often = not selective enough. Too rare = not enough opportunities.)
4. **Consistency** — Does it work across sectors, or only in certain sectors?
5. **vs. Our Stack** — Is the Minervini system better, worse, or complementary to our data-driven indicator stack?

---

## Implementation Notes

### File Dependencies
- Uses `config.py` for ticker loading
- Uses `data_fetcher.py` for data fetching
- Self-contained otherwise (does not depend on indicators.py or indicators_expanded.py)

### Python 3.9 Compatibility
- Must include `from __future__ import annotations` at the top
- No walrus operators or other 3.10+ syntax

### Data Considerations
- Needs at least 274 rows per ticker (200-day SMA + 52-week high/low + slope lookback)
- Fetch with `period="2y"` to ensure enough data
- Crypto tickers trade 7 days/week — the 50/150/200 SMA values will reflect calendar days, not trading days. This is fine; the same applies to metals futures.
- WOLF only has ~115 rows and will be automatically excluded from results requiring 274+ rows

### VCP Detection Edge Cases
- Some stocks may never form a clean VCP (e.g., straight up-trending stocks with no pullbacks)
- The swing detection algorithm should handle this gracefully (return `detected: False`)
- In thinly-traded stocks (some small crypto tokens), volume data may be unreliable for the volume confirmation step

---

## Expected Results

Based on the trading literature and our existing backtest data:

- **Level 1 (Trend Template)** should show meaningful edge, probably in the +5-10% range at 63 days, with a moderate fire rate (~20-35% of observations). The 200-day rising requirement will be the most restrictive filter.

- **Level 2 (Full SEPA)** should show higher edge but with far fewer signals. Expect maybe 50-150 events across the full backtest window, but with win rates potentially in the 75-85% range.

- **The most interesting finding** will be whether Minervini's system catches different stocks than our indicator stack, or the same ones. If there's low overlap, combining the two could be very powerful.

---

## How To Run (for Claude Code)

```bash
# After implementation:
cd ~/Claude-Workspace/breakout-tracker
python3 minervini_backtest.py
```

Expected runtime: 5-10 minutes (similar to the full indicator analysis).
