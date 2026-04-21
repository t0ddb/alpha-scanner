# Portfolio Simulation Backtest Specification
## For Claude Code Implementation

---

## Overview

Simulate a real portfolio with capital constraints, position sizing rules, and entry/exit logic to answer:

1. What's the right entry threshold (9.5+, 9.0+, 8.5+)?
2. Does 2-day persistence filtering improve outcomes?
3. How should capital be rotated as new signals appear and old ones fade?

---

## Simulation Parameters

### Starting Conditions
- **Starting capital:** $100,000
- **Benchmark:** SPY (track portfolio return vs SPY return over same period)

### Position Sizing
- **Max per position:** 20% of starting capital ($20,000)
- **Initial position size:** Equal weight at entry (divide available cash equally, up to $20k cap)
- **No leverage:** Can only invest cash on hand
- **Fractional shares:** Allowed (simplifies math)

### Entry Rules
- **Threshold:** Test three variants: 9.5+, 9.0+, 8.5+
- **Persistence filter:** Test two variants for each threshold:
  - **No filter:** Buy on the first day score crosses threshold
  - **2-day confirmation:** Buy only if score stays above threshold for 2 consecutive scoring dates
- **Buy timing:** At next day's open price after the signal (no lookahead bias)

### Exit Rules (whichever triggers first)
1. **Score exit:** Score drops below 5 → sell at next day's open
2. **Stop loss:** Price drops 15% below entry price → sell at next day's open
3. **No time-based exit:** Positions can run indefinitely as long as score stays above 5 and stop isn't hit

### Capital Rotation
When a new signal appears but no cash is available:
- **Test Strategy A — No rotation:** Skip the new signal entirely. Hold existing positions.
- **Test Strategy B — Trim weakest:** Sell the position with the lowest current score (if its score is below the new signal's score), redeploy proceeds to the new signal.
- **Test Strategy C — Trim smallest gainer:** Sell the position with the lowest unrealized P&L, redeploy to the new signal.

---

## Data Requirements

Use the existing backfill data in SQLite (`breakout_tracker.db`, table `ticker_scores`) for daily scores. For price data, fetch from yfinance.

**Backtest period:** Use all available historical data in the database (approximately 6 months of daily snapshots).

**Scoring frequency:** Use whatever frequency exists in the database (every 5 trading days based on backfill). On days between scoring snapshots, carry forward the last known score for each ticker.

---

## Simulation Logic (Pseudocode)

```
portfolio = {cash: 100000, positions: {}, history: []}

for each trading_day in backtest_period:

    # 1. Check exits on existing positions
    for each position in portfolio.positions:
        current_price = get_price(ticker, trading_day)
        current_score = get_score(ticker, trading_day)  # carried forward if no new score

        if current_score < 5:
            sell(position, reason="score_exit")
            portfolio.cash += position.value

        elif current_price <= position.entry_price * 0.85:
            sell(position, reason="stop_loss")
            portfolio.cash += position.value

    # 2. Check for new entry signals
    todays_scores = get_all_scores(trading_day)
    new_signals = [t for t in todays_scores if t.score >= THRESHOLD]

    # Apply persistence filter if enabled
    if persistence_filter:
        new_signals = [t for t in new_signals
                       if previous_score(t.ticker) >= THRESHOLD]

    # Exclude tickers already in portfolio
    new_signals = [t for t in new_signals
                   if t.ticker not in portfolio.positions]

    # Sort by score descending (prioritize highest scores)
    new_signals.sort(by=score, descending=True)

    # 3. Buy new positions
    for signal in new_signals:
        if portfolio.cash >= 1000:  # minimum buy threshold
            position_size = min(portfolio.cash, 20000)  # 20% cap
            buy(signal.ticker, position_size)
            portfolio.cash -= position_size

        elif rotation_strategy == "trim_weakest":
            weakest = find_weakest_position()  # lowest current score
            if weakest.current_score < signal.score:
                sell(weakest)
                position_size = min(proceeds, 20000)
                buy(signal.ticker, position_size)

        elif rotation_strategy == "trim_smallest_gain":
            worst_pnl = find_worst_pnl_position()
            sell(worst_pnl)
            position_size = min(proceeds, 20000)
            buy(signal.ticker, position_size)

    # 4. Record daily snapshot
    portfolio.history.append({
        date: trading_day,
        total_value: portfolio.cash + sum(position.values),
        cash: portfolio.cash,
        num_positions: len(portfolio.positions),
        positions: snapshot(portfolio.positions),
    })
```

---

## Output

### Section 1: Entry Threshold Comparison

```
ENTRY THRESHOLD ANALYSIS
──────────────────────────────────────────────────────
                    ≥ 9.5       ≥ 9.0       ≥ 8.5
──────────────────────────────────────────────────────
Total return        +XX.X%      +XX.X%      +XX.X%
SPY return          +XX.X%      +XX.X%      +XX.X%
Alpha               +XX.X%      +XX.X%      +XX.X%
Max drawdown        -XX.X%      -XX.X%      -XX.X%
Total trades        XX          XX          XX
Win rate            XX.X%       XX.X%       XX.X%
Avg hold (days)     XX          XX          XX
Avg gain/trade      +XX.X%      +XX.X%      +XX.X%
Max concurrent pos  XX          XX          XX
Capital utilization XX.X%       XX.X%       XX.X%
──────────────────────────────────────────────────────
```

### Section 2: Persistence Filter Impact

For each threshold, compare "no filter" vs "2-day confirmation":

```
PERSISTENCE FILTER IMPACT (threshold ≥ 9.5)
──────────────────────────────────────────────────────
                    No Filter   2-Day Confirm
──────────────────────────────────────────────────────
Total return        +XX.X%      +XX.X%
Signals generated   XX          XX
Signals filtered    N/A         XX (XX.X% removed)
Win rate            XX.X%       XX.X%
Avg drawdown/trade  -XX.X%      -XX.X%
False positive rate XX.X%       XX.X%
──────────────────────────────────────────────────────
```

### Section 3: Capital Rotation Strategy Comparison

Using the best threshold from Section 1:

```
ROTATION STRATEGY COMPARISON
──────────────────────────────────────────────────────
                    No Rotate   Trim Weakest  Trim Worst PnL
──────────────────────────────────────────────────────
Total return        +XX.X%      +XX.X%        +XX.X%
Alpha vs SPY        +XX.X%      +XX.X%        +XX.X%
Max drawdown        -XX.X%      -XX.X%        -XX.X%
Signals missed      XX          XX            XX
Rotation events     N/A         XX            XX
Avg hold (days)     XX          XX            XX
──────────────────────────────────────────────────────
```

### Section 4: Trade Log

Full log of every trade:

```
TRADE LOG (top 30 by P&L)
──────────────────────────────────────────────────────
Ticker    Entry Date   Entry Score   Entry $    Exit Date   Exit Reason     P&L      Hold Days
AAOI      2025-08-15   9.8           $20,000    2025-10-03  score_exit      +$4,200  35
KRKNF     2025-09-22   10.0          $18,500    2025-09-30  stop_loss       -$2,775  6
...
```

### Section 5: Portfolio Equity Curve

Print daily portfolio values so we can plot them:

```
DATE, PORTFOLIO_VALUE, SPY_VALUE, CASH, NUM_POSITIONS
2025-07-01, 100000, 100000, 100000, 0
2025-07-02, 100000, 100150, 80000, 1
...
```

(Save this as a CSV file: `portfolio_equity_curve.csv`)

### Section 6: Summary Statistics

```
BEST TRADE:     AAOI +42.3% ($8,460) held 35 days
WORST TRADE:    KRKNF -15.0% (-$2,775) held 6 days (stop loss)
AVG WINNER:     +XX.X% ($X,XXX)
AVG LOSER:      -XX.X% (-$X,XXX)
WIN/LOSS RATIO: X.Xx (avg winner / avg loser)
PROFIT FACTOR:  X.Xx (total gains / total losses)
```

---

## Implementation Notes

### File: `portfolio_backtest.py`

- Self-contained script (imports from config.py, data_fetcher.py, indicators.py, subsector_store.py)
- Uses SQLite data for scores where available, falls back to computing live scores for dates not in DB
- Fetches price data from yfinance for entry/exit execution prices
- Python 3.9 compatible (`from __future__ import annotations`)

### Run Command
```bash
python3 portfolio_backtest.py
```

### Expected Runtime
5-10 minutes (fetching price data + running 6 simulation variants: 3 thresholds × 2 persistence filters, plus 3 rotation strategies for the winning config).

### Key Edge Cases to Handle
- **Ticker with no price data on entry day:** Skip the signal
- **Score gap between snapshots:** Carry forward last known score
- **Multiple exits on same day:** Process all exits before entries
- **Position size when cash < $20k:** Use whatever cash is available (minimum $1,000)
- **Same ticker re-entry:** If a stock exits and later re-signals, treat it as a new trade
- **Stock splits / adjustments:** yfinance handles adjusted prices by default

---

## What We're Trying to Learn

1. **Is 9.5+ actually the optimal threshold**, or are we leaving money on the table by being too selective?
2. **Does 2-day confirmation meaningfully reduce false positives** (like Kraken Robotics), or does it just delay good entries?
3. **Is active rotation worth the complexity**, or does "buy and hold until exit" with no rotation perform just as well?
4. **What does realistic portfolio performance look like** with capital constraints, versus the theoretical "buy every signal" backtest numbers we've seen before?
