# Entry/Exit Threshold Optimization Backtest
## For Claude Code Implementation

---

## Overview

Our current paper trading system uses **entry ≥ 9.5** and **exit < 5**. We want to test whether loosening the entry threshold (more trades, faster rotation) and/or tightening the exit threshold (shorter holds, less drawdown) produces better results.

This is a **grid search** across entry/exit combinations using the existing portfolio simulation framework.

---

## Test Matrix

### Entry Thresholds to Test
- **9.5** (current baseline)
- **9.0** (slightly more trades)
- **8.5** (significantly more trades)

### Exit Thresholds to Test
- **< 5** (current baseline — hold until momentum is mostly gone)
- **< 6** (exit earlier — cut at first sign of fading)
- **< 7** (exit aggressively — only hold while score stays strong)

### Full Grid: 9 Combinations

| | Exit < 5 | Exit < 6 | Exit < 7 |
|---|---|---|---|
| **Entry ≥ 9.5** | BASELINE | Test | Test |
| **Entry ≥ 9.0** | Test | Test | Test |
| **Entry ≥ 8.5** | Test | Test | Test |

---

## Simulation Rules (same as portfolio_backtest.py)

All rules stay the same across all 9 tests — only entry threshold and exit score threshold change:

- **Starting capital:** $100,000
- **Max per position:** 20% of starting capital ($20,000)
- **Stop loss:** 15% below entry price (unchanged across all tests)
- **Rotation strategy:** No rotation (skip new signals if no cash)
- **No persistence filter** at 9.5; no persistence filter at lower thresholds either (keep it consistent)
- **Buy timing:** Next day's open after signal
- **Sell timing:** Next day's open after exit trigger
- **Wash sale:** Log only, don't block

---

## Implementation

### Option A: Modify existing `portfolio_backtest.py`

If the existing simulation already supports configurable entry/exit thresholds via parameters, just loop over the grid:

```python
ENTRY_THRESHOLDS = [9.5, 9.0, 8.5]
EXIT_THRESHOLDS = [5, 6, 7]

for entry_thresh in ENTRY_THRESHOLDS:
    for exit_thresh in EXIT_THRESHOLDS:
        results = run_simulation(
            entry_threshold=entry_thresh,
            exit_score_threshold=exit_thresh,
            # all other params unchanged
        )
        store_results(entry_thresh, exit_thresh, results)
```

### Option B: Create `threshold_optimizer.py`

If `portfolio_backtest.py` isn't easily parameterized, create a new script that imports the simulation logic and wraps it in the grid search.

---

## Output

### Section 1: Grid Summary Table

The main deliverable — one table showing all 9 combinations:

```
ENTRY/EXIT THRESHOLD OPTIMIZATION — GRID RESULTS
══════════════════════════════════════════════════════════════════════════════════════════
Entry    Exit    Total Return    SPY Return    Alpha    Max DD    Win Rate    Trades    Avg Hold    Avg Gain    Profit Factor
─────    ────    ────────────    ──────────    ─────    ──────    ────────    ──────    ────────    ────────    ─────────────
≥ 9.5    < 5     +356.6%        +33.3%        +323%    -30.3%    70.2%       47        XX days     +XX.X%      15.28x
≥ 9.5    < 6     +XXX.X%        +33.3%        +XXX%    -XX.X%    XX.X%       XX        XX days     +XX.X%      X.Xx
≥ 9.5    < 7     +XXX.X%        +33.3%        +XXX%    -XX.X%    XX.X%       XX        XX days     +XX.X%      X.Xx
≥ 9.0    < 5     +XXX.X%        +33.3%        +XXX%    -XX.X%    XX.X%       XX        XX days     +XX.X%      X.Xx
≥ 9.0    < 6     +XXX.X%        +33.3%        +XXX%    -XX.X%    XX.X%       XX        XX days     +XX.X%      X.Xx
≥ 9.0    < 7     +XXX.X%        +33.3%        +XXX%    -XX.X%    XX.X%       XX        XX days     +XX.X%      X.Xx
≥ 8.5    < 5     +XXX.X%        +33.3%        +XXX%    -XX.X%    XX.X%       XX        XX days     +XX.X%      X.Xx
≥ 8.5    < 6     +XXX.X%        +33.3%        +XXX%    -XX.X%    XX.X%       XX        XX days     +XX.X%      X.Xx
≥ 8.5    < 7     +XXX.X%        +33.3%        +XXX%    -XX.X%    XX.X%       XX        XX days     +XX.X%      X.Xx
══════════════════════════════════════════════════════════════════════════════════════════

BEST COMBINATION:  Entry ≥ X.X / Exit < X  →  +XXX.X% return, XX.X% win rate, -XX.X% max DD
BASELINE:          Entry ≥ 9.5 / Exit < 5  →  +356.6% return, 70.2% win rate, -30.3% max DD
```

### Section 2: Dimension Analysis

Isolate the effect of each dimension:

```
ENTRY THRESHOLD EFFECT (holding exit constant at < 5)
──────────────────────────────────────────────
Entry ≥ 9.5:    +356.6%    47 trades    70.2% WR    -30.3% DD
Entry ≥ 9.0:    +XXX.X%    XX trades    XX.X% WR    -XX.X% DD
Entry ≥ 8.5:    +XXX.X%    XX trades    XX.X% WR    -XX.X% DD

→ Lowering entry from 9.5 to 9.0: [better/worse] by XX.X%
→ Lowering entry from 9.5 to 8.5: [better/worse] by XX.X%


EXIT THRESHOLD EFFECT (holding entry constant at ≥ 9.5)
──────────────────────────────────────────────
Exit < 5:    +356.6%    avg hold XX days    70.2% WR    -30.3% DD
Exit < 6:    +XXX.X%    avg hold XX days    XX.X% WR    -XX.X% DD
Exit < 7:    +XXX.X%    avg hold XX days    XX.X% WR    -XX.X% DD

→ Tightening exit from 5 to 6: [better/worse] by XX.X%
→ Tightening exit from 5 to 7: [better/worse] by XX.X%
```

### Section 3: Trade Frequency & Capital Utilization

```
CAPITAL UTILIZATION BY CONFIGURATION
──────────────────────────────────────────────
Entry    Exit    Avg Positions    Avg Cash Idle    Signals Skipped (no cash)    Turnover
≥ 9.5    < 5     X.X              $XX,XXX          XX                           X.Xx/yr
≥ 9.5    < 6     X.X              $XX,XXX          XX                           X.Xx/yr
...
```

This answers the core question: "Do lower entry thresholds and faster exits put my capital to work more efficiently?"

### Section 4: Risk-Adjusted Comparison

```
RISK-ADJUSTED METRICS
──────────────────────────────────────────────
Entry    Exit    Total Return    Max DD    Return/DD Ratio    Calmar Ratio    Worst Trade
≥ 9.5    < 5     +356.6%        -30.3%    11.8x              X.Xx            -XX.X%
...
```

Return/DD Ratio = Total Return / Max Drawdown. Higher is better — it measures how much return you get per unit of risk.

### Section 5: Recommendation

Based on the data, provide a clear recommendation:

```
RECOMMENDATION
──────────────────────────────────────────────
Best overall:           Entry ≥ X.X / Exit < X
Best risk-adjusted:     Entry ≥ X.X / Exit < X  (if different)
Best for active trader:  Entry ≥ X.X / Exit < X  (most trades, still positive alpha)
Safest:                 Entry ≥ X.X / Exit < X  (lowest max drawdown)

Rationale: [2-3 sentences explaining the tradeoffs]
```

---

## File: `threshold_optimizer.py`

```bash
python3 threshold_optimizer.py
```

Expected runtime: 5-15 minutes (9 simulation runs, each similar to portfolio_backtest.py).

Uses existing SQLite data for scores and yfinance for price data.

---

## Key Questions This Answers

1. **Is 9.5 actually optimal, or are we leaving alpha on the table?** Lower thresholds mean more trades and faster rotation — if the scoring system has edge at 8.5+, we should capture it.

2. **Are we holding losers too long?** If exiting at < 7 instead of < 5 produces similar returns with less drawdown, our current exit is too patient.

3. **What's the interaction effect?** Maybe entry 8.5 with exit < 5 is terrible (too many weak trades held too long), but entry 8.5 with exit < 7 is great (more trades but cut quickly if they don't work). The grid search finds these interactions.

4. **Does faster turnover improve capital efficiency?** With $100k and a 20% cap, you can hold 5 positions max. If exits are faster, capital recycles into new signals more often — potentially compounding more aggressively.
