# Quarterly System Review — Automated Report Specification
## For Claude Code Implementation

---

## Overview

Build an automated quarterly review script that validates all assumptions in the Alpha Scanner scoring system. The report is **diagnostic, not prescriptive** — it flags concerns but does not automatically change anything. Changes require human review and should only be made when the same concern persists across 2 consecutive quarters.

The script should be runnable on demand and also scheduled via GitHub Actions to run quarterly.

---

## File: `quarterly_review.py`

```bash
# Run manually
python3 quarterly_review.py

# Run for a specific lookback window
python3 quarterly_review.py --months 12

# Default: last 12 months of data
```

---

## Report Structure

The report has 6 sections. Each section ends with a **STATUS** of:
- ✅ **HEALTHY** — no concerns
- ⚠️ **WATCH** — something shifted, monitor next quarter
- 🔴 **ACTION NEEDED** — persistent issue, recommend change

---

## Section 1: Indicator Edge Validation

Rerun the conditional edge analysis on the most recent 12 months of data. For each of the 7 scored indicators, measure:
- Standalone 63-day forward return edge (fired vs not fired)
- Incremental edge after controlling for Relative Strength
- Compare to the original calibration values

```
SECTION 1: INDICATOR EDGE VALIDATION
══════════════════════════════════════════════════════════════════════
                        Original Edge    Current Edge    Delta    Status
                        (calibration)    (last 12mo)
────────────────────────────────────────────────────────────────────
Relative Strength       +13.31%          +XX.XX%         +X.XX%   ✅/⚠️/🔴
Ichimoku Cloud          +10.89%          +XX.XX%         +X.XX%   ✅/⚠️/🔴
Higher Lows             +7.73%           +XX.XX%         +X.XX%   ✅/⚠️/🔴
MA Alignment (dropped)  -9.30%           +XX.XX%         +X.XX%   ✅/⚠️/🔴
Rate of Change          +6.84%           +XX.XX%         +X.XX%   ✅/⚠️/🔴
Chaikin Money Flow      +6.57%           +XX.XX%         +X.XX%   ✅/⚠️/🔴
ATR Expansion           +4.58%           +XX.XX%         +X.XX%   ✅/⚠️/🔴
Dual-TF RS              +5.20%           +XX.XX%         +X.XX%   ✅/⚠️/🔴
Near 52w High (dropped) -3.30%           +XX.XX%         +X.XX%   ✅/⚠️/🔴

DROPPED INDICATORS — SHOULD THEY BE RECONSIDERED?
────────────────────────────────────────────────────────────────────
Volume Spike            +0.72%           +XX.XX%         +X.XX%   ✅/⚠️
BB Squeeze              +0.38%           +XX.XX%         +X.XX%   ✅/⚠️
MACD Crossover          +0.95%           +XX.XX%         +X.XX%   ✅/⚠️
ADX Trend               +2.66%           +XX.XX%         +X.XX%   ✅/⚠️

STATUS RULES:
  ✅ HEALTHY:        Current edge within 3% of original (same direction)
  ⚠️ WATCH:          Edge dropped by >3% OR flipped sign
  🔴 ACTION NEEDED:  Edge flipped sign for 2 consecutive quarters
                     OR a dropped indicator now shows >5% positive edge

ALSO CHECK:
  - Has any currently-dropped indicator developed strong positive
    incremental edge? If so, flag for potential inclusion.
  - Has any currently-scored indicator's incremental edge gone negative?
    If so, flag for potential removal.
```

---

## Section 2: Weight Calibration Check

Compare current indicator edge ranking to the weight ranking. Weights should be roughly proportional to edge.

```
SECTION 2: WEIGHT CALIBRATION
══════════════════════════════════════════════════════════════════════
                        Current Weight    Edge Rank    Weight Rank    Aligned?
────────────────────────────────────────────────────────────────────
Relative Strength       3.0 pts           #X           #1             ✅/⚠️
Ichimoku Cloud          2.0 pts           #X           #2             ✅/⚠️
Chaikin Money Flow      1.5 pts           #X           #3             ✅/⚠️
Rate of Change          1.5 pts           #X           #4             ✅/⚠️
Higher Lows             1.0 pts           #X           #5             ✅/⚠️
Dual-TF RS              0.5 pts           #X           #6             ✅/⚠️
ATR Expansion           0.5 pts           #X           #7             ✅/⚠️

STATUS RULES:
  ✅ HEALTHY:  Weight rank matches edge rank (within 1 position)
  ⚠️ WATCH:    Weight rank differs from edge rank by 2+ positions

SUGGESTED WEIGHTS (if recalibration were applied):
  [Calculate weights proportional to current edge values]
  [Show side-by-side with current weights]
  [DO NOT auto-apply — this is informational only]
```

---

## Section 3: Entry/Exit Threshold Validation

Rerun the 3×3 grid search on the latest 12-month window.

```
SECTION 3: ENTRY/EXIT THRESHOLD GRID (last 12 months)
══════════════════════════════════════════════════════════════════════
Entry    Exit    Return    Alpha    Max DD    Win%    Trades    Ret/DD
────────────────────────────────────────────────────────────────────
≥ 9.5    < 5     +XX.X%    +XX.X%   -XX.X%   XX.X%   XX        X.Xx
≥ 9.5    < 6     +XX.X%    +XX.X%   -XX.X%   XX.X%   XX        X.Xx
≥ 9.5    < 7     +XX.X%    +XX.X%   -XX.X%   XX.X%   XX        X.Xx
≥ 9.0    < 5     +XX.X%    +XX.X%   -XX.X%   XX.X%   XX        X.Xx
≥ 9.0    < 6     +XX.X%    +XX.X%   -XX.X%   XX.X%   XX        X.Xx
≥ 9.0    < 7     +XX.X%    +XX.X%   -XX.X%   XX.X%   XX        X.Xx
≥ 8.5    < 5     +XX.X%    +XX.X%   -XX.X%   XX.X%   XX        X.Xx
≥ 8.5    < 6     +XX.X%    +XX.X%   -XX.X%   XX.X%   XX        X.Xx
≥ 8.5    < 7     +XX.X%    +XX.X%   -XX.X%   XX.X%   XX        X.Xx

CURRENT CONFIG:     Entry ≥ 9.5 / Exit < 5
BEST THIS QUARTER:  Entry ≥ X.X / Exit < X

STATUS RULES:
  ✅ HEALTHY:        Current config is #1 or #2 in the grid
  ⚠️ WATCH:          Current config dropped to #3-4
  🔴 ACTION NEEDED:  Current config is #5+ for 2 consecutive quarters
```

---

## Section 4: Live Performance vs. Backtest

Compare actual trade results (from `trade_history.json`) against what the backtest predicted for the same period.

```
SECTION 4: LIVE vs BACKTEST COMPARISON
══════════════════════════════════════════════════════════════════════
Period: [first trade date] → [today]

                        Live (actual)    Backtest (predicted)    Gap
────────────────────────────────────────────────────────────────────
Total Return            +XX.X%           +XX.X%                  XX.X%
Win Rate                XX.X%            XX.X%                   XX.X%
Avg Gain/Trade          +XX.X%           +XX.X%                  XX.X%
Avg Loss/Trade          -XX.X%           -XX.X%                  XX.X%
Avg Hold (days)         XX               XX                      XX
Trades Executed         XX               XX                      XX
Max Drawdown            -XX.X%           -XX.X%                  XX.X%

SLIPPAGE ANALYSIS:
  Avg entry slippage:   +XX.X% (actual fill vs signal close)
  Avg exit slippage:    +XX.X%
  Total slippage cost:  $X,XXX

STATUS RULES:
  ✅ HEALTHY:        Live return within 20% of backtest return
  ⚠️ WATCH:          Live return 20-40% below backtest
  🔴 ACTION NEEDED:  Live return >40% below backtest for 2 consecutive quarters

NOTE: During paper trading phase, "live" = paper trades.
      This section will be most valuable once real trading begins.
      If no trades have been executed yet, skip this section.
```

---

## Section 5: Universe Review

Check the health and relevance of the ticker universe.

```
SECTION 5: UNIVERSE REVIEW
══════════════════════════════════════════════════════════════════════
Total tickers in config:    XXX
Successfully fetched data:  XXX
Failed to fetch:            XXX

FAILED TICKERS (may be delisted or renamed):
────────────────────────────────────────────────────────────────────
Ticker    Subsector                    Last Known Price    Issue
XXXX      XXXXX                        $XX.XX              No data since XXXX-XX-XX
XXXX      XXXXX                        N/A                 Ticker not found

NEVER-SCORED TICKERS (always score 0-2, may not belong):
────────────────────────────────────────────────────────────────────
Ticker    Subsector                    Avg Score (12mo)    Max Score    Times ≥ 6
XXXX      XXXXX                        X.X                 X.X          0

SUBSECTOR SIZE CHECK:
────────────────────────────────────────────────────────────────────
Subsector                         Tickers    Concern
Quantum Hardware & Software       4          ⚠️ Small — breadth signals unreliable
eVTOL / Urban Air Mobility        3          ⚠️ Very small
...

POTENTIAL ADDITIONS (manual research needed):
────────────────────────────────────────────────────────────────────
  - Check for recent IPOs in covered sectors
  - Check if any covered companies were acquired/delisted
  - Review if any sector thesis has changed fundamentally

STATUS RULES:
  ✅ HEALTHY:  <5% of tickers failing, no subsectors below 3 tickers
  ⚠️ WATCH:    5-10% failing, or subsectors with <4 tickers
  🔴 ACTION:   >10% failing, or a sector thesis has fundamentally changed
```

---

## Section 6: Subsector Breakout State Machine Validation

Check if the state machine parameters still produce meaningful signals.

```
SECTION 6: STATE MACHINE HEALTH
══════════════════════════════════════════════════════════════════════
Period: last 12 months

State Transitions Summary:
────────────────────────────────────────────────────────────────────
Total transitions:              XX
Quiet → Emerging:               XX
Emerging → Confirmed:           XX (XX% confirmation rate)
Emerging → Quiet (failed):      XX
Confirmed → Fading:             XX
Fading → Revival:               XX
Fading → Quiet:                 XX

Forward Returns by State (63-day, SPY-adjusted):
────────────────────────────────────────────────────────────────────
                    Original    Current     Delta    Status
Emerging            +13.1%      +XX.X%      +X.X%    ✅/⚠️
Confirmed           +21.7%      +XX.X%      +X.X%    ✅/⚠️
Revival             +27.3%      +XX.X%      +X.X%    ✅/⚠️

Parameter Check:
────────────────────────────────────────────────────────────────────
  Breadth threshold (6):     Still appropriate?  ✅/⚠️
  Confirm days (3):          Too fast/slow?      ✅/⚠️
  Fade cool days (5):        Too fast/slow?      ✅/⚠️

STATUS RULES:
  ✅ HEALTHY:  Confirmed and Revival still show positive edge
  ⚠️ WATCH:    Edge dropped by >50% from original
  🔴 ACTION:   Confirmed signals show zero or negative edge
```

---

## Section 7: Executive Summary

Roll up all section statuses into a single scorecard.

```
══════════════════════════════════════════════════════════════════════
  QUARTERLY REVIEW — EXECUTIVE SUMMARY
  Period: XXXX-XX-XX → XXXX-XX-XX
══════════════════════════════════════════════════════════════════════

  Section                          Status    Notes
  ─────────────────────────────    ──────    ──────────────────────
  1. Indicator Edge                ✅/⚠️/🔴   [one-line summary]
  2. Weight Calibration            ✅/⚠️/🔴   [one-line summary]
  3. Entry/Exit Thresholds         ✅/⚠️/🔴   [one-line summary]
  4. Live vs Backtest              ✅/⚠️/🔴   [one-line summary]
  5. Universe Health               ✅/⚠️/🔴   [one-line summary]
  6. State Machine                 ✅/⚠️/🔴   [one-line summary]

  OVERALL SYSTEM HEALTH:           ✅/⚠️/🔴

  RECOMMENDED ACTIONS:
  ─────────────────────────────────────────────────────────────────
  [List only items with ⚠️ or 🔴 status]
  [For ⚠️: "Monitor next quarter"]
  [For 🔴: "Consider [specific change], pending 2nd quarter confirmation"]

  NEXT REVIEW DATE:  XXXX-XX-XX

══════════════════════════════════════════════════════════════════════
```

---

## Historical Tracking

### File: `quarterly_reviews/`

Each review should be saved as a dated file:
```
quarterly_reviews/
├── review_2026_Q2.txt
├── review_2026_Q3.txt
└── ...
```

This enables the "2 consecutive quarters" rule — the script should automatically load the previous quarter's review (if it exists) and check for persistent issues.

### File: `quarterly_reviews/review_history.json`

Track status history per section:
```json
{
    "2026-Q2": {
        "date": "2026-07-01",
        "indicator_edge": "healthy",
        "weight_calibration": "healthy",
        "thresholds": "watch",
        "live_vs_backtest": "skip",
        "universe": "healthy",
        "state_machine": "healthy",
        "overall": "healthy"
    }
}
```

When a section shows ⚠️ WATCH, the script checks `review_history.json` to see if the previous quarter also showed WATCH for the same section. If yes → upgrade to 🔴 ACTION NEEDED.

---

## GitHub Actions: Quarterly Schedule

Add to `.github/workflows/`:

```yaml
name: Quarterly System Review
on:
  schedule:
    - cron: '0 14 1 1,4,7,10 *'  # 1st of Jan/Apr/Jul/Oct at 2PM UTC (6AM PT)
  workflow_dispatch:  # manual trigger

jobs:
  quarterly-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -r requirements.txt
      - run: python quarterly_review.py --months 12
        env:
          ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}
          ALPACA_SECRET_KEY: ${{ secrets.ALPACA_SECRET_KEY }}
      - name: Commit review results
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add quarterly_reviews/
          git diff --staged --quiet || git commit -m "Quarterly review $(date +%Y-Q$(( ($(date +%-m)-1)/3+1 )))"
          git pull --rebase
          git push
      - name: Send review email
        if: env.GMAIL_ADDRESS != ''
        run: python quarterly_review.py --email-only
        env:
          GMAIL_ADDRESS: ${{ secrets.GMAIL_ADDRESS }}
          GMAIL_APP_PASSWORD: ${{ secrets.GMAIL_APP_PASSWORD }}
```

---

## Data Requirements

The script needs:
- **yfinance** — fetch 12-15 months of OHLCV data for all tickers + SPY
- **SQLite DB** (`breakout_tracker.db`) — for subsector state machine history
- **trade_history.json** — for live vs backtest comparison (Section 4)
- **Previous quarter's review** (`quarterly_reviews/review_history.json`) — for 2-quarter persistence checks

---

## Implementation Notes

- Python 3.9 compatible (`from __future__ import annotations`)
- Reuses existing modules: `config.py`, `data_fetcher.py`, `indicators.py`, `indicators_expanded.py`, `subsector_breakout.py`, `subsector_store.py`
- Runtime: 15-25 minutes (indicator edge analysis is the slowest part — scoring all tickers across ~50 weekly test dates)
- Save both console output and a plain text file to `quarterly_reviews/`
- The review should be self-contained — someone reading just the review file should understand the system's health without needing other context

---

## Decision Framework: When to Act

| Situation | Action |
|-----------|--------|
| All sections ✅ | No changes. System is healthy. |
| One section ⚠️ | Note it. Monitor next quarter. Do nothing. |
| Same section ⚠️ for 2 quarters | Upgrade to 🔴. Investigate and propose a specific change. |
| One section 🔴 (first time) | Investigate. Prepare a change but don't implement yet. |
| Same section 🔴 for 2 quarters | Implement the change. Rerun backtest to validate. |
| Indicator flips to negative edge | If incremental edge is negative for 2 quarters, drop it. |
| Dropped indicator shows >5% edge | If persistent for 2 quarters, consider re-adding. |
| Live performance >40% below backtest | Urgent — investigate immediately for data/execution bugs. |

**The core principle: never react to a single quarter. Always wait for confirmation.**
