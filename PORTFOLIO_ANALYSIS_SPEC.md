# Portfolio Backtest Follow-Up Analysis
## For Claude Code Implementation

---

## Context

The portfolio simulation has been run successfully. Results are in:
- `portfolio_backtest.py` — the simulation code
- `portfolio_equity_curve.csv` — daily portfolio values
- `portfolio_trade_log.csv` — all 47 trades with entry/exit details

**Headline results:** $100k → $456.6k (+356.6%) vs SPY +33.3%. 47 trades, 70.2% win rate, 7.91x win/loss ratio. Best config: ≥ 9.5 threshold, no persistence filter, no rotation.

---

## Task 1: Separate Realized vs. Unrealized P&L

11 of the 47 trades are still open at end of backtest (exit reason = `end_of_backtest`). The +356.6% total return includes unrealized gains that could evaporate.

**What to do:**
- Read `portfolio_trade_log.csv`
- Split trades into two groups: closed (36 trades) and open (11 trades)
- Report separately:

```
REALIZED vs UNREALIZED P&L
──────────────────────────────────────────
                    Closed Trades    Open Trades    Total
Trades              36               11             47
Total P&L ($)       $XXX,XXX         $XXX,XXX       $356,600
Total P&L (%)       +XX.X%           +XX.X%         +356.6%
Win Rate            XX.X%            XX.X%          70.2%
Avg P&L/Trade       $X,XXX           $X,XXX         —
```

This tells us how much of the return is "banked" vs. "at risk."

---

## Task 2: AXTI Concentration Risk Analysis

AXTI was a monster trade: +1,242.9% ($107k gain). That single trade accounts for ~30% of total portfolio gains. We need to understand how dependent the results are on this one outlier.

**What to do:**
- Recalculate total portfolio return EXCLUDING AXTI entirely
- Recalculate win rate, avg gain, profit factor without AXTI
- Report:

```
CONCENTRATION RISK — AXTI IMPACT
──────────────────────────────────────────
                    With AXTI       Without AXTI    Difference
Total Return        +356.6%         +XX.X%          -XX.X%
Total P&L ($)       $356,600        $XXX,XXX        -$XXX,XXX
Win Rate            70.2%           XX.X%           —
Avg Gain/Trade      $X,XXX          $X,XXX          —
Profit Factor       15.28x          X.Xx            —
Max Drawdown        -30.3%          -XX.X%          —
```

Also identify the top 5 trades by P&L contribution and show what % of total gains each represents. This reveals whether the portfolio is diversified or dependent on a few outliers.

---

## Task 3: Monthly Return Breakdown

Were the gains spread evenly across the backtest period, or did one or two months drive most of the performance? This matters for understanding whether the strategy works consistently or got lucky in a specific window.

**What to do:**
- Using `portfolio_equity_curve.csv`, calculate month-over-month returns
- Also calculate SPY's monthly return for comparison
- Report:

```
MONTHLY RETURNS
──────────────────────────────────────────
Month           Portfolio    SPY         Alpha       Trades Opened    Trades Closed
2025-07         +XX.X%       +XX.X%      +XX.X%      XX               XX
2025-08         +XX.X%       +XX.X%      +XX.X%      XX               XX
...
──────────────────────────────────────────
Best Month:     XXXX-XX (+XX.X%)
Worst Month:    XXXX-XX (-XX.X%)
Profitable Months: X of X (XX.X%)
```

---

## Task 4: Wash Sale Flag Audit

Retroactively identify any trades in the log where wash sale rules would have applied. The rule: if a ticker was sold at a LOSS and then re-bought within 30 calendar days, the loss is not tax-deductible.

**What to do:**
- Read `portfolio_trade_log.csv`
- For every trade that exited at a loss (P&L < 0):
  - Check if the same ticker was re-entered within 30 calendar days of the loss exit
  - If yes, flag it as a wash sale
- Report:

```
WASH SALE AUDIT
──────────────────────────────────────────
Total loss exits:           XX
Wash sale violations:       XX

Flagged Trades:
Ticker    Loss Exit Date    Loss Amount    Re-Entry Date    Days Between    Wash Sale?
VIAV      2026-03-09        -$X,XXX        2026-03-09       0 days          YES
...
──────────────────────────────────────────
Total disallowed losses:    $X,XXX
Impact on tax basis:        [explain briefly]
```

**Important:** The VIAV trade is a known example — stopped out at -21% on 3/9, then re-entered the same day. This should absolutely be flagged.

---

## Task 5: Drawdown Analysis

The -30.3% max drawdown needs more context. When did it happen, how long did it last, and how quickly did it recover?

**What to do:**
- From `portfolio_equity_curve.csv`, calculate the drawdown series (% below the running peak at each point)
- Identify:
  - Max drawdown: magnitude, start date, trough date, recovery date
  - Top 3 drawdowns by magnitude
  - Average time in drawdown
- Report:

```
DRAWDOWN ANALYSIS
──────────────────────────────────────────
Max Drawdown:       -30.3%
Peak Date:          XXXX-XX-XX ($XXX,XXX)
Trough Date:        XXXX-XX-XX ($XXX,XXX)
Recovery Date:      XXXX-XX-XX (or "not yet recovered")
Duration:           XX days peak-to-trough, XX days trough-to-recovery

Top 3 Drawdowns:
#    Magnitude    Peak Date       Trough Date     Duration
1    -30.3%       XXXX-XX-XX      XXXX-XX-XX      XX days
2    -XX.X%       XXXX-XX-XX      XXXX-XX-XX      XX days
3    -XX.X%       XXXX-XX-XX      XXXX-XX-XX      XX days

Time Spent in Drawdown:
  > -5%:   XX days (XX% of backtest)
  > -10%:  XX days (XX% of backtest)
  > -20%:  XX days (XX% of backtest)
```

---

## Implementation

Create a single script: `portfolio_analysis.py`

- Reads `portfolio_trade_log.csv` and `portfolio_equity_curve.csv`
- Runs all 5 analyses above
- Prints results to console
- No new data fetching required — everything comes from the existing CSV files

```bash
python3 portfolio_analysis.py
```

Expected runtime: seconds (just CSV processing, no API calls).
