# Alpha Scanner — Technical Documentation & Methodology

**Subsector Momentum Breakout Detection & Automated Paper-Trading System**
*April 2026*

---

## 1. Executive Summary

Alpha Scanner is a momentum breakout scoring system that detects when entire subsectors — not just individual stocks — are experiencing coordinated technical breakouts, and automatically executes paper trades against the resulting signals. The system operates on a three-layer architecture:

- **Layer 1 — Ticker Scoring:** Each of 180 tickers is scored 0–10 using 7 technical indicators. Tells you *what to do* once you're looking at a stock.
- **Layer 2 — Subsector Breakout Detection:** A state machine aggregates individual scores into subsector breadth metrics, detecting when 50%+ of a subsector is "hot." Tells you *where to look.*
- **Layer 3 — Automated Execution:** `trade_executor.py` runs nightly (after US market close, via GitHub Actions) and submits limit orders to Alpaca paper trading when signals satisfy a 3-day persistence filter. Places real GTC stop orders for downside protection.

The core thesis: when multiple stocks in the same subsector start firing technical signals simultaneously, it signals a sector rotation or thematic breakout wave worth trading.

### Key Results from Backtesting

- Tickers scoring **8+** produce **+18.4% SPY-adjusted alpha** at 63 days
- **Confirmed** subsector breakouts generate **+21.7% alpha** with +6.9% edge over baseline
- **Revival** signals (fading → confirmed) show highest alpha: **+27.3%**, 70.8% win rate
- **Current live configuration** (validated on `entry_mode_backtest.py`, 2026-04-17): 12-position cap, 8.3% dynamic sizing, 20% fixed stop, 3% limit orders, 5% cash floor. Backtest result over the validation window: **+433.7% total return, 3.16 Sharpe, 4.88 Sortino**, vs baseline market-order config +417.1% / 2.86 Sharpe and 70 days with negative cash.

---

## 2. Universe & Structure

The system tracks **180 tickers** across **9 sectors** and **31 subsectors**, spanning themes from AI infrastructure to space, biotech, nuclear, and crypto.

| Sector | Tickers | Subsectors |
|--------|---------|------------|
| AI & Tech Capex Cycle | 101 | Compute, Memory, Networking/Photonics, Power & Energy, Data Center Infra, Alt AI Compute/GPU Hosts, Hyperscalers, AI Software, Enterprise AI, AI Security, Healthcare AI, Physical AI, Power Semis, Semi Equipment, Semi Test |
| Metals | 16 | Gold & Silver Direct, Gold Miners, Silver Miners |
| Crypto | 11 | Crypto Majors (incl. GLXY), Crypto/AI Crossover Tokens |
| Robotics & Automation | 14 | Surgical/Medical, Industrial, Subsea/Ocean |
| Space & Satellite | 11 | Launch & Spacecraft, Satellite Comms & Data |
| Nuclear & Uranium | 10 | Nuclear Reactors/SMR, Uranium Miners |
| Biotechnology | 8 | Gene Editing/CRISPR, Synthetic Biology |
| eVTOL & Drones | 5 | eVTOL/Urban Air Mobility |
| Quantum Computing | 4 | Quantum Hardware & Software |

**Benchmark:** SPY (S&P 500 ETF). All alpha calculations are raw forward returns minus SPY's return over the same period.

Recent universe additions (April 2026): BE, CRWV, APLD, BTDR, RIOT, CLSK, WYFI, KEEL, SNDK, EQT, SEI, LBRT, PUMP, PSIX, BW, GLXY — net +15 tickers. Adding new tickers is YAML-only (`ticker_config.yaml`), no code changes needed.

---

## 3. Scoring Engine

Each ticker is scored on a 0–10 scale using 7 technical indicators. Weights were determined through a **3-year conditional edge analysis** — each indicator's weight reflects its incremental predictive edge after controlling for Relative Strength (the strongest individual predictor).

### 3.1 The 7 Scored Indicators

| Indicator | Weight | Type | Edge | Win Rate | Logic |
|-----------|--------|------|------|----------|-------|
| Relative Strength | 0–3.0 | Gradient | +13.31% | 77.4% | 63-day outperformance vs SPY, percentile-ranked across universe |
| Ichimoku Cloud | 2.0 | Binary | +11.9% | 76.9% | Price above cloud AND cloud bullish (Senkou A > B) |
| Chaikin Money Flow | 1.5 | Binary | +8.9% | 73.2% | 20-day CMF > 0.05 (institutional buying pressure) |
| Rate of Change | 1.5 | Binary | +7.5% | 72.6% | 21-day ROC > 5% (strong recent momentum) |
| Higher Lows | 0–1.0 | Gradient | +7.73% | 75.9% | Consecutive weekly higher-low periods (staircase uptrend) |
| Dual-TF RS | 0.5 | Binary | +5.2% | N/A | Multi-timeframe RS strong AND accelerating |
| ATR Expansion | 0.5 | Binary | +5.2% | N/A | 14-day ATR in top 20% of 50-day range |

### 3.2 Gradient vs Binary Scoring

- **Gradient:** Relative Strength scores proportionally from 0 to 3.0 based on percentile (90th→3.0, 80th→2.4, 70th→1.8, 60th→1.2, 50th→0.6). Higher Lows scores 0.25 per consecutive higher low up to 1.0.
- **Binary:** Ichimoku, CMF, ROC, Dual-TF RS, and ATR Expansion award full weight if triggered, zero otherwise. Determined by analyzing forward return distributions — these indicators showed no meaningful improvement from proportional scoring.

### 3.3 Indicators Tested but NOT Included

Two additional indicators were evaluated but excluded from the final scoring model:

- **Moving Average Alignment:** −9.3% incremental edge. Harmful when Relative Strength is already strong — it adds noise without predictive value. Still computed and displayed on the dashboard for visual context. **Watch note**: in the 2026-Q2 quarterly review its 12-month edge flipped to +7.76%. Do not re-include yet — await confirmation in next quarterly review before reweighting.
- **Near 52-Week High:** −3.3% incremental edge. Fully redundant with Relative Strength — tickers with high RS are almost always near their 52-week highs. Displayed on dashboard but not scored.

### 3.4 Score Tiers (dashboard + email-digest display)

| Tier | Score Range | Color (dashboard) | Interpretation |
|------|-------------|---|----------------|
| 🔴 **Fire** | 9.5+ | `#E15759` | Exceptional — nearly every signal firing at once |
| 🟠 **Hot** | 8.5 – 9.5 | `#F28E2B` | Actionable breakout, high conviction (live entry threshold) |
| 🟡 **Warm** | 7 – 8.5 | `#F1CE63` | Setup building but not high-conviction |
| 🔵 **Tepid** | 5 – 7 | `#A0CBE8` | Some signals — watchlist territory |
| 🟦 **Cold** | < 5 | `#4E79A7` | Few or no technical signals — live exit threshold |

Dashboard uses the Tableau 20 palette ("heat" metaphor: red=hot, blue=cold). The email digest uses a separate green/amber/orange/red palette for P&L-style semantic coloring; the two are intentionally distinct and should not be unified.

### 3.5 Scoring Process (Two-Pass)

1. **Pass 1:** Compute raw RS values at 4 timeframes (21d, 63d, 126d, primary) across all tickers → percentile-rank each
2. **Pass 2:** For each ticker, run all 7 indicators → apply gradient/binary scoring → sum to weighted score → sort descending

---

## 4. Subsector Breakout Detection

The second layer aggregates individual ticker scores into subsector-level metrics and uses a state machine to detect coordinated breakout waves.

### 4.1 Subsector Metrics

- **Breadth %** — Percentage of tickers in the subsector scoring ≥ 6 (the "hot" threshold)
- **Z-Score** — How unusual today's avg score is vs 60-day rolling history (standard deviations above mean)
- **Acceleration** — 2nd derivative of avg score (velocity of velocity)
- **Signal Consensus** — How coordinated the indicators are across tickers (High = 75%+ share the same top indicator)

### 4.2 State Machine

```
quiet → warming → emerging → confirmed → fading → quiet
                               ↑           ↓
                               ←───────────┘ (recovery/revival)
```

The state machine has **7 states**, progressing from cold to hot:

| Status | Definition | Entry Condition |
|--------|-----------|-----------------|
| **Quiet** | No notable activity — breadth < 25% | Default / fading cools |
| **Warming** | Early signs of life — breadth 25–49% | Breadth rises |
| **Emerging** | Breadth spiked, watching for confirmation | Breadth ≥ 50% + z > 1.0 + accel ≥ 0 |
| **Confirmed** | Sustained hot breadth for 3+ consecutive readings — actionable signal | 3+ hot readings |
| **Steady Hot** | Breadth ≥ 50% but z-score too low for Emerging (strength is baseline, not new) | Quiet + breadth ≥ 50% |
| **Fading** | Breadth declining from a confirmed breakout | Breadth < 50% or score declining |
| **Revival** | Recovery from fading back to confirmed — highest alpha signal | Fading + breadth recovers + accel > 0 |

### 4.3 Configuration Parameters

```yaml
breakout_detection:
  breadth_threshold: 6          # score threshold for "hot" ticker
  breadth_trigger: 0.5          # 50% of subsector must be hot
  z_score_trigger: 1.0          # std devs above mean for emerging
  lookback_days: 60             # rolling window for mean/std
  confirm_days: 3               # consecutive hot readings for confirmation
  fade_cool_days: 5             # consecutive cool readings to return to quiet
  history_retention_days: 180   # SQLite retention
```

---

## 5. Automated Trade Execution (Layer 3)

Once scoring identifies a signal and the subsector state machine confirms a breakout wave, `trade_executor.py` takes over: it submits orders to Alpaca paper trading, attaches GTC stop orders, and manages exits.

### 5.1 Current Live Configuration

```yaml
trade_execution:
  entry_threshold:    8.5       # score ≥ 8.5 (Hot tier)
  persistence_days:   3         # sustained for 3 prior trading days
  exit_threshold:     5.0       # exit when score drops below 5
  stop_loss_pct:      0.20      # 20% trailing stop (via Alpaca GTC order)
  max_positions:      12        # hard cap on concurrent positions
  max_position_pct:   0.083     # 8.3% of equity per position (= 1/12)
  min_position_size:  500       # don't open positions smaller than $500
```

**Non-YAML constants** (in `trade_executor.py`):
- `CASH_FLOOR_PCT = 0.05` — reserve 5% of equity uncommitted
- `LIMIT_ORDER_BUFFER = 0.03` — limit price = sizing_price × 1.03

### 5.2 Execution Pipeline (per daily run)

Triggered by `.github/workflows/daily-trade-execution.yml` at 21:30 UTC (5:30 PM ET) Mon-Fri. Sequence:

1. **Connect to Alpaca** (paper-account safety check — account number must start with `PA`)
2. **Snapshot**: equity, cash, open positions, last-equity (for day-over-day P&L)
3. **Detect filled stops**: scan Alpaca's closed-order history for GTC stops that hit since the last run; log as sell exits in `trade_history.json`
4. **Ensure stops in place**: idempotently backfill a GTC stop order for any held position without one (this is how stops get attached to positions opened via limit orders that filled overnight)
5. **Detect unfilled limits**: flag prior-day limit orders that canceled without filling (for email reporting)
6. **Score all tickers**: yfinance fetch + `score_all()`; persist today's scores to SQLite
7. **Evaluate exits**: close positions where score < 5 (market sell next open) — stops fire independently via Alpaca GTC orders
8. **Evaluate entries**: rank candidates by score, apply persistence filter (3 prior days ≥ 8.5 in SQLite), then sizing with cash floor
9. **Submit limit orders**: for each accepted candidate, `submit_order(LimitOrderRequest, limit=sizing×1.03, TIF=DAY)`
10. **Email digest**: send daily summary (if GMAIL_* env vars configured)

### 5.3 Sizing Formula with Cash Floor

The live sizing calculation in `evaluate_entries()`:

```python
raw_max_position  = equity × 0.083                           # 8.3% per slot
remaining_slots   = max_positions - current_position_count
floor_budget      = equity × 0.95 - projected_committed      # unused of 95% cap
per_slot_cap      = floor_budget / remaining_slots
max_position      = min(raw_max_position, per_slot_cap)       # cash-floor bounded
target_size       = min(projected_cash, max_position)         # also cash-bounded
qty               = floor(target_size / sizing_price)
```

The cash-floor constraint **mathematically guarantees** that total commitment never exceeds 95% of equity, regardless of how much individual fills gap up on the open.

### 5.4 Alpaca-First Pricing

Sizing uses Alpaca's latest-trade price first, falling back to yfinance close only if Alpaca data is unavailable:

```python
est_price = get_alpaca_latest_price(data_client, ticker)
if not est_price or est_price <= 0:
    est_price = _estimated_price(rec) or 0.0   # yfinance close
```

**Why**: yfinance closes are regular-session-only. They don't reflect extended-hours moves (news, earnings pre-market). When `trade_executor.py` runs at 5:30 PM ET, Alpaca's latest-trade may already reflect after-hours moves that will drive the next-day open. Using yfinance first caused the AEHR 2026-04-16 incident: yfinance close was $73.22, Alpaca-latest was already near $80, actual next-morning fill was $84.58.

### 5.5 Limit Orders vs Market Orders

Prior to 2026-04-17 the executor submitted `MarketOrderRequest` with DAY TIF. This filled at next-day open regardless of gap size. The `entry_mode_backtest.py` study compared 4 configurations over the full backtest window:

| Config | Total Return | Max DD | Sharpe | Sortino | Max Slippage | Neg-Cash Days |
|---|---|---|---|---|---|---|
| Baseline (market, no floor) | +417.1% | -21.7% | 2.86 | 4.47 | +27.7% | 70 |
| Defensive (market + 5% floor) | +351.4% | -19.8% | 2.93 | 4.52 | +27.7% | 0 |
| Limit-2% + 5% floor | +423.7% | -20.5% | 3.15 | 4.87 | +2.0% | 0 |
| **Limit-3% + 5% floor (live)** | **+433.7%** | **-20.8%** | **3.16** | **4.88** | **+2.6%** | **0** |

Limit-3% was chosen because it combines:
- Highest Sharpe/Sortino of the four configs
- Zero negative-cash days (mathematically guaranteed by floor)
- Capped slippage at ~2.6% (vs +27.7% baseline)
- Best path-dependency stability across 10 staggered start dates (+386% to +459% range)

The Limit-2% config had a notable +229% path outlier — the diagnostic analysis traced it to 6 gap-up momentum winners (LITE, TSEM, PL, HUT, GSAT, EXK) that the 2% limit filtered out but Limit-3% caught. 3% strikes the best balance between filtering real slippage and catching genuine breakouts.

### 5.6 Persistence Filter Semantics

The filter checks **the N most recent DB rows** for a ticker, not strict trading days. This intentionally handles genuine data gaps (holidays, CI glitches) gracefully:

```sql
SELECT date, score FROM ticker_scores
WHERE ticker = ? AND date < ?
ORDER BY date DESC LIMIT 3
```

If the DB has rows for 4/6, 4/2, 4/1 and the query day is 4/7, those three rows are treated as the "prior 3 days" even though 4/3 (Good Friday) is absent. This is correct — there was no trading day to record on 4/3.

### 5.7 Stop Order Management

Stop orders are real GTC sell orders at Alpaca, placed at `entry_price × 0.80`. They persist across the weekend and fire if the position trades below the stop price at any point during market hours.

Order lifecycle:

- **Attached post-fill**: `ensure_stops_for_positions()` runs at the top of every daily cycle. Any held position without an open GTC stop gets one attached. This handles both newly-filled limit orders and any edge cases where a stop got orphaned.
- **Canceled on score-based exit**: when `evaluate_exits()` triggers a sell (score < 5), the corresponding GTC stop is explicitly canceled via `cancel_stop_orders_for_ticker()` before submitting the market sell.
- **Detected when filled**: `detect_filled_stops()` scans closed orders for filled STOP+SELL types and logs them as exits. Wash-sale tracking is triggered for losing stop exits.

### 5.8 Skip-Reason Categories (for email digest)

When the executor rejects a candidate, the reason is classified into one of six categories:

- `insufficient cash` — raw cash or min-position-size constraint
- `position cap reached` — would exceed max_positions
- `wash sale cooldown` — logged advisory only (never blocks)
- `cash floor cap` — would exceed 95% equity commit budget
- `limit unfilled` — prior-day limit order didn't fill and the ticker's score has dropped below threshold
- `other` — persistence filter, tradeability, no-price, etc.

---

## 6. Backtesting & Validation

Extensive backtesting validates indicator selection, scoring weights, sector weighting, subsector detection, exit strategies, and portfolio construction.

### 6.1 Individual Indicator Backtesting

94 tickers over 2 years (4,048 observations). Each indicator was tested for its standalone 63-day forward return edge:

| Indicator | 63d Edge | Win Rate | Incremental Edge |
|-----------|----------|----------|-----------------|
| Relative Strength | +13.31% | 77.4% | Baseline |
| Ichimoku Cloud | +10.89% | 76.9% | +11.9% |
| Higher Lows | +7.73% | 75.9% | +7.73% |
| Moving Avg Alignment | +7.54% | 73.8% | −9.3% (dropped) |
| Rate of Change | +6.84% | 72.6% | +7.5% |
| Chaikin Money Flow | +6.57% | 73.2% | +8.9% |

**Key insight:** Moving Average Alignment had positive standalone edge but **negative** incremental edge — it was redundant with RS and actually degraded combined performance. This is why conditional edge analysis matters: standalone performance can be misleading.

**Optimal Stack** (RS + Ichimoku + Higher Lows, require 3/3): **84.4% win rate**, +37.94% avg 63-day return.

### 6.2 SPY-Adjusted Alpha by Score Tier

Forward returns measured relative to SPY over the same holding period:

| Score Threshold | 5-Day Alpha | 21-Day Alpha | 63-Day Alpha |
|----------------|-------------|--------------|--------------|
| ≥ 6 | +0.8% | +3.2% | +8.5% |
| ≥ 7 | +1.2% | +5.1% | +12.3% |
| ≥ 8 | +1.8% | +7.4% | +18.4% |
| ≥ 9 | +2.3% | +9.1% | +22.7% |

Higher score thresholds consistently produce higher alpha, confirming the scoring system's predictive value.

### 6.3 Sector Weighting Analysis

Five comprehensive tests were run to determine if sectors should receive different scoring multipliers:

1. Equal-weight baseline vs sector-adjusted weights
2. Optimized multipliers from in-sample data
3. Out-of-sample validation of optimized multipliers
4. Simple sector exclusion (remove worst performers)
5. Cross-validated multiplier stability

**Conclusion: No multipliers.** Sector-specific multipliers consistently overfit to in-sample data and degraded out-of-sample performance.

### 6.4 Subsector State Machine Validation

| Signal Type | Avg Alpha | Edge vs Baseline | Win Rate |
|-------------|-----------|-----------------|----------|
| Emerging | +13.1% | −1.7% (worse) | Below baseline |
| Confirmed | +21.7% | +6.9% | Above baseline |
| **Revival** | **+27.3%** | **+12.5%** | **70.8%** |

**Key finding:** Emerging signals do NOT beat random — they're noise. Confirmed signals work. Revival signals (fading → confirmed recovery) have the highest alpha and win rate, representing the highest-conviction entry point.

### 6.5 Portfolio Construction Backtest (`sizing_comparison_backtest.py`)

Head-to-head comparison of 4 sizing strategies over 2 years:

| Strategy | Return | Max DD | Sharpe | Sortino | Trades |
|---|---|---|---|---|---|
| Fixed/5-Max (20% × 5) | +148.9% | -28.8% | 1.34 | 1.82 | 36 |
| Fixed/10-Max (10% × 10) | +459.0% | -22.6% | 2.07 | 3.26 | 65 |
| Dynamic/Trim (equity/(N+1)) | +205.9% | -20.8% | 1.57 | 2.18 | 157 |
| Fixed/10+Swap (10% × 10, swap weakest at cap) | +229.0% | -29.8% | 1.54 | 2.21 | 134 |

**Winner: Fixed/10-Max.** A position-cap sweep over 3-20 revealed a plateau of similar-quality results between 10-12. The current live config uses 12 positions × 8.3% to maximize deployment flexibility without sacrificing risk-adjusted returns.

An **entry-threshold sweep** on the corrected simulator validated 8.5 as the sweet spot:

| Threshold | Return | Max DD | Sharpe | Sortino | Win Rate |
|---|---|---|---|---|---|
| 7.5 | +356.0% | -21.1% | 1.84 | 2.83 | 46.1% |
| 8.0 | +343.9% | -20.4% | 1.91 | 2.93 | 48.8% |
| **8.5** | **+428.1%** | **-22.1%** | **1.98** | **3.11** | **53.2%** |
| 9.0 | +424.6% | -24.0% | 1.96 | 3.12 | 61.0% |
| 9.5 | +264.0% | -21.4% | 1.66 | 2.65 | 35.1% |

A **stop-loss sweep** over 13 configurations (fixed 10/15/20%, trailing, ATR-based) showed fixed 20% stops eliminate whipsaws (stop-outs followed by recovery to above entry) without materially reducing capture.

### 6.6 Entry-Mode Backtest (`entry_mode_backtest.py`)

See Section 5.5 above. Validated limit-3% + 5% cash floor as the live configuration, specifically addressing the overnight-gap slippage problem.

---

## 7. Exit Strategy

10+ exit strategies were backtested to determine the optimal approach for maximizing returns while managing risk. The system combines a **score-based primary exit** with a **fixed stop-loss safety net**:

- **Score < 5 exit**: Momentum fading signal. Average hold ~26 days when triggered. Market sell at next open via `evaluate_exits()`.
- **20% fixed stop loss**: Safety net. Real GTC sell order at Alpaca, fires independently of daily execution cycle. Applied to ~25% of trades.
- **No fixed time limit**: Positions can run for months if score stays high. Only exit on deterioration.

The 20% threshold (widened from the original 15%) was chosen via the `sizing_comparison_backtest.py --sweep-stop-loss` analysis (2026-04-16). The primary finding: trailing stops caused catastrophic whipsaw rates (60-82%), while fixed-20% had the best combination of trade count, capture, and whipsaw elimination.

---

## 8. Universe-Wide Signal Diagnostics

A 2026-04-17/18 diagnostic pass evaluated whether the score actually predicts returns universe-wide, with statistical rigor.

### 8.1 Methodology

Three diagnostic scripts produce:

1. **`signal_diagnostics.py`** — aggregate bucket tables (forward return by score bucket) across raw + sm3/sm5/sm10/sm20 smoothing × 7/21/63-day horizons; Spearman rank correlations; score autocorrelation at lags 1/3/5/10/20.
2. **`signal_diagnostics_subsector.py`** — same analysis stratified by all 31 subsectors; ranked summary.
3. **`signal_diagnostics_significance.py`** — bootstrap 95% CIs on ρ (1000 iterations, observation-level resample) with significance classification.

### 8.2 Key Findings

**Smoothing improves predictive rank correlation** at every horizon:

| Smoothing | h=7d | h=21d | h=63d |
|---|---|---|---|
| raw | +0.058 | +0.122 | +0.277 |
| sm10 | +0.076 | +0.144 | +0.331 |
| sm20 | +0.091 | +0.152 | +0.353 |

**Score has ~10-day memory** (autocorrelation ρ at lag 10 = +0.462, at lag 20 = +0.173) — sm10 smoothing captures most persistent signal.

**26 of 31 subsectors have statistically significant signal** at 95% CI:
- 15 significantly positive (Chips — Networking/Photonics has the tightest CI and largest N, making it the highest-confidence positive finding at ρ=+0.264)
- 11 significantly negative (Industrial Robotics at ρ=−0.507 is the strongest anti-predictive signal)
- 5 inconclusive (genuine zeros, not underpowered)

### 8.3 Caveats

The observation-level bootstrap overstates precision because it assumes independence of (ticker, date) rows — score is autocorrelated across time and overlapping forward returns induce cross-observation dependence. Marginal findings (|ρ| < 0.15) should be treated with extra skepticism. Large-effect findings (|ρ| > 0.25) are robust even after mental correction.

### 8.4 Status

**Findings documented but not yet acted on.** No sector-weighted scoring, universe pruning, or signal-conditional sizing has been implemented. The diagnostics exist as reference for future strategy decisions.

---

## 9. Dashboard

The system is presented through a Streamlit web dashboard deployed on Streamlit Community Cloud, with 3 pages.

### 9.1 Subsectors Page

- Status banner showing counts for all 7 states with tooltip definitions
- Active Signals section with expandable subsector panels
- Each panel shows: avg score, breadth, z-score, acceleration, signal consensus, and individual ticker table
- All Subsectors summary table sorted by avg score

### 9.2 Tickers Page

- Summary metrics card: 5-tier counts (Total / Fire / Hot / Warm / Tepid / Cold) with tier colors + bold labels
- Breakouts table: all tickers scoring ≥ 7 with 5-tier color-coded Score column, Δ Since Hot metric, and all indicator columns
- Ticker Deep Dive: candlestick chart (4-month view, 50/200 SMA) with indicator breakdown table
- All Tickers table with dropdown filters for Sector, Subsector, and Score tier

### 9.3 Historical Charts Page

- **Ticker Score History:** Line chart tracking score evolution, with tier-band overlays at 9.5 / 8.5 / 7 / 5
- **Subsector Avg Score Trends:** Multi-select comparison of subsector avg score over time (up to 8 subsectors)
- **Score Heatmap:** Ticker × date heatmap in the 5-tier heat colorscale (blue → red), 2-month default window
- **Universe Score Distribution:** Stacked area chart with 5-bucket counts (Fire / Hot / Warm / Tepid / Cold) over time

---

## 10. Email Digest

`trade_executor.py --email` sends a daily summary via Gmail SMTP (requires `GMAIL_ADDRESS` / `GMAIL_APP_PASSWORD` env vars).

**Subject**: `Alpha Scanner 04/18: 3.14% | 2 buy / 1 sell` (MM/DD, no year, no "Today" prefix).

**Body** includes:
- Two-column summary cards: P&L (Today / All-Time / vs SPY) and Account (Equity / Cash / Positions x/12)
- Exits table, Entries table
- Skipped Signals table: Ticker | Subsector (2-letter code) | Score | Days ≥ 8.5 | Skip Reason — sorted by persistence
- Current Positions table: Ticker | Subsector | Days | Current | Day % | P&L % | P&L $ | Score — sorted by P&L % desc
- Exit Watch (positions in danger zones)
- Wash Sale Cooldowns (advisory)
- Subsectors referenced legend — only codes appearing in today's email

Subsector codes are defined in `trade_executor.py:SUBSECTOR_CODES` (31 entries) — e.g. `CN = Chips — Networking/Photonics`, `AX = AI Security`, `SO = Subsea/Ocean Robotics`, etc. Unknown codes fall back to the full name for graceful degradation.

---

## 11. Technical Architecture

### 11.1 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9.6 |
| Market Data | yfinance (free) + Alpaca StockHistoricalDataClient (latest-trade for sizing) |
| Broker | Alpaca Paper Trading API (`alpaca-py` SDK) |
| Database | SQLite (local persistence for historical scores) |
| Dashboard | Streamlit (deployed on Streamlit Community Cloud) |
| Charts | Plotly (interactive candlestick, heatmap, area charts) |
| Analysis | pandas, numpy, scipy, scikit-learn |
| Email | Gmail SMTP (direct, no service dependency) |
| Automation | GitHub Actions (3 scheduled workflows) |
| Source Control | GitHub (`t0ddb/alpha-scanner`) |

### 11.2 Database Schema (SQLite)

**`subsector_daily`** — One row per subsector per snapshot date
- date, subsector, subsector_name, sector, ticker_count, avg_score, max_score, breadth, hot_count, ticker_scores (JSON)

**`subsector_breakout_state`** — Current state machine state per subsector
- subsector (PK), status, status_since, consecutive_hot, consecutive_cool, peak_avg_score, peak_breadth, updated_at

**`ticker_scores`** — One row per ticker per snapshot date
- date, ticker, name, subsector, sector, score, signals (JSON), signal_weights (JSON)

### 11.3 Trade Audit Log (`trade_history.json`)

JSON append-only log with three entry types:

- `side: "buy"` — captures ticker, date, sizing_price, qty, cost_basis, score_at_entry, Alpaca order_id, stop_order_id, stop_price, reason
- `side: "sell"` — captures ticker, date, fill_price, qty, proceeds, pnl, pnl_pct, score_at_entry (auto-carried from matching buy), score_at_exit, reason, hold_days, Alpaca order_id
- `side: "trim"` with `action: "transition_trim"` — reserved for one-time migration operations (e.g., the 2026-04-20 sizing migration from old 20%-of-equity to new 8.33%-of-equity basis)

`trade_log.bucket_stats()` groups realized P&L by entry-score bucket (8.5-9.0, 9.0-9.5, 9.5+) for quick validation of per-tier performance against backtest expectations.

### 11.4 GitHub Actions Workflows

- `daily-backfill.yml` — cron `0 21 * * 1-5` (21:00 UTC Mon-Fri). Runs `backfill_subsector.py --days 180 --frequency 5`, commits `breakout_tracker.db`.
- `daily-trade-execution.yml` — cron `30 21 * * 1-5` (21:30 UTC Mon-Fri). Runs `trade_executor.py`, commits `trade_history.json`, `wash_sale_log.json`, `breakout_tracker.db`. Supports `workflow_dispatch` inputs `entry_threshold` and `persistence_days` as env-var overrides.
- `quarterly-review.yml` — cron `0 14 1 1,4,7,10 *` (14:00 UTC on the 1st of Jan/Apr/Jul/Oct). Runs `quarterly_review.py --months 12`, writes `quarterly_reviews/review_YYYY_QN.txt`.

Crons are staggered so backfill's DB commit lands before trade-exec pulls the repo.

### 11.5 File Map

**Core pipeline:**
- `ticker_config.yaml` — Central config (180 tickers, sectors, indicator params, weights, thresholds, trade-exec config)
- `config.py` — YAML parser with loader functions
- `data_fetcher.py` — yfinance wrapper
- `indicators.py` — 7-indicator scoring engine
- `subsector_breakout.py` — Breakout detection (aggregation, state machine)
- `subsector_store.py` — SQLite persistence layer

**Execution:**
- `trade_executor.py` — Alpaca paper-trading execution: cash floor, limit orders, GTC stops, email digest
- `trade_log.py` — Trade history persistence + bucket stats
- `wash_sale_tracker.py` — Log-only loss exits + cooldowns

**Dashboard & review:**
- `dashboard.py` — Streamlit UI (3 pages, 5-tier Tableau 20 palette)
- `quarterly_review.py` — 7-section automated health report

**Backtesting:**
- `sizing_comparison_backtest.py` — 4-strategy portfolio-construction backtest + sweep flags
- `entry_mode_backtest.py` — market-vs-limit entry-order comparison
- `backtester.py` — shared backtest primitives
- `portfolio_backtest.py` — legacy portfolio simulation (superseded but retained for reference)

**Diagnostics:**
- `signal_diagnostics.py` — aggregate score→forward-return analysis
- `signal_diagnostics_subsector.py` — per-subsector decomposition
- `signal_diagnostics_significance.py` — bootstrap 95% CIs on ρ

**Operations:**
- `backfill_subsector.py` — Historical data population
- `backfill_stop_orders.py` — One-time GTC stop backfill for pre-existing positions
- `transition_trim.py` — One-time position-sizing migration (dry-run + execute modes)

---

## 12. Key Design Decisions

1. **Three-layer architecture.** Scoring → subsector detection → automated execution. Each layer is testable in isolation and has its own validation artifacts.

2. **Regime-independent scoring.** All indicators and weights remain the same regardless of market environment. Testing showed RS-regime-dependent weighting hurt out-of-sample performance.

3. **Breadth-based subsector detection.** Detects breakouts at the subsector level, not individual stock level. When 50%+ of a subsector is "hot," it signals a thematic momentum wave.

4. **State machine with hysteresis.** Confirm_days (3) and fade_cool_days (5) prevent rapid state cycling. Widened from (2, 3) based on backtest results showing too much noise.

5. **No sector multipliers.** 5 comprehensive tests showed sector-specific weight adjustments overfit to historical data. Equal weighting across sectors is more robust.

6. **Conditional edge for indicator selection.** Indicators were evaluated for their incremental edge after controlling for RS, not standalone performance. This prevented including redundant signals (MA alignment, 52w high) that degrade combined accuracy.

7. **Position-trading timeframe.** System designed for after-market-close analysis. All forward return analysis uses 5/10/21/63 day windows. Not suitable for intraday trading.

8. **Limit orders over market orders.** The entry-mode backtest validated that 3% limit orders filter out the worst overnight gap-ups (e.g., AEHR +15.5%) without materially reducing fill rate (87%) or sacrificing real breakout momentum. 83% of missed signals re-qualify within 5 trading days.

9. **Cash floor guarantees solvency.** The 5% cash floor is not a guideline — the sizing formula mathematically guarantees total commitment ≤ 95% of equity regardless of how individual fills come in.

10. **Alpaca-first pricing for execution.** yfinance closes are stale by 5:30 PM ET. Alpaca's latest-trade reflects extended-hours moves that drive the next-day open. Using the more-current price for sizing eliminates a class of slippage surprises.

11. **Real GTC stops, not software-simulated.** Stop orders are placed at Alpaca as real GTC sell orders. If the dashboard / executor / network is down during a crash, the stops still fire. Simulated stops in backtest match this semantics.

12. **Documented diagnostics, un-acted findings.** The 31-subsector signal analysis identified 26 subsectors with statistically significant signal direction (15 positive, 11 negative). Findings are documented for future revisit; the universe has not been pruned or reweighted based on them. This is a deliberate hold — diagnostic insight is cheap; strategy changes are expensive.

---

*Built with Claude Code · April 2026*
