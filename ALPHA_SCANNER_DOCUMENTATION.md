# Alpha Scanner — Technical Documentation & Methodology

**Subsector Momentum Breakout Detection System**
*March 2026*

---

## 1. Executive Summary

Alpha Scanner is a momentum breakout scoring system designed to detect when entire subsectors — not just individual stocks — are experiencing coordinated technical breakouts. The system operates on a two-layer architecture:

- **Layer 1 — Ticker Scoring:** Each of 165 tickers is scored 0–10 using 7 technical indicators. This tells you *what to do* once you're looking at a stock.
- **Layer 2 — Subsector Breakout Detection:** A state machine aggregates individual scores into subsector breadth metrics, detecting when 50%+ of a subsector is "hot." This tells you *where to look.*

The core thesis: when multiple stocks in the same subsector start firing technical signals simultaneously, it signals a sector rotation or thematic breakout wave worth trading.

### Key Results from Backtesting

- Tickers scoring **8+** produce **+18.41% SPY-adjusted alpha** at 63 days
- **Confirmed** subsector breakouts generate **+21.7% alpha** with +6.9% edge over baseline
- **Revival** signals (fading → confirmed) show highest alpha: **+27.3%**, 70.8% win rate
- Optimal exit strategy: **15% stop loss + exit when score drops below 5**

---

## 2. Universe & Structure

The system tracks **165 tickers** across **9 sectors** and **31 subsectors**, spanning themes from AI infrastructure to space, biotech, quantum computing, and crypto.

| Sector | Tickers | Subsectors |
|--------|---------|------------|
| AI & Tech Capex Cycle | ~55 | Compute, Memory, Networking/Photonics, Power, Data Center, Alt AI Compute, Hyperscalers, AI Software, Enterprise AI, AI Security, Healthcare AI, Physical AI, Power Semis, Semi Equipment, Semi Test |
| Metals | 14 | Gold & Silver Direct, Gold Miners, Silver Miners |
| Crypto | 10 | Crypto Majors, Crypto/AI Crossover Tokens |
| Robotics & Automation | 14 | Surgical/Medical, Industrial, Subsea/Ocean |
| Biotechnology | 8 | Gene Editing/CRISPR, Synthetic Biology |
| Space & Satellite | 11 | Launch & Spacecraft, Satellite Comms & Data |
| Quantum Computing | 4 | Quantum Hardware & Software |
| Nuclear & Uranium | 9 | Nuclear Reactors/SMR, Uranium Miners |
| eVTOL & Drones | 5 | eVTOL/Urban Air Mobility |

**Benchmark:** SPY (S&P 500 ETF). All alpha calculations are raw forward returns minus SPY's return over the same period.

---

## 3. Scoring Engine

Each ticker is scored on a 0–10 scale using 7 technical indicators. Weights were determined through a **3-year conditional edge analysis** — each indicator's weight reflects its incremental predictive edge after controlling for Relative Strength (the strongest individual predictor).

### 3.1 The 7 Scored Indicators

| Indicator | Weight | Type | Edge | Win Rate | Logic |
|-----------|--------|------|------|----------|-------|
| Relative Strength | 0–3.0 | Gradient | +13.31% | 77.4% | 63-day outperformance vs SPY, percentile-ranked across universe |
| Higher Lows | 0–1.0 | Gradient | +7.73% | 75.9% | Consecutive weekly higher-low periods (staircase uptrend) |
| Ichimoku Cloud | 2.0 | Binary | +11.9% | 76.9% | Price above cloud AND cloud bullish (Senkou A > B) |
| Chaikin Money Flow | 1.5 | Binary | +8.9% | 73.2% | 20-day CMF > 0.05 (institutional buying pressure) |
| Rate of Change | 1.5 | Binary | +7.5% | 72.6% | 21-day ROC > 5% (strong recent momentum) |
| Dual-TF RS | 0.5 | Binary | +5.2% | N/A | Multi-timeframe RS strong AND accelerating |
| ATR Expansion | 0.5 | Binary | +5.2% | N/A | 14-day ATR in top 20% of 50-day range |

### 3.2 Gradient vs Binary Scoring

- **Gradient:** Relative Strength scores proportionally from 0 to 3.0 based on percentile (90th→3.0, 80th→2.4, 70th→1.8, 60th→1.2, 50th→0.6). Higher Lows scores 0.25 per consecutive higher low up to 1.0.
- **Binary:** Ichimoku, CMF, ROC, Dual-TF RS, and ATR Expansion award full weight if triggered, zero otherwise. Determined by analyzing forward return distributions — these indicators showed no meaningful improvement from proportional scoring.

### 3.3 Indicators Tested but NOT Included

Two additional indicators were evaluated but excluded from the final scoring model:

- **Moving Average Alignment:** −9.3% incremental edge. Harmful when Relative Strength is already strong — it adds noise without predictive value. Still computed and displayed on the dashboard for visual context.
- **Near 52-Week High:** −3.3% incremental edge. Fully redundant with Relative Strength — tickers with high RS are almost always near their 52-week highs. Displayed on dashboard but not scored.

### 3.4 Score Tiers

| Tier | Score Range | Interpretation |
|------|-------------|----------------|
| **Fire** | 9.0+ | Exceptional — nearly all indicators firing simultaneously |
| **Strong** | 7.0 – 8.9 | Actionable breakout setup with multiple confirming signals |
| **Moderate** | 5.0 – 6.9 | Some signals present but not yet a high-conviction setup |
| **Weak** | Below 5.0 | Few or no technical signals — not actionable |

### 3.5 Scoring Process (Two-Pass)

1. **Pass 1:** Compute raw RS values at 4 timeframes (21d, 63d, 126d, primary) across all tickers → percentile rank each
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
quiet → emerging → confirmed → fading → quiet
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

## 5. Backtesting & Validation

Extensive backtesting was conducted to validate indicator selection, scoring weights, sector weighting, subsector detection, and exit strategies.

### 5.1 Individual Indicator Backtesting

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

### 5.2 SPY-Adjusted Alpha by Score Tier

Forward returns measured relative to SPY over the same holding period:

| Score Threshold | 5-Day Alpha | 21-Day Alpha | 63-Day Alpha |
|----------------|-------------|--------------|--------------|
| ≥ 6 | +0.8% | +3.2% | +8.5% |
| ≥ 7 | +1.2% | +5.1% | +12.3% |
| ≥ 8 | +1.8% | +7.4% | +18.41% |
| ≥ 9 | +2.3% | +9.1% | +22.7% |

Higher score thresholds consistently produce higher alpha, confirming the scoring system's predictive value.

### 5.3 Sector Weighting Analysis

Five comprehensive tests were run to determine if sectors should receive different scoring multipliers:

1. Equal-weight baseline vs sector-adjusted weights
2. Optimized multipliers from in-sample data
3. Out-of-sample validation of optimized multipliers
4. Simple sector exclusion (remove worst performers)
5. Cross-validated multiplier stability

**Conclusion: No multipliers.** Sector-specific multipliers consistently overfit to in-sample data and degraded out-of-sample performance. Simple sector exclusion (removing crypto and biotech) modestly improved alpha by ~4–5%, but this was not implemented to avoid reducing universe coverage.

### 5.4 Subsector State Machine Validation

| Signal Type | Avg Alpha | Edge vs Baseline | Win Rate |
|-------------|-----------|-----------------|----------|
| Emerging | +13.1% | −1.7% (worse) | Below baseline |
| Confirmed | +21.7% | +6.9% | Above baseline |
| **Revival** | **+27.3%** | **+12.5%** | **70.8%** |

**Key finding:** Emerging signals do NOT beat random — they're noise. Confirmed signals work. Revival signals (fading → confirmed recovery) have the highest alpha and win rate, representing the highest-conviction entry point.

---

## 6. Exit Strategy Analysis

10+ exit strategies were backtested to determine the optimal approach for maximizing returns while managing risk.

### 6.1 Strategies Tested

**Score-Based Exits:**
- Exit when score drops below 5, 6, or by 2–3 points

**Price/Technical Exits:**
- Price below 20 SMA, 50 SMA, Ichimoku cloud break, CMF negative

**Stop Losses:**
- 5%, 10%, 15% fixed stop losses; 2x and 3x ATR trailing stops

**RS-Based Exits:**
- RS underperforming SPY over 10-day or 21-day windows

**Time-Based Baselines:**
- Fixed hold periods: 10, 21, 42, 63 days

### 6.2 Winning Strategy: 15% Stop Loss + Score < 5

The optimal exit combines a 15% stop loss as crash protection with a score-based exit for momentum fading:

- **15% stop loss:** Safety net. Triggers on ~25% of trades, limiting worst-case scenarios.
- **Score < 5 exit:** Momentum fading signal. Average hold ~26 days when triggered.
- **No fixed time limit:** Positions can run for months if score stays high. Only exits on deterioration.

For entries at score ≥ 9 (Fire tier): highest alpha per trade with tightest max loss (−22.8%), making it the recommended entry threshold for new users of the system.

---

## 7. Dashboard & Presentation

The system is presented through a Streamlit web dashboard deployed on Streamlit Community Cloud, with 3 pages:

### 7.1 Subsectors Page

- Status banner showing counts for all 7 states with tooltip definitions
- Active Signals section with expandable subsector panels
- Each panel shows: avg score, breadth, z-score, acceleration, signal consensus, and individual ticker table
- All Subsectors summary table sorted by avg score

### 7.2 Tickers Page

- Summary metrics banner (Fire/Strong/Moderate/Weak counts) with color coding
- Breakouts table: all tickers scoring 7+ with color-coded scores, Δ Since Hot metric, and all indicator columns
- Ticker Deep Dive: candlestick chart (4-month view, 50/200 SMA) with indicator breakdown table
- All Tickers table with dropdown filters for Sector, Subsector, and Score tier

### 7.3 Historical Charts Page

- **Ticker Score History:** Line chart tracking score evolution over time
- **Subsector Breadth Trends:** Multi-select comparison of subsector breadth over time
- **Score Heatmap:** Ticker × date heatmap (white-to-green scale, 3-month default)
- **Universe Score Distribution:** Stacked area chart showing Fire/Strong/Moderate/Weak counts over time

---

## 8. Technical Architecture

### 8.1 Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9.6 |
| Data Source | yfinance (free, no API key required) |
| Database | SQLite (local persistence for historical scores) |
| Dashboard | Streamlit 1.50 (deployed on Streamlit Community Cloud) |
| Charts | Plotly 6.6 (interactive candlestick, heatmap, area charts) |
| Analysis | pandas, numpy, scipy, scikit-learn |
| Source Control | GitHub (t0ddb/alpha-scanner) |

### 8.2 Database Schema (SQLite)

**`subsector_daily`** — One row per subsector per snapshot date
- date, subsector, subsector_name, sector, ticker_count, avg_score, max_score, breadth, hot_count, ticker_scores (JSON)

**`subsector_breakout_state`** — Current state machine state per subsector
- subsector (PK), status, status_since, consecutive_hot, consecutive_cool, peak_avg_score, peak_breadth, updated_at

**`ticker_scores`** — One row per ticker per snapshot date
- date, ticker, name, subsector, sector, score, signals (JSON), signal_weights (JSON)

### 8.3 File Map

**Core Pipeline:**
- `ticker_config.yaml` — Central config (tickers, sectors, indicator params, weights, thresholds)
- `config.py` — YAML parser with loader functions
- `data_fetcher.py` — yfinance wrapper for data retrieval
- `indicators.py` — Scoring engine (7 indicators, weighted composite scoring)
- `subsector_breakout.py` — Breakout detection (aggregation, state machine)
- `subsector_store.py` — SQLite persistence layer (3 tables)
- `dashboard.py` — Streamlit UI (3 pages, live + historical)

**Analysis & Backtesting:**
- `backtester.py` — Core backtesting framework
- `exit_signal_backtest.py` — 10+ exit strategy comparison
- `exit_combo_backtest.py` — Stop loss + score exit combination testing
- `indicator_optimizer.py` — 3-year conditional edge analysis for weight determination
- `minervini_backtest.py` — Minervini SEPA comparison
- `gradient_analysis.py` — Gradient vs binary scoring analysis

**Operations:**
- `backfill_subsector.py` — Historical data population
- `email_alerts.py` — Resend API email digest (not yet configured)

---

## 9. Key Design Decisions

1. **Regime-independent scoring.** All indicators and weights remain the same regardless of market environment. Testing showed RS-regime-dependent weighting hurt out-of-sample performance.

2. **Breadth-based subsector detection.** Detects breakouts at the subsector level, not individual stock level. When 50%+ of a subsector is "hot," it signals a thematic momentum wave.

3. **State machine with hysteresis.** Confirm_days (3) and fade_cool_days (5) prevent rapid state cycling. Widened from (2, 3) based on backtest results showing too much noise.

4. **No sector multipliers.** 5 comprehensive tests showed sector-specific weight adjustments overfit to historical data. Equal weighting across sectors is more robust.

5. **Conditional edge for indicator selection.** Indicators were evaluated for their incremental edge after controlling for RS, not standalone performance. This prevented including redundant signals (MA alignment, 52w high) that degrade combined accuracy.

6. **Position-trading timeframe.** System designed for after-market-close analysis. All forward return analysis uses 5/10/21/63 day windows. Not suitable for intraday trading.

7. **Free data, no dependencies.** Uses yfinance for market data (no API key, no cost). Raw OHLCV data is not stored — only scored results persist in SQLite. System can be run from any machine with Python installed.

---

*Built with Claude Code · March 2026*
