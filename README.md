[README.md](https://github.com/user-attachments/files/26392157/README.md)
# Alpha Scanner

**A momentum breakout detection system that identifies when entire market sectors — not just individual stocks — are experiencing coordinated technical breakouts.**

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B) ![License](https://img.shields.io/badge/License-MIT-green) ![Tickers](https://img.shields.io/badge/Universe-165_Tickers-orange)

<!-- 
📸 SCREENSHOT: Add a hero screenshot of the dashboard here
Recommended: Full-width screenshot of the Tickers page showing the breakouts table
Save as: docs/images/dashboard-hero.png
-->
![Alpha Scanner Dashboard](docs/images/dashboard-hero.png)

---

## The Idea

I built Alpha Scanner to answer a question I kept running into as an investor: **how do you systematically catch a sector rotation before it's obvious?**

In early 2025, I watched AI infrastructure stocks break out in sequence — first GPUs, then networking, then memory, then power. Each wave was visible in hindsight, but hard to catch in real time. The stocks that moved first showed the same technical signatures: rising relative strength, expanding volatility, accumulating institutional volume. And crucially, they moved *together* — when one stock in a subsector broke out, the rest often followed.

Alpha Scanner automates that pattern recognition. It scores 165 stocks across 9 sectors and 31 subsectors on a 0–10 scale, then aggregates those scores to detect when an entire subsector is breaking out simultaneously. The result is a daily signal that tells you both **where to look** (which subsectors are heating up) and **what to do** (which individual stocks are scoring highest).

---

## How It Works

Alpha Scanner operates in two layers:

<!-- 
📸 SCREENSHOT: Architecture diagram
Save as: docs/images/architecture.png
Or use the generated architecture-diagram.html file, screenshot it, and save here
-->
![System Architecture](docs/images/architecture-diagram.html)

### Layer 1 — Ticker Scoring

Each stock is scored daily using 7 technical indicators, weighted by their empirically-tested predictive power:

| Indicator | Weight | What It Detects |
|-----------|--------|-----------------|
| **Relative Strength vs. S&P 500** | 3.0 pts | Stocks outperforming the market — the single strongest predictor of future returns |
| **Ichimoku Cloud** | 2.0 pts | Confirmed uptrend with price above a bullish cloud formation |
| **Chaikin Money Flow** | 1.5 pts | Institutional buying pressure — "smart money" accumulating shares |
| **Rate of Change** | 1.5 pts | Strong recent price momentum over 21 days |
| **Higher Lows** | 1.0 pt | Staircase uptrend pattern — each pullback is shallower than the last |
| **Dual-Timeframe RS** | 0.5 pts | Momentum *acceleration* — strong AND getting stronger |
| **ATR Expansion** | 0.5 pts | Expanding volatility, often signaling the start of a big move |

**Max score: 10.** Stocks scoring 8+ are in a strong breakout setup. Stocks scoring 9+ have nearly every signal firing simultaneously.

### Layer 2 — Subsector Breakout Detection

Individual ticker scores are aggregated into subsector-level metrics. A state machine monitors each of the 31 subsectors and tracks their progression through a breakout lifecycle:

<!-- 
📸 SCREENSHOT: State machine diagram
Save as: docs/images/state-machine.png
Or use the generated state-machine-diagram.html file
-->
![Subsector State Machine](docs/images/state-machine-diagram.html)

When 50%+ of the stocks in a subsector are scoring 6 or higher ("hot"), that subsector enters the breakout pipeline. If it sustains that breadth for 3 consecutive readings, it's **Confirmed** — the highest-conviction signal.

---

## What I Tested — and What I Learned

The indicator weights above weren't chosen by intuition. They're the result of a systematic backtesting process that started with 16 candidate indicators, tested each one against 3 years of historical data, and progressively narrowed to the 7 that actually predict future returns.

### Starting with 16 indicators, keeping 7

I began with every commonly-used technical indicator I could find: moving average crossovers, RSI, MACD, Bollinger Bands, volume spikes, On-Balance Volume, ADX, Donchian channels, and more. Each was tested for a simple metric: **when this indicator fires, do stocks perform better over the next 63 trading days than when it doesn't?**

The results surprised me:

<!-- 
📸 SCREENSHOT: Indicator ranking chart or table
Save as: docs/images/indicator-ranking.png
-->
![Indicator Ranking by Predictive Edge](docs/images/indicator-methodology-diagram.html)

**Top performers** — Relative Strength (+13.3% edge), Ichimoku Cloud (+10.9%), Higher Lows (+7.7%). These earned the highest weights.

**Near-useless** — Volume Spike (+0.7% edge), Bollinger Band Squeeze (+0.4%), MACD Crossover (+0.9%). These were dropped entirely.

**The biggest surprise** — Moving Average Alignment had a +7.5% standalone edge, but when I tested it *controlling for Relative Strength*, it actually had a **negative** incremental edge of −9.3%. It was completely redundant with RS and added noise. This is why conditional testing matters: an indicator that looks good in isolation can be harmful in combination.

### Validating that scores predict returns

The scoring system was validated against SPY-adjusted forward returns. Higher scores consistently produce higher alpha:

| Score Threshold | 21-Day Alpha | 63-Day Alpha |
|----------------|--------------|--------------|
| ≥ 6 | +3.2% | +8.5% |
| ≥ 7 | +5.1% | +12.3% |
| ≥ 8 | +7.4% | +18.4% |
| ≥ 9 | +9.1% | +22.7% |

The monotonic increase across thresholds confirms the scoring system is capturing real signal, not noise.

### Testing whether subsector detection adds value

The subsector layer was validated separately. The key finding: **not all states are equal.**

| Subsector State | Avg Alpha (63-day) | Edge vs. Baseline |
|-----------------|-------------------|-------------------|
| Emerging | +13.1% | −1.7% (no better than random) |
| Confirmed | +21.7% | +6.9% |
| **Revival** | **+27.3%** | **+12.5%** |

Emerging signals — when a subsector first starts to heat up — are actually noise. They don't beat baseline. But **Confirmed** breakouts (sustained for 3+ readings) and **Revival** signals (a subsector that faded and then recovered) show strong, actionable alpha.

### Testing exit strategies

I backtested 10+ exit strategies including fixed time periods, trailing stops, moving average breaks, and score-based exits. The winner: **a 15% stop loss combined with exiting when a stock's score drops below 5**. This gives positions room to run while cutting losses on the ~25% of trades that don't work out.

---

## Universe Coverage

The system tracks 165 tickers across 9 sectors and 31 subsectors:

| Sector | Tickers | Subsectors | Key Themes |
|--------|---------|------------|------------|
| AI & Tech Capex | ~55 | 15 | GPUs, memory, networking, power, data centers, hyperscalers |
| Precious Metals | 14 | 3 | Gold/silver ETFs, miners |
| Crypto | 10 | 2 | BTC, ETH, SOL, AI/DePIN tokens |
| Robotics & Automation | 14 | 3 | Surgical, industrial, subsea |
| Biotechnology | 8 | 2 | CRISPR, synthetic biology |
| Space & Satellite | 11 | 2 | Launch vehicles, satellite data |
| Quantum Computing | 4 | 1 | Quantum hardware & software |
| Nuclear & Uranium | 9 | 2 | SMRs, uranium miners |
| eVTOL & Drones | 5 | 1 | Urban air mobility |

Adding new sectors or tickers requires only editing a YAML config file — no code changes.

---

## Dashboard

The live dashboard is built with Streamlit and provides three views:

### Tickers — Current Breakout Signals

<!-- 
📸 SCREENSHOT: Tickers page showing the breakouts table with score badges
Save as: docs/images/dashboard-tickers.png
-->
![Tickers Page](docs/images/dashboard-tickers.png)

Shows every stock's current score, which indicators are firing, and a candlestick chart with technical overlays for any selected ticker.

### Subsectors — Breakout State Tracking

<!-- 
📸 SCREENSHOT: Subsectors page showing state machine status banners and active signals
Save as: docs/images/dashboard-subsectors.png
-->
![Subsectors Page](docs/images/dashboard-subsectors.png)

Monitors all 31 subsectors through the breakout lifecycle. Shows breadth, z-scores, acceleration, and which subsectors are Emerging, Confirmed, or in Revival.

### Historical Charts — Score Evolution Over Time

<!-- 
📸 SCREENSHOT: Historical page showing the score heatmap or subsector breadth trends
Save as: docs/images/dashboard-historical.png
-->
![Historical Charts](docs/images/dashboard-historical.png)

Tracks how scores and subsector breadth have evolved, with heatmaps, trend lines, and distribution charts pulled from the SQLite database.

---

## Technical Architecture

| Component | Technology | Role |
|-----------|-----------|------|
| **Language** | Python 3.9 | Core pipeline |
| **Data** | yfinance | Free market data — no API key required |
| **Database** | SQLite | Stores historical scores and subsector states |
| **Dashboard** | Streamlit + Plotly | Interactive web UI with candlestick charts |
| **Config** | YAML | Single source of truth for tickers, weights, thresholds |
| **Analysis** | pandas, numpy, scipy | Backtesting and statistical analysis |

### Project Structure

```
alpha-scanner/
├── ticker_config.yaml          # Universe, indicator params, scoring weights
├── config.py                   # Configuration loader
├── data_fetcher.py             # Market data retrieval (yfinance)
├── indicators.py               # 7-indicator scoring engine
├── subsector_breakout.py       # Subsector aggregation + state machine
├── subsector_store.py          # SQLite persistence layer
├── dashboard.py                # Streamlit web dashboard
├── backfill_subsector.py       # Historical data population
├── backtester.py               # Backtesting framework
└── docs/
    └── images/                 # Dashboard screenshots
```

---

## Key Design Decisions

**Why subsector breadth instead of individual stock signals?**
A single stock breaking out could be a one-off event (earnings surprise, acquisition rumor). When 50%+ of a subsector breaks out simultaneously, it signals a thematic wave — AI spending acceleration, a gold breakout, crypto cycle turning. These are tradeable trends, not isolated events.

**Why drop indicators with positive standalone performance?**
Moving Average Alignment had a +7.5% standalone edge — but a −9.3% *incremental* edge after controlling for Relative Strength. In a multi-indicator system, what matters is whether an indicator adds information beyond what the other indicators already capture. Testing only standalone performance leads to redundant, noisy systems.

**Why no sector-specific weights?**
I ran 5 separate tests on sector-adjusted scoring (in-sample optimization, out-of-sample validation, cross-validation). Sector multipliers consistently overfit to historical data and degraded live performance. Equal weighting across sectors is more robust.

**Why a state machine instead of simple thresholds?**
Thresholds are stateless — they can't distinguish between a subsector that just crossed 50% breadth today versus one that's been above 50% for three weeks. The state machine adds memory, requiring sustained signals before upgrading to "Confirmed" and allowing for Revival detection (the highest-alpha signal).

---

## Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the dashboard
streamlit run dashboard.py

# Score all tickers from the command line
python3 -c "
from config import load_config
from data_fetcher import fetch_all
from indicators import score_all, print_scorecard
cfg = load_config()
data = fetch_all(cfg, period='1y', verbose=True)
results = score_all(data, cfg)
print_scorecard(results, min_score=7)
"

# Backfill historical data
python3 backfill_subsector.py
```

---

## About

Built by [Todd Bruschwein](https://linkedin.com/in/toddbruschwein) — Revenue Operations & Analytics leader with 13+ years at Tesla and Lucid Motors. Alpha Scanner started as a side project to systematically track the AI infrastructure investment cycle and grew into a full cross-asset breakout detection system.

Built with Python and [Claude](https://claude.ai) · March 2026

---

*Alpha Scanner is a personal research tool. Nothing here constitutes financial advice. Past backtested performance does not guarantee future results.*
