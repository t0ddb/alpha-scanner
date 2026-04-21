# Cross-Asset Breakout Tracker — Context Handoff

**Last updated:** 2026-03-16
**Project root:** `/Users/toddbruschwein/Claude-Workspace/breakout-tracker/`

---

## 1. What This Project Is

A **momentum breakout scoring system** that tracks 139 tickers across 9 sectors and 31 subsectors. It scores each stock on a 0-10 scale using 7 technical indicators, then aggregates those scores to detect **subsector-level breakout waves** using a state machine.

The core thesis: when multiple stocks in the same subsector start firing technical signals simultaneously, it signals a sector rotation or thematic breakout worth paying attention to.

**Tech stack:** Python 3.9.6 · pandas/numpy · yfinance (free market data) · SQLite (local persistence) · Streamlit (dashboard UI) · Plotly (interactive charts)

**Python compatibility note:** All files use `from __future__ import annotations` for 3.9 compatibility.

---

## 2. Universe Coverage

### Sectors (9) → Subsectors (31) → Tickers (139)

| Sector | Subsectors | Tickers | Key Names |
|--------|-----------|---------|-----------|
| AI & Tech Capex | Chips—Compute, Chips—Memory, Chips—Networking/Photonics, Power & Energy, Data Center Infra, Alt AI Compute, Hyperscalers | 37 | NVDA, AMD, AVGO, VRT, VST, CEG |
| Enterprise Software | Cloud/Data Infra, AI/ML Platforms, SaaS, Cybersecurity | 20 | PLTR, SNOW, CRWD, PANW, CRM |
| Semiconductors | Auto/Industrial Chips, Foundries, Equipment & Test | 14 | TSM, ASML, ON, LRCX |
| Precious Metals | Gold ETFs/Futures, Gold Miners, Silver Miners | 14 | GLD, NEM, GDX, AG |
| Crypto | Layer 1, AI/DePIN | 10 | BTC-USD, ETH-USD, SOL-USD, RENDER-USD |
| Robotics & Automation | Surgical/Medical, Industrial, Subsea/Ocean | 9 | ISRG, SYK, KRKNF, FANUY |
| Biotechnology | Gene Editing/CRISPR, Synthetic Biology | 7 | CRSP, NTLA, BEAM, TWST |
| Space & Satellite | Launch & Spacecraft, Satellite Comms & Data | 7 | RKLB, LUNR, ASTS, PL |
| Quantum Computing | Quantum Hardware & Software | 4 | IONQ, RGTI, QBTS |
| Nuclear & Uranium | Nuclear Reactors/SMR, Uranium Miners | 8 | SMR, OKLO, CCJ, UEC |
| eVTOL & Drones | eVTOL / Urban Air Mobility | 3 | JOBY, ACHR, RCAT |

**Benchmark:** SPY (S&P 500 ETF), used for relative strength calculations.

All defined in `ticker_config.yaml` — the single source of truth. No code changes needed to add/remove tickers.

**Ticker notes:**
- PURR is a Nasdaq-listed Digital Asset Treasury (holds HYPE/Hyperliquid), not "PURR-USD"
- WOLF (Wolfspeed) only has ~116 rows (recently relisted)
- ABB was changed to ABBNY (US ADR) because ABB returned no data from yfinance
- LILM (Lilium) was removed — company went bankrupt/delisted

---

## 3. Scoring Engine (`indicators.py`)

### 7 Scored Indicators — Max 10.0 Points

Weights were set by a **3-year conditional edge analysis** — each indicator's weight reflects its incremental predictive edge *after* controlling for Relative Strength.

**GRADIENT indicators** (proportional to signal strength):

| Indicator | Max Pts | What It Measures |
|-----------|---------|------------------|
| Relative Strength vs SPY | 3.0 | 63-day outperformance vs SPY, scored by percentile rank across the universe. Breakpoints: 90th→3.0, 80th→2.4, 70th→1.8, 60th→1.2, 50th→0.6 |
| Higher Lows | 1.0 | Consecutive weekly higher-low periods (staircase uptrend). Breakpoints: 5+→1.0, 4→0.75, 3→0.5, 2→0.25 |

**BINARY indicators** (full weight if triggered, zero otherwise):

| Indicator | Points | Trigger Condition | Incremental Edge |
|-----------|--------|-------------------|------|
| Ichimoku Cloud | 2.0 | Price above cloud AND cloud bullish (Senkou A > B) | +11.9% |
| Chaikin Money Flow | 1.5 | 20-day CMF > 0.05 (institutional buying pressure) | +8.9% |
| Rate of Change | 1.5 | 21-day ROC > 5% (strong recent momentum) | +7.5% |
| Dual-TF RS Acceleration | 0.5 | Multi-timeframe RS strong AND accelerating (126d≥70th+accelerating, or 63d+21d both≥80th) | +5.2% |
| ATR Expansion | 0.5 | 14-day ATR in top 20% of 50-day range | +5.2% |

**Dropped indicators** (still computed for dashboard display, NOT scored):
- MA Alignment: −9.3% incremental edge (harmful when RS is strong)
- Near 52w High: −3.3% incremental edge (redundant with RS)
- Dropped earlier in development: Volume Spike, BB Squeeze, MACD Crossover, ADX Trend, RSI Momentum, OBV Trend, Donchian Breakout, Consolidation Tightness

**Score thresholds:**
- **8+** → Strong breakout setup
- **6-8** → Moderate signal
- **3-6** → Weak/early
- **<3** → Minimal signals

### How `score_all()` Works (Two-Pass)

1. **Pass 1:** Compute raw RS values at 4 timeframes (21d, 63d, 126d, primary) across all tickers → percentile rank each
2. **Pass 2:** For each ticker, run all 7 indicators → apply gradient/binary scoring → sum to weighted score → sort descending

---

## 4. Subsector Breakout Detection (`subsector_breakout.py`)

### Aggregation: Ticker Scores → Subsector Metrics

For each subsector, we compute:
- **Breadth %** — fraction of tickers scoring ≥ 6 (the `breadth_threshold` in config)
- **Avg Score** — subsector mean score
- **Max Score** — highest individual score
- **Hot Count** — number of tickers above threshold

### Derived Metrics (from rolling history in SQLite)

Using 60-day rolling window:
- **Z-Score** — how unusual is today's avg score vs its recent history
- **Acceleration** — 2nd derivative of avg score (velocity of velocity)
- **Score changes** — 5d, 10d, 21d deltas for trend detection

### State Machine — 4 States

```
quiet → emerging → confirmed → fading → quiet
                      ↑          ↓
                      ←──────────┘ (recovery)
```

**Transition rules:**

| Transition | Trigger |
|-----------|---------|
| quiet → emerging | breadth ≥ 50% AND (z_score > 1.0 OR first run) AND acceleration ≥ 0 |
| emerging → confirmed | hot for ≥ 3 consecutive readings (`confirm_days: 3`) |
| emerging → quiet (failed) | cool for ≥ 2 consecutive readings |
| confirmed → fading | breadth drops below 50% OR avg score declining > 1.0 over 5 days |
| fading → quiet | breadth < 30% OR cool for ≥ 5 consecutive readings (`fade_cool_days: 5`) |
| fading → confirmed (recovery) | breadth recovers AND acceleration positive |

This is **breadth-based** (not price/MA-based like Weinstein Stage Analysis). It operates on shorter timeframes and detects thematic momentum waves. We discussed adding a 5th "Declining" state but decided it wasn't needed — "fading" already captures the decline, and a separate state would add complexity without clear value.

The `confirm_days` and `fade_cool_days` were widened from (2, 3) to (3, 5) to reduce noise and prevent rapid state cycling.

**Current states (as of 2026-03-16):**
- 2 subsectors **emerging**: Gene Editing (since Mar 3), Subsea/Ocean Robotics (since Feb 17)
- 29 subsectors **quiet**

---

## 5. Database (`breakout_tracker.db` — SQLite)

### 3 Tables

**`subsector_daily`** — One row per subsector per snapshot date
```sql
date TEXT, subsector TEXT, subsector_name TEXT, sector TEXT,
ticker_count INTEGER, avg_score REAL, max_score REAL,
breadth REAL, hot_count INTEGER, ticker_scores TEXT (JSON)
-- PRIMARY KEY (date, subsector)
-- Indexes: idx_daily_subsector (subsector, date), idx_daily_date (date)
```

**`subsector_breakout_state`** — Current state machine state per subsector
```sql
subsector TEXT PRIMARY KEY, status TEXT, status_since TEXT,
consecutive_hot INTEGER, consecutive_cool INTEGER,
peak_avg_score REAL, peak_breadth REAL, updated_at TEXT
```

**`ticker_scores`** — One row per ticker per snapshot date
```sql
date TEXT, ticker TEXT, name TEXT, subsector TEXT, sector TEXT,
score REAL, signals TEXT (JSON), signal_weights TEXT (JSON)
-- PRIMARY KEY (date, ticker)
-- Indexes: idx_ticker_scores_ticker (ticker, date), idx_ticker_scores_score (score, date)
```

**Current data:** 36 snapshots from 2025-06-27 → 2026-03-10, sampled every 5 trading days. All 139 tickers are represented (backfill was re-run on 2026-03-16 with the expanded universe).

**Important:** Raw OHLCV data from yfinance is **NOT stored** — it's fetched on demand and kept in memory only. Only scored results persist. yfinance is free, no API key needed.

The database can be browsed with **DB Browser for SQLite** (GUI, installed on Todd's Mac via Homebrew) or queried with Python/pandas.

---

## 6. File Map

### Core Pipeline
| File | Purpose |
|------|---------|
| `ticker_config.yaml` | Central config — tickers, sectors, indicator params, scoring weights, breakout detection thresholds. **Single source of truth.** |
| `config.py` | YAML parser — `load_config()`, `get_all_tickers()`, `get_tickers_by_sector()`, `get_ticker_metadata()`, `get_indicator_config()`, `get_scoring_config()` |
| `data_fetcher.py` | yfinance wrapper — `fetch_all(cfg, period)`, `fetch_ticker(ticker, period)`, `fetch_batch(tickers, period)`, `fetch_sector(cfg, sector_key)`, `data_summary()` |
| `indicators.py` | Scoring engine — `score_all(data, cfg)`, `compute_all_indicators()`, `score_ticker()`, `print_scorecard()`. Contains `INDICATOR_WEIGHTS`, `RS_GRADIENT`, `HIGHER_LOWS_GRADIENT` constants. |
| `subsector_breakout.py` | Breakout detection — `run_breakout_detection(results, cfg)`, `compute_subsector_metrics()`, `compute_derived_metrics()`, `detect_breakout_state()`, `print_breakout_summary()` |
| `subsector_store.py` | SQLite persistence — `init_db()`, `upsert_daily()`, `upsert_ticker_scores()`, `get_history()`, `get_all_history()`, `get_breakout_states()`, `update_breakout_state()`, `get_ticker_history()`, `get_high_scores()`, `get_scores_on_date()`, `cleanup_old_records()` |

### Operations
| File | Purpose |
|------|---------|
| `backfill_subsector.py` | Populate SQLite with historical data. Fetches 2yr data, walks 180 days at 5-day intervals: slice → score → compute metrics → persist → run state machine. ~10-20 sec runtime. |
| `minervini_backtest.py` | Minervini SEPA trend template backtest — checks 8 criteria. **Created but never fully run.** |
| `email_alerts.py` | Email digest via Resend API — HTML formatted. **Created but not yet configured** (needs `RESEND_API_KEY` and `ALERT_EMAIL_TO` env vars). |

### Analysis (used during development, not part of daily pipeline)
| File | Purpose |
|------|---------|
| `indicators_expanded.py` | Earlier version with 11 additional indicators (16 total). Superseded by current `indicators.py`. |
| `indicator_optimizer.py` | 3-year conditional edge analysis — tested each indicator's incremental value |
| `conditional_edge_analysis.py` | Deep dive into indicator edge by RS regime |
| `indicator_analysis.py` / `indicator_analysis_full.py` | Earlier indicator screening |
| `gradient_analysis.py` | Analyzed gradient vs binary scoring approaches |
| `first_hit_analysis.py` | Forward returns from first time a ticker hits score threshold |
| `all_occurrences_analysis.py` | Forward returns from all occurrences above threshold |
| `backtester.py` / `run_full_backtest.py` | Earlier backtesting framework |

### Presentation
| File | Purpose |
|------|---------|
| `dashboard.py` | Streamlit dashboard — main UI, 4 pages (see Section 7) |
| `architecture_diagram.html` | Visual system architecture diagram (open in browser) |

---

## 7. Dashboard (`dashboard.py`)

Run with: `streamlit run dashboard.py` (opens at localhost:8501)

### 4 Pages

1. **🏠 Dashboard** — Live scores for all 139 tickers. Fetches current data from yfinance and scores on demand. Shows: top 15 breakout signals with score badges + signal pills, score distribution bar chart, sector overview table with progress bars. Filters: min score slider, sector dropdown.

2. **🔍 Sector Drill-Down** — Select a sector → see all subsectors as expandable panels. Each shows per-ticker scores, signal pills, and a full indicator detail table. Panels auto-expand if avg score ≥ 5.

3. **📋 Full Scorecard** — Sortable/filterable table of all tickers with all indicator values. CSV download button. Ticker deep dive: select any ticker for candlestick chart (50/200 SMA overlays + volume) and full indicator breakdown with actual contributed weights.

4. **📊 Historical Charts** — Reads from SQLite (not live data):
   - **Ticker Score History** — line chart with color zones at 6 and 8 thresholds, expandable signal details table
   - **Subsector Breadth Trends** — multi-select up to 8 subsectors, shows breadth % and avg score over time with 50% trigger line
   - **Score Heatmap** — Plotly heatmap (ticker × date), filterable by "Top 20" or "By subsector", custom green colorscale
   - **Universe Score Distribution Over Time** — stacked area chart of score buckets + mean/median trend lines

Data is cached for 1 hour (`@st.cache_data(ttl=3600)`).

---

## 8. How to Run Things

```bash
cd /Users/toddbruschwein/Claude-Workspace/breakout-tracker

# Score all tickers right now (console output)
python3 -c "
from config import load_config
from data_fetcher import fetch_all
from indicators import score_all, print_scorecard
cfg = load_config()
data = fetch_all(cfg, period='1y', verbose=True)
results = score_all(data, cfg)
print_scorecard(results, min_score=6)
"

# Run breakout detection (scores + subsector state machine + persist to DB)
python3 -c "
from config import load_config
from data_fetcher import fetch_all
from indicators import score_all
from subsector_breakout import run_breakout_detection, print_breakout_summary
cfg = load_config()
data = fetch_all(cfg, period='1y', verbose=True)
results = score_all(data, cfg)
summary = run_breakout_detection(results, cfg)
print_breakout_summary(summary, cfg)
"

# Backfill historical data into SQLite (re-run after adding tickers)
python3 backfill_subsector.py                # 6 months, every 5 days (~10-20 sec)
python3 backfill_subsector.py --days 90      # 90 days only
python3 backfill_subsector.py --frequency 3  # every 3 trading days

# Launch dashboard
streamlit run dashboard.py
```

---

## 9. Key Design Decisions

1. **Regime-independent scoring** — All indicators and weights are the same regardless of market environment. The 3-year conditional edge analysis showed that RS-regime-dependent weighting actually hurt performance.

2. **Gradient vs Binary** — RS and Higher Lows use proportional scoring (stronger signal = more points). The other 5 use binary (triggered or not). This was determined by analyzing which approach produced better forward returns.

3. **Subsector breadth-based detection** — We detect breakouts at the subsector level (not individual stock). The thesis is that when 50%+ of a subsector is "hot," that's a thematic momentum wave, not just a single stock event.

4. **State machine with hysteresis** — The confirm_days (3) and fade_cool_days (5) prevent rapid cycling between states. We deliberately widened these from (2, 3) to reduce noise.

5. **yfinance for data** — Free, no API key, fetched on demand. Raw data is NOT stored. Only scored results persist in SQLite. This keeps the system simple and zero-cost.

6. **Single config file** — `ticker_config.yaml` defines everything: tickers, sectors, indicator parameters, scoring weights, breakout thresholds. No code changes needed to adjust the universe or parameters.

7. **Position-trading timeframe** — All forward return analysis uses 5/10/21/63 day windows (~1 week to 3 months). Todd is not day-trading. The system is designed for after-market-close runs, as mid-day scores are unstable.

---

## 10. Backtest Results (Summary)

From the full 16-indicator analysis (94 tickers, 2 years, 4048 observations):

**Individual indicator ranking by 63-day forward return edge:**
1. Relative Strength: +13.31% edge, 77.4% win rate
2. Ichimoku Cloud: +10.89%, 76.9% win rate
3. Higher Lows: +7.73%, 75.9% win rate
4. MA Alignment: +7.54%, 73.8% win rate (later dropped due to negative *incremental* edge)
5. Rate of Change: +6.84%, 72.6% win rate
6. Chaikin Money Flow: +6.57%, 73.2% win rate

**Optimal stack (RS + Ichimoku + Higher Lows, require 3/3):** 84.4% win rate, +37.94% avg 63-day return

**Score ≥ 8 (all occurrences):** ~2.8% avg 5-day return, ~13.1% avg 63-day return
**Score ≥ 9 (all occurrences):** Higher returns, fewer events (more selective)

**Note:** Returns are raw price returns from signal date, NOT SPY-adjusted. Win rate = % of events with positive return.

---

## 11. Pending / Future Work

- [ ] **Run Minervini backtest** (`minervini_backtest.py`) — created in prior session but never fully executed
- [ ] **SPY-adjusted returns** — backtest returns currently raw, not benchmarked against S&P 500
- [ ] **Email alerts** (`email_alerts.py`) — code exists using Resend API, needs API key configuration
- [ ] **Web deployment** — currently local only; would need cloud hosting (~$5-10/month for basic VPS)
- [ ] **Daily automation** — no scheduler set up yet; backfill and scoring are run manually
- [ ] **Cooldown mechanism** for breakout state machine — to prevent re-triggering too quickly after fading
- [ ] **Weight rebalancing** — 3yr conditional edge data changed the picture; current weights work but could be further optimized
- [ ] **Finalize scoring weights** — current weights are based on analysis but not formally "locked in" via a final review

---

## 12. Current High Scorers (as of 2026-03-16 scoring run)

| Ticker | Score | Subsector |
|--------|-------|-----------|
| KRKNF (Kraken Robotics) | 9.8 | Subsea / Ocean Robotics |
| AAOI | 9.8 | Chips — Networking/Photonics |
| CIEN | 9.0 | Chips — Networking/Photonics |
| RCAT | 8.0 | eVTOL / Urban Air Mobility |
| PL | 7.2 | Satellite Communications & Data |
| BKSY | 7.0 | Satellite Communications & Data |

**Distribution:** 6 strong (8+), 12 moderate (6-8), 48 weak (3-6), 73 minimal (<3)
**Hottest new sector:** Space & Satellite (avg 4.7)
**2 emerging subsectors:** Gene Editing, Subsea/Ocean Robotics

---

## 13. Development History (Chronological)

1. Started with 5 equally-weighted indicators, 94 tickers, 3 sectors
2. Expanded to 16 indicators and ran full backtesting analysis
3. Performed 3-year conditional edge analysis → determined optimal weights
4. Refactored to regime-independent 7-indicator weighted scoring (0-10 scale)
5. Built subsector breakout detection with state machine
6. Added SQLite persistence layer (3 tables)
7. Built backfill system for historical data population
8. Added Historical Charts page to dashboard (4 chart types)
9. Expanded universe: added 6 new sectors (Robotics, Biotech, Space, Quantum, Nuclear, eVTOL) → 139 tickers, 9 sectors, 31 subsectors
10. Re-ran backfill with expanded universe → all 139 tickers in DB

---

## 14. Config Reference (`ticker_config.yaml` — key sections)

```yaml
# Breakout detection thresholds
breakout_detection:
  breadth_threshold: 6          # score threshold for "hot" ticker
  breadth_trigger: 0.5          # 50% of subsector must be hot
  z_score_trigger: 1.0          # std devs above mean for emerging
  lookback_days: 60             # rolling window for mean/std
  confirm_days: 3               # consecutive hot readings for confirmation
  fade_cool_days: 5             # consecutive cool readings to return to quiet
  history_retention_days: 180   # SQLite retention

# Scoring config
scoring:
  max_score: 10.0
  email_threshold: 6
  indicators: [relative_strength, higher_lows, ichimoku_cloud, cmf, roc, dual_tf_rs, atr_expansion]
```
