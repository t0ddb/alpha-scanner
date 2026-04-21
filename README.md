# Alpha Scanner

**An automated paper-trading system that runs every weekday at 4:30 PM ET, scores 180 stocks with a daily momentum signal, and places limit orders on Alpaca for the next session's open — with real GTC stop orders for downside protection.**

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Broker](https://img.shields.io/badge/Broker-Alpaca_Paper-purple) ![Automation](https://img.shields.io/badge/Runtime-GitHub_Actions-black) ![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B) ![License](https://img.shields.io/badge/License-MIT-green)

[Alpha Scanner Dashboard](https://alphascanner.streamlit.app/)

---

## What it does

Every trading day at 4:30 PM ET — 30 minutes after market close — a GitHub Actions workflow runs `trade_executor.py`. That one script:

1. Pulls today's OHLC data for 180 tickers across 31 subsectors
2. Scores every ticker 0–10 using 7 momentum indicators
3. Checks which positions to exit (score dropped below 5, or stop fired)
4. Checks which new entries qualify (score ≥ 8.5 for 3 consecutive days)
5. Submits limit orders to Alpaca paper trading, sized at 8.3% of equity with a 5% cash floor
6. Attaches a real GTC stop order to each filled position at entry × 0.80
7. Emails a daily summary with P&L, positions, and next-day candidates

That's the entire product. The scoring, the subsector state machine, the dashboard — they all exist to feed or observe this one pipeline.

The underlying thesis is that individual stock return prediction is noisy, but a multi-indicator technical score aggregated across ~180 names consistently identifies ~5-15 tickers per day where momentum is concentrated. Running those signals through a disciplined execution layer (persistence filter, cash floor, limit orders, fixed stops) turns a noisy academic edge into an actually-tradeable system.

---

## The trading rules

| Rule | Value | Why |
|---|---|---|
| **Entry threshold** | score ≥ 8.5 | 3-year backtest winner; highest Sharpe on the `entry-threshold` sweep |
| **Persistence filter** | 3 prior trading days ≥ 8.5 | Filters out false-start signals (one-day score spikes that fade) |
| **Entry order type** | 3% LIMIT, DAY TIF | Caps overnight gap-up slippage at 3%; 87% fill rate in backtest |
| **Position sizing** | 8.3% of equity | = 1/12, gives full utilization at the cap |
| **Max concurrent positions** | 12 | Plateau optimum on the position-cap sweep |
| **Cash floor** | 5% of equity reserved | Guarantees total commit ≤ 95% even on simultaneous gap-ups |
| **Exit threshold** | score < 5.0 | Score-based exit; positions run indefinitely above this |
| **Stop loss** | 20% below entry (GTC) | Real order at Alpaca; fires independently of the daily cycle |
| **Pricing for sizing** | Alpaca latest-trade, yfinance fallback | Reflects extended-hours moves that drive the next open |

Every parameter above was validated by a backtest. The backtest scripts are in the repo:
- `sizing_comparison_backtest.py` — position cap + stop-loss + entry-threshold sweeps
- `entry_mode_backtest.py` — market vs limit order comparison (4 configs)

The specific results from `entry_mode_backtest.py`:

| Config | Return | Max DD | Sharpe | Sortino | Max single-trade slippage | Neg-cash days |
|---|---|---|---|---|---|---|
| Market + no cash floor (original) | +417% | -22% | 2.86 | 4.47 | +27.7% | 70 |
| Market + 5% floor | +351% | -20% | 2.93 | 4.52 | +27.7% | 0 |
| Limit-2% + 5% floor | +424% | -21% | 3.15 | 4.87 | +2.0% | 0 |
| **Limit-3% + 5% floor (live)** | **+434%** | **-21%** | **3.16** | **4.88** | **+2.6%** | **0** |

The current configuration won on risk-adjusted return, eliminated negative-cash days, and capped per-trade slippage at ~2.6%.

---

## The daily pipeline

```
 4:00 PM ET  ─ Market close
     ↓
 4:00 PM ET  ─ GitHub Actions: daily-backfill.yml
              └─ backfill_subsector.py  (historical DB rows, 180d, every 5d)
              └─ commits breakout_tracker.db
     ↓
 4:30 PM ET  ─ GitHub Actions: daily-trade-execution.yml
              └─ trade_executor.py
                  ├─ detect_filled_stops()          ← close positions that hit GTC stop
                  ├─ ensure_stops_for_positions()   ← backfill any missing stops
                  ├─ score_all()                    ← yfinance fetch + 7 indicators
                  ├─ evaluate_exits()               ← score < 5 → market sell next open
                  ├─ evaluate_entries()             ← score ≥ 8.5 + persistence → limit order
                  ├─ submit LIMIT orders            ← at sizing × 1.03, DAY TIF
                  └─ send daily email digest
              └─ commits trade_history.json, breakout_tracker.db
     ↓
 4:35 PM ET  ─ Daily email summary in inbox
     ↓
 Next 9:30 AM ET  ─ Market opens, limit orders attempt fill
```

The whole thing is idempotent. If a run fails, rerunning it the next day picks up cleanly — pending limit orders auto-cancel at close under DAY TIF, and `ensure_stops_for_positions()` reattaches any orphaned stops.

---

## How the scoring works

The 0–10 score is a weighted sum of 7 technical indicators, chosen by conditional-edge analysis (each indicator's contribution **after** controlling for the others):

| Indicator | Weight | What it measures |
|---|---|---|
| Relative Strength vs S&P 500 | 0–3.0 (gradient) | 63-day outperformance, percentile-ranked across universe |
| Ichimoku Cloud | 2.0 (binary) | Price above cloud AND cloud bullish |
| Chaikin Money Flow | 1.5 (binary) | 20-day CMF > 0.05 (institutional buying) |
| Rate of Change | 1.5 (binary) | 21-day ROC > 5% |
| Higher Lows | 0–1.0 (gradient) | Consecutive weekly higher-lows |
| Dual-Timeframe RS | 0.5 (binary) | Momentum accelerating across 21/63/126-day windows |
| ATR Expansion | 0.5 (binary) | 14-day ATR in top 20% of 50-day range |

Two indicators tested but **not** included: Moving Average Alignment (−9.3% incremental edge — redundant with RS) and Near 52-Week High (−3.3% — same story). Both are computed for dashboard display but don't count toward the score.

The score maps to a 5-tier display palette:

| Tier | Score | Meaning |
|---|---|---|
| 🔴 Fire | 9.5+ | Nearly every signal firing |
| 🟠 Hot | 8.5 – 9.5 | **Live entry threshold** |
| 🟡 Warm | 7 – 8.5 | Setup building |
| 🔵 Tepid | 5 – 7 | Watchlist only |
| 🟦 Cold | < 5 | **Live exit threshold** |

---

## Universe

180 tickers across 9 sectors and 31 subsectors. The universe was chosen to span durable growth themes (AI infrastructure, nuclear, space, crypto equities) where momentum signals have historically concentrated.

| Sector | Tickers | Subsectors |
|---|---|---|
| AI & Tech Capex | 101 | 15 — compute, memory, networking/photonics, power, data centers, AI-native clouds, hyperscalers, AI software, enterprise AI, AI security, healthcare AI, physical AI, power semis, semi equipment, semi test |
| Metals | 16 | 3 — gold/silver direct, gold miners, silver miners |
| Crypto | 11 | 2 — crypto majors (incl. GLXY), crypto/AI crossover tokens |
| Robotics & Automation | 14 | 3 — surgical, industrial, subsea/ocean |
| Space & Satellite | 11 | 2 — launch, satellite comms |
| Nuclear & Uranium | 10 | 2 — SMRs, uranium miners |
| Biotechnology | 8 | 2 — gene editing, synthetic biology |
| eVTOL & Drones | 5 | 1 — urban air mobility |
| Quantum Computing | 4 | 1 — quantum H/W+S/W |

Adding or removing tickers is YAML-only (`ticker_config.yaml`) — no code changes.

---

## Dashboard

[alphascanner.streamlit.app](https://alphascanner.streamlit.app/) — observability layer for the automated system, not a strategy control center. Three pages:

- **Tickers** — current scores in the 5-tier palette, drill-down candlestick charts, indicator breakdown per ticker
- **Subsectors** — 31 subsectors in a 7-state lifecycle (quiet → warming → emerging → confirmed → fading → revival); shows which themes are coordinated today
- **Historical Charts** — score history per ticker, subsector avg-score trends, 2-month universe heatmap, score distribution over time

The subsector state machine is an informational layer. It does **not** drive trade decisions — entries are purely score-based on the individual ticker. The subsector dashboard helps interpret *why* a cluster of tickers is scoring high on a given day (e.g. "Chips — Networking has 12/13 tickers hot → data center capex wave").

---

## Daily email digest

Lands in inbox ~4:35 PM ET each trading day. Format:

- **Subject**: `Alpha Scanner MM/DD: X.XX% | N buy / N sell`
- **P&L card**: today / all-time / vs SPY (since account inception)
- **Account card**: equity / cash / positions (x / 12)
- **Exits, Entries, Skipped Signals** — action tables (Skipped shows persistence streak, categorized reason: persistence / cash floor / position cap / wash sale / limit unfilled / other)
- **Current Positions** — sorted by P&L %
- **Exit Watch** — positions with declining score or close to stop
- **Subsectors referenced legend** — 2-letter codes expand to full names

---

## Running it

```bash
# Install
pip install -r requirements.txt

# Dry-run the executor (reads Alpaca, computes everything, places no orders)
python3 trade_executor.py --dry-run

# Preview the email HTML without sending
python3 trade_executor.py --dry-run --preview-email
open email_preview.html

# Dashboard
streamlit run dashboard.py

# Re-run any of the backtests that validated the live config
python3 sizing_comparison_backtest.py
python3 entry_mode_backtest.py

# Signal diagnostics (statistical significance of score→return correlation)
python3 signal_diagnostics_significance.py
```

### Required environment variables

Stored in `.env` (local) or GitHub Actions secrets (production):

- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` — paper trading credentials (account number must start with `PA`; enforced at runtime)
- `GMAIL_ADDRESS`, `GMAIL_APP_PASSWORD`, `ALERT_EMAIL_TO` — email digest (optional)
- `ENTRY_THRESHOLD`, `EXIT_THRESHOLD`, `PERSISTENCE_DAYS`, `STOP_LOSS_PCT`, `MAX_POSITIONS`, `MAX_POSITION_PCT` — override `ticker_config.yaml` for manual test runs

---

## Architecture

| Component | Technology | Role |
|---|---|---|
| Runtime | GitHub Actions | 2 daily workflows (backfill + trade exec), 1 quarterly review |
| Language | Python 3.9 | Single codebase; no microservices |
| Market data | yfinance + Alpaca Market Data | yfinance for batch historical, Alpaca latest-trade for pre-open pricing |
| Broker | Alpaca Paper Trading API | All orders + position management |
| Database | SQLite | Historical score rows + subsector state |
| Dashboard | Streamlit + Plotly | Deployed on Streamlit Community Cloud |
| Config | YAML | Universe + indicator params + trade execution config |
| Email | Gmail SMTP | Direct, no service dependency |

### Project structure

```
alpha-scanner/
├── trade_executor.py                  # THE main script — runs daily
├── ticker_config.yaml                 # Universe + all tunable parameters
├── indicators.py                      # 7-indicator scoring engine
├── data_fetcher.py                    # yfinance batch fetch
├── config.py                          # YAML loader
├── subsector_breakout.py              # State machine (dashboard layer)
├── subsector_store.py                 # SQLite persistence
├── trade_log.py                       # Trade audit log
├── wash_sale_tracker.py               # Log-only advisory
├── dashboard.py                       # Streamlit UI (3 pages)
├── quarterly_review.py                # Quarterly health report
├── sizing_comparison_backtest.py      # Portfolio-construction validation
├── entry_mode_backtest.py             # Market vs limit validation
├── signal_diagnostics*.py             # Universe signal-quality stats
├── transition_trim.py                 # One-time sizing migration
└── .github/workflows/                 # 3 cron-scheduled workflows
```

---

## About

Built by [Todd Bruschwein](https://linkedin.com/in/toddbruschwein). Started as a research tool for tracking the AI infrastructure investment cycle; evolved into a full automated paper-trading system. Built with Python and [Claude](https://claude.ai). Ongoing.

All trading is on Alpaca's paper trading environment — no real capital. This is a personal research / learning tool, not financial advice.
