# Alpha Scanner — Technical Documentation

**Automated momentum trading system driven by a daily score across 180 tickers.**
*April 2026*

---

## 1. What the system does

Alpha Scanner is a single script (`trade_executor.py`) that runs once per weekday via GitHub Actions, 30 minutes after US market close. In one execution it:

1. Pulls OHLC data for 180 tickers from yfinance
2. Scores every ticker 0–10 using 7 weighted technical indicators
3. Exits positions where the score has dropped below 5 (market sell at next open)
4. Cancels the GTC stop for each exited position
5. Enters positions where the score ≥ 8.5 has persisted for 3 consecutive trading days, passing wash-sale and cash-floor checks
6. Submits 3% limit orders to Alpaca for tomorrow's open
7. On the next day's run, attaches a real GTC stop at entry × 0.80 to each newly-filled position
8. Sends a daily email digest summarizing everything that happened

That is the product. Everything else in the repo — the subsector state machine, the Streamlit dashboard, the backtest framework, the signal diagnostics — either feeds this script or observes its output.

### 1.1 Thesis

Individual stock return prediction is noisy. Aggregating multiple technical indicators into a single score across a ~180-ticker universe consistently identifies 5–15 names per day where technical momentum is concentrated. But a noisy academic edge doesn't automatically turn into a tradeable system — most of the value came from the execution layer, not the score: the persistence filter that rejects one-day score spikes, the limit order that caps overnight gap-up slippage, the cash floor that prevents catastrophic over-commitment on a bad gap day, the fixed-20% stop that cuts losing trades without whipsawing on winners.

### 1.2 Backtest performance of the current config

`entry_mode_backtest.py` over a 1-year validation window (250 trading days, $100k starting equity, 10 staggered start dates for path dependency):

| Metric | Value |
|---|---|
| Total return | +434% |
| CAGR | ~+440% |
| Max drawdown | −21% |
| Sharpe ratio | 3.16 |
| Sortino ratio | 4.88 |
| Win rate | 53% |
| Max single-trade slippage | +2.6% |
| Days with negative cash | 0 |

These are backtest results against historical data. Live performance will diverge — the rest of this doc explains the system mechanics well enough to understand why.

---

## 2. The daily pipeline

```
 4:00 PM ET — market close
     ↓
 4:00 PM ET — daily-backfill.yml
              └─ backfill_subsector.py (freq=5, window=180d)
              └─ commits breakout_tracker.db
     ↓
 4:30 PM ET — daily-trade-execution.yml
              └─ trade_executor.main()
                  1. connect_alpaca()                  # paper-account check
                  2. get_account_snapshot()            # equity, cash, positions
                  3. detect_filled_stops()             # close positions whose GTC stop fired
                  4. ensure_stops_for_positions()      # backfill GTC stops on any unstopped position
                  5. detect_unfilled_limits_since()    # prior-day limits that canceled
                  6. score_all() + upsert_ticker_scores()
                  7. evaluate_exits()                  # score<5 → queue market sell
                  8. evaluate_entries()                # score≥8.5 + persistence + cash floor
                  9. execute_entries()                 # submit LIMIT BUY, DAY TIF
                  10. send_trade_digest()              # email via Gmail SMTP
              └─ commits trade_history.json, breakout_tracker.db
     ↓
 ~4:35 PM ET — email lands in inbox
     ↓
 Next 9:30 AM ET — limit orders attempt to fill at market open
     ↓
 Next 4:30 PM ET — ensure_stops_for_positions() attaches GTC stops to fresh fills
```

The entire pipeline is idempotent. If a daily run fails, the next day's run picks up cleanly: pending limit orders auto-cancel at close under DAY TIF, `ensure_stops_for_positions()` reattaches any orphaned stops, and the persistence filter keeps working because backfill fills any missing score rows.

---

## 3. Trading rules

All values live in `ticker_config.yaml` under `trade_execution:`, with env-var overrides available for any manual run.

### 3.1 Entry

A candidate ticker is entered if and only if all of the following are true:

1. **Score ≥ 8.5 today** (the Hot tier threshold; validated by `--sweep-entry-threshold`)
2. **Score ≥ 8.5 for each of the 3 prior trading days** (persistence filter — "N most recent DB rows", not strict trading days, so holidays don't block legitimate signals)
3. **Not already held** (skips tickers in current positions)
4. **Not excluded** (crypto spot tickers and futures contracts are filtered; only Alpaca-tradeable equities + ETFs considered)
5. **Sized above `min_position_size` ($500)** after applying the cash floor
6. **Tradeable on Alpaca** (the `alpaca.get_asset()` check passes)
7. **Has a valid price** (Alpaca latest-trade, falling back to yfinance close)

Candidates are sorted by score descending, so the highest-scoring signals fill first when capacity is tight.

**Wash sale advisory**: If the ticker is in a wash-sale cooldown (a prior losing exit within the last 30 days), the trade proceeds but the loss is flagged in `wash_sale_log.json` for tax tracking. Wash sale never blocks an entry — this is a personal research tool, not a tax-compliance product.

### 3.2 Exit

Two mechanisms can close a position. They are independent:

**Score-based exit (`evaluate_exits`)**:
- Triggers when today's score drops below 5.0
- Submits a market sell at tomorrow's open (DAY TIF)
- Cancels the GTC stop order first so we don't have a stray sell order sitting at Alpaca

**Stop-loss exit (Alpaca-side GTC)**:
- A real Good-Til-Canceled stop sell order is placed at entry × 0.80 (20% below entry) on every new position
- Fires independently of the daily cycle, whenever the market trades below the stop price
- `detect_filled_stops()` on the next run scans Alpaca for filled stops and logs them as sells in `trade_history.json`

### 3.3 Position sizing (with cash floor)

```python
raw_max_position  = equity × 0.083                           # 8.3% per slot
remaining_slots   = max_positions - current_position_count
floor_budget      = equity × 0.95 - projected_committed      # unused 95%-of-equity cap
per_slot_cap      = floor_budget / remaining_slots
max_position      = min(raw_max_position, per_slot_cap)
target_size       = min(projected_cash, max_position)        # also cash-bounded
qty               = floor(target_size / sizing_price)
```

The `per_slot_cap` term is what enforces the 5% cash floor **mathematically**. With 12 positions at 8.3% each = 99.6% of equity, there's essentially zero cash buffer under naive sizing — one small gap-up pushes cash negative. The floor caps total commitment at 95% of equity, guaranteeing enough slack to absorb even simultaneous gap-ups.

This is why 2026-04-16's AEHR scenario (15.5% overnight gap, $3,158 cost overrun on one trade, account cash went to −$2,626) is not reproducible under the current logic even if someone hand-disables the limit order. The cash floor is a separate, redundant safeguard.

### 3.4 Order mechanics

**Entries are LIMIT orders with DAY TIF**:
- `limit_price = sizing_price × 1.03`
- If the next open is at or below the limit → fills at the open price (or lower if the ticker opens below)
- If the next open is above the limit → order sits DAY and cancels at close
- Canceled orders don't need cleanup — DAY TIF handles it. The ticker will simply be re-evaluated next run if it still qualifies.

**Exits are MARKET orders with DAY TIF** (for score-based exits). Stop losses are GTC, attached separately.

**Sizing price source**: `get_alpaca_latest_price()` first (Alpaca's `StockLatestTradeRequest`), which reflects extended-hours trading. Falls back to yfinance close (`_estimated_price()`) if Alpaca data is unavailable. At 4:30 PM ET, Alpaca's latest-trade may already reflect after-hours news that will drive the next open — yfinance only has the regular-session close.

### 3.5 Stop-order attachment sequence

Because entries are limit orders submitted at 4:30 PM today and they fill (or don't) at 9:30 AM tomorrow, stops get attached on a 1-day lag:

- **Day T, 4:30 PM**: `execute_entries()` submits LIMIT BUY for ticker XYZ
- **Day T+1, 9:30 AM**: market opens, limit either fills or doesn't
- **Day T+1, 4:30 PM**: `ensure_stops_for_positions()` sees XYZ in positions, sees no GTC stop attached, places one at `entry_price × 0.80`

During the ~24-hour gap between fill and stop attachment, the position is technically unstopped. Acceptable because:
1. The fill is bounded — buy limit at `sizing × 1.03`, so worst case you bought at 3% above sizing reference
2. Regular market-session risk is small vs overnight risk
3. Complete protection during the ~16-hour overnight/pre-market window, which is when 99% of gap risk occurs

### 3.6 Skip categories (for email digest)

When `evaluate_entries()` rejects a candidate, the reason is categorized into one of:

| Category | Meaning |
|---|---|
| `persistence` | Score ≥ 8.5 today but not for all 3 prior days |
| `cash floor cap` | Would exceed 95% equity commit budget |
| `position cap reached` | At max_positions (12) |
| `insufficient cash` | Raw cash or min-size fail |
| `wash sale cooldown` | Advisory log (never actually blocks — see 3.1) |
| `limit unfilled` | Prior-day limit canceled and score has since dropped |
| `other` | Not tradeable, no price, etc. |

---

## 4. Signal generation (scoring)

The 0–10 score is a weighted sum of 7 technical indicators. Weights were determined by 3-year conditional-edge analysis — each indicator's incremental predictive edge **after controlling for the others**.

### 4.1 Scored indicators

| Indicator | Weight | Type | Incremental edge | Logic |
|---|---|---|---|---|
| Relative Strength | 0–3.0 | Gradient | +13.31% | 63-day return vs SPY, percentile-ranked across universe. 90th→3.0, 80th→2.4, 70th→1.8, 60th→1.2, 50th→0.6 |
| Ichimoku Cloud | 2.0 | Binary | +11.9% | Price > cloud AND Senkou A > Senkou B |
| Chaikin Money Flow | 1.5 | Binary | +8.9% | 20-day CMF > 0.05 |
| Rate of Change | 1.5 | Binary | +7.5% | 21-day ROC > 5% |
| Higher Lows | 0–1.0 | Gradient | +7.73% | 0.25 per consecutive higher-low period (4 periods = 1.0) |
| Dual-Timeframe RS | 0.5 | Binary | +5.2% | RS strong AND accelerating across 21/63/126-day windows |
| ATR Expansion | 0.5 | Binary | +5.2% | 14-day ATR in top 20% of 50-day range |

Maximum score: 10.0.

### 4.2 Not scored (but computed for dashboard display)

- **Moving Average Alignment** — standalone edge +7.5%, but **incremental edge −9.3%** after controlling for RS. Almost fully redundant with relative strength. Adding it actively degrades the signal. (Note: the 2026-Q2 quarterly review showed a 12-month window where MA Alignment's edge flipped to +7.76% — watch for re-inclusion if 2026-Q3 also shows positive.)
- **Near 52-Week High** — incremental edge −3.3%. Tickers with high RS are almost always near their 52w high; this indicator adds no additional information.

Both are displayed in the dashboard's ticker detail view for visual context but don't count toward the score.

### 4.3 Scoring process (two-pass)

**Pass 1**: For every ticker, compute raw relative-strength values at 4 timeframes (21d, 63d, 126d, primary) in parallel. Then percentile-rank each timeframe across the universe.

**Pass 2**: For each ticker, run all 7 indicators, apply gradient/binary scoring, sum to the weighted score, sort results by score descending.

### 4.4 Display tiers

Dashboard and email use the same 5-tier bucketing:

| Tier | Range | Dashboard color | Action |
|---|---|---|---|
| 🔴 Fire | 9.5+ | `#E15759` | — |
| 🟠 Hot | 8.5 – 9.5 | `#F28E2B` | **Entry threshold** |
| 🟡 Warm | 7 – 8.5 | `#F1CE63` | — |
| 🔵 Tepid | 5 – 7 | `#A0CBE8` | — |
| 🟦 Cold | < 5 | `#4E79A7` | **Exit threshold** |

Dashboard palette is Tableau 20 heat colors. The email digest uses a semantic green/amber/orange/red palette for P&L signals, intentionally different.

---

## 5. The subsector layer (observability only)

`subsector_breakout.py` implements a state machine that aggregates individual ticker scores into subsector-level breadth metrics, then classifies each of the 31 subsectors into one of 7 states:

```
quiet → warming → emerging → confirmed → fading → quiet
                               ↑           ↓
                               └───────────┘  (revival)
```

| State | Definition |
|---|---|
| Quiet | Breadth < 25% |
| Warming | Breadth 25–49% |
| Emerging | Breadth ≥ 50% + z-score > 1.0 + accel ≥ 0 |
| Confirmed | 3+ consecutive hot readings |
| Steady Hot | Breadth ≥ 50% but z-score too low for Emerging |
| Fading | Breadth declining from Confirmed |
| Revival | Fading + breadth recovers + accel > 0 |

**The subsector state machine does not drive trades.** Entries are purely score-based on the individual ticker. The state machine is a Streamlit dashboard layer — useful for humans to understand why a cluster of tickers is scoring high on a given day ("Chips — Networking is Confirmed → the data center capex wave is active"), but it never gates an entry.

If you ever want to wire it into execution (e.g., "only enter if subsector is Confirmed"), that's a material strategy change requiring its own backtest validation. The current live-trade record is under score-only entry logic.

---

## 6. Universe

180 tickers across 9 sectors and 31 subsectors. The universe targets durable growth themes where multi-indicator momentum signals have historically concentrated.

| Sector | Tickers | Subsectors |
|---|---|---|
| AI & Tech Capex Cycle | 101 | 15 (chips, data center, AI clouds, hyperscalers, AI software, semi equipment/test) |
| Metals | 16 | 3 |
| Crypto | 11 | 2 |
| Robotics & Automation | 14 | 3 |
| Space & Satellite | 11 | 2 |
| Nuclear & Uranium | 10 | 2 |
| Biotechnology | 8 | 2 |
| eVTOL & Drones | 5 | 1 |
| Quantum Computing | 4 | 1 |

Universe is defined entirely in `ticker_config.yaml`. Adding a ticker is YAML-only — no code changes. Recent April 2026 additions: BE, CRWV, APLD, BTDR, RIOT, CLSK, WYFI, KEEL, SNDK, EQT, SEI, LBRT, PUMP, PSIX, BW, GLXY (+15).

If you add a new subsector, also add a 2-letter code to `SUBSECTOR_CODES` in `trade_executor.py` for the email-display column.

---

## 7. Backtest validation

Every parameter in section 3 was validated by a backtest. Scripts are in the repo and can be re-run any time.

### 7.1 `sizing_comparison_backtest.py`

Compares 4 portfolio-construction strategies over 2 years:

| Strategy | Return | Max DD | Sharpe | Trades |
|---|---|---|---|---|
| Fixed/5-Max (20% × 5) | +149% | −29% | 1.34 | 36 |
| Fixed/10-Max (10% × 10) | +459% | −23% | 2.07 | 65 |
| Dynamic/Trim | +206% | −21% | 1.57 | 157 |
| Fixed/10+Swap | +229% | −30% | 1.54 | 134 |

Fixed/N-Max wins. Position-cap sweep (3–20) shows a plateau between 10–12; current config uses 12 × 8.3% to maximize deployment flexibility without sacrificing risk-adjusted return.

**Entry-threshold sweep** on the corrected simulator:

| Threshold | Return | Sharpe |
|---|---|---|
| 7.5 | +356% | 1.84 |
| 8.0 | +344% | 1.91 |
| **8.5** | **+428%** | **1.98** |
| 9.0 | +425% | 1.96 |
| 9.5 | +264% | 1.66 |

**Stop-loss sweep** (13 configs: fixed 10/15/20%, trailing, ATR-based): fixed 20% eliminates whipsaws (stop-outs followed by recovery to above entry) without meaningful return reduction. Trailing stops in particular had catastrophic whipsaw rates (60–82%).

### 7.2 `entry_mode_backtest.py`

Head-to-head of 4 entry execution modes:

| Config | Return | Sharpe | Max Slip | Neg-cash days |
|---|---|---|---|---|
| Market, no floor (original) | +417% | 2.86 | +27.7% | 70 |
| Market + 5% floor | +351% | 2.93 | +27.7% | 0 |
| Limit-2% + 5% floor | +424% | 3.15 | +2.0% | 0 |
| **Limit-3% + 5% floor** | **+434%** | **3.16** | **+2.6%** | **0** |

Limit-3% won on every metric except max slippage (Limit-2% had slightly tighter). The path-dependency test across 10 staggered start dates revealed Limit-2%'s weakness: one start window returned only +229% because 6 gap-up momentum winners (LITE, TSEM, PL, HUT, GSAT, EXK) were filtered by the tight 2% limit. Limit-3% caught them. 3% was the minimum limit buffer that consistently preserved breakout-momentum capture.

---

## 8. Universe-wide signal diagnostics

Three scripts characterize whether the score actually predicts returns, with statistical rigor.

### 8.1 Methodology

- **`signal_diagnostics.py`** — bucket tables: forward return by score bucket at 7/21/63d horizons across raw + sm3/5/10/20 smoothing. Spearman rank correlations. Score autocorrelation at lags 1/3/5/10/20.
- **`signal_diagnostics_subsector.py`** — same analysis stratified by each of the 31 subsectors. Ranked summary.
- **`signal_diagnostics_significance.py`** — bootstrap 95% CIs on Spearman ρ (1000 iterations, observation-level resample) with significance classification.

### 8.2 Findings

**Smoothing improves rank correlation at every horizon.** At h=63d, raw ρ = +0.277, sm20 ρ = +0.353. Score autocorrelation is ρ = +0.462 at lag 10, ρ = +0.173 at lag 20 — the score has ~10-day memory, so sm10 smoothing captures most of the persistent signal without over-smoothing.

**26 of 31 subsectors have statistically significant signal** at 95% CI: 15 positive, 11 negative, 5 inconclusive. Strongest positive: **Chips — Networking/Photonics** (ρ = +0.264, CI = [+0.219, +0.306], N = 1,824 — tightest CI and largest N, highest-confidence positive finding). Strongest negative: **Industrial Robotics & Automation** (ρ = −0.507, CI = [−0.566, −0.446]).

### 8.3 Caveats

The observation-level bootstrap overstates precision — it assumes independence of (ticker, date) rows, but score is autocorrelated and overlapping forward returns induce cross-observation dependence. Marginal findings (|ρ| < 0.15) should be treated with extra skepticism. Large effects (|ρ| > 0.25) are robust even after mental correction.

### 8.4 Status

**Findings documented but not acted on.** The universe has not been pruned or reweighted. No sector-conditional sizing. No signal-weighted position sizing. These diagnostics exist as reference for future strategy decisions — intentionally held while the current config runs live.

---

## 9. Architecture

### 9.1 Tech stack

| Component | Technology |
|---|---|
| Runtime | GitHub Actions (3 scheduled workflows) |
| Language | Python 3.9.6 |
| Broker | Alpaca Trading API (`alpaca-py` SDK) |
| Market data | yfinance (historical batch) + Alpaca `StockHistoricalDataClient` (latest-trade) |
| Database | SQLite (local, committed to repo) |
| Dashboard | Streamlit + Plotly, deployed on Streamlit Community Cloud |
| Email | Gmail SMTP (direct) |
| Config | YAML |

### 9.2 Database schema (SQLite)

**`ticker_scores`** — one row per ticker per snapshot date
- `date`, `ticker`, `name`, `subsector`, `sector`, `score`, `signals` (JSON), `signal_weights` (JSON)

**`subsector_daily`** — one row per subsector per snapshot date
- `date`, `subsector`, `subsector_name`, `sector`, `ticker_count`, `avg_score`, `max_score`, `breadth`, `hot_count`, `ticker_scores` (JSON)

**`subsector_breakout_state`** — current state per subsector
- `subsector` (PK), `status`, `status_since`, `consecutive_hot`, `consecutive_cool`, `peak_avg_score`, `peak_breadth`, `updated_at`

### 9.3 Trade audit log (`trade_history.json`)

Append-only JSON with three entry types (`side` field):

- `"buy"` — ticker, date, sizing_price, qty, cost_basis, score_at_entry, Alpaca order_id, reason
- `"sell"` — ticker, date, fill_price, qty, proceeds, pnl, pnl_pct, score_at_entry (auto-carried from matching buy), score_at_exit, reason, hold_days, Alpaca order_id
- `"trim"` with `action: "transition_trim"` — reserved for one-time migration operations (used once on 2026-04-20 for the sizing migration; not a recurring schema)

`trade_log.bucket_stats()` groups realized P&L by entry-score bucket for validating per-tier performance vs backtest expectations.

### 9.4 GitHub Actions workflows

| Workflow | Cron (UTC, DST) | Purpose |
|---|---|---|
| `daily-backfill.yml` | `0 20 * * 1-5` (4:00 PM ET) | Backfill any missing historical DB rows |
| `daily-trade-execution.yml` | `30 20 * * 1-5` (4:30 PM ET) | The main pipeline (`trade_executor.py`) |
| `quarterly-review.yml` | `0 14 1 1,4,7,10 *` | 7-section system health report |

Daily crons are staggered so backfill commits the DB before trade-exec checks out the repo — no merge contention. Both daily crons need `+1 hour` when US returns to Standard Time (~Nov 2026).

### 9.5 File map

**Core pipeline:**
- `trade_executor.py` — the daily script (pipeline, order helpers, email builder)
- `ticker_config.yaml` — universe + all tunable parameters
- `indicators.py` — scoring engine
- `data_fetcher.py` — yfinance wrapper
- `config.py` — YAML loader
- `subsector_breakout.py` — state machine (dashboard only)
- `subsector_store.py` — SQLite persistence
- `trade_log.py` — trade audit log
- `wash_sale_tracker.py` — advisory logs

**Dashboard + review:**
- `dashboard.py` — Streamlit, 3 pages
- `quarterly_review.py` — quarterly health report

**Backtesting:**
- `sizing_comparison_backtest.py` — portfolio construction + sweep flags
- `entry_mode_backtest.py` — market vs limit comparison
- `backtester.py` — shared primitives
- `portfolio_backtest.py` — legacy (superseded)

**Diagnostics:**
- `signal_diagnostics.py`
- `signal_diagnostics_subsector.py`
- `signal_diagnostics_significance.py`

**One-time / utility:**
- `backfill_subsector.py` — historical DB fill (also runs daily from CI)
- `backfill_stop_orders.py` — one-time stop attachment (now covered by `ensure_stops_for_positions()`)
- `transition_trim.py` — one-time sizing migration (executed 2026-04-20)

---

## 10. Key design decisions

1. **Automated execution as the primary product.** Scoring is a subsystem; the pipeline is the product. Every design choice optimizes for "what does the daily execution do?" not "what's the best individual-stock prediction model?"

2. **Subsector layer is observability, not strategy.** Entries are score-only on the individual ticker. The state machine exists for human interpretability (dashboard), not to gate trades. Any attempt to wire subsector state into execution is a material strategy change requiring its own backtest.

3. **Real stops, not software-simulated.** GTC stop orders live at Alpaca. If the dashboard, executor, or network is down during a crash, the stops still fire.

4. **Cash floor is a hard mathematical guarantee, not a guideline.** The sizing formula provably caps total commitment at ≤ 95% of equity. No runtime check needed — the math does it.

5. **Limit orders over market orders.** Sacrifices ~13% of signals to cap slippage at 3%. Of those missed, 83% re-qualify within 5 trading days. The few big gap-up breakouts that escape the filter (AXTI 2026-01-27 type events) are an acceptable cost vs the AEHR-class blow-up scenario.

6. **Alpaca-first pricing for sizing.** yfinance close is stale by 4:30 PM ET. Alpaca's latest reflects extended-hours moves that drive the next open. Using the more-current price eliminates a class of sizing surprises.

7. **Conditional-edge indicator selection.** Indicators are evaluated for incremental edge after controlling for RS, not standalone edge. Otherwise you end up with redundant signals (MA Alignment, 52w High) that degrade combined accuracy.

8. **Regime-independent scoring.** Same weights across all market environments. RS-regime-dependent weighting was tested and hurt out-of-sample performance.

9. **No sector multipliers.** 5 tests of sector-weighted scoring all overfit to in-sample and degraded out-of-sample. Equal weighting across sectors.

10. **Documented but un-acted-on diagnostics.** Universe-level signal analysis identified 26 subsectors with statistically significant signal direction (15 positive, 11 negative). These are documented for future reference. No pruning, reweighting, or signal-conditional sizing has been shipped — deliberate hold.

11. **Three-layer commit sequencing** (backfill → trade-exec → dashboard rerender) plus DAY TIF limit orders plus idempotent stop-backfill means the system self-heals from any single-day failure. A missed cron doesn't leave orphaned state.

12. **Account-type safety check at the API boundary.** `connect_alpaca()` verifies the account number prefix on every connection — currently enforcing `PA` (paper) during the validation phase. When switching to live, this check and the `paper=True` flag on the `TradingClient` must be updated in coordinated fashion. Bypassing either on its own is a deliberate act, not a confused mistake.

---

*Built with Claude Code · April 2026*
