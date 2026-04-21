# Alpha Scanner — Session Handoff

Last updated: 2026-04-21

This doc exists so a fresh Claude Code thread can pick up where the
last one left off. Read it first, then read the code. Anything a
careful code reader could figure out is **not** here — only
non-obvious state, decisions, and reasoning.

---

## What Alpha Scanner is (1 paragraph)

**Alpha Scanner is an automated momentum trading system.** Every
trading day at 4:30 PM ET (30 min after market close), a GitHub
Actions workflow runs `trade_executor.py`, which scores 180 tickers
with a 7-indicator momentum signal, evaluates exits on held
positions, evaluates entries on candidates that pass a 3-day
persistence filter, and places 3% limit orders on Alpaca for the
next session's open. Real GTC stop orders at Alpaca protect each
position at entry × 0.80. A daily email summary lands in inbox
~5 min after the run. Everything else in this repo — the subsector
state machine, the Streamlit dashboard, the backtest scripts, the
signal diagnostics — exists to feed or observe or validate this one
pipeline.

The system is designed for real-capital trading. It currently runs
against an Alpaca paper-trading account for live-market validation;
the live-capital switchover is the next planned step.

Repo: `/Users/toddbruschwein/Claude-Workspace/breakout-tracker`
GitHub: `github.com/t0ddb/alpha-scanner`
Dashboard: `alphascanner.streamlit.app`

---

## Current live config (deployed 2026-04-17)

All values below are in `ticker_config.yaml` under `trade_execution:`.
Env vars override YAML (precedence: env > yaml > hardcoded default in
`trade_executor.py:_DEFAULTS`).

| Parameter | Value | Source |
|---|---|---|
| Entry threshold | **8.5** | `--sweep-entry-threshold` optimum (Sharpe 1.98) |
| Persistence | **3 prior days ≥ 8.5** | 3yr backtest + Pareto vs 9.5 baseline |
| Exit threshold | **< 5.0** | Fixed since project start |
| Stop loss | **20%** | `--sweep-stop-loss` winner (fixed beats trailing) |
| Max positions | **12** | `--sweep-max-positions` plateau optimum |
| Position size | **8.3% of equity** | = 1/12 per slot |
| **Cash floor** | **5% of equity reserved** | `entry_mode_backtest.py` validated |
| **Entry order** | **3% LIMIT @ DAY TIF** | `entry_mode_backtest.py` — Limit-3% won |
| **Sizing price** | **Alpaca latest-trade** (yfinance fallback) | Post-AEHR-gap fix |
| Min position | $500 | Never hit in practice |

**Constants not in YAML** (hardcoded in `trade_executor.py`, too
tightly validated to expose):
- `CASH_FLOOR_PCT = 0.05`
- `LIMIT_ORDER_BUFFER = 0.03`

---

## Daily execution pipeline (what happens in the 4:30 PM ET run)

Inside `trade_executor.main()`:

```
1. connect_alpaca()                  # paper-account check (acct # starts with "PA")
2. get_account_snapshot()            # equity, cash, last_equity, all positions
3. detect_filled_stops()             # find GTC stops that fired → log as sells
4. ensure_stops_for_positions()      # idempotent: backfill GTC stops for any held
                                       position lacking one (covers limit orders
                                       that filled overnight)
5. detect_unfilled_limits_since()    # prior-day limit orders that canceled (for email)
6. score_all() + upsert_ticker_scores()   # compute + persist today's scores
7. evaluate_exits()                  # positions where score < 5 → market sell next open
8. evaluate_entries()                # candidates: score ≥ 8.5 + persistence + cash floor
9. execute_entries()                 # submit LIMIT BUY at sizing_price × 1.03, DAY TIF
10. send_trade_digest()              # email summary via Gmail SMTP
```

**Key sequencing note**: limit orders submitted at 4:30 PM ET today
fill (or don't) at the NEXT morning's open. When they fill, the
position appears in Alpaca at 9:30 AM ET. Tomorrow's `trade_executor`
run at 4:30 PM ET is the first one that sees the position — that's
when `ensure_stops_for_positions()` attaches the GTC stop. So for
~24 hours after a limit fills, the position is technically
unstopped. Acceptable because the fill is bounded to ≤ sizing × 1.03
(max 3% above sizing), and Alpaca's market-session risk is small
relative to overnight risk.

---

## GitHub Actions workflows

Three cron-scheduled workflows. All three commit state back to the
repo on completion.

| Workflow | Cron (UTC, DST) | What it does |
|---|---|---|
| `daily-backfill.yml` | `0 20 * * 1-5` (4:00 PM ET) | `backfill_subsector.py` fills any missing historical DB rows |
| `daily-trade-execution.yml` | `30 20 * * 1-5` (4:30 PM ET) | `trade_executor.py` scores + trades |
| `quarterly-review.yml` | `0 14 1 1,4,7,10 *` | `quarterly_review.py --months 12` writes 7-section health report |

**⚠️ DST note**: GitHub Actions cron is UTC-only. Current UTC values
are tuned for US Eastern DAYLIGHT TIME (March–November). When the US
returns to Standard Time (~Nov 2026), both daily crons need to shift
+1 hour:
- backfill: `0 21 * * 1-5`
- trade-exec: `30 21 * * 1-5`

YAML comments in both workflow files document this.

**Crons are staggered** (backfill 4:00, trade-exec 4:30) so backfill's
DB commit lands before trade-exec checks out the repo. Eliminated the
original race condition that needed a `-X ours` merge strategy.

**Manual override**: `workflow_dispatch` on `daily-trade-execution.yml`
exposes `entry_threshold` and `persistence_days` as inputs → env vars
that `load_trade_config()` picks up.

---

## Key non-obvious decisions

### Persistence filter uses N most recent DB rows, not strict trading days

`trade_executor.check_persistence()`:
```sql
SELECT date, score FROM ticker_scores
WHERE ticker = ? AND date < ?
ORDER BY date DESC LIMIT ?
```

**Intentional.** If the DB has a gap (holiday, CI glitch, DB rebase
conflict), we'd rather let a legit signal through than block it. Good
Friday 2026 (4/3) would have broken a strict-trading-day interpretation
for any VIAV-like ticker that qualified on 4/7. User confirmed this
design; don't tighten it.

### Entries use live-scored "today" + DB rows for "prior N"

Inside the same main() call:
1. `score_all()` computes today's scores in memory
2. `upsert_ticker_scores(db, today, results)` writes them to DB
3. `evaluate_entries` queries the DB for the prior N days per candidate

Today's live scores become "prior" data for tomorrow's run. This is
why the workflow must commit the DB at the end — otherwise the next
run's persistence check would have a missing row.

### The subsector state machine does NOT drive trades

Entries are purely score-based on the individual ticker. The
subsector state machine (`subsector_breakout.py`) is an observability
layer for the Streamlit dashboard. It tracks when a cluster of
tickers in the same subsector all score high simultaneously
(quiet → warming → emerging → confirmed → fading → revival), which is
useful context for humans but never gates an entry. If you ever want
to wire it into execution (e.g., "only enter if subsector is
Confirmed"), that would be a material strategy change requiring its
own backtest.

### The email digest palette is intentionally NOT the dashboard palette

Dashboard uses a 5-tier **heat** palette (Tableau 20:
`#E15759`/`#F28E2B`/`#F1CE63`/`#A0CBE8`/`#4E79A7`) for the
Fire/Hot/Warm/Tepid/Cold tiers.

Email digest uses a **semantic** palette (green/amber/orange/red —
"good/bad" scale) for P&L-style coloring.

Do NOT unify them. The dashboard tells you where signals are; the
email tells you what's winning/losing.

### Limit orders over market orders — why

Market orders at next-open fill at whatever price opens, gap or not.
99th-percentile overnight gap in the 1y backtest window is +8.5%. One
AEHR trade gapped +15.5% overnight and turned an estimated $20.4k
entry into an actual $23,513 cost — a $3,158 overrun on a single
trade that pushed account cash to -$2,626.

`entry_mode_backtest.py` compared 4 configs. Limit-3% + 5% floor
produced the best Sharpe (3.16), Sortino (4.88), and zero negative-cash
days (vs 70 for the market-order baseline). 87% of signals still fill;
83% of missed signals re-qualify within 5 trading days.

### Zero-score rows are valid output, not data failures

A row like `NVDA 2026-04-06 score=0.0 signals=[]` means the scorer
ran but no indicator fired. It is NOT a fetch failure — fetch
failures don't write rows at all. A rising share of zero-score rows
across the universe is a **breadth signal** (universe-wide
deterioration), not a data bug.

---

## Current live state (2026-04-21)

**trade_history.json:**
- 5 original buys (2026-04-10 → 04-15) under OLD 20%-of-equity sizing
- 5 transition trims executed 2026-04-20 via `transition_trim.py`
  (one-time migration to new 8.33% basis — AEHR/FORM/IRDM/VIAV/WDC
  all trimmed, stops preserved at original entry × 0.80)
- Any entries that filled overnight from tonight's automated run
  will appear in tomorrow's snapshot

**Account state after transition trim:**
- 5 positions at new 8.33% × (1 + P&L) sizing totaling ~$37k
- ~$49k cash available (after trim proceeds)
- 7 open position slots out of 12
- All stops in place at original entry × 0.80

---

## Universe (180 tickers, 31 subsectors)

Committed to `ticker_config.yaml`. Summary:

- **AI & Tech Capex** — 101 tickers, 15 subsectors (semis, data center, AI-native clouds, hyperscalers, software)
- **Metals** — 16 tickers, 3 subsectors (gold/silver direct, miners)
- **Crypto** — 11 tickers, 2 subsectors (majors incl. GLXY equity, crossover tokens)
- **Robotics** — 14 tickers, 3 subsectors
- **Space & Satellite** — 11 tickers, 2 subsectors
- **Nuclear & Uranium** — 10 tickers, 2 subsectors
- **Biotech** — 8 tickers, 2 subsectors
- **eVTOL** — 5 tickers, 1 subsector
- **Quantum** — 4 tickers, 1 subsector

Recent additions (April 2026): BE, CRWV, APLD, BTDR, RIOT, CLSK,
WYFI, KEEL, SNDK, EQT, SEI, LBRT, PUMP, PSIX, BW, GLXY.

Adding a ticker: edit `ticker_config.yaml`, commit. No code change.

If you add a new subsector, also add its 2-letter code to
`SUBSECTOR_CODES` in `trade_executor.py` (email display uses it). The
code `_subsector_code()` falls back to full name if a code is
missing, so forgetting won't crash — just looks ugly.

---

## Universe-wide signal diagnostics (2026-04-17/18, documented not acted on)

Three scripts, run once to characterize signal quality:

- `signal_diagnostics.py` — aggregate forward return by score bucket
  at 7/21/63d horizons across raw + sm3/5/10/20 smoothing. **Finding**:
  smoothing consistently improves rank correlation (sm20 ρ=+0.353 vs
  raw ρ=+0.277 at h=63d).
- `signal_diagnostics_subsector.py` — per-subsector decomposition.
  **Finding**: the AI/Tech aggregate result is driven by ~4
  subsectors (Chips — Networking/Photonics, Memory, Alt AI Compute,
  Hyperscalers). "Other" aggregate is not uniform — Nuclear Reactors
  (ρ=+0.487) is the single strongest subsector.
- `signal_diagnostics_significance.py` — bootstrap 95% CIs.
  **Finding**: 26 of 31 subsectors statistically significant (15
  positive, 11 negative, 5 inconclusive). Chips — Networking/Photonics
  has the highest-confidence positive signal (tight CI, N=1824,
  ρ=+0.264). Industrial Robotics is the strongest negative (ρ=−0.507).

**No strategy changes made based on these findings.** The universe
has not been pruned or reweighted. Signal-conditional sizing has
not been implemented. User has the data for future decisions but
intentionally held.

---

## Open threads / next work

Priority roughly top-down:

1. **Monitor the new limit-order + cash-floor regime.** First few
   weeks of entries will tell us if fill rates and slippage match the
   `entry_mode_backtest.py` predictions (87% fill, ≤3% slippage). If
   reality diverges meaningfully (e.g. fill rate < 70%, slippage > 5%
   regularly), that's a signal the market is behaving differently
   than the backtest window and needs re-tuning.

2. **Reconcile actual fill prices back into `trade_history.json`.**
   `log_buy()` currently records `price = sizing_price` (Alpaca
   latest at submit time), not the actual fill price. For accurate
   per-position cost basis, a next-day reconciliation step could look
   up `filled_avg_price` from Alpaca's order history and update the
   trade log entry. Not shipped.

3. **MA Alignment re-inclusion watch.** Its 12-month edge flipped to
   +7.76% in the 2026-Q2 quarterly review (was −9.3% in the 3yr
   window). Do NOT re-add yet — wait for 2026-Q3 review to confirm
   it's not a one-window fluke. If Q3 also shows positive edge, add
   back at weight ~1.5 and rerun `--sweep-entry-threshold`.

4. **Daily-backfill frequency.** Currently `frequency=5` (samples
   every 5 trading days, 180-day window). The persistence filter
   works fine because the executor fills in the per-day rows, but
   `frequency=1` would give a robust fallback if trade-exec commits
   fail. Tradeoff: runtime goes 10-20 min → ~50+ min. CI budget
   permitting.

5. **Quarterly review Section 4 completion.** Needs live sells to be
   meaningful. Wire up bucket-grouped realized P&L comparison vs
   backtest expectations once 20+ sells accumulate.

6. **DST cron shift**. Both daily workflows need `+1` hour around
   Nov 2, 2026. YAML comments document the replacement values.

---

## Files to know

### THE one script

- `trade_executor.py` — the daily pipeline. 5:30 PM ET entry point.
  Contains: connection helpers, order helpers (market/limit/stop),
  `evaluate_exits()`, `evaluate_entries()` with cash-floor sizing +
  Alpaca-first pricing, `execute_entries()` (submits LIMIT orders),
  email digest builder, skip-reason categorizer, SUBSECTOR_CODES map.

### Pipeline components

- `indicators.py` — 7-indicator scoring engine + `score_all()` +
  `score_ticker()`. Gradient + binary weighting.
- `config.py` — YAML loader
- `data_fetcher.py` — yfinance batch fetch
- `subsector_store.py` — SQLite CRUD (3 tables: `subsector_daily`,
  `subsector_breakout_state`, `ticker_scores`)
- `subsector_breakout.py` — state machine (dashboard-only — does not
  gate trades)
- `trade_log.py` — JSON audit log. `log_buy` / `log_sell` /
  `log_trim` (transition_trim records). Auto-carries
  `score_at_entry` onto sells for bucket analysis.
- `wash_sale_tracker.py` — logs loss-exit cooldowns. Never blocks.

### Dashboard + review

- `dashboard.py` — Streamlit, 3 pages, 5-tier Tableau 20 palette
- `quarterly_review.py` — 7-section quarterly health report

### Backtesting (validates live config)

- `sizing_comparison_backtest.py` — 4-strategy comparison + sweep
  flags (`--sweep-max-positions`, `--sweep-stop-loss`,
  `--sweep-entry-threshold`)
- `entry_mode_backtest.py` — 4-way market vs limit comparison (the
  backtest that chose Limit-3% + 5% floor)
- `backtester.py` — shared backtest primitives
- `portfolio_backtest.py` — legacy, superseded by
  `sizing_comparison_backtest.py`; kept for reference

### Diagnostics

- `signal_diagnostics.py`
- `signal_diagnostics_subsector.py`
- `signal_diagnostics_significance.py`

### Utility / one-time

- `backfill_subsector.py` — historical DB fill (runs daily from CI)
- `backfill_stop_orders.py` — one-time GTC stop attachment for any
  position without one (ran once in April; now idempotently covered
  by `ensure_stops_for_positions()` in the daily pipeline)
- `transition_trim.py` — one-time sizing migration (ran 2026-04-20)

### Config + CI

- `ticker_config.yaml` — 180 tickers, indicator weights,
  `trade_execution:`, `breakout_detection:`, `scoring:`
- `.github/workflows/daily-backfill.yml` (20:00 UTC M-F)
- `.github/workflows/daily-trade-execution.yml` (20:30 UTC M-F)
- `.github/workflows/quarterly-review.yml` (14:00 UTC quarterly)

### Outputs (committed)

- `breakout_tracker.db` — SQLite
- `trade_history.json` — trade audit log
- `wash_sale_log.json`
- `quarterly_reviews/review_YYYY_QN.txt` + `review_history.json`

### Docs

- `README.md` — public overview
- `HANDOFF.md` — **this file**
- `ALPHA_SCANNER_DOCUMENTATION.md` — technical deep dive

---

## Gotchas

- **`fetch_all(cfg, period=...)`**, not `fetch_all(tickers, ...)`. Not
  the other way round. Easy to get wrong.
- **Python 3.9 compatibility**: `from __future__ import annotations`
  and `str | None` still needs the future import.
- **Score 0 with empty signals** is valid, not a bug.
- **Max score is 10.0**, not 13.0. An old plan proposed 13-point; never
  adopted.
- **MA Alignment and Near 52w High** are computed but NOT scored.
- **`check_persistence` needs a `db_conn` param** — pass it through
  from `main()`, don't create a new connection.
- **Alpaca account-type safety check.** `connect_alpaca()` currently
  enforces that the account number starts with `PA` (paper), and
  `paper=True` is passed to `TradingClient`. This protects the
  validation phase from accidental live-account use. When switching
  to live trading, both the account-prefix check and `paper=True`
  need to be updated together — don't bypass just one.
- **`--preview-email`** renders HTML to a local file (default
  `email_preview.html`, gitignored). Use for visual testing without
  sending.
- **Dashboard 5-tier palette uses black text on Hot/Warm/Tepid**,
  white on Fire/Cold. Contrast matters if you change a bg color.
- **`SUBSECTOR_CODES` lives in `trade_executor.py`**, not
  ticker_config.yaml. Update it when subsectors change.
- **Limit orders auto-cancel at close next day (DAY TIF).** Don't try
  to cancel them manually — stale limit IDs from yesterday are
  already gone by the time today's run starts.
- **`transition_trim.py` is DONE** — ran 2026-04-20. Do not re-run.

---

## How to run things locally

```bash
cd /Users/toddbruschwein/Claude-Workspace/breakout-tracker

# Dry-run the daily pipeline
python3 trade_executor.py --dry-run

# Dry-run + render email HTML for visual preview
python3 trade_executor.py --dry-run --preview-email
open email_preview.html

# Force an entry (bypass threshold + persistence, keep cash floor + sizing)
python3 trade_executor.py --dry-run --force-entry HL

# Trade log summary + bucket stats
python3 trade_log.py

# Dashboard
streamlit run dashboard.py

# Quarterly review
python3 quarterly_review.py --months 12

# Backtests
python3 sizing_comparison_backtest.py
python3 sizing_comparison_backtest.py --sweep-entry-threshold 7.5,8.0,8.5,9.0,9.5
python3 entry_mode_backtest.py

# Signal diagnostics
python3 signal_diagnostics.py
python3 signal_diagnostics_subsector.py
python3 signal_diagnostics_significance.py
```

---

## Working style for the next thread

- User prefers concise, direct responses. No preamble, no "great
  question" intros.
- Prefer editing existing files over creating new ones. This project
  has accumulated enough scripts.
- Investigation > delegation. When the user asks "why is X happening,"
  query the DB or Alpaca directly, run small scripts, show the data.
  Don't theorize without evidence.
- User is a fluent programmer. No need to explain basics, but do
  show the reasoning behind design choices.
- Ground every config change in a specific backtest or diagnostic
  artifact.
- Don't call something a "data issue" or "bug" without verifying
  against raw data first.
- Commit every meaningful code change. When in doubt: present, ask.
