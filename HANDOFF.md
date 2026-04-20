# Alpha Scanner — Session Handoff

Last updated: 2026-04-19

This document exists so a fresh Claude Code thread can pick up where we
left off without re-deriving context. Read this first, then read the
code. Anything a careful code-reader could figure out is **not** here;
only non-obvious state, decisions, and reasoning.

---

## What this project is

**Alpha Scanner** (formerly "Breakout Tracker") — a subsector momentum
breakout detection system that runs nightly, scores 180 tickers across
31 subsectors using 7 scored indicators, tracks subsector-level
"breakout waves" via a state machine, and executes paper trades on
Alpaca against the resulting signals with real GTC stop orders for
downside protection.

The thesis: individual stock return prediction is noisy, but when a
cluster of stocks in the same theme (Photonics, Memory, etc.) all break
out simultaneously, that's a high-signal event worth trading.

Repo: `/Users/toddbruschwein/Claude-Workspace/breakout-tracker` (the
checked-in `.git` directory is authoritative).

GitHub: `github.com/t0ddb/alpha-scanner`. Dashboard:
`alphascanner.streamlit.app`.

---

## Current live strategy (deployed 2026-04-17)

**Config** (`ticker_config.yaml` → `trade_execution:`):

| Parameter | Value | Source |
|---|---|---|
| Entry threshold | **8.5** | 3yr backtest + entry-threshold sweep on corrected simulator |
| Persistence filter | **3 prior trading days ≥ 8.5** | Backtest |
| Exit threshold | **< 5.0** | Unchanged from original |
| Stop loss | **20%** | `sizing_comparison_backtest.py` stop-loss sweep (2026-04-16) |
| Max positions | **12** | Position-cap sweep optimum |
| Position size | **8.3% of equity** (= 1/12) | Dynamic per-entry sizing |
| **Cash floor** | **5% of equity reserved** | `entry_mode_backtest.py` (2026-04-17) |
| Min position | $500 | Unchanged |
| **Entry order type** | **3% LIMIT @ DAY TIF** | `entry_mode_backtest.py` — Limit-3% best-performing config |
| **Sizing price source** | **Alpaca latest-trade** (yfinance fallback) | Post-AEHR-gap bug |

Env vars override config: `ENTRY_THRESHOLD`, `EXIT_THRESHOLD`,
`PERSISTENCE_DAYS`, `STOP_LOSS_PCT`, `MAX_POSITIONS`,
`MAX_POSITION_PCT`, `MIN_POSITION_SIZE`. Precedence: env > yaml >
hardcoded default.

**`CASH_FLOOR_PCT` and `LIMIT_ORDER_BUFFER`** are module constants in
`trade_executor.py` (not YAML-configurable — tightly validated).

**Key evolution vs original config** (see `trade_executor.py` git
history for details):

1. **Limit orders replaced market orders** on 2026-04-17 after a
   15.5% overnight gap on AEHR caused a +$3,049 fill-price overrun on
   a single entry, pushing cash to -$2,626 on a 5-position book. The
   `entry_mode_backtest.py` study confirmed Limit-3% as the right fix.
2. **5% cash floor** enforced via
   `max_position = min(8.3% × equity, (0.95 × equity − committed) / remaining_slots)`.
   Mathematically guarantees total commitment ≤ 95% of equity.
3. **Alpaca-first pricing**: sizing now uses Alpaca's latest-trade
   (which reflects extended-hours moves) before falling back to
   yfinance close.
4. **Stops deferred to post-fill**: `execute_entries` no longer places
   the stop immediately. `ensure_stops_for_positions()` runs at the
   top of each daily cycle and idempotently backfills a GTC stop for
   any held position without one — including limit orders that filled
   overnight.

**Still-open follow-up**: as of 2026-04-19 the account holds 5
positions from the OLD 20%-sizing regime (~$20k each, cost basis
~$100k). Cash is -$2,626. `transition_trim.py` exists to migrate
these positions to the new 8.33% sizing basis proportional to accrued
P&L. **It has been dry-run-verified but NOT executed.** User plans to
run `python transition_trim.py --execute` Monday 2026-04-20 before
the 5:30 PM scoring run. Until that happens, the cash floor blocks
all new entries (over-committed state).

---

## Architecture at a glance

```
Data sources
  └─ data_fetcher.fetch_all(cfg, period='Xy')       yfinance batch
  └─ alpaca.StockHistoricalDataClient               live prices (preferred for sizing)
  └─ alpaca.TradingClient                           orders + positions

Scoring pipeline
  └─ indicators.compute_all_indicators(df, bench, cfg, all_rs_values, multi_tf_rs)
        └─ RS (gradient 0-3.0), Ichimoku (2.0), Higher Lows (1.0),
           CMF (1.5), ROC (1.5), Dual-TF RS (0.5), ATR (0.5)
        └─ Max score 10.0. MA Alignment + Near 52w High computed
           for display but NOT scored.
  └─ indicators.score_all(data, cfg)                returns sorted list[dict]
  └─ indicators.score_ticker(indicators)            applies weights

Subsector state machine
  └─ subsector_breakout.run_breakout_detection(results, cfg)
        └─ quiet → warming → emerging → confirmed → fading → quiet/revival
        └─ Params in ticker_config.yaml → breakout_detection:

Persistence (SQLite: breakout_tracker.db)
  └─ subsector_store.py — 3 tables:
      - subsector_daily         per-date per-subsector metrics + ticker_scores JSON
      - subsector_breakout_state current state per subsector
      - ticker_scores            per-date per-ticker score row

Live trading
  └─ trade_executor.py                              runs nightly via CI
        └─ load_trade_config(cfg)                   env > yaml > defaults
        └─ detect_filled_stops()                    close positions that hit the GTC stop
        └─ ensure_stops_for_positions()             idempotent GTC stop backfill
        └─ detect_unfilled_limits_since()           previous-day limit orders that didn't fill
        └─ score_all() + upsert_ticker_scores()     persist today's scores
        └─ evaluate_exits()                         score<5 or stop-loss
        └─ check_persistence()                      3 prior DB rows ≥ threshold
        └─ evaluate_entries()                       cash-floor + Alpaca-first pricing
        └─ execute_entries()                        submit LIMIT orders @ sizing × 1.03
  └─ trade_log.py                                   JSON history of buys/sells/trims
  └─ wash_sale_tracker.py                           log-only, never blocks

Backtesting framework
  └─ sizing_comparison_backtest.py                  4-strategy comparison + sweeps:
        --sweep-max-positions 3,5,8,10,12,15,20
        --sweep-stop-loss fixed-10,fixed-15,fixed-20,trail-10,atr-2x,...
        --sweep-entry-threshold 7.5,8.0,8.5,9.0,9.5
  └─ entry_mode_backtest.py                         4-config market-vs-limit comparison
        baseline / defensive / limit-2% / limit-3%

Diagnostics
  └─ signal_diagnostics.py                          bucket tables, rank corr, autocorr
  └─ signal_diagnostics_subsector.py                same stratified by all 31 subsectors
  └─ signal_diagnostics_significance.py             bootstrap 95% CIs on ρ

Review
  └─ quarterly_review.py                            7-section automated review
        └─ Runs quarterly on GitHub Actions
        └─ Outputs saved to quarterly_reviews/review_YYYY_QN.txt

Dashboard
  └─ dashboard.py                                   Streamlit; 3 pages:
        Tickers · Subsectors · Historical Charts
        5-tier Tableau 20 palette (Fire/Hot/Warm/Tepid/Cold)
```

---

## GitHub Actions workflows

Three scheduled workflows, all runs commit state back to the repo:

| Workflow | Cron | What it does |
|---|---|---|
| `daily-backfill.yml` | `0 21 * * 1-5` (21:00 UTC Mon-Fri) | `backfill_subsector.py` (default freq=5, 180d) — maintains historical state machine + subsector metrics, commits `breakout_tracker.db` |
| `daily-trade-execution.yml` | `30 21 * * 1-5` (21:30 UTC Mon-Fri) | `trade_executor.py` — scores, exits, enters, commits `trade_history.json`, `wash_sale_log.json`, `breakout_tracker.db` |
| `quarterly-review.yml` | `0 14 1 1,4,7,10 *` | `quarterly_review.py --months 12` — writes `quarterly_reviews/review_YYYY_QN.txt` + updates `review_history.json` |

**Workflow race resolved**: crons were staggered on an earlier update
(backfill 21:00, trade-exec 21:30) so trade-exec checks out the
backfill's DB commit before running. No `-X ours` hack needed in
current workflow.

**Manual overrides** on `daily-trade-execution.yml`: `entry_threshold`
and `persistence_days` as workflow_dispatch inputs, exported as env
vars that `load_trade_config()` reads.

---

## Key non-obvious decisions

### Persistence filter uses "N most recent DB rows," not strict trading days

`trade_executor.check_persistence()` runs:

```sql
SELECT date, score FROM ticker_scores
WHERE ticker = ? AND date < ?
ORDER BY date DESC LIMIT ?
```

This is **intentional**. It does NOT walk a trading calendar to require
exactly the last N trading days. If the DB has a genuine gap (holiday,
CI glitch), we'd rather let a legit signal through than block it.

**Example:** VIAV on 2026-04-07 has rows 4/6, 4/2, 4/1 as its last 3 —
4/3 is missing from the DB because **Good Friday 2026 was 4/3, not a
trading day**. So those 3 DB rows *are* the 3 most recent trading days.
The loose interpretation matches strict interpretation for this case,
and degrades gracefully when they differ.

User confirmed this design (feedback memory:
`feedback_persistence_filter_semantics.md`). Don't tighten it.

### Live trading uses live-scored values for "today" + DB rows for "prior N"

Sequence inside `trade_executor.main()`:
1. `score_all()` computes today's scores in-memory
2. `upsert_ticker_scores(db, today, results)` writes them to DB
3. `evaluate_entries` queries the DB for prior N days per candidate

The same run's live scores become "prior" data for the next day. This
is why the executor must commit the DB at the end (see workflow note
above).

### Limit orders land *tomorrow* — today's run submits, tomorrow's run verifies

Because `trade_executor.py` runs at 21:30 UTC (5:30 PM ET, after close),
any limit order it submits is for **the next session's open** with
`TimeInForce.DAY`. If the limit fills overnight, the position appears
in Alpaca the next day. `ensure_stops_for_positions()` at the top of
tomorrow's run is what finally attaches the GTC stop.

**Consequence**: on the day a limit order is submitted, we don't know
if it filled. The trade_log entry is written with `price=sizing_price`
(not fill_price) as a sizing reference. The actual fill price lives
in Alpaca's order history and is not currently auto-reconciled back
into `trade_history.json`. Left as a future enhancement.

### Zero-score rows are scoring-pipeline output, not data failures

A row like `NVDA 2026-04-06 score=0.0 signals=[]` means the scorer ran
successfully but no indicator fired. It is NOT a fetch failure. Fetch
failures don't write rows at all. Zero-score counts across the
universe are a **breadth signal** — universe-wide deterioration shows
up as rising zero-score share.

### Dashboard uses a 5-tier heat metaphor, not 4 tiers

`dashboard.score_tier()` returns `fire` (≥9.5), `hot` (≥8.5),
`warm` (≥7), `tepid` (≥5), or `cold` (<5). Colors come from the
Tableau 20 palette (#E15759 / #F28E2B / #F1CE63 / #A0CBE8 / #4E79A7).
Text is white on Fire+Cold, black on Hot/Warm/Tepid for contrast.

The same tier system is used in:
- Tickers-page summary card (counts per tier)
- Score column cells on All Tickers table + Subsector drill-downs
  (`score_cell_style()` helper)
- Score History line chart (tier bands via `add_hrect`)
- Score heatmap colorscale (2-month window, was 3-month)
- Universe distribution stacked area (5 buckets)

The email digest from `trade_executor.py` uses a DIFFERENT palette
(green/amber/orange/red — semantic "good/bad" scale, not heat). Don't
accidentally unify them.

### Dual-TF RS fix in quarterly_review.py

`quarterly_review.py:collect_indicator_events()` was passing
`multi_tf_rs=None` to `compute_all_indicators`, causing dual_tf_rs to
always report `triggered=False` → 0 events collected → Section 2 fell
back to the hardcoded baseline (+5.20%). Fix: replicate `score_all()`'s
multi-timeframe percentile computation inside the event loop, at 126d,
63d, 21d windows. Next quarterly review will have a real dual_tf_rs
edge value.

### Trade log schema carries score_at_entry on sells + supports trims

`trade_log.log_sell()` auto-populates `score_at_entry` on the sell
record by walking back to the last matching buy. This lets
`trade_log.bucket_stats()` group realized P&L by entry bucket
(8.5-9.0, 9.0-9.5, 9.5+) without cross-referencing records.

`transition_trim.py` writes records with `side: "trim"` and
`action: "transition_trim"` to preserve audit trail for the one-time
sizing migration (not a recurring operation).

Run `python trade_log.py` for a quick bucket table.

---

## Universe-wide signal diagnostics (2026-04-17/18)

Three diagnostic scripts were added to answer "where does the score
actually work?" before considering any signal-based sizing/pyramiding.

**`signal_diagnostics.py`** — aggregate-level: forward return by score
bucket across raw/sm3/sm5/sm10/sm20 smoothing × 7/21/63-day horizons.
Key finding: smoothing consistently improves predictive rank
correlation (sm20 ρ=+0.353 vs raw ρ=+0.277 at h=63d). Signal is
concentrated in AI/Tech — ρ=+0.407 at 63d sm10 for AI/Tech vs +0.239
for Other.

**`signal_diagnostics_subsector.py`** — decomposed by all 31 subsectors.
Revealed that "AI/Tech aggregate works" is driven by ~4 subsectors
(Chips — Networking/Photonics, Chips — Memory, Alt AI Compute,
Hyperscalers), and "Other fails" is also not uniform (Nuclear Reactors
ρ=+0.487 is the single best subsector).

**`signal_diagnostics_significance.py`** — bootstrap 95% CIs on ρ.
26 of 31 subsectors are statistically significant (15 positive,
11 negative, 5 inconclusive). Chips — Networking/Photonics is the
highest-confidence positive signal (tightest CI, N=1824, ρ=+0.264).
Industrial Robotics & Automation is the strongest negative signal
(ρ=-0.507).

User has **not yet acted on these findings**. They're documented for
future revisit. No pyramiding, no sector-weighted scoring, no universe
pruning has been implemented.

---

## Trade history so far

`trade_history.json` at time of writing:

- **5 buy entries** under OLD 20%-of-equity sizing (2026-04-10
  through 2026-04-15): FORM, IRDM, VIAV (all $19.9k cost), AEHR
  ($20.4k), WDC ($19.3k)
- **0 sells** yet (stops in place, all positions alive)
- Total cost basis ≈ $99,580; total market value ≈ $108,028;
  unrealized P&L ≈ +$8,448 (+8.5%)
- Cash: **-$2,626** (negative due to AEHR gap-up slippage — see AEHR
  note below)

**AEHR incident (2026-04-16)**: AEHR gapped up +15.5% overnight (yfinance
close 4/15 = $73.22, Alpaca fill 4/16 = $84.58). Our sizing math
assumed $73.22, bought 278 shares @ target $20.4k, actual cost
$23,513. Single-trade overrun of $3,158 turned a small cash buffer
into a negative balance. This triggered the limit-order / cash-floor
redesign.

**Transition trim plan** (pending Monday 2026-04-20 execution):
resize all 5 positions from old 20%-of-$100k basis to new
8.33%-of-$100k × (1 + pnl_pct) basis. Expected outcome:
~$65k cash freed, 5 positions at P&L-weighted new sizes totaling
~$45k, 7 open slots. Script: `transition_trim.py --execute`. Stops
preserved at `original_entry × 0.80` (unchanged — same thesis).

---

## Memory files

At `~/.claude/projects/-Users-toddbruschwein-Claude-Workspace/memory/`:

- `MEMORY.md` — index
- `feedback_persistence_filter_semantics.md` — "N most recent DB rows,
  not strict trading days. Holidays aren't gaps."

Also note the `CLAUDE.md` project memory:
- Read `HANDOFF.md` first before any work
- Terse style, no preamble
- Investigation > delegation (query DB, don't theorize)

---

## Open threads / next work

Priority roughly top-down:

1. **Execute `transition_trim.py` Monday 2026-04-20 at market open.**
   Dry-run was verified 2026-04-18 (5 trims, ~$65k freed, stops
   preserved). Must complete before the 5:30 PM scoring run or cash
   floor keeps blocking new entries. Script halts loudly on any
   failure — manual intervention only if that happens.

2. **Monitor the new limit-order + cash-floor regime.** First few
   entries under the new rules will tell us if Alpaca-first pricing
   + 3% limit + 5% floor behaves as modeled. Specifically: fill rate
   on limits, slippage on actual fills, cash utilization. Compare
   against `entry_mode_backtest.py` predictions.

3. **Reconcile actual limit-fill prices back into trade_history.json.**
   Currently `log_buy()` records `price=sizing_price` (Alpaca latest
   at submit time), not the actual fill price. For accurate
   per-position basis tracking, a next-day reconciliation step could
   look up the filled_avg_price from Alpaca's order history and
   update the trade log entry.

4. **MA Alignment re-inclusion watch.** Its edge flipped +17% in the
   last year (Q2 quarterly review). Do NOT re-add yet — wait one more
   quarterly review (2026-Q3) to confirm it's not a one-window fluke.
   If Q3 also shows positive MA Alignment edge, add it back with
   weight ~1.5 and rerun the threshold sweep.

5. **Daily-backfill frequency.** Currently `frequency=5` (samples every
   5 trading days, 180-day window). The persistence filter works fine
   because the trade executor fills in the per-day rows, but backfill
   at freq=1 would give us a robust fallback if trade-exec commits
   fail. Tradeoff: runtime goes from 10-20 min to ~50+ min per day.

6. **Signal-diagnostics-driven strategy.** We have 26 subsectors with
   statistically significant signal direction, but haven't acted on
   the findings. Plausible next steps if/when user wants to:
   - Sector-conditional entry rules (tighter threshold in
     anti-predictive subsectors like Industrial Robotics / Power Semis
     / Gene Editing)
   - Explicit weighting of positions by subsector ρ (advanced; needs
     its own backtest)
   - Pyramiding on confirmed-signal subsectors only (ρ-filtered)

7. **Quarterly review Section 4 & 6 completion.**
   - Section 4: becomes meaningful once live sells exist. Wire up
     bucket-grouped realized P&L comparison against backtest
     expectations.
   - Section 6: implement per-state 63-day forward returns via per-date
     ticker replay. Expensive — consider sampling.

---

## Files to know

### Production code

- `indicators.py` — all indicator functions + `score_all` + `score_ticker`
- `trade_executor.py` — live execution: Alpaca-first pricing, 5% cash
  floor, 3% limit orders, DAY TIF, GTC stops via `ensure_stops_for_positions`
- `trade_log.py` — `log_buy`, `log_sell` (auto-carries score_at_entry),
  `bucket_stats()` for per-tier analysis. Supports `side: "trim"` records.
- `subsector_store.py` — SQLite CRUD for `ticker_scores`,
  `subsector_daily`, `subsector_breakout_state`
- `subsector_breakout.py` — state machine
- `backfill_subsector.py` — historical DB population
- `data_fetcher.py` — yfinance batch fetch
- `config.py` — yaml loader + `get_all_tickers`, `get_indicator_config`
- `wash_sale_tracker.py` — logs violations but never blocks entries
- `quarterly_review.py` — 7-section quarterly review runner
- `dashboard.py` — Streamlit UI; 5-tier Tableau 20 palette

### Backtesting & diagnostics

- `sizing_comparison_backtest.py` — portfolio-construction framework;
  sweeps over `--sweep-max-positions`, `--sweep-stop-loss`,
  `--sweep-entry-threshold`
- `entry_mode_backtest.py` — 4-way market vs limit comparison (the
  backtest that validated the current Limit-3% + cash-floor config)
- `signal_diagnostics.py` — aggregate score→forward-return analysis
- `signal_diagnostics_subsector.py` — per-subsector decomposition
- `signal_diagnostics_significance.py` — bootstrap CIs on ρ

### One-time / utility scripts

- `backfill_stop_orders.py` — places GTC stops for any held position
  lacking one (ran once in April to attach stops to pre-existing
  positions)
- `transition_trim.py` — one-time sizing migration script (dry-run
  verified, execute pending Monday)

### Config

- `ticker_config.yaml` — 180 tickers across 31 subsectors, indicator
  weights, `trade_execution:` section (entry_threshold=8.5,
  persistence_days=3, stop_loss_pct=0.20, max_positions=12,
  max_position_pct=0.083), `breakout_detection:` section,
  `scoring:` section (max_score: 10.0)

### CI

- `.github/workflows/daily-backfill.yml` — cron `0 21 * * 1-5`
- `.github/workflows/daily-trade-execution.yml` — cron `30 21 * * 1-5`,
  env-var overrides via `workflow_dispatch`
- `.github/workflows/quarterly-review.yml` — cron `0 14 1 1,4,7,10 *`

### Outputs (committed to repo)

- `breakout_tracker.db` — SQLite; ~34k+ ticker_score rows
- `trade_history.json` — 5 buys under old sizing, 0 sells
- `wash_sale_log.json` — probably empty
- `quarterly_reviews/review_2026_Q2.txt` — baseline review
- `quarterly_reviews/review_history.json` — index for trend tracking

### Specs (read-only historical context)

- `ALPHA_SCANNER_DOCUMENTATION.md` — main technical methodology doc
  (kept in sync with this handoff)
- `QUARTERLY_REVIEW_SPEC.md`, `THRESHOLD_OPTIMIZER_SPEC.md`,
  `PORTFOLIO_BACKTEST_SPEC.md`, `ALPACA_PAPER_TRADING_SPEC.md`,
  `PORTFOLIO_ANALYSIS_SPEC.md` — original specs for shipped features

---

## Gotchas

- **`fetch_all` signature is `fetch_all(cfg, period=...)`**, not
  `fetch_all(tickers, period=...)`.
- **Python 3.9 compatibility**: use `from __future__ import annotations`
  and `str | None` still needs the future import.
- **Score 0 with `signals=[]`** is valid output, not a bug.
- **Max score is 10.0**, not 13.0. An older plan
  (`glistening-prancing-reddy.md`) proposed a 13-point scoring system
  that was never adopted.
- **MA Alignment is computed but NOT scored.** Same with Near 52w High.
  Both exist in the `indicators` dict for dashboard display only.
- **`check_persistence` requires a `db_conn` param** — pass it through
  from `main()`. Don't create a new connection inside the function.
- **Alpaca orders are paper only.** `connect_alpaca()` enforces a
  paper-account check (account number must start with "PA") — don't
  bypass it. Production is `paper=True` in every call.
- **Dashboard 5-tier palette uses black text on Hot/Warm/Tepid**,
  white on Fire/Cold. If you change a background color, verify text
  contrast before shipping.
- **The email digest palette is GREEN/AMBER/ORANGE/RED** — semantic,
  not the dashboard's heat palette. They're intentionally different.
- **`trade_executor.py` is the source of truth for `SUBSECTOR_CODES`**
  — the 2-letter email-column codes for all 31 subsectors. Update the
  dict there if a subsector is added or renamed; `_subsector_code()`
  falls back to the full name if a code is missing.

---

## How to run things locally

```bash
cd /Users/toddbruschwein/Claude-Workspace/breakout-tracker

# Score once, print top signals (no trading)
python3 -c "from config import load_config; from data_fetcher import fetch_all; from indicators import score_all; cfg=load_config(); data=fetch_all(cfg, period='1y', verbose=False); res=score_all(data, cfg); [print(f\"{r['ticker']:8} {r['score']:.1f}\") for r in res[:20]]"

# Dry-run trade executor (needs .env with ALPACA_API_KEY/ALPACA_SECRET_KEY)
python3 trade_executor.py --dry-run

# Dry-run trade executor with email preview
python3 trade_executor.py --dry-run --email

# Trade log summary + bucket stats
python3 trade_log.py

# Dashboard
streamlit run dashboard.py

# Run a quarterly review (writes to quarterly_reviews/)
python3 quarterly_review.py --months 12

# Backfill the DB (long-running)
python3 backfill_subsector.py --days 180 --frequency 5

# Portfolio sizing comparison (baseline backtest)
python3 sizing_comparison_backtest.py

# Entry-mode backtest (market vs limit)
python3 entry_mode_backtest.py

# Signal diagnostics (aggregate → subsector → bootstrap significance)
python3 signal_diagnostics.py
python3 signal_diagnostics_subsector.py
python3 signal_diagnostics_significance.py

# ONE-TIME: resize 5 old-sizing positions to new 8.33% basis
python3 transition_trim.py --dry-run    # preview
python3 transition_trim.py --execute    # actually run, market must be open
```

---

## Tone / working style notes for the next thread

- User prefers concise, direct responses. No preamble, no "great
  question" intros.
- When making changes, prefer editing existing files over creating
  new ones. This project has accumulated enough scripts.
- Investigation > delegation: when a user asks "why is X happening",
  query the DB or Alpaca directly, run small scripts, show the data.
  Don't theorize without evidence.
- User is a fluent programmer — no need to explain basics, but do
  show the reasoning behind design choices.
- When suggesting config changes, always ground them in a specific
  backtest or diagnostic artifact.
- **Don't call something a "data issue" or "bug" without verifying
  against the raw data first.** NVDA/IREN zero scores looked like
  "data issues" once — they were legitimate score decays during a
  breadth pullback.
- Commit every meaningful code change; the user will usually ask.
  When in doubt: present the change, ask before committing.
