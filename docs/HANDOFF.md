# Alpha Scanner — Project Handoff

Last updated: 2026-10-13 (catch-up refresh covering 2026-05-02 → 2026-10-13)

This doc exists so a fresh Claude Code thread can pick up where the
last one left off. Read it first, then read the code. Anything a
careful code reader could figure out is **not** here — only
non-obvious state, decisions, and reasoning.

**Companion docs:**
- `DECISIONS.md` — append-only log of major decisions with rationale +
  commit references. Read this when you need to know **why** something
  is the way it is.
- `ALPHA_SCANNER_DOCUMENTATION.md` — technical deep dive on architecture.
- `SCHEME_M_SCORING.md` — scoring spec for the Scheme M shadow scheme.

---

## What Alpha Scanner is (1 paragraph)

**Alpha Scanner is an automated momentum trading system.** Every
trading day after market close (paper at 4:30 PM ET, live at 4:35 PM
ET), a GitHub Actions workflow runs `trade_executor.py`, which scores
195 tickers with a 7-indicator momentum signal, evaluates exits on held
positions, evaluates entries on candidates that pass a 3-day
persistence filter, and places 3% limit orders on Alpaca for the
next session's open. Real GTC stop orders at Alpaca protect each
position at entry × 0.80. A daily email summary lands in inbox
when the run finishes, plus a Claude-written weekly recap on Sunday
mornings (live account only). Everything else in this repo — the
subsector state machine, the Streamlit dashboard, the shadow/spillover
trackers, the backtest scripts, the signal diagnostics — exists to
feed or observe or validate this one pipeline.

The system is designed for real-capital trading and **runs against
both a paper and a live Alpaca account in parallel**. Paper started
2026-04-10 with $100k synthetic; live started 2026-04-28 with $5k
real capital (~$6,000 equity as of 2026-10-13). Both run daily via
separate GitHub Actions workflows that share the scoring DB but
maintain independent trade histories and wash-sale logs.

Repo: `/Users/toddbruschwein/Claude-Workspace/breakout-tracker`
GitHub: `github.com/t0ddb/alpha-scanner`
Dashboard: `alphascanner.streamlit.app`

---

## Current live config (Scheme C deployed 2026-04-24)

**Unchanged since 2026-04-24.** Exit-threshold, swap, and
exit-persistence sweeps were all run in May and rejected (DECISIONS
2026-05-05 and 2026-05-11).

All values below are in `ticker_config.yaml` under `trade_execution:`.
Env vars override YAML (precedence: env > yaml > hardcoded default in
`trade_executor.py:_DEFAULTS`).

| Parameter | Value | Source |
|---|---|---|
| Entry threshold | **9.0** | Scheme C audit (was 8.5) — matches prior ~5% selectivity under new weights |
| Persistence | **3 prior days ≥ 9.0** | unchanged |
| Exit threshold | **< 5.0** | Fixed since project start; re-validated 2026-05-05 |
| Stop loss | **20%** | `--sweep-stop-loss` winner (fixed beats trailing) |
| Max positions | **12** | `--sweep-max-positions` plateau optimum |
| Position size | **8.3% of equity** | = 1/12 per slot |
| **Cash floor** | **5% of equity reserved** | `entry_mode_backtest.py` validated |
| **Entry order** | **3% LIMIT @ DAY TIF** | `entry_mode_backtest.py` — Limit-3% won |
| **Sizing price** | **Alpaca latest-trade** (yfinance fallback) | Post-AEHR-gap fix |
| Min position | $500 (YAML) / **$250 live** (env override) | Live lowered $300 → $250 on 2026-09-30 (DECISIONS) |
| Full-cap behavior | **Skip** (no swapping) | Swap-gap sweeps rejected 2026-05-05/11 |

### Indicator weights — Scheme C (deployed 2026-04-24)

| Indicator | Weight | Change from prior |
|---|---|---|
| Relative Strength | 3.0 | unchanged |
| Ichimoku Cloud | 2.0 | unchanged |
| **Dual-TF RS Acceleration** | **2.5** | **was 0.5** (+5x; highest under-weighted signal per audit) |
| Rate of Change | 1.5 | unchanged |
| **Higher Lows** | **0.5** | **was 1.0** (halved; near-zero incremental edge) |
| ATR Expansion | 0.5 | unchanged |
| **Chaikin Money Flow** | **0.0** | **dropped from 1.5** (negative incremental edge in 3yr + 12mo) |
| **Total (max score)** | **10.0** | unchanged |

**Backtest validation (13mo window, 5 path starts, Scheme C @ 9.0 vs baseline @ 8.5):**
- Return: +438% → +598% (+160 pp)
- Sharpe: 1.97 → 2.21
- Max DD: −21.2% → −20.3%
- Win rate: 50.7% → 59.0%
- Path std: 8.3% → 11.9% (proportional to higher return; normalized dispersion same)
- Scheme C's worst start beats Baseline's best. No overlap in return distributions.

Decomposition: dropping CMF drove +90 pp alone, boosting Dual-TF drove +47 pp
alone, reducing HL was ~neutral; combined synergy added another +29 pp.
Extensions tested and rejected: MACD rescue indicator (no lift at threshold
9.0), `pct_from_52w_high` filter (signal-level win but portfolio-level fail
due to whipsaw/concentration/timing).

Audit artifacts: `backtest_results/audit_*.{txt,log,parquet}`.

**Constants not in YAML** (hardcoded in `trade_executor.py`, too
tightly validated to expose):
- `CASH_FLOOR_PCT = 0.05`
- `LIMIT_ORDER_BUFFER = 0.03`

**Cash-floor per-slot cap interacts with min position.** When several
slots are open, the floor rule splits the remaining under-floor budget
evenly (`per_slot_cap = floor_budget / remaining_slots`). On a ~$6k
account that can land below the min-position gate and produce "Cap Hit"
skips even with plenty of cash. That's why live min is $250.

---

## Daily execution pipeline (what happens in the daily run)

Inside `trade_executor.main()`:

```
1. connect_alpaca()                  # mode-aware account check (paper = "PA" prefix)
2. get_account_snapshot()            # equity, cash, last_equity, all positions
3. detect_filled_stops()             # find GTC stops that fired → log as sells
4. ensure_stops_for_positions()      # idempotent: backfill GTC stops for any held
                                       position lacking one (covers limit orders
                                       that filled overnight)
5. detect_unfilled_limits_since()    # prior-day limit orders that canceled (for email)
6. score_all() + upsert_ticker_scores()   # compute + persist today's scores
7. evaluate_exits() + execute_exits()     # score < 5 → cancel GTC stop, wait 2s,
                                            THEN market sell (see below)
8. evaluate_entries()                # candidates: score ≥ 9.0 + persistence + cash floor
9. execute_entries()                 # submit LIMIT BUY at sizing_price × 1.03, DAY TIF
10. _write_pending_spillover()       # capacity skips → pending_spillover_{mode}.json
11. send_trade_digest()              # email summary via Gmail SMTP
```

Then, as separate `continue-on-error` workflow steps: `shadow_m.py`
(paper workflow only) and `spillover_tracker.py --mode {paper|scheme_m|live}`.

**Exit ordering (fixed 2026-05-13, `61d3e87`).** The held GTC stop
locks the shares, so a sell submitted while the stop is live is
rejected by Alpaca with `insufficient qty ... held_for_orders`.
`execute_exits` now cancels the stop first, sleeps 2s for the async
cancel to propagate, then sells. `log_sell` and
`wash_sale_tracker.record_loss_exit` only run if the sell order was
actually accepted — before the fix, failed sells still wrote phantom
sell records and wash-sale cooldowns.

**Key sequencing note**: limit orders submitted after today's close
fill (or don't) at the NEXT morning's open. When they fill, the
position appears in Alpaca at 9:30 AM ET. Tomorrow's `trade_executor`
run is the first one that sees the position — that's when
`ensure_stops_for_positions()` attaches the GTC stop. So for
~24 hours after a limit fills, the position is technically
unstopped. Acceptable because the fill is bounded to ≤ sizing × 1.03
(max 3% above sizing), and Alpaca's market-session risk is small
relative to overnight risk.

---

## Daily email digest

`send_trade_digest()` in `trade_executor.py`. Current layout:

- **P&L card**: Today / All-Time / vs SPY / vs SMH rows, 1-decimal
  percentages. SMH is a **comparison benchmark** declared via
  `comparison_benchmarks: [SMH]` in `ticker_config.yaml`;
  `data_fetcher.fetch_all` prepends it to the fetch. It is NOT used in
  scoring (SPY is still the RS benchmark). All-Time uses the
  `STARTING_EQUITY` env var as its baseline.
- **Section order**: Current Positions → Skipped Signals → Entries →
  Exits → Exit Watch → Wash Sale → Sectors Referenced.
- **Skipped Signals** has a 7d % column between Score and Streak; the
  column header is "Sector" (was "Sub Sector").
- **Current Positions Day %** is close-over-close from yfinance
  (`_compute_recent_return(lookback=1)`), falling back to Alpaca only
  if that fails. Do NOT go back to Alpaca's `position.change_today`:
  Alpaca rolls its reference to today's close at 4:00 PM ET, so by
  the time our run fires it reports after-hours movement only
  (`cb42aec`, 2026-10-12).
- `_get_benchmark_return_since` drops NaN Close rows, guards with
  `math.isfinite`, and prints one-line warnings instead of rendering
  `nan%` (`4f52f99`).

---

## GitHub Actions workflows

Five cron-scheduled workflows + one one-time recovery workflow. All
daily workflows commit state back to the repo on completion.

| Workflow | Cron (UTC, DST) | What it does |
|---|---|---|
| `daily-backfill.yml` | `0 20 * * 1-5` (4:00 PM ET) | `backfill_subsector.py --frequency 1` fills missing DB rows (skips dates already scored) |
| `daily-trade-execution.yml` | `30 20 * * 1-5` (4:30 PM ET) | Paper trade execution (`ALPACA_MODE=paper`) + Scheme M shadow (`shadow_m.py`) + spillover (paper, scheme_m) |
| `daily-trade-execution-live.yml` | `35 20 * * 1-5` (4:35 PM ET) | **Live** trade execution (`ALPACA_MODE=live`) + spillover (live) |
| `weekly-recap.yml` | `0 13 * * 0` (Sun 9:00 AM ET) | `weekly_recap.py` — Claude-written live-account recap email. Read-only |
| `quarterly-review.yml` | `0 14 1 1,4,7,10 *` | `quarterly_review.py --months 12` writes 7-section health report |
| `reconstruct-live-trade-log.yml` | manual only | One-time recovery for `trade_history_live.json`. Used 2026-04-28 to replay the failed first live run; harmless residue, can be deleted |

**⚠️ DST note**: GitHub Actions cron is UTC-only. Current UTC values
are tuned for US Eastern DAYLIGHT TIME (March–November). When the US
returns to Standard Time (Nov 1, 2026), all of these need +1 hour:
- backfill: `0 21 * * 1-5`
- trade-exec paper: `30 21 * * 1-5`
- trade-exec live: `35 21 * * 1-5`
- weekly recap: `0 14 * * 0`

YAML comments in each workflow file document the replacement value.

**Paper and live are staggered 5 min (4:30 / 4:35)** to avoid
concurrent-push races on the shared repo (`9c1aaef`). Both daily
trade workflows also retry the push up to 3× with a 10s backoff
(`git pull --rebase -X ours` → `git push`), and fail the job loudly
if all three fail. They use shared infrastructure (DB, scoring code,
indicators) but maintain independent state files:
`trade_history_{paper,live}.json`, `wash_sale_log_{paper,live}.json`,
`spillover_*_{mode}.json`. The DB is **paper/backfill-owned** — the
live workflow does `git checkout -- breakout_tracker.db` before its
rebase to discard local DB mutations from `score_all()`. (This was
the bug that caused the first live run on 2026-04-28 to lose its
commit; fixed in 4814177.)

**Crons are staggered** (backfill 4:00, paper 4:30, live 4:35) so
backfill's DB commit lands before trade-exec checks out the repo.

**Scheduled runs fire late.** GitHub's scheduler routinely delays
these crons by hours — automated commits in Sept/Oct 2026 land
between ~00:00 and ~03:30 UTC (8–11:30 PM ET), not ~20:45 UTC.
Anything that assumes "the run happens at 4:30 PM ET" (e.g. intraday
vs after-hours Alpaca fields) should not.

**Manual override**: `workflow_dispatch` on either trade-execution
workflow exposes `dry_run`, `entry_threshold`, `persistence_days` as
inputs. Live workflow additionally exposes `max_entries_per_day`
(default 4). `weekly-recap.yml` exposes `preview_only`.

### Live-mode safety controls

| Control | Where | Trigger |
|---|---|---|
| Manual kill switch (paper) | repo variable `KILL_SWITCH=1` | Halts paper entries |
| Manual kill switch (live) | repo variable `KILL_SWITCH_LIVE=1` | Halts live entries |
| Daily-loss circuit breaker | `DAILY_LOSS_CIRCUIT_BREAKER_PCT = 0.10` | Auto-halts entries if equity drops > 10% intraday |
| Account-prefix safety | `connect_alpaca()` | Rejects mode/account-type mismatches |
| Daily entries cap (live) | env `MAX_ENTRIES_PER_DAY=4` | Caps new entries per run |
| Min position size (live) | env `MIN_POSITION_SIZE=250` | Was 300 until 2026-09-30; see DECISIONS |
| Starting equity (live) | env `STARTING_EQUITY='5000'` | All-Time P&L baseline — **set in both `daily-trade-execution-live.yml` and `weekly-recap.yml`** |

All halts gate ENTRIES only — exits and stop-loss management always run.

### Weekly recap (deployed 2026-05-03, `5080cc5`)

`weekly_recap.py` pulls live positions, the week's trade activity, and
recent price action, then makes **one** Anthropic API call
(`claude-sonnet-4-6`, prompt-cached system block) for per-holding
outlooks + an AI-infrastructure sector summary, and emails it. Needs
the `ANTHROPIC_API_KEY` repo secret. Live account only; never places
trades or writes state; `contents: read` permission. Cost ~$0.05–0.20/run.

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

### Trade-exec scores are authoritative; backfill only fills gaps

Since 2026-05-19 (`a94bcf9`), `backfill_subsector.py` skips any date
that already has rows in `ticker_scores` (`date_already_scored()`).
Before that, backfill could overwrite a trade-exec score using a
yfinance fetch that was missing the target-date bar —
`slice_data_to_date` silently used the prior day's row and keyed it
to the target date. That turned LUNR's 5/15 score from 7.10 into 9.60
and triggered a bad live entry. Now `slice_data_to_date()` returns
`(sliced, missing_tickers)`, excludes tickers whose last row pre-dates
the target, and backfill aborts the whole date if > 5% of tickers are
missing. LUNR 5/15 = 9.60 is still in the DB (known, decayed artifact).

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

### Backtest methodology bar for rule changes

Any future rule-change validation in `sizing_comparison_backtest.py`
needs **`--price-period 3y` (or more) and `--path-starts 20` minimum**.
The default 2y fetch caps `compute_path_start_dates` at ~10 candidates
regardless of `--path-starts`, which proved insufficient (DECISIONS
2026-05-11 — a "winner" at 2y/10 didn't survive 3y/20).

---

## Current state (2026-10-13)

**Production trading (Scheme C):**
- Live and paper Alpaca accounts running daily on Scheme C scoring
  (entry 9.0 / exit < 5.0 / persistence 3 / stop 20% / cap 12).
- Live launched 2026-04-28 with $5k; ~$6,000 equity on 2026-10-13
  (no deposits recorded — `STARTING_EQUITY` still `'5000'`). Paper has
  been running since 2026-04-10. See `trade_history_paper.json` and
  `trade_history_live.json`; dashboard for visualization.
- 2026-05-13 data cleanup: 7 phantom sells scrubbed from trade
  histories (live: FORM × 2; paper: COHU × 3, FORM × 2) and the live
  FORM wash-sale cooldown removed (it would have blocked re-entry
  through 2026-06-12 for a sell that never happened).

**Scheme M shadow tracker (deployed 2026-05-01):**
- Hypothetical portfolio using Scheme M scoring (Layer 1 indicator
  buckets + Layer 2 sequence overlay). Threshold 7.5 / exit 4.5.
- Runs daily in `daily-trade-execution.yml`, `continue-on-error: true`.
- State: `shadow_m_state.json`, `shadow_m_trades.json`, `shadow_m_log.json`.
- SQLite table `ticker_scores_m` stores per-day Scheme M scores and
  fire flags (needed for Layer 2 streak computation).
- Dashboard tab "🧪 Scheme M Shadow". Spec: `SCHEME_M_SCORING.md`.

**Spillover trackers (deployed 2026-05-03, `b0caf2f`):**
- Three parallel hypothetical portfolios — `live`, `paper`, `scheme_m`
  — that open a position for every **capacity-skipped** candidate of
  the parent stream (skip reasons: Full, Cap Hit, No Cash, Daily Cap).
  Answers "what are we leaving on the table by being size-constrained?"
- Sized at the parent's intended slot dollars (whole shares at close).
  Same exit threshold + 20% stop as parent; no position cap, no cash
  limit. Re-entry for an already-held ticker is a no-op.
- Handoff: `trade_executor.py` / `shadow_m.py` write
  `pending_spillover_{mode}.json`; `spillover_tracker.py --mode X`
  consumes it. State in `spillover_state_{mode}.json`,
  `spillover_trades_{mode}.json`, `spillover_log_{mode}.json`.
- Runs as a `continue-on-error` step after trade_executor (+ shadow_m).
- Dashboard tab "💧 Spillover Trackers" (`b78cc11`): stream selector
  (Live / Paper / Scheme M), 5 headline metrics incl. "vs Parent
  Realized", cumulative P&L chart, open positions, recent 14-day
  activity, closed trades.
- As of 2026-10-13 the live spillover stream holds 6 positions with
  cumulative P&L of −$2,546 — i.e. so far the skipped trades have
  *lost* money, which argues against loosening capacity constraints.

---

## Universe (195 tickers, 33 subsectors)

Committed to `ticker_config.yaml`. Summary:

- **AI & Tech Capex** — 111 tickers, 16 subsectors (semis, foundries, data center, AI-native clouds, hyperscalers, software)
- **Metals** — 19 tickers, 4 subsectors (gold/silver direct, miners, copper miners)
- **Robotics & Automation** — 16 tickers, 3 subsectors
- **Crypto** — 11 tickers, 2 subsectors (majors incl. GLXY equity, crossover tokens)
- **Space & Satellite** — 11 tickers, 2 subsectors
- **Nuclear & Uranium** — 10 tickers, 2 subsectors
- **Biotech** — 8 tickers, 2 subsectors
- **eVTOL & Drones** — 5 tickers, 1 subsector
- **Quantum** — 4 tickers, 1 subsector

Plus `benchmark: SPY` (RS calculations) and `comparison_benchmarks:
[SMH]` (email only).

Additions since May 2026 (`7cdf4f0`, `c5403dd`):
- New subsector **`foundries`** (AI/Tech) — split out of
  `semi_equipment`: TSM, TSEM, GFS.
- New subsector **`copper_miners`** (Metals): FCX, SCCO, TECK.
- `semi_equipment`: UCTT, VECO. `chips_networking`: LASR, CRDO.
  `datacenter_infra`: SMCI. `alt_ai_compute`: NBIS. `power_energy`:
  TE, HUBB. `industrial_robotics`: SYM, OUST. `physical_ai`: OSS.
- `SUBSECTOR_CODES` additions: `"Foundries": "FD"`, `"Copper Miners": "CU"`.

Earlier additions (April 2026): BE, CRWV, APLD, BTDR, RIOT, CLSK,
WYFI, KEEL, SNDK, EQT, SEI, LBRT, PUMP, PSIX, BW, GLXY.

Adding a ticker: edit `ticker_config.yaml`, commit. No code change.

If you add a new subsector, also add its 2-letter code to
`SUBSECTOR_CODES` in `trade_executor.py` (email display uses it). The
code `_subsector_code()` falls back to full name if a code is
missing, so forgetting won't crash — just looks ugly.

---

## Universe-wide signal diagnostics (2026-04-17/18, documented not acted on)

Three scripts, run once to characterize signal quality (on the
then-180-ticker / 31-subsector universe):

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

1. **DST cron shift — due Nov 1, 2026.** Five crons need +1 hour
   (values in the workflows section above and in each YAML comment).
   Do it the week of Oct 26 so the first STD-time run is correct.

2. **MA Alignment re-inclusion decision is now due.** Q2 flagged a
   12-month edge flip; Q3 (+13.77%) and Q4 (+10.06%) both confirmed
   it positive (`quarterly_reviews/review_2026_Q{3,4}.txt`), meeting
   the "two windows" bar previously set. Next step is a backtest, not
   a config edit: add back at weight ~1.5 and rerun
   `--sweep-entry-threshold` at `--price-period 3y --path-starts 20`.

3. **Update `STARTING_EQUITY` when capital is added.** Still `'5000'`
   in BOTH `daily-trade-execution-live.yml` and `weekly-recap.yml`.
   Fine while equity growth is P&L only (~$6k today); must be bumped
   on any deposit or All-Time P&L / vs-benchmark numbers go wrong.

4. **Remove `MAX_ENTRIES_PER_DAY=4` cap from live workflow.** Was
   meant as a 2–4 week phased-rollout guard; still in place 5+ months
   later. Check spillover "Daily Cap" skips to see if it ever binds.

5. **Spillover read-out.** ~5 months of data across three streams.
   Worth a formal comparison vs parent (realized + open) to close the
   path-dependency question raised in DECISIONS 2026-05-05/11.

6. **Fractional shares** — discussed and deferred 2026-04-27. Would
   capture high-priced tickers (GEV/CIEN/SNDK) excluded at small
   scale but requires switching from Alpaca-side GTC stops to software-
   monitored stops (Alpaca rejects GTC stops on fractional positions).
   Revisit when capital grows past ~$15k.

Resolved since the last refresh: live-trading 2–4 week monitoring
window; Q3 2026 quarterly review (ran 2026-07-01, Q4 on 2026-10-01);
daily-backfill frequency (now `--frequency 1`).

---

## Files to know

### THE one script

- `trade_executor.py` — the daily pipeline. Contains: connection
  helpers, order helpers (market/limit/stop), `evaluate_exits()` /
  `execute_exits()`, `evaluate_entries()` with cash-floor sizing +
  Alpaca-first pricing, `execute_entries()` (submits LIMIT orders),
  `_write_pending_spillover()`, email digest builder, skip-reason
  categorizer, SUBSECTOR_CODES map.

### Pipeline components

- `indicators.py` — 7-indicator scoring engine + `score_all()` +
  `score_ticker()`. Gradient + binary weighting.
- `config.py` — YAML loader
- `data_fetcher.py` — yfinance batch fetch; `fetch_all` also pulls
  `comparison_benchmarks`.
- `subsector_store.py` — SQLite CRUD (4 tables: `subsector_daily`,
  `subsector_breakout_state`, `ticker_scores`, `ticker_scores_m`).
- `sequence_overlay.py` — Layer 2 sequence-pattern bonuses/penalties
  for Scheme M. Pure functions; consumed by `shadow_m.py`.
- `subsector_breakout.py` — state machine (dashboard-only — does not
  gate trades)
- `trade_log.py` — JSON audit log. `log_buy` / `log_sell` /
  `log_trim` (transition_trim records). Auto-carries
  `score_at_entry` onto sells for bucket analysis.
- `wash_sale_tracker.py` — logs loss-exit cooldowns. Never blocks.

### Shadow / spillover / recap

- `shadow_m.py` — Scheme M daily runner (hypothetical portfolio).
- `spillover_tracker.py` — capacity-skip spillover portfolios
  (`--mode live|paper|scheme_m`, `--dry-run`).
- `weekly_recap.py` — Sunday Claude-written live recap email.
- `compute_scheme_m_scores.py` — pre-compute Scheme M scores from
  `audit_dataset.parquet` for backtester consumption.
- `backfill_scheme_m_history.py` — one-time backfill of `ticker_scores_m`
  from parquet. Already run; don't re-run.
- `indicators.score_ticker_v2()` — Scheme M Layer 1 scoring. See
  `SCHEME_M_SCORING.md`.

### Dashboard + review

- `dashboard.py` — Streamlit, **5 pages** (Subsectors / Tickers /
  Historical Charts / 🧪 Scheme M Shadow / 💧 Spillover Trackers),
  5-tier Tableau 20 palette
- `quarterly_review.py` — 7-section quarterly health report

### Backtesting (validates live config)

- `sizing_comparison_backtest.py` — strategy comparison + sweeps:
  `--sweep-max-positions`, `--sweep-stop-loss`, `--sweep-entry-threshold`,
  `--sweep-exit-threshold`, `--sweep-swap-gap`, `--sweep-exit-persistence`,
  and `--price-period` (default 2y; use 3y+ for rule changes).
  `StrategyConfig` gained `swap_score_gap`, `swap_min_persistence`,
  `exit_persistence_days`.
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

- `backfill_subsector.py` — historical DB gap fill (runs daily from CI;
  never overwrites already-scored dates)
- `backfill_stop_orders.py` — one-time GTC stop attachment for any
  position without one (ran once in April; now idempotently covered
  by `ensure_stops_for_positions()` in the daily pipeline)
- `transition_trim.py` — one-time sizing migration (ran 2026-04-20)

### Config + CI

- `ticker_config.yaml` — 195 tickers, indicator weights, `benchmark`,
  `comparison_benchmarks`, `trade_execution:`, `breakout_detection:`,
  `scoring:`
- `.github/workflows/daily-backfill.yml` (20:00 UTC M-F)
- `.github/workflows/daily-trade-execution.yml` (20:30 UTC M-F, paper)
- `.github/workflows/daily-trade-execution-live.yml` (20:35 UTC M-F, live)
- `.github/workflows/weekly-recap.yml` (13:00 UTC Sunday, live recap)
- `.github/workflows/quarterly-review.yml` (14:00 UTC quarterly)
- `.github/workflows/reconstruct-live-trade-log.yml` (manual-only recovery)

### Outputs (committed)

- `breakout_tracker.db` — SQLite
- `trade_history_{paper,live}.json` — trade audit logs
- `wash_sale_log_{paper,live}.json`
- `shadow_m_{state,trades,log}.json`
- `spillover_{state,trades,log}_{live,paper,scheme_m}.json`
- `quarterly_reviews/review_YYYY_QN.txt` + `review_history.json`

### Docs

See `docs/README.md` for the directory map. Key entries:
- `README.md` (root) — public overview
- `docs/HANDOFF.md` — **this file** (project state)
- `docs/DECISIONS.md` — append-only log of major decisions + rationale
- `docs/ALPHA_SCANNER_DOCUMENTATION.md` — technical deep dive
- `docs/SCHEME_M_SCORING.md` — Scheme M scoring spec (Layer 1 + Layer 2)
- `docs/*_SPEC.md` — design specs for various modules (most are reference)

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
- **Alpaca account-type safety check.** `connect_alpaca()` reads
  `ALPACA_MODE` (paper/live), passes `paper=(mode == "paper")` to
  `TradingClient`, and rejects mismatches: paper mode requires a `PA`
  account prefix, live mode rejects one. Don't bypass either half.
- **Sell while a GTC stop is live = rejected.** Alpaca holds the shares
  for the stop (`held_for_orders`). Cancel the stop first (as
  `execute_exits` does). Same applies to any new manual-sell tooling.
- **Never log a trade or wash-sale event before confirming the order
  was accepted.** That's how phantom sells happened.
- **`slice_data_to_date()` returns a tuple** `(sliced, missing_tickers)`
  since 2026-05-19. Old call sites expecting a dict will break.
- **Alpaca `position.change_today` is not "today's move" by the time
  we run** — use yfinance close-over-close.
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

# Backtests (use --price-period 3y --path-starts 20 for rule changes)
python3 sizing_comparison_backtest.py
python3 sizing_comparison_backtest.py --sweep-entry-threshold 7.5,8.0,8.5,9.0,9.5 --price-period 3y --path-starts 20
python3 entry_mode_backtest.py

# Signal diagnostics
python3 signal_diagnostics.py
python3 signal_diagnostics_subsector.py
python3 signal_diagnostics_significance.py

# Scheme M shadow tracker (runs daily in CI; can also run manually)
python3 shadow_m.py            # writes to shadow_m_*.json
python3 shadow_m.py --dry-run  # print decisions without writing state

# Spillover trackers
python3 spillover_tracker.py --mode live --dry-run

# Weekly recap (needs ANTHROPIC_API_KEY)
python3 weekly_recap.py --preview-email   # render HTML, don't send
python3 weekly_recap.py --dry-run-llm     # skip the Anthropic call
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
- **Ask before every `git push`.**
