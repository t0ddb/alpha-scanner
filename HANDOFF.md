# Alpha Scanner — Session Handoff

Last updated: 2026-04-11

This document exists so a fresh Claude Code thread can pick up where we
left off without re-deriving context. Read this first, then read the
code. Anything a careful code-reader could figure out is **not** here;
only non-obvious state, decisions, and reasoning.

---

## What this project is

**Alpha Scanner** (formerly "Breakout Tracker") — a subsector momentum
breakout detection system that runs nightly, scores ~164 tickers across
~40 subsectors using 7 scored indicators, tracks subsector-level
"breakout waves" via a state machine, and executes paper trades on
Alpaca against the resulting signals.

The thesis: individual stock return prediction is noisy, but when a
cluster of stocks in the same theme (Photonics, Memory, etc.) all break
out simultaneously, that's a high-signal event worth trading.

Repo: `/Users/toddbruschwein/Claude-Workspace/breakout-tracker` (not a
git worktree — the checked-in `.git` directory is authoritative).

---

## Current live strategy (deployed 2026-04-11)

**Config** (`ticker_config.yaml` → `trade_execution:`):

| Parameter | Value | Source |
|---|---|---|
| Entry threshold | **8.5** | 3yr backtest winner |
| Persistence filter | **3 prior trading days ≥ 8.5** | Backtest + user approval |
| Exit threshold | **< 5.0** | Unchanged from original |
| Stop loss | **15%** | Unchanged |
| Max position | 20% of equity | Unchanged |
| Min position | $500 | Unchanged |

Env vars override config: `ENTRY_THRESHOLD`, `EXIT_THRESHOLD`,
`PERSISTENCE_DAYS`, `STOP_LOSS_PCT`, `MAX_POSITION_PCT`,
`MIN_POSITION_SIZE`. Precedence: env > yaml > hardcoded default.

**How we arrived at this config:**

- Ran `threshold_optimizer.py` against 3 years of data → 8.5/<5 had
  highest return but worse DD than 9.5/<5
- `analyze_8_5_config.py` bucketed trades by entry score:
  the 8.5-9.0 bucket added alpha primarily via earlier entries on
  shared tickers (AXTI, WDC, QUBT, PLTR, KTOS), not via unique names
- `axti_breakdown.py` confirmed: ~110% of the 8.5 alpha came from
  earlier entries on shared tickers, only ~-10% from tickers unique to
  8.5
- `persistence_test.py` tested 11 configs (baseline + 5×8.5 + 5×9.0
  with persistence days 0-5). Winner: **8.5 + 3-day persistence** at
  +767% return, -39% DD, 49.3% WR — a Pareto improvement on the 9.5
  baseline (+683%, -41%, 48.9%).
- The 3-day persistence specifically filters out "false start" tickers
  (IREN, AAOI, BTBT, RCAT, RGTI) while retaining early-catch tickers.

**Important nuance:** the most recent 12-month quarterly review (see
`quarterly_reviews/review_2026_Q2.txt`) showed 9.5/<5 winning at
+612.9% vs 8.5/<5 at +319.3%. The 3-year advantage of 8.5 came from
opportunities earlier in the window. User chose to deploy the 8.5+3p
config anyway based on the 3yr test; monitor this quarter-by-quarter.

---

## Architecture at a glance

```
Data sources
  └─ data_fetcher.fetch_all(cfg, period='Xy')       yfinance batch
  └─ alpaca.StockHistoricalDataClient               live prices (fallback)

Scoring pipeline
  └─ indicators.compute_all_indicators(df, bench, cfg, all_rs_values, multi_tf_rs)
        └─ RS (gradient 0-3.0), Ichimoku (2.5), Higher Lows (2.0),
           ROC (1.5), CMF (1.0), Dual-TF RS (0.5), ATR (0.5)
        └─ Max score 10.0. MA Alignment + Near 52w High computed
           for display but NOT scored.
  └─ indicators.score_all(data, cfg)                returns sorted list[dict]
  └─ indicators.score_ticker(indicators)            applies weights

Subsector state machine
  └─ subsector_breakout.run_breakout_detection(results, cfg)
        └─ quiet → emerging → confirmed → fading → quiet/revival
        └─ Params in ticker_config.yaml → breakout_detection:

Persistence (SQLite: breakout_tracker.db)
  └─ subsector_store.py — 3 tables:
      - subsector_daily         per-date per-subsector metrics + ticker_scores JSON
      - subsector_breakout_state current state per subsector
      - ticker_scores            per-date per-ticker score row

Live trading
  └─ trade_executor.py                              runs nightly via CI
        └─ load_trade_config(cfg)                   env > yaml > defaults
        └─ evaluate_exits(snapshot, scores, cfg)    score<5 or stop-loss
        └─ check_persistence(db, ticker, thresh, N, today)
        └─ evaluate_entries(..., trade_cfg, db_conn)
        └─ After scoring: upsert_ticker_scores()    so tomorrow has today's row
  └─ trade_log.py                                   JSON history of buys/sells
  └─ wash_sale_tracker.py                           log-only, never blocks

Backtesting
  └─ portfolio_backtest.run_simulation(...)         capital-constrained sim
        └─ Supports persistence_days param (added recently)

Review
  └─ quarterly_review.py                            7-section automated review
        └─ Runs quarterly on GitHub Actions
        └─ Outputs saved to quarterly_reviews/review_YYYY_QN.txt
```

---

## GitHub Actions workflows

Three scheduled workflows, all runs commit state back to the repo:

| Workflow | Cron | What it does |
|---|---|---|
| `daily-trade-execution.yml` | `30 21 * * 1-5` (21:30 UTC Mon-Fri) | `trade_executor.py` — scores, exits, enters, commits `trade_history.json`, `wash_sale_log.json`, `breakout_tracker.db` |
| `daily-backfill.yml` | `30 21 * * 1-5` | `backfill_subsector.py` (default freq=5, 180d) — maintains historical state machine + subsector metrics, commits `breakout_tracker.db` |
| `quarterly-review.yml` | `0 14 1 1,4,7,10 *` | `quarterly_review.py --months 12` — writes `quarterly_reviews/review_YYYY_QN.txt` + updates `review_history.json` |

**Known race condition**: trade-execution and daily-backfill are both
scheduled at 21:30 UTC and both commit `breakout_tracker.db`. Trade
executor uses `git pull --rebase -X ours || git rebase --abort` in its
commit step, preferring its fresh live-EOD scores on conflict. Not a
bug, but worth staggering crons eventually (e.g., backfill at 21:00,
trade exec at 21:30).

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

### Zero-score rows are scoring-pipeline output, not data failures

A row like `NVDA 2026-04-06 score=0.0 signals=[]` means the scorer ran
successfully but no indicator fired. It is NOT a fetch failure. Fetch
failures don't write rows at all. The recent 44% zero-score rate
(2026-04-06) is a **breadth signal** — the universe has been
deteriorating since mid-March (started at ~11% zeros on 2026-03-16).

### Dual-TF RS fix in quarterly_review.py (just landed)

`quarterly_review.py:collect_indicator_events()` was passing
`multi_tf_rs=None` to `compute_all_indicators`, causing dual_tf_rs to
always report `triggered=False` → 0 events collected → Section 2 fell
back to the hardcoded baseline (+5.20%). Fix: replicate `score_all()`'s
multi-timeframe percentile computation inside the event loop, at 126d,
63d, 21d windows. Smoke-tested — dual_tf_rs now fires 227/1458 in a
2-month window.

Next quarterly review (2026-07-01) will have a real dual_tf_rs edge
value; Section 2 will no longer mark it WATCH on fallback alone.

### Trade log schema carries score_at_entry on sells

`trade_log.log_sell()` auto-populates `score_at_entry` on the sell
record by walking back to the last matching buy. This lets
`trade_log.bucket_stats()` group realized P&L by entry bucket
(8.5-9.0, 9.0-9.5, 9.5+) without cross-referencing records.

Run `python trade_log.py` for a quick bucket table — it's wired into
the CLI summary.

---

## Recent quarterly review (2026-Q2)

File: `quarterly_reviews/review_2026_Q2.txt`

Headlines:

- **Overall: [ACTION]**
- Section 1: 10/13 indicators healthy. Biggest finding:
  **MA Alignment edge flipped from -9.30% (why we dropped it) to +7.76%
  in the last 12 months (+17.06% delta)** → flagged for re-inclusion
  consideration. BB Squeeze also improved (+0.38% → +4.79%).
- Section 2: WATCH. Dual-TF RS was flagged because it fell back to
  baseline — **this was a bug, now fixed** (see above). Next quarter
  will have a real number.
- Section 3: HEALTHY. Notably, 9.5/<5 ranked **#1** in the last 12
  months at +612.9% / 21.17x Ret/DD. The 8.5/<5 config we just
  deployed ranked lower in this window. The 3yr backtest favored 8.5;
  the 12mo quarterly view favors 9.5. Worth watching.
- Section 4: SKIP. Only 3 buys, 0 sells at review time.
- Section 5: HEALTHY. 0 fetch failures, 11 dormant tickers.
- Section 6: HEALTHY. 520 state transitions, 62.4% confirmation rate.

Known limitations in `quarterly_review.py` (punt to next quarter):
- Section 4 requires realized sells before it computes anything
- Section 6 logs transition counts but does NOT yet compute per-state
  forward returns (needs per-date ticker replay — expensive)

---

## Trade history so far

`trade_history.json` at time of writing:

- 3 buys on 2026-04-10 (first live paper trade day under old 9.5/<5
  config): **FORM** @ $123.80, **IRDM** @ $34.20, **VIAV** @ $41.79 —
  all at score 9.8, ~$20k cost basis each.
- 0 sells yet.

First run under the new 8.5+3p config will execute on the next
scheduled trading-day close after the code lands. The DB has enough
recent history for the persistence filter to evaluate from day 1 —
no warm-up period required.

---

## Memory files

At `/Users/toddbruschwein/.claude/projects/-Users-toddbruschwein-Claude-Workspace-experiments/memory/`:

- `MEMORY.md` — index
- `feedback_persistence_filter_semantics.md` — "N most recent DB rows,
  not strict trading days. Holidays aren't gaps."

---

## Open threads / next work

Priority roughly top-down:

1. **Monitor the 8.5+3p config in live trading.** First few entries will
   tell us whether the backtest alpha is real in the current market.
   Use `trade_log.bucket_stats()` to see per-bucket performance once
   sells accumulate.

2. **MA Alignment re-inclusion watch.** Its edge flipped +17% in the
   last year. Do NOT re-add yet — wait one more quarterly review
   (2026-Q3) to confirm it's not a one-window fluke. If Q3 also shows
   positive MA Alignment edge, add it back with weight ~1.5 and rerun
   `threshold_optimizer.py`.

3. **Daily-backfill frequency.** Currently `frequency=5` (samples every
   5 trading days, 180-day window). The persistence filter works fine
   because the trade executor fills in the per-day rows, but backfill
   at freq=1 would give us a robust fallback if trade-exec commits
   fail. Tradeoff: runtime goes from 10-20 min to ~50+ min per day.
   Consider if CI budget allows.

4. **Workflow race condition.** Stagger crons: backfill at 21:00 UTC,
   trade execution at 21:30 UTC. Backfill completes, commits, pushes;
   trade executor checks out the updated DB. No more
   `-X ours` rebase workaround.

5. **Quarterly review Section 4 & 6 completion.**
   - Section 4: becomes meaningful once live sells exist. Wire up
     bucket-grouped realized P&L comparison against backtest expectations.
   - Section 6: implement per-state 63-day forward returns via per-date
     ticker replay. Expensive — consider sampling.

6. **Dashboard subsector breakout page.** The plan in
   `~/.claude/plans/glistening-prancing-reddy.md` mentions a 4th
   dashboard page for subsector breakouts. Not started yet. Not
   blocking anything.

---

## Files to know

### Production code

- `indicators.py` — all indicator functions + `score_all` + `score_ticker`
- `portfolio_backtest.py` — `run_simulation()` (capital-constrained);
  supports `persistence_days` param
- `trade_executor.py` — live execution; `check_persistence()`,
  `load_trade_config()`, `evaluate_entries()` with DB-backed persistence
- `trade_log.py` — `log_buy`, `log_sell` (auto-carries score_at_entry),
  `bucket_stats()` for per-tier analysis
- `subsector_store.py` — SQLite CRUD for `ticker_scores`,
  `subsector_daily`, `subsector_breakout_state`
- `subsector_breakout.py` — state machine
- `backfill_subsector.py` — historical DB population
- `data_fetcher.py` — yfinance batch fetch
- `config.py` — yaml loader + `get_all_tickers`, `get_indicator_config`
- `wash_sale_tracker.py` — logs violations but never blocks entries
- `quarterly_review.py` — 7-section quarterly review runner
- `email_alerts.py`, `dashboard.py` — not touched recently

### Investigation / analysis scripts (kept in repo, not part of daily run)

- `analyze_8_5_config.py` — bucket analysis of 8.5/<5 vs 9.5/<5
- `axti_breakdown.py` — shared-ticker vs unique decomposition
- `check_axti.py` — one-off peak-score verification
- `persistence_test.py` — the 11-config grid that validated 8.5+3p
- `threshold_optimizer.py` — entry/exit threshold grid search
- `gradient_analysis.py`, `conditional_edge_analysis.py`,
  `indicator_optimizer.py` — earlier analysis that informed the
  three-tier scoring

### Config

- `ticker_config.yaml` — ~164 tickers across ~40 subsectors,
  indicator weights, `trade_execution:` section (new),
  `breakout_detection:` section, `scoring:` section (max_score: 10.0)

### CI

- `.github/workflows/daily-trade-execution.yml` — env-var overrides
  exposed via `workflow_dispatch` inputs
- `.github/workflows/daily-backfill.yml`
- `.github/workflows/quarterly-review.yml`

### Outputs (committed to repo)

- `breakout_tracker.db` — SQLite; 50,332 rows across 313 dates
  (2023-04-14 → 2026-04-06)
- `trade_history.json` — 3 buys, 0 sells
- `wash_sale_log.json` — probably empty
- `quarterly_reviews/review_2026_Q2.txt` — first baseline review
- `quarterly_reviews/review_history.json` — index for trend tracking

### Specs (read-only context)

- `QUARTERLY_REVIEW_SPEC.md`, `THRESHOLD_OPTIMIZER_SPEC.md`,
  `PORTFOLIO_BACKTEST_SPEC.md`, `ALPACA_PAPER_TRADING_SPEC.md`,
  `PORTFOLIO_ANALYSIS_SPEC.md`

---

## Gotchas

- **`fetch_all` signature is `fetch_all(cfg, period=...)`**, not
  `fetch_all(tickers, period=...)`. I've gotten this wrong before.
- **Python 3.9 compatibility**: use `from __future__ import annotations`
  and `str | None` still needs the future import.
- **Score 0 with `signals=[]`** is valid output, not a bug.
- **Subsector state machine** uses absolute dates; if you convert
  relative dates (from user messages) to absolute, use the project's
  "today" not your training cutoff.
- **Max score is 10.0**, not 13.0. An older plan
  (`glistening-prancing-reddy.md`) proposed a 13-point scoring system
  that was never adopted — the current system is 7 indicators summing
  to 10.0.
- **MA Alignment is computed but NOT scored.** Same with Near 52w High.
  They exist in `indicators` dict for dashboard display only. Scoring
  uses the list in `cfg['scoring']['indicators']`.
- **`check_persistence` requires a `db_conn` param** — pass it through
  from `main()`. Don't create a new connection inside the function.

---

## How to run things locally

```bash
cd /Users/toddbruschwein/Claude-Workspace/breakout-tracker

# Score once, print top signals (no trading)
python3 -c "from config import load_config; from data_fetcher import fetch_all; from indicators import score_all; cfg=load_config(); data=fetch_all(cfg, period='1y', verbose=False); res=score_all(data, cfg); [print(f\"{r['ticker']:8} {r['score']:.1f}\") for r in res[:20]]"

# Dry-run trade executor (needs .env with ALPACA_API_KEY/ALPACA_SECRET_KEY)
python3 trade_executor.py --dry-run

# Trade log summary + bucket stats
python3 trade_log.py

# Run a quarterly review (writes to quarterly_reviews/)
python3 quarterly_review.py --months 12

# Backfill the DB (long-running)
python3 backfill_subsector.py --days 180 --frequency 5
```

---

## Tone / working style notes for the next thread

- User prefers concise, direct responses. No preamble, no "great
  question" intros.
- When making changes, prefer editing existing files over creating
  new ones. This project has accumulated enough scripts.
- Investigation > delegation: when a user asks "why is X happening",
  query the DB directly, run small scripts, show the data. Don't
  theorize without evidence.
- User is a fluent programmer — no need to explain basics, but do
  show the reasoning behind design choices.
- When suggesting config changes, always ground them in a specific
  backtest or analysis artifact.
- **Don't call something a "data issue" or "bug" without verifying
  against the raw data first.** I mislabeled NVDA/IREN zero scores as
  "data issues" once — they were legitimate score decays during a
  breadth pullback.
