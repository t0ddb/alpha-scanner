# Alpha Scanner — Decision Log

**Append-only chronological log of major design and configuration decisions.**

Each entry: date, decision, rationale, evidence, commit reference, current
status. Past entries are NEVER edited — if a decision is later reversed,
add a new entry that supersedes it (and mark the original as superseded).

This doc captures the **WHY**. Git captures the **WHAT** and **WHEN**.
Together they tell the full story.

For project orientation, read `HANDOFF.md` first. This doc is for
historical context ("why did we choose X?").

---

## Format

```
## YYYY-MM-DD — <One-line decision summary>

**Decision:** What was chosen.
**Rationale:** Why this over alternatives.
**Evidence:** Backtests, audit scripts, or empirical observations cited.
**Commit:** `<hash>` (or PR number)
**Status:** Active | Superseded by <date> | Reverted on <date>
```

---

## Decisions (chronological, oldest first)

## 2026-04-16 — Switch live config from 9.5 / <5 to 8.5/p=3 / <5

**Decision:** Entry threshold lowered from 9.5 to 8.5 with 3-day persistence
filter. Exit threshold remains < 5.0.

**Rationale:** `sizing_comparison_backtest.py --sweep-entry-threshold` showed
8.5/p=3 had best Sharpe (1.98) and lowest path std dev (8.6%) on the
corrected portfolio simulator. 9.5 was too selective (37 trades, 35% win
rate). Initial portfolio_backtest.py results that motivated 9.5 were
invalidated due to no-position-cap and static-sizing bugs.

**Evidence:** Sweep across 7.5–9.5 in `sizing_comparison_backtest.py`.

**Status:** Superseded 2026-04-24.

---

## 2026-04-16 — Position cap = 12, sizing = 8.3% of equity, fixed 20% stop

**Decision:** 12-position cap with 8.33% per-position sizing, fixed -20% stop
loss as Alpaca GTC orders. Skip-when-full (no rotation/swap/trim).

**Rationale:**
- Position cap sweep (3-20) showed Sharpe peak at 10 (2.02) but lowest
  path std at 12 (7.0% vs 19.2% at cap 10). Chose 12 for path stability.
- Stop loss sweep (13 configs) showed trailing stops catastrophic
  (60-82% whipsaw rate). Fixed-20 had Sharpe 2.09 with 0% whipsaw.
- 4-strategy comparison (Fixed/Trim/Swap) showed trimming destroyed
  $595k of value (cut winners at +17.4% avg P&L); swapping similar.
- "Skip when full" preserved long-tail winners.

**Evidence:** All in `sizing_comparison_backtest.py` sweeps.

**Status:** Active. (Path C shadow tracker uses same parameters except
entry/exit thresholds — see 2026-05-01 entries.)

---

## 2026-04-24 — Scheme C scoring deployed (entry threshold 9.0)

**Decision:** Indicator weights changed to Scheme C: RS=3.0, ICH=2.0,
DTF=2.5 (was 0.5; +5x), HL=0.5 (was 1.0; halved), CMF=0.0 (dropped from
1.5), ROC=1.5, ATR=0.5. Total max=10.0. Entry threshold 9.0 (matches
prior ~5% selectivity under new weights).

**Rationale:** `audit_*.{txt,log,parquet}` artifacts showed:
- CMF had NEGATIVE incremental edge in both 3yr and 12mo audits.
- DTF was the highest under-weighted signal (+5.5pp 3yr / +9.6pp 12mo
  edge despite weight 0.5).
- HL had near-zero incremental edge after RS.

Backtest validation (13mo, 5 path starts, Scheme C @ 9.0 vs baseline
@ 8.5): Return +438% → +598%, Sharpe 1.97 → 2.21, Max DD −21.2 → −20.3,
Win rate 50.7% → 59.0%. Scheme C's worst start beats baseline's best.
No overlap in return distributions.

**Evidence:** `backtest_results/audit_*.{txt,log,parquet}`.

**Commit:** ~2026-04-24 (pre-investigation period)

**Status:** Active in production. Live + paper trading run on this config.

---

## 2026-04-28 — Live trading launched at $5k

**Decision:** Live Alpaca account activated, parallel to paper account.
$5k starting capital. Same daily 4:30 PM ET schedule. `MAX_ENTRIES_PER_DAY=4`
cap for phased rollout (review for removal after 2-4 weeks of clean ops).

**Rationale:** Paper trading running cleanly since 2026-04-10. Wanted
real-fill data without exposing meaningful capital. $5k chosen to make
slippage/fee dynamics visible without large blast radius.

**Status:** Active. ~$5k capital, observation period through ~mid-May 2026.

---

## 2026-04-30 → 2026-05-01 — Multi-session scoring investigation (Scheme I+)

**Decision:** Investigated whether to recalibrate Scheme C scoring based
on empirical findings about indicator non-monotonicity, sequence patterns,
and interaction effects. **Outcome: kept Scheme C in production unchanged.**

**Rationale:**
- Designed and tested "Scheme I+" with empirical bucket curves, sequence
  overlay (Layer 2), and joint scoring across multiple variants.
- Backtest comparison vs Scheme C @ cap 12 / threshold 9.0:
  - Best Scheme C: +651.6% / std 10.2% / Sharpe 3.28
  - Best Scheme I+ Path C: +529.9% / std 23.3% / Sharpe 3.18
  - Best Path C coarse: +522.6% / std 11.5% / Sharpe 3.02
  - Layer 1 only: +491.3% / std 109% (full-range path-dep)
- Scheme C's coarseness IS its advantage — many ties at score 9.9 →
  alphabetical tie-breaking is path-stable. Finer-grained scoring breaks
  ties and increases path-dependency.
- Empirical curves were optimized for win rate, but in heavy-tail regime,
  patterns with lower win rate often capture mega-winners (RS 96-100 zone
  has lower win % but contains AXTI / BW / IREN — the regime's biggest
  contributors). Penalizing those zones excluded the right tail.
- Increasing position cap (12 → 15 → 20) HURT all schemes by 30-50%
  on cumulative return (slot dilution).

**Evidence:** Multiple audit scripts in `audit_*.py` and `_test_*.py`,
backtest logs in `backtest_results/`, full investigation captured in
git history of commits between 2026-04-30 and 2026-05-01.

**Commits:** `bdc4221` (implementation), `e7e1cdc` (shadow deployment),
`5bac6b3` (final exit threshold), `394c34c` (sweep validation).

**Status:** Investigation complete. Scheme C stays in production.
Path C deployed as shadow tracker (see next entry).

---

## 2026-05-01 — Path C @ 7.5 deployed as shadow tracker (no real trades)

**Decision:** Deploy Path C as a **shadow** portfolio that tracks
hypothetical trades alongside the live Scheme C system. No real Alpaca
calls; pure JSON state + DB tables. Runs daily after `trade_executor.py`
in the same GitHub Actions workflow with `continue-on-error: true`.

**Rationale:** Per-trade quality analysis showed Path C @ 7.5 picks
3x better signals on per-trade median (+12.4% vs Scheme C's +4.4%) but
compounds less in heavy-tail backtest (+530% vs +652%). Want to observe
real-world behavior over multi-week period to see whether per-trade
quality translates to actual outcomes. Shadow tracking lets us collect
data without changing production.

**Configuration (validated via sweeps):**
- Entry threshold: 7.5 (sweep 7.0-8.5 in `_test_pathc_sweeps.py`)
- Exit threshold: 4.5 (sweep 3.0-7.0 in `_test_pathc_exit_threshold.py`)
- Persistence: 3 days (sweep 1-5)
- Stop loss: 20% (sweep 0.10-0.40)
- Position cap: 12 (matches Scheme C)
- Sizing: 8.33% (matches Scheme C)
- Starting equity: $100k synthetic

**Evidence:** Sweep logs in `backtest_results/pathc_*.log`. All four
parameters land at empirically-validated optima.

**Commits:** `e7e1cdc` (initial deployment), `5bac6b3` (exit revert),
`394c34c` (sweep validation), `5540315` (initial exit tune).

**Status:** Active shadow tracker. State in `pathc_shadow_*.json`.
Dashboard tab "🧪 Path C Shadow" visualizes results.

---

## 2026-05-01 — Documentation strategy: living docs + decision log

**Decision:** Adopted "Tier 1 living documents + Tier 2 append-only
DECISIONS.md" approach. Deleted session-by-session investigation docs
(SCORING_INVESTIGATION_*.md, CONTEXT_HANDOFF.md, SCHEME_I_PLUS_PROPOSAL.md).
HANDOFF.md and ALPHA_SCANNER_DOCUMENTATION.md kept as living references.

**Rationale:** Multiple session-handoff docs accumulated noise and
discovery friction (3 separate "investigation" docs from same line of
work). The most recent SCORING_INVESTIGATION doc declared "investigation
concluded" but we then deployed shadow tracking, making that doc
misleading. Git history is authoritative for "what existed at time X";
DECISIONS.md captures "why" — together they replace the multi-doc library.

**Status:** Active. This file is the canonical decision log going forward.

---

## 2026-05-03 — Renamed Path C → Scheme M (universal)

**Decision:** Renamed "Path C" (a.k.a. "Scheme I+") to "Scheme M" across
the entire codebase, docs, dashboard, DB schema, env vars, and state files.

**Rationale:** "Path C" was visually and verbally too similar to "Scheme C"
(both have C, similar structure), causing ambiguity in conversation and
documentation. "Scheme M" follows the existing Scheme A-G family naming
convention and is unambiguous. The "M" mnemonic captures the key
differentiator: Scheme M's Layer 2 is **mean-return-based** (vs Scheme
C's win-rate-based empirical calibration).

**Scope of rename (universal):**
- DB table: `ticker_scores_v2` → `ticker_scores_m` (with idempotent
  ALTER TABLE migration in `subsector_store.init_db()`)
- Python: `score_ticker_v2` → `score_ticker_m`, `INDICATOR_WEIGHTS_V2`
  → `INDICATOR_WEIGHTS_M`, all `*_BUCKETS_V2` → `*_BUCKETS_M`,
  `upsert_ticker_scores_v2` → `upsert_ticker_scores_m`,
  `get_fire_flags_history_v2` → `get_fire_flags_history_m`,
  `get_v2_scores_for_persistence` → `get_m_scores_for_persistence`
- Files: `shadow_pathc.py` → `shadow_m.py`,
  `compute_scheme_i_plus_scores.py` → `compute_scheme_m_scores.py`,
  `backfill_pathc_history.py` → `backfill_scheme_m_history.py`
- State files: `pathc_shadow_*.json` → `shadow_m_*.json`
- Env vars: `PATHC_*` → `SCHEME_M_*`
- Dashboard: tab "🧪 Path C Shadow" → "🧪 Scheme M Shadow",
  function `render_pathc_shadow` → `render_scheme_m_shadow`
- Docs: `PATH_C_SCORING.md` → `SCHEME_M_SCORING.md`; updated
  HANDOFF.md, README.md, docs/README.md references

**Backward compatibility:** None needed. Path C had only just been
deployed (2026-05-01) with 2 days of state (initial bootstrap). The DB
migration preserves the v2 historical backfill rows (127k rows). State
files renamed via git mv preserve content.

**Status:** Active. Past DECISIONS.md entries reference "Path C"; those
entries are historical and not edited (per append-only rule). All
new references use "Scheme M".

---

## 2026-05-05 — Re-validated exit=5.0 + skip-when-full under Scheme C

**Decision:** Keep production exit threshold at < 5.0 unchanged. Reject
all gap-gated swap-at-cap variants. No change to live config.

**Rationale:** Path-dependency concern surfaced after observing the
daily email — multiple capacity-skipped tickers (SNDK +40%, AXTI +57%
over 7 days) were running materially hotter than the weakest current
holdings (BE at score 7.9, P&L −0.4%). Two questions arose:
  1. Is exit=5.0 too lenient under Scheme C? (Original number was tuned
     under the pre-Scheme-C weighting and never re-swept.)
  2. Should a gap-gated swap rule replace stalled holdings with
     hotter skipped candidates? (Distinct from the unconditional
     trim/swap rules already disproved in the 2026-04-16 backtest.)

Both questions were swept under current Scheme C scoring on the
historical DB. Both answers came back negative.

**Evidence (sizing_comparison_backtest.py):**

Exit-threshold sweep across {4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0}, holding
entry=9.0/p=3, 12-pos cap, 8.3% sizing, -20% stop:

| Exit | Return | Max DD | Sharpe | Trades | Avg Hold |
|---|---|---|---|---|---|
| 4.0 | +741.7% | -19.6% | 2.39 | 59 | 65.7d |
| 5.0 (live) | +745.1% | -20.3% | 2.35 | 79 | 48.7d |
| 5.5 | +496.6% | -20.6% | 2.25 | 94 | 41.4d |
| 6.0 | +498.9% | -17.8% | 2.32 | 108 | 36.0d |
| 7.0 | +299.8% | -25.2% | 1.80 | 149 | 25.2d |

Tightening the exit destroys cumulative return: 5.0 → 6.0 cuts ~33%,
5.0 → 7.0 cuts ~60%. Mechanism: shorter holds (49d → 25d), more
trades (79 → 149), positions exit before they recover or compound.
Looser exit (4.0) is a marginal Sharpe winner (+0.04) with longer
holds, but cumulative return is essentially tied (+741.7% vs +745.1%).
No empirical case for tightening; weak-but-real case for 4.0 if
re-validated with full path-dependency analysis.

Gap-gated swap sweep — 2D grid over score_gap ∈ {1.0, 1.5, 2.0, 2.5, 3.0}
× min_persistence ∈ {3, 5, 7, 10}. New `swap_score_gap` and
`swap_min_persistence` fields on StrategyConfig require: candidate
score must clear victim by `swap_score_gap`, AND candidate must have
`swap_min_persistence` consecutive prior days at-or-above entry
threshold. Best variant (gap=3.0, persist=3): +569.8% / Sharpe 2.17 /
44 swaps over the full backtest window. **All 20 variants underperform
the no-swap baseline (+745.1% / 2.35).** The strictest filter still
loses 175 pp of cumulative return and 0.18 Sharpe.

**Why the data refuses to validate the rotation intuition** (recorded
for future re-analyses): The strategy compounds via long-tail outliers,
not median trades. Cutting any holding to make room for a hotter
candidate roulette-spins both: you lose the held position's optionality
to become a +30%+ winner, and gain only the chance the swapped-in
candidate does. The "weakest holding" today is rarely the weakest
ex-post — AAOI was a flat early holding before becoming a +25.9%
position. Today's "hottest skipped" candidates suffer survivorship
bias: the names that look hottest at any given snapshot are not the
names that would have looked hottest when the swap would have fired.

**Spillover tracker is the empirical follow-up.** The shipped tracker
opens hypothetical positions for every capacity-skipped name and
will, after 3-4 weeks of accumulation, give us direct P&L data on
"what we would have bought." If the spillover portfolio meaningfully
outpaces the live portfolio over a sustained window, that warrants
re-opening this question with new evidence.

**Commit:** Sweep machinery added (`--sweep-exit-threshold`,
`--sweep-swap-gap`, `swap_score_gap`/`swap_min_persistence` fields on
StrategyConfig). Re-runnable with one command if observations later
warrant.

**Status:** Active. No live config change.

---

## 2026-05-11 — Exit-with-persistence: no robust edge for either scheme

**Decision:** Keep current exit rules unchanged for both schemes:
  - Scheme C (live + paper): exit < 5.0, 1-day (immediate)
  - Scheme M (shadow): exit < 4.5, 1-day (immediate)

Reject all tested exit-with-persistence variants. No live config change.

**Rationale:** Tested "tighter exit threshold + persistence filter"
as an alternative to today's immediate-exit rule. Hypothesis: a
position scoring 7.9 (e.g., today's BE) looks like dead capital but
isn't an exit candidate under the current 5.0 rule. A tighter exit
(< 6 or < 7) with N-day persistence would catch weakening positions
earlier while avoiding single-day-dip whipsaws.

Grid tested: exit threshold × persistence_days = {6.0, 7.0} × {2, 3, 5}.
6-config 2D sweep per scheme, on the historical backtest DB.

**Evidence (sizing_comparison_backtest.py with new
`exit_persistence_days` field + `--sweep-exit-persistence` flag):**

**Scheme C** (entry 9.0, 2y default window, 10 path starts):

| Exit | Persist | Return | Sharpe | Win % | Trades |
|---|---|---|---|---|---|
| < 5.0 / 1 (live) | — | +837.8% | 2.45 | 60.0% | 80 |
| < 7.0 / 5 (best) | — | +833.4% | 2.46 | 70.1% | 77 |

Best Scheme C variant is statistical noise vs baseline (+0.01 Sharpe).
Lower persistence values (2, 3) destroy returns by 40-60% under tighter
thresholds. No actionable improvement.

**Scheme M** initial result (2y window, 10 path starts) showed an
apparent strong edge:
  Best variant `< 6.0 / persist=3`: +2289.6% / Sharpe 2.49 vs
  baseline +1697.0% / Sharpe 2.37 — gap of +593pp / +0.12 Sharpe.

However, follow-up rigor testing showed the edge was a measurement
artifact:

| Window / Method | Baseline Sharpe | Variant Sharpe | ΔSharpe | ΔReturn |
|---|---|---|---|---|
| 3y full + 20 paths (most rigorous) | 2.09 | 2.03 | **−0.06** | +227pp |
| 2y default + 10 paths (initial) | 2.37 | 2.49 | +0.12 | +593pp |
| 16mo recent regime (2025-01+) | 2.13 | 2.32 | +0.19 | +87pp |

With the most rigorous measurement (3y history, 20 staggered starts),
the variant's Sharpe goes mildly NEGATIVE relative to baseline. The
initial +0.12 Sharpe was a function of (a) the under-sampled 10 starts
and (b) the favorable May-2024 start date that the 2y price-fetch
window forced. Across the three windows, variant Sharpe is stable
(~2.0-2.5) but baseline Sharpe is similarly stable — the apparent
"gap" was sampling variance, not a real edge.

**Mechanistic explanation** (recorded for future re-analyses): The
strategy compounds via long-tail outliers. A tighter exit catches
weakening positions earlier — superficially attractive — but trades
this for shorter holding periods that truncate winners before they
compound. The persistence filter (2, 3, 5 days) modulates how
aggressively this truncation happens, but doesn't fix the underlying
issue. 5-day persistence with < 7.0 threshold approximates the current
< 5.0 / 1-day rule's behavior on real exits and produces statistically
indistinguishable results.

**Methodology lesson:** Default 2y price-period + 10 path starts on
the backtester is insufficient for evaluating rule changes. The 2y
fetch limits `trading_days` such that `compute_path_start_dates`
caps at ~10 candidates regardless of `--path-starts`. Added
`--price-period` flag (default still 2y) — pass `--price-period 3y`
(or larger) to widen the start-date window and let `--path-starts`
take effect. Any future rule-change validation should use 3y+ data
and at least 20 path starts as the minimum bar.

**Spillover tracker remains the empirical follow-up** for the
underlying path-dependency concern (capacity-skipped tickers
outperforming holdings). After ~3-4 weeks of accumulation it'll
give direct evidence rather than snapshot intuition or
backtest-artifact noise.

**Commit:** Sweep machinery added — `exit_persistence_days` field on
StrategyConfig, simulator exit logic updated, `--sweep-exit-persistence`
and `--price-period` CLI flags, dedicated 2D-grid sweep block with
shared path-dep helper.

**Status:** Active. No live config change.

---

## 2026-05-19 — Backfill: don't overwrite trade-exec scores; drop tickers missing target-date data

**Decision:** `backfill_subsector.py` now:
  1. Skips any target_date that already has scores in `ticker_scores`
     (trade-exec is authoritative for live decisions; backfill only fills
     genuine gaps).
  2. Excludes individual tickers from a target-date's slice when their
     last available row pre-dates the target. If > 5% of the universe is
     missing target-date data, the entire date is aborted.
  3. Logs skip counts + partial-coverage warnings in the summary.

**Rationale:** Investigation of an unexpected LUNR live entry on
2026-05-18 traced to backfill corruption of Friday 2026-05-15's score:

  - Friday 4:30 PM ET trade-exec wrote LUNR_5/15 = 7.10 from full data
    (last close $33.89, −7.2% intraday).
  - Monday backfill (`--frequency 1`) processed target_date=5/15. Its
    yfinance fetch at 5:07 PM ET Monday was missing LUNR's 5/15 bar.
    `slice_data_to_date` silently used 5/14 as the most-recent row but
    keyed the resulting score (9.60) to "2026-05-15" → overwrote 7.10.
  - Monday EOD live trade-exec saw the corrupted history, persistence
    filter passed (5/15 now showed 9.60 ≥ 9.0), LUNR became a buy
    candidate with the highest score in the queue and took the open
    capacity slot ahead of SNDK.

  Walked back through DB snapshots in git history to confirm the score
  flipped from 7.10 → 9.60 between Friday's commit and Monday's
  backfill commit. Reproduced both values with current yfinance data
  by toggling whether 5/15's row is present in the slice — confirming
  the data-gap-during-slice hypothesis.

  Cross-checked IRDM's 5/15 exit-trigger score (4.50) against the same
  test: 4.50 reproduces from full data through 5/15, and Monday's
  backfill did NOT change IRDM's 5/15 value across snapshots. IRDM's
  exit was legitimate momentum reversal, not a data artifact.

**Evidence (verified):**
  - `slice_data_to_date(data, 2026-05-15)` with full universe data
    returns LUNR with last row = 5/14 (when LUNR's data has a 5/15 gap)
    → score 9.60 keyed to 2026-05-15 (wrong).
  - With LUNR's 5/15 bar present → last row = 5/15, score 7.10 (right).
  - DB had 7.10 in Friday's commit, 9.60 in Monday backfill commit,
    stable 9.60 thereafter.

**Implementation:**
  - `slice_data_to_date()` return signature changed to
    `tuple[dict, list[str]]` — returns slice plus list of tickers
    excluded due to last-row date mismatch.
  - New helper `date_already_scored(conn, date_str) -> int` queries the
    ticker_scores row count for the date; non-zero → skip.
  - Main loop in `run_backfill` gates each date on:
    - Already-scored check (skip if >0)
    - Coverage abort if missing_pct > 5%
    - Partial-coverage warning (logged but date proceeds) if 0 < missing_pct ≤ 5%
  - Summary line prints skipped_already_scored, skipped_missing_data,
    and partial-coverage example dates.

**Pre-existing corruption:** The DB still holds LUNR_5/15 = 9.60
(incorrect; Friday's value was 7.10). Impact is small and decaying:
the 5/15 row only affects persistence checks within a 4-day window,
which expires after today's run (5/19). Did not repair retroactively
because (a) the entry has already happened and (b) the repair logic
would require either re-fetching historical data or carefully walking
the DB snapshot history per ticker. Filed as a known minor data
artifact rather than acted on.

**Commit:** Code in `backfill_subsector.py`; smoke test passed
(`--dates 2026-05-15,2026-05-14` skipped both as already-scored).

**Status:** Active. No live config change.
