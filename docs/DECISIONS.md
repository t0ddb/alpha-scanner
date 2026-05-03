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
