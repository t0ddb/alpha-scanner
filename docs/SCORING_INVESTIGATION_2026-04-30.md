# Scoring System Investigation — Handoff

**Date:** 2026-04-30
**Status:** Investigation paused. Scheme H (sequence-aware scoring) designed but not built or tested. Production unchanged (still Scheme C).

This document captures a multi-hour rigorous investigation of whether the Scheme C scoring system (currently live) can be made more sensitive to differentiate top performers. Read this BEFORE making any scoring changes.

---

## TL;DR

1. **Production is still Scheme C.** Live trading account is unaffected by anything in this investigation. No code changes were committed.
2. **The original concern was real**: Scheme C clusters multiple tickers at score=9.9 (typically 3-9 per day), and the portfolio has 80%+ path-dependency variance across staggered start dates because of how ties are filled.
3. **Five new scoring schemes were tested** (Schemes D, D-soft, D-mild/mod/med, E, F variants, G variants). **All except G1 (persistence=1) underperformed Scheme C significantly.** G1 was a marginal improvement.
4. **The single biggest unexplored finding is sequence-aware scoring (Scheme H)** — the order in which indicators fire predicts forward returns dramatically. The same indicator (RS) at different sequence positions has +20pp difference in win rate. Not yet built.
5. **Local artifacts exist** (multiple DB copies, audit scripts, logs) but production is untouched. See "Local State" section below.

---

## Why this investigation started

User observation on dashboard: 9 tickers tied at score=9.9. With 12-position portfolio cap, fill order among ties is essentially alphabetical (Python's stable sort on dict iteration). The system makes important decisions based on alphabetical luck.

User's concern (validated by data):
> "With only 12 slots in our portfolio, my performance will really just depend on timing luck. Even if we have a better tie breaker."

This is genuine: 15-start path dependency for Scheme C @ threshold 9.0:
- Mean return: +613% (matches the original audit)
- **Path std: 85.6%** (was 11.9% with only 5 starts — original Scheme C audit under-tested this)
- **Path range: 268.6%** — runs varied from ~+450% to ~+900% depending on start date
- Coefficient of variation: 14.0%

The original Scheme C audit reported path std of 11.9% with 5 starts — that was an under-sampling artifact. Real path-dependency is much higher than we thought.

---

## Production state (unchanged through this investigation)

- **Scheme C deployed 2026-04-24**: weights RS 3.0, Ichimoku 2.0, ROC 1.5, Dual-TF RS 2.5, ATR 0.5, HL 0.5, CMF 0.0
- **Entry threshold**: 9.0
- **Persistence filter**: 3 prior trading days at ≥9.0
- **Live launched 2026-04-28** with $5k, paper continues with ~$112k equity
- Both paper + live workflows scheduled at 4:30 PM ET M-F
- Daily automation working correctly (commit ee0ce63 schedule enabled)

---

## Schemes tested — full results table

All backtests use 12 positions × 8.3% sizing, 20% stop, 3-day persistence (unless noted), against the production DB rescored with the scheme's weights. Returns are over the ~13-month DB window (2025-03-25 → 2026-04-23).

| Scheme | Description | Threshold | Return | Sharpe | Max DD | Win % | Trades | Path std (15-start) |
|---|---|---|---|---|---|---|---|---|
| **C (current production)** | Binary thresholds | 9.0 | **+613%** | **2.21** | **−20.3%** | **59.5%** | 79 | **85.6%** |
| C+5-start | (same scheme) | 9.0 | +598% | 2.21 | −21.2% | 50.7% | 75 | 11.9% (sampling artifact) |
| **D-orig** | Aggressive gradient (RS 90→2.0, Ich all-or-nothing 0/0/0/2, etc.) | 8.0 | +299% | 1.76 | −24.5% | 48.3% | 89 | 19.5% |
| D-orig | (same) | 9.0 | +130% | 1.25 | −28.7% | 47.8% | 46 | 4.5% |
| D-mild | RS 90→2.85, less aggressive | 8.5 | +308% | 1.79 | −23.2% | 56.2% | 89 | — |
| D-mod | RS 90→2.70 | 8.5 | +310% | 1.76 | −22.6% | 56.8% | 88 | — |
| D-med | RS 90→2.50 | 8.5 | +269% | 1.67 | −22.8% | 55.7% | 88 | — |
| **D-soft** | All gradients dialed back, two-tier Ichimoku | 8.5 | +379% | 1.94 | −23.7% | 56.7% | 97 | — |
| **E** | Non-monotonic RS curve (peaks at 90-91, dips at 94-97, recovers 99+) + Scheme C binaries | 9.0 | +466% | 1.96 | −23.4% | 51.1% | 90 | — |
| F1 | Scheme C + persistence=10 | 9.0 | +339% | 1.77 | −24.5% | 53.4% | 58 | — |
| F2 | Scheme C + persistence=20 | 9.0 | +102% | 1.06 | −31.8% | 30.6% | 36 | — |
| F3 | Scheme E + persistence=10 | 9.0 | +321% | 1.75 | −24.0% | 54.8% | 62 | — |
| G0 | Scheme C + persistence=0 | 9.0 | +621% | 2.17 | −23.8% | 57.1% | 84 | — |
| **G1** | Scheme C + persistence=1 | 9.0 | **+636.7%** | 2.18 | −22.0% | 59.5% | 84 | **80.4%** |
| G2 | Scheme C + persistence=2 | 9.0 | +566% | 2.08 | −21.8% | 58.8% | 85 | — |
| H (proposed, NOT BUILT) | Scheme C + sequence-aware position bonuses | TBD | TBD | TBD | — | — | — | — |

**Key takeaway: every scheme except G1 underperforms or matches Scheme C.** G1 is a marginal improvement (+24pp return, slightly lower path std, very slightly lower Sharpe).

---

## Investigations that did NOT pan out

### Scheme D (gradient curves)
**Hypothesis**: Convert binary thresholds to smooth gradient curves so 99-percentile RS earns more than 90-percentile.
**Result**: All variants severely underperformed (best D-soft at +379% vs Scheme C +613%).
**Why it failed**:
- The gradient framework biases toward "exceptional/already-extended" stocks (RS 99+ has +29.5% mean / 51.6% win — fat-tailed)
- Scheme C's binary thresholds catch breakouts EARLIER in their phase (RS 90+ has +17.4% mean / 51% win — more consistent compounding)
- "All-or-nothing Ichimoku" rule penalized 2/3 stocks that ARE legit Scheme C entries (above + bullish, no tenkan/kijun yet)
- Cross-validation with Spearman ρ said all curve families were equivalent, but Spearman is rank-based — portfolio P&L depends on the EXACT cutoff selection, which differs between schemes

### Scheme F (extended persistence)
**Hypothesis**: Firing-streak data showed 20+ day setups outperform 1-3 day setups, so longer persistence = better entries.
**Result**: All variants severely underperformed (F1 +339%, F2 +102%).
**Why it failed**:
- Confound: the streak forward-return measurement was FROM the current date. 20+ day streak stocks had already gained massively in those 20 days; the forward measurement is "what's left of the run."
- In a portfolio context, longer persistence means we ENTER LATE. We pay the price of missing the early move.
- The 3-day persistence is empirically near-optimal for catching breakouts in formation.

### Curve calibration via cross-validation
**Hypothesis**: Different curve families (piecewise-linear, power, isotonic) might have different OOS predictiveness.
**Result**: For continuous indicators (RS, Dual-TF, ROC), every curve family produced **identical OOS Spearman ρ** because rank order is preserved by any monotonic transform.
**Conclusion**: Curve family choice is irrelevant for rank-based prediction. Only matters for distribution shape and interpretability. Piecewise-linear is the most readable.

---

## Investigations that DID find signal

### G1 — persistence=1 is marginally better than persistence=3
- Same Sharpe, slightly better mean, slightly lower path std
- 15-start path-dep: G1 +636.7% / Sharpe 2.18 / path std 80.4% vs Scheme C +613% / 2.21 / 85.6%
- Score-streak data showed: day-1 of crossing 9.0 has +3.17% median forward return; day 3-5 has near-zero or negative; day 11-20 has +4.46%
- The 3-day persistence enters in the trough of forward returns
- Could ship G1 alone for a small consistent improvement

### Scheme E — non-monotonic RS curve has empirical support
RS percentile bucket data, fine-grained 2-pt buckets:
- 88-90: median +3.07%, win 53.5%
- **90-92: median +4.45%, win 55.3% — PEAK**
- 92-94: median +1.09%, win 51.6%
- **94-96: median −1.14%, win 48.0% — DIP**
- 96-98: median −1.53%, win 48.6%
- **98-100: median +3.15%, win 52.1% — recovery**

The 95-99 cohort is THE WORST performers in the high-RS range — extended stocks prone to mean reversion. The 99+ tail recovers. Scheme E rewards this empirical shape (peak at 90, dip at 94-97, recovery at 99) but underperformed at portfolio level (+466% vs +613%) — likely because it still mucks with the threshold-passing dynamics.

### Score-streak analysis (production-equivalent thresholds)
Using SCORE-based streaks at threshold 9.0:

| Days at score ≥9.0 | n | Median 63d xSPY | Win % |
|---|---|---|---|
| 1 day (just crossed) | 1,177 | **+3.17%** | **53.7%** |
| 2 days | 782 | +1.94% | 53.2% |
| 3 days (= production persistence) | 589 | +0.41% | 50.3% |
| **4-5 days** | 877 | **−0.42%** | **49.6%** |
| 6-10 days | 1,373 | +0.55% | 50.6% |
| **11-20 days** | 1,252 | **+4.46%** | **55.0%** |
| 21-40 days | 692 | −0.15% | 49.7% |
| 41+ days | 136 | −9.90% | 44.1% |

**Two sweet spots**: day 1 (just crossed) and days 11-20 (established). Production enters at day 3-4 — in the WORST zone. Persistence variants tested didn't fully exploit this because longer persistence misses the early move (Scheme F failed).

### Sequence pattern analysis — THE BIGGEST UNTAPPED FINDING

For each (date, ticker) row at score ≥9.0, computed each indicator's current consecutive-firing streak length, then the ORDER (longest streak = first to fire, shortest = last to fire).

#### First-firer (longest streak indicator) determines forward returns dramatically:

| First-firer | n | Median 63d xSPY | Win % |
|---|---|---|---|
| **Higher Lows** | 85 | **+17.72%** | **68.2%** |
| **CMF** | 262 | **+12.37%** | **59.2%** |
| **ROC** | 971 | **+10.10%** | **58.8%** |
| **Ichimoku** | 970 | **+6.43%** | **60.0%** |
| **RS** | **4,364** (most common!) | **−1.47%** | **48.4%** |
| ATR | 226 | **−15.66%** | 38.9% |
| Dual-TF RS | 0 | (structurally never first) | — |

When RS is FIRST to fire (i.e., RS is the longest-standing indicator in the setup), forward returns are NEGATIVE on median. **63% of all signals have RS-first** — these are essentially coin flips.
When non-RS technicals fire first (HL, CMF, ROC, Ich), forward returns are strongly positive.

**Mechanistic intuition**: RS-first means the stock had a big run BEFORE other indicators confirmed. By entry time, it's extended. Non-RS-first means the technical pattern was forming first; RS catches up later as price strengthens. The latter catches breakouts in formation.

Dual-TF RS is structurally NEVER first because its trigger requires higher percentile thresholds than rs_fired's 50th-pctl. So it always lights up after RS in any uptrend.

#### Position-by-rank: each indicator has an OPTIMAL position

| Indicator | BEST rank | WORST rank | Δ win rate |
|---|---|---|---|
| **RS** | rank 4+ (late) — 68% win | rank 1 (first) — 48% win | **+20 pp** |
| **Ichimoku** | rank 3 — 61% | rank 6 — 42% | +19 pp |
| **Higher Lows** | rank 1 — 68% | rank 7 — 47% | +21 pp |
| **CMF** | rank 1 — 59% | rank 3 — 46% | +13 pp |
| **ROC** | rank 1 — 59% | rank 2 — 48% | +11 pp |
| **ATR** | rank 2 — 62% | rank 1 — **39% win** | **+23 pp** |
| **Dual-TF RS** | rank 7 — 59% | rank 2 — 50% | +9 pp |

**The "good" pattern**: technicals (Ichimoku, HL, CMF, ROC) fire FIRST → RS catches up LATER → Dual-TF confirms LAST.
**The "bad" pattern**: RS first (already extended) → ATR first (volatility expanding before move).

#### Top sequence pairs (first → last):

| First → Last | n | Median | **Win %** |
|---|---|---|---|
| **ROC → Dual-TF RS** | 135 | **+31.22%** | **84.4%** |
| Higher Lows → Ichimoku | 55 | +48.39% | 80.0% |
| **CMF → Higher Lows** | 117 | +20.83% | 70.1% |
| ATR → Dual-TF RS | 57 | +22.58% | 73.7% |
| Ichimoku → Higher Lows | 302 | +10.82% | 63.9% |
| Ichimoku → CMF | 138 | +6.73% | 61.6% |
| **ATR → Ichimoku** | 55 | **−36.57%** | **14.5%** ← AVOID |
| ATR → Higher Lows | 80 | −22.02% | 31.2% |

**These are massive edge differences** that Scheme C completely ignores.

---

## Scheme H (proposed, NOT BUILT)

### Design

Add a sequence-aware bonus on top of Scheme C base scoring:

```python
def sequence_pts(ranks):  # dict of indicator → rank (1=first to fire)
    s = 0.0
    # RS position
    if ranks.get('rs') == 1:        s -= 0.3   # RS-first = late entry, bad
    if ranks.get('rs', 0) >= 4:     s += 0.3   # RS-late = real breakout
    # ATR position (most extreme delta)
    if ranks.get('atr') == 1:       s -= 0.4   # ATR-first = avoid (39% win)
    if ranks.get('atr') == 2:       s += 0.2   # ATR-2nd = great
    # Early non-RS firers
    if ranks.get('hl', 99) <= 2:    s += 0.3
    if ranks.get('cmf') == 1:       s += 0.2
    if ranks.get('roc') == 1:       s += 0.2
    if ranks.get('ich') in (1, 3):  s += 0.15
    # Late Dual-TF
    if ranks.get('dtf', 0) >= 6:    s += 0.2
    return max(-0.5, min(s, 1.0))
```

Scheme H score = Scheme C base + sequence_pts. Total max becomes 11.0; threshold likely 9.5-10.0 for matched selectivity.

### Open questions for Scheme H

1. **Bonus magnitude**: −0.5 to +1.0 may be too aggressive. Could overfit to historical patterns. Consider conservative variant ±0.3.
2. **Threshold calibration**: with extra range, threshold needs to move. Test 9.0, 9.5, 10.0.
3. **Tie-breaker only variant**: instead of changing scores, use sequence pattern as a SECONDARY SORT. Same Scheme C universe of threshold-passers, but better-sequence stocks fill slots first. Less invasive, same intent.
4. **Filter variant**: skip entries with bad sequences (RS-first, ATR-first, ATR→Ichimoku pair). Most aggressive — could meaningfully reduce signal count.
5. **Validation**: needs portfolio backtest with 15-start path-dep at multiple thresholds. ~30 min compute.

### Why we paused before building Scheme H
The user wants a fresh thread / context window. The investigation has accumulated significant context that should be summarized before continuing.

---

## Key empirical findings (numbered for reference)

1. **Scheme C has high path-dependency** (15-start path std 85.6%, range 268.6%). Original audit's 11.9% std was from only 5 starts.
2. **All gradient curve families are equivalent on rank-based metrics.** Spearman ρ identical for piecewise-linear, power, isotonic on continuous indicators.
3. **RS curve is non-monotonic.** Peak at 90-91 (median +5.6%, win 56%), dip at 94-97 (negative median, 47% win), recovery at 98-100 (+3% median, 52% win).
4. **3-day persistence enters at the WORST point.** Days 3-5 have median 0% to −0.4%. Day 1 (+3.17% median) and days 11-20 (+4.46% median) are the sweet spots.
5. **The OUTSIZED finding: indicator firing SEQUENCE predicts forward returns dramatically.** RS-first (63% of signals) has 48% win rate. Non-RS-first has 58-68% win rate. This is +10 to +20 pp edge that Scheme C ignores entirely.
6. **Each indicator has an optimal sequence position.** ATR-first is catastrophic (39% win); ATR-2nd is great (62% win). Same indicator, different positions, +23pp delta.
7. **CMF is more useful than Scheme C suggests.** Dropped from scoring due to negative incremental edge as a binary, but as a FIRST-firer it's a strong signal (59% win, +12% median).
8. **Top sequence pairs have 70-87% win rates.** ROC→Dual-TF: 84%. HL→Ichimoku: 80%. CMF→HL: 70%. ATR→Ichimoku: 14% (AVOID).
9. **Persistence=1 (G1) is marginally better than persistence=3.** +24pp return, 5pp lower path std, same Sharpe (2.18 vs 2.21). Consider shipping alone or with Scheme H.

---

## Local state (uncommitted)

### `indicators.py` — currently has SCHEME D code
Scheme D weights/anchors/scoring logic was implemented in `indicators.py` for testing but **was never committed**. The file on disk is in Scheme D state. **Do NOT push from this state**. Either:
- `git checkout indicators.py` to revert to Scheme C (last committed version)
- OR write the new Scheme H from scratch on top of committed Scheme C

### Scheme D databases (uncommitted, gitignored)
Multiple DB copies were created for backtesting variants:
- `breakout_tracker_schemeD.db`, `_schemeD_mild.db`, `_schemeD_mod.db`, `_schemeD_med.db`, `_schemeD_soft.db`
- `breakout_tracker_schemeE.db`
- These are large (12 MB each) and gitignored. Safe to delete to free disk.
- Backup of pre-investigation production DB: `breakout_tracker.db.pre-schemeC.bak` (also gitignored)

### Audit scripts created during this investigation (uncommitted)
All in `/Users/toddbruschwein/Claude-Workspace/breakout-tracker/`:

| Script | Purpose |
|---|---|
| `audit_curve_calibration.py` | OOS time-series CV of curve families per indicator |
| `audit_gradient_buckets.py` | Bucket analysis (mean/median fwd ret per indicator value) |
| `audit_rs_full_range.py` | Fine-grained 2-pt RS percentile bucket analysis |
| `audit_rs_fine_grained.py` | Earlier version (90+ only) |
| `audit_scheme_d_distribution.py` | Score-distribution comparison Scheme C vs D |
| `audit_rescore_db_scheme_d.py` | Rescore DB with full Scheme D |
| `audit_rescore_scheme_d_variant.py` | Rescore with parametrized RS anchors (mild/mod/med) |
| `audit_rescore_scheme_d_soft.py` | Rescore with all-conservative gradients |
| `audit_rescore_scheme_e.py` | Rescore with non-monotonic RS curve |
| `audit_topN_comparison.py` | Top-N selection comparison Scheme C vs D |
| `audit_firing_sequence.py` | Initial firing-count streak analysis (used `>=5 firing`, somewhat arbitrary) |
| `audit_score_streaks.py` | **Score-based streak analysis at production thresholds (the right one)** |
| `audit_indicator_sequence.py` | First-firer / last-firer analysis (single-position) |
| `audit_full_sequence.py` | Full sequence ordering analysis |
| `audit_sequence_position.py` | Position-by-rank per indicator (the cleanest sequence view) |

### Audit logs in `backtest_results/` (gitignored)
Many `audit_portfolio_scheme*.log` files with raw backtest output. Useful for verification.

### Production code change (UNCOMMITTED) — `sizing_comparison_backtest.py`
A single env-var hook was added to support `PERSISTENCE_DAYS_OVERRIDE`:
```python
_PERSISTENCE_OVERRIDE = os.environ.get("PERSISTENCE_DAYS_OVERRIDE")
PERSISTENCE_DAYS = int(_PERSISTENCE_OVERRIDE) if _PERSISTENCE_OVERRIDE else 3
```
And all hardcoded `persistence_days=3` were replaced with `persistence_days=PERSISTENCE_DAYS`.

**This change is benign** (purely additive — defaults to 3 if env unset) but is uncommitted. Could be committed safely if a future investigation needs persistence experimentation.

### Background tasks
None currently running. Last task `b5hcbu0ik` (Scheme G1 15-start path-dep) completed earlier.

---

## Recommended next steps for fresh thread

The user wants to continue the Scheme H investigation. Concrete next actions:

1. **Read this doc.** Then `git checkout indicators.py ticker_config.yaml` to revert any local Scheme D pollution.

2. **Decide on Scheme H variant to test**:
   - **H1 (additive bonus)**: most direct application of sequence findings. Range −0.5 to +1.0. Threshold likely 9.5.
   - **H2 (tie-breaker only)**: same Scheme C scoring, but secondary sort by sequence quality. Less invasive.
   - **H3 (filter)**: skip entries with bad sequences (RS-first + bad last-firer combinations). Reduces signal count.

3. **Build the chosen variant**:
   - Compute per-indicator streaks per (date, ticker) cell
   - Compute rank assignments
   - Compute sequence_pts using the formula above (or chosen alternative)
   - Add to Scheme C base score
   - Rescore production DB copy (`audit_rescore_*.py` template exists)
   - Run portfolio backtest at multiple thresholds with `--no-path-test` first
   - If promising, run with `--path-starts 15` for path-dep validation

4. **Compare to baseline + G1**:
   - Scheme C @ 9.0: +613% / Sharpe 2.21 / path std 85.6%
   - Scheme G1 @ 9.0 (persistence=1): +637% / Sharpe 2.18 / path std 80.4%
   - Scheme H @ ?: ?

5. **Decision criteria for shipping Scheme H**:
   - Return ≥ Scheme C
   - Sharpe ≥ 2.0
   - Path std meaningfully lower (e.g., < 50%)
   - Win rate ≥ 55%

6. **Open question worth revisiting**: combine winning elements? E.g., Scheme E's non-monotonic RS curve + Scheme G1's persistence=1 + Scheme H's sequence bonus. Risk: combining changes makes it hard to isolate which actually helps.

7. **If Scheme H also fails**: accept that the 12-position cap is the structural source of path-dependency, and the path forward is **risk management** (e.g., position sizing as a function of conviction) rather than scoring sensitivity.

---

## Reference: Scheme C scoring (for any new design)

```python
INDICATOR_WEIGHTS = {
    "relative_strength": 3.0,    # gradient: 50→0.6, 60→1.2, 70→1.8, 80→2.4, 90→3.0
    "ichimoku_cloud":    2.0,    # binary: above_cloud AND cloud_bullish
    "dual_tf_rs":        2.5,    # binary: cond_a OR cond_b (see indicators.py)
    "roc":               1.5,    # binary: 21d % change > 5%
    "atr_expansion":     0.5,    # binary: 14d ATR in top 20% of 50d range
    "higher_lows":       0.5,    # gradient: 2→0.125, 3→0.25, 4→0.375, 5→0.5
    "cmf":               0.0,    # DROPPED in Scheme C audit (negative incremental edge)
}
# Total max: 10.0
# Production threshold: 9.0
# Persistence: 3 prior trading days at ≥9.0
```

---

## Other things to know

- **The audit parquet** (`backtest_results/audit_dataset.parquet`) covers 2023-05-15 → 2026-04-30, 127k rows, 174 tickers. Built with `--include-recent` flag to score through today (forward-return columns NaN where insufficient future data). Regeneratable via `python3 audit_build_dataset.py --years 3 --frequency 1 --include-recent`.

- **Score column in parquet**: was Scheme D when this analysis ran (because `indicators.py` had Scheme D code). Several scripts (`audit_topN_comparison.py`, `audit_score_streaks.py`, `audit_indicator_sequence.py`, etc.) reconstruct Scheme C scores explicitly from raw inputs to avoid this confound. Use those scripts as templates if computing scores from parquet in a new analysis.

- **The `audit_dataset_scheme_d_scored.parquet`** is an artifact from one early test — has Scheme D scores explicitly. Can ignore.

- **GitHub repo state**: All commits up to and including 2026-04-28 daily automation are pushed. Nothing from this investigation is pushed. Last pushed commit: `06af7b9 Automated LIVE trade execution 2026-04-28`.

- **Production live trading** is unaffected and continues normally on its scheduled 4:30 PM ET runs.
