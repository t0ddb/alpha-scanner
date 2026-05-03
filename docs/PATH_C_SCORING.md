# Path C Scoring Spec (Scheme I+ Layer 1 + Layer 2)

**Status:** **DEPLOYED as shadow tracker** (2026-05-01). Not in production
trading; runs in parallel to live Scheme C as `shadow_pathc.py`.

> Scheme C remains the production scoring system (entry 9.0). Path C is
> a parallel hypothetical portfolio that catches different (often
> higher-quality per trade) signals. See `DECISIONS.md` for the full
> investigation history that led here.

**Source data calibration:** `backtest_results/audit_dataset.parquet`,
regime-filtered to ≥ 2025-05-01.
**Baseline win rate (current regime, score ≥ 9.0):** 74.2%.
**Live config (validated via sweeps):** entry 7.5 / exit 4.5 / persist 3
/ stop 20% / cap 12 / sizing 8.33% / starting equity $100k synthetic.

---

## Philosophy

1. **Regime-tuned.** Scoring is calibrated to the current AI-infrastructure bull regime (2025-Q2 onward, n=2,450 score≥9.0 signals). When the regime fades, recalibrate.

2. **Empirical bucketing, not assumed monotonicity.** Each indicator's points reflect actual win-rate-by-bucket data. Where the data shows non-monotonicity (e.g., RS dip at 96-98), the scoring follows.

3. **Three-layer architecture, fully additive:**
   - **Layer 1 — Indicator scoring** (positive-only buckets, total 0-10): the base score
   - **Layer 2 — Sequence overlay** (additive bonuses + penalties): score adjustment based on indicator firing patterns
   - **Layer 3 — Threshold gate**: single threshold determined empirically by backtest sweep
   
   No hard filters or threshold overrides — everything expresses through the score.

4. **Conservative on interaction-confounded indicators.** HL and CMF showed strong isolated curves but didn't survive interaction control; they get smaller weights and the v1 backtest will tell us whether to invest more in them later.

5. **Joint scoring deferred.** v1 tests Layer 1 + Layer 2 in isolation. Joint scoring (e.g., ICH × ATR, DTF cross-timeframe) added in v1.5 only if v1 outperforms Scheme C and we want incremental refinement.

---

## Layer 1 — Indicator Scoring (Total 0-10)

Weights derived from empirical peak-Δ in current regime, modulated by interaction-confound confidence (lower confidence → lower weight). Positive-only — no bucket awards negative points.

| Indicator | Max weight | Confidence basis |
|---|---|---|
| RS | **2.30** | Strong empirical, mostly survives interaction control |
| DTF_63d | **1.75** | TOP SHAP importance in regime; survives interaction control |
| ICH composite | **1.55** | Trust-isolated verdict; clean monotonic in regime |
| DTF_126d | **1.50** | Strong empirical peak at 85-90 pctl |
| ROC | **1.00** | Mixed verdict; clear peak zone but interaction-affected |
| CMF | **0.85** | Re-included from 0; isolated curve clean but interaction-confounded |
| ATR | **0.65** | Mostly-trust verdict; direction now matches regime (high = good) |
| HL | **0.40** | Use-PDP verdict (interaction-confounded); kept low |
| **TOTAL MAX** | **10.00** | |

### RS (max 2.30) — non-monotonic with 88-94 peak and 96-98 dip

Coarser buckets below 80 (where shape is flat-ish), finer 2pp buckets in critical 88-100 zone where the empirical dip lives.

| RS percentile | Pts | Empirical Δ |
|---|---|---|
| < 50 | 0.00 | -9.2pp |
| [50, 60) | 0.85 | +3-10pp |
| [60, 70) | 1.00 | +6-9pp |
| [70, 80) | 1.20 | +8-10pp |
| [80, 85) | 1.50 | +11pp |
| [85, 88) | 1.30 | +10pp |
| **[88, 90)** | **2.30** | **+17.6pp PEAK** |
| [90, 92) | 2.00 | +15.4pp |
| [92, 94) | 2.00 | +15.3pp |
| [94, 96) | 1.60 | +12.1pp |
| **[96, 98)** | **0.20** | **+1.7pp DIP** |
| [98, 100] | 0.60 | +4.6pp |

### Higher Lows (max 0.40)

| HL count | Pts | Empirical Δ |
|---|---|---|
| 0-1 | 0.00 | -2 to -4pp |
| 2 | 0.05 | +1.2pp |
| 3 | 0.30 | +5.6pp |
| 4+ | 0.40 | +7.3pp |

### Ichimoku composite (max 1.55)

| ICH composite | Pts | Empirical Δ |
|---|---|---|
| 0/3 | 0.00 | -21.2pp |
| 1/3 | 0.00 | -7.6pp |
| 2/3 | 0.05 | +0.8pp |
| **3/3** | **1.55** | **+11.4pp** |

### Rate of Change (max 1.00)

Sweet spot 15-50%, tail-off above 50%, zero above 75%.

| ROC % | Pts | Empirical Δ |
|---|---|---|
| ≤ 5 | 0.00 | -4 to -15pp |
| [5, 7.5) | 0.05 | +0.9pp |
| [7.5, 10) | 0.50 | +5.0pp |
| [10, 15) | 0.75 | +7.8pp |
| **[15, 50)** | **1.00** | **+8-10pp PEAK** |
| [50, 75) | 0.35 | +3.5pp |
| [75, 100) | 0.00 | -0pp |
| ≥ 100 | 0.00 | -13.3pp |

### Chaikin Money Flow (max 0.85)

| CMF | Pts | Empirical Δ |
|---|---|---|
| < 0.05 | 0.00 | -2 to -11pp |
| [0.05, 0.20) | 0.25 | +3-4pp |
| [0.20, 0.30) | 0.50 | +7.5pp |
| ≥ 0.30 | 0.85 | +12.5pp |

### ATR Expansion (max 0.65) — DIRECTION MATCHES CURRENT REGIME

| ATR percentile | Pts | Empirical Δ |
|---|---|---|
| < 50 | 0.00 | -2.5 to -4.7pp |
| [50, 60) | 0.00 | -0.5pp |
| [60, 70) | 0.30 | +2.8pp |
| [70, 80) | 0.40 | +3.4pp |
| [80, 90) | 0.45 | +2-5pp |
| [90, 95) | 0.65 | +5.5pp |
| [95, 100] | 0.50 | +4.3pp |

### Dual-TF — 126d RS percentile (max 1.50)

| 126d pctl | Pts | Empirical Δ |
|---|---|---|
| < 50 | 0.00 | -7.6pp |
| [50, 65) | 0.20 | +2-3pp |
| [65, 75) | 0.75 | +7-10pp |
| [75, 85) | 0.75 | +7-9pp |
| **[85, 90)** | **1.50** | **+15.5pp PEAK** |
| [90, 95) | 0.95 | +9.7pp |
| [95, 100] | 0.20 | +1.7pp |

### Dual-TF — 63d RS percentile (max 1.75)

| 63d pctl | Pts | Empirical Δ |
|---|---|---|
| < 50 | 0.00 | -9.2pp |
| [50, 60) | 1.00 | +7.1pp |
| [60, 70) | 1.10 | +7.7pp |
| [70, 80) | 1.30 | +9-10pp |
| [80, 90) | 1.60 | +11-12pp |
| **[90, 95)** | **1.75** | **+12.8pp PEAK** |
| [95, 100] | 0.70 | +5.3pp |

### Layer 1 score range

- **Min:** 0.00 (a stock with RS<50, ICH 0/3, ROC<5%, CMF<0.05, ATR<50, DTF_126d<50, DTF_63d<50, HL<2 — i.e., not even close to an entry candidate)
- **Max:** 10.00 (all bucket maxes hit simultaneously — theoretical only; in practice rare to hit max on RS+DTF_63d+ICH+DTF_126d+ROC+CMF+ATR+HL all at once)
- **Realistic candidate range:** ~5 to ~9 for stocks passing a reasonable threshold

---

## Layer 2 — Sequence Overlay (Additive Adjustments)

All adjustments are simple additive arithmetic on the Layer 1 base score. No hard filters, no threshold overrides — the score reflects all signal context.

### 2A — Penalties (subtract points for catastrophic patterns)

| ID | Pattern | Magnitude | Empirical Δ | Notes |
|---|---|---|---|---|
| **PEN1** | 3 indicators leading Ichimoku | **−3.0** | -21pp | Universally bad; n=137 in regime |
| **PEN2** | ROC → DTF as first-2 firers | **−3.0** | -26pp at score≥9.0 | High-score trap; n=85 |
| **PEN3** | ICH → DTF as first-2 firers | **−2.5** | -19pp | High-score trap; n=211 |
| **PEN4** | ICH → RS as first-2 firers | **−2.0** | -12pp | High-score trap; n=144 |
| **PEN5** | TREND → MOMENTUM type (N1) | **−0.5** | -6pp | Most common pattern (n=935); modest penalty |

A score-9.5 signal hitting PEN1 or PEN2 drops to 6.5 — well below any reasonable threshold. PEN5 is small enough to be a tie-breaker rather than a disqualifier.

### 2B — Fast-track bonuses (sub-threshold patterns with empirical edge)

| ID | Pattern | Magnitude | Empirical Δ (sub-thresh) | Notes |
|---|---|---|---|---|
| **FT1** | ATR is first-firer | **+1.5** | +6.6pp | n=47 sub-threshold |
| **FT2** | ROC → ICH first-2 firers | **+1.5** | +7.0pp | n=133 sub-threshold |
| **FT3** | ICH → CMF first-2 firers | **+1.5** | +7.9pp | n=168 sub-threshold |
| **FT4** | RS → ATR last-2 firers | **+1.0** | +6.7pp | n=21 sub-threshold (small) |

A sub-threshold signal at base 7.5 with FT2 fires at 9.0 — passes most reasonable thresholds.

### 2C — Tie-breaker bonuses (signals that work primarily at high base score)

| ID | Pattern | Magnitude | Empirical Δ | Notes |
|---|---|---|---|---|
| **B1** | MOMENTUM → TREND first-2 type | **+0.75** | +9pp | n=418 in regime |
| **B2** | DTF is first-firer | **+1.0** | +21pp at score≥9.0 | n=77; mostly tie-breaker (sub-thresh DTF-first underperforms) |
| **B3** | 1 OR 2 indicators leading Ich | **+0.75** | +9pp | n=764 (546+218) combined |
| **B4** | dtf → rs as last-2 firers | **+0.4** | +5.5pp | n=277 |
| **B5** | All-3 firing combo (B1 ∧ B3 ∧ B4) | **+0.5 EXTRA** | +21pp combined cell | n=46, 95.7% win |

The "All-3 firing" bonus is on TOP of the individual bonuses. So the all-3 cell stacks: B1 (+0.75) + B3 (+0.75) + B4 (+0.4) + B5 (+0.5) = **+2.4 total** for these 46 ultra-high-confidence signals.

### Layer 2 score range

- **Min adjustment:** −5.5 (multiple penalties stack: PEN1 + PEN2 + PEN5)
- **Max adjustment:** +5.4 (multiple positive patterns stack: FT3 + B1 + B3 + B4 + B5 = +1.5 + +0.75 + +0.75 + +0.4 + +0.5)
- **Typical:** ±1 to ±2 for most signals

### Layer 1 + Layer 2 final score range

- **Theoretical min:** ~−5
- **Theoretical max:** ~15
- **Realistic candidate range:** ~5 to ~12
- **Bottom 25% of candidates:** likely 5-7
- **Top 25% of candidates:** likely 9.5-12

---

## Layer 3 — Threshold Gate (DEFERRED to backtest)

The threshold will be determined empirically via threshold sweep:

```bash
python3 sizing_comparison_backtest.py \
    --scheme i_plus \
    --sweep-entry-threshold 7.0,7.5,8.0,8.5,9.0,9.5,10.0 \
    --start-date 2025-05-01
```

For each threshold candidate, measure:
- Total return
- Sharpe ratio
- Max drawdown
- Win rate
- # trades executed
- Path std dev (5+ start dates)

Choose the threshold that best balances return × stability × selectivity.

### Persistence

Keep at **3 prior trading days at ≥ entry threshold**, same as Scheme C. The score-streak findings from prior investigation were regime-dependent and didn't justify changing this.

---

## Final entry decision logic

```python
def compute_score_v2(signal_indicators):
    # Layer 1: positive-only bucket scoring
    layer_1 = (
        rs_pts(signal_indicators)          # 0 to 2.30
        + hl_pts(signal_indicators)        # 0 to 0.40
        + ich_pts(signal_indicators)       # 0 to 1.55
        + roc_pts(signal_indicators)       # 0 to 1.00
        + cmf_pts(signal_indicators)       # 0 to 0.85
        + atr_pts(signal_indicators)       # 0 to 0.65
        + dtf_126d_pts(signal_indicators)  # 0 to 1.50
        + dtf_63d_pts(signal_indicators)   # 0 to 1.75
    )  # total 0 to 10

    # Layer 2: additive sequence adjustments
    sequence_features = compute_sequence_features(signal_indicators)
    layer_2 = (
        # Penalties
        (-3.0 if matches(sequence_features, "3_led_ich") else 0)
        + (-3.0 if matches(sequence_features, "roc_dtf_first2") else 0)
        + (-2.5 if matches(sequence_features, "ich_dtf_first2") else 0)
        + (-2.0 if matches(sequence_features, "ich_rs_first2") else 0)
        + (-0.5 if matches(sequence_features, "trend_mom_type") else 0)
        # Fast-track bonuses
        + (+1.5 if matches(sequence_features, "atr_first") else 0)
        + (+1.5 if matches(sequence_features, "roc_ich_first2") else 0)
        + (+1.5 if matches(sequence_features, "ich_cmf_first2") else 0)
        + (+1.0 if matches(sequence_features, "rs_atr_last2") else 0)
        # Tie-breaker bonuses
        + (+0.75 if matches(sequence_features, "mom_trend_type") else 0)
        + (+1.0 if matches(sequence_features, "dtf_first") else 0)
        + (+0.75 if matches(sequence_features, "one_or_two_led_ich") else 0)
        + (+0.4 if matches(sequence_features, "dtf_rs_last2") else 0)
        # All-3 compound bonus
        + (+0.5 if all_three_firing(sequence_features) else 0)
    )

    return layer_1 + layer_2  # final score


def should_enter(ticker, signal):
    score = compute_score_v2(signal)

    if score < ENTRY_THRESHOLD:
        return SKIP, f"score {score:.2f} < threshold {ENTRY_THRESHOLD}"

    if not check_persistence(ticker, ENTRY_THRESHOLD, days=3):
        return SKIP, "persistence not met"

    return ENTER, score
```

---

## Implementation Plan

### Phase 1 — Build Layer 1 (~3 hours)
- Add `score_ticker_v2()` to `indicators.py` alongside the existing `score_ticker()` (don't replace yet)
- New `INDICATOR_WEIGHTS_V2` dict and bucket lookup tables (per the tables above)
- Unit tests against a few known-value cases to verify correctness

### Phase 2 — Build Layer 2 (~3 hours)
- New module: `sequence_overlay.py`
- Functions to compute sequence features per (ticker, date) row: `first_firer()`, `last_two()`, `n_led_ich()`, `first_two_distinct_types()`, etc. (logic already exists in audit scripts)
- `compute_sequence_adjustments()` returning the additive offset
- Unit tests against pattern-match cases

### Phase 3 — Integrate into trade_executor.py (~1 hour)
- Add `SCHEME` env var (or YAML config flag) to select between `c` (current) and `v2` (Scheme I+)
- Wire `compute_score_v2()` into the entry evaluation path
- Keep Scheme C path intact for backwards compatibility

### Phase 4 — Backtest threshold sweep (~2 hours)
- Update `sizing_comparison_backtest.py` to support `--scheme v2`
- Run sweep at threshold values 7.0 through 10.0 in 0.5 increments
- Output comparison table: Scheme C vs Scheme I+ at each threshold

### Phase 5 — Decide on production deployment
- Compare best Scheme I+ threshold to Scheme C (current 9.0)
- Decision criteria:
  - Total return ≥ Scheme C
  - Sharpe ≥ 2.0
  - Max DD ≤ -25%
  - Win rate ≥ 60%
  - Path std meaningfully lower than Scheme C
  - # trades not dramatically lower
- If Scheme I+ wins on most criteria → deploy to paper account first, then live

**Total estimated effort:** 1-2 days of focused work to get to a backtest result.

---

## What's NOT in v1 (deferred to v1.5+)

- **Joint scoring** (J1: ICH×ATR; J2: DTF cross-timeframe). Add only if v1 outperforms and we want incremental refinement.
- **Single-indicator catastrophic penalties.** Could add Layer 2 penalties for ICH 0/3, ROC ≥ 100%, very negative CMF, etc. Currently captured only by getting 0 Layer 1 points (no penalty). If v1 underperforms because catastrophic single-indicator values aren't being penalized enough, add in v1.5.
- **Variable position sizing by signal tier.** Currently all entries use 8.3% of equity. Could vary by score (e.g., higher-scoring signals get larger positions). Major architectural change; defer.
- **Regime detection / auto-recalibration.** Currently scoring is hardcoded for AI-bull regime. When regime fades, recalibration is manual. Auto-detection is much later work.

---

## Open data caveats

- Sub-threshold sample sizes for fast-track patterns (FT1-FT4) are modest. FT4 (RS→ATR last-2) has only n=21 sub-threshold. Confidence is moderate; live data will refine.
- Model R² out-of-sample within current regime was -0.05 (close to zero). Scheme I+ should be expected to MODESTLY improve on Scheme C, not transform performance dramatically.
- The sub-threshold baseline win rate isn't directly known; the +6-8pp lifts for fast-track patterns compare to the score≥9.0 baseline (74.2%), which may slightly overstate vs the proper sub-threshold baseline.

These caveats are acceptable for a v1 trial. If backtest shows clear improvement, deploy. If marginal or worse, the data caveats become reasons to investigate further.

---

## Investigation that led here

After 4 sessions of analysis (2026-04-30 to 2026-05-01) we tested Path C
against Scheme C across multiple variants (v1.0, v1.1, Path C, Path C
coarse), position caps (12, 15, 20), and threshold values, with 10-start
within-regime path-dep validation. Result on cumulative return × stability:
Scheme C @ 9.0 wins (+651.6% / std 10.2% / Sharpe 3.28). Path C best:
+529.9% / std 23.3% / Sharpe 3.18 — slightly lower cumulative.

**But Path C picks 3x better signals on per-trade median** (+12.4% vs
Scheme C's +4.4%). This per-trade quality advantage didn't translate to
cumulative return in backtest due to lower trade frequency — but
backtest is single-regime. We deployed Path C as a SHADOW tracker
(no real money) to collect out-of-sample data on whether the per-trade
quality advantage persists in real-world conditions.

See `DECISIONS.md` (entries 2026-04-30 → 2026-05-01) for the full
investigation arc and `shadow_pathc.py` / `dashboard.py` for the
shadow tracking implementation.

