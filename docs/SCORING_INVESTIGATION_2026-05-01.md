# Scoring Investigation — Session 3 (2026-05-01) Handoff

**Status:** Investigation CONCLUDED. Scheme C @ cap 12 / threshold 9.0 (current LIVE) confirmed regime-optimal. No production changes shipped.

This is the third (and final) session in the multi-session scoring investigation that began 2026-04-30. Read these in order:
1. `HANDOFF.md` — base project orientation
2. `SCORING_INVESTIGATION_2026-04-30.md` — session 1 (sequence patterns discovery)
3. `SCORING_INVESTIGATION_2026-04-30_session2.md` — session 2 (Scheme I+ design proposal)
4. **This doc — session 3 (Scheme I+ implementation, backtest, and rejection)**

---

## TL;DR

1. **Production unchanged.** Scheme C continues to drive paper + live trading at cap 12 / threshold 9.0.
2. Built Scheme I+ end-to-end: Layer 1 (recalibrated indicator scoring) + Layer 2 (mean-return-based sequence overlay).
3. Tested 12+ configurations across schemes (Scheme C, Scheme I+ v1.0, v1.1, Path C, Path C coarse), position caps (12/15/20), thresholds (6.0-10.5), with 10-start within-regime path-dep validation.
4. **No tested configuration beats Scheme C @ cap 12 / threshold 9.0** on the combined return × stability metric.
5. Investigation produced significant analytical infrastructure for future regime-shift detection, but no actionable scoring change for current regime.

---

## What was built and tested

### Code added (alongside existing, not replacing)
- **`indicators.py`**: `score_ticker_v2()` function + `INDICATOR_WEIGHTS_V2` + per-indicator bucket tables (RS, HL, ICH, ROC, CMF, ATR, DTF_126d, DTF_63d). All non-monotonic curves with regime-tuned shapes.
- **`sequence_overlay.py`** (new): Layer 2 framework with `compute_sequence_features()` and `compute_layer_2_adjustment()`. Path C version uses mean-return-based bonuses/penalties, capped at ±0.5 per pattern.
- **`compute_scheme_i_plus_scores.py`** (new): pre-computes v2 scores from `audit_dataset.parquet` into a parquet for backtester consumption.
- **`sizing_comparison_backtest.py`**: added `--score-source` flag (sqlite or parquet:<path>) and `--scheme-label`. Lets the backtester run any scheme as long as scores are pre-computed.

### Audit scripts created during investigation
- `audit_indicator_curves.py` — empirical bucket curves per indicator (`--start-date` for regime filtering)
- `audit_indicator_interactions_full.py` — gradient-boosted regression + partial dependence + SHAP interaction analysis
- `audit_signal_score_distribution.py` — score-distribution-conditional pattern analysis (which patterns work at sub-threshold)
- `audit_sequence_total_return.py` — patterns ranked by mean fwd return (the right metric for portfolio optimization)
- All accept `--start-date 2025-05-01` for regime filtering.

### Test diagnostics
- `_test_caps_and_bucketing.py` — custom grid: Scheme C and Path C variants at caps 12/15/20
- `_diag_pathdep.py` — verified skip-when-full simulator behavior
- `_compute_layer1_only.py` — quick parquet generator for Layer 1 ablation test

---

## The investigation arc

### Phase 1: Build Layer 1 (recalibrated indicator scoring)
v1.0 used aggressive non-monotonic curves directly from full-dataset bucket analysis. **Backtest showed v1.0 missed the regime's biggest winners** (AXTI, BW, IREN, NVTS) because they live in RS 96-100 zone that v1.0 penalized to 0.20 points. The mega-winners contribute most of total return through heavy-tail dynamics; penalizing their score zone catastrophically reduced returns.

Fix: v1.1 softened the dips (RS 96-98 from 0.20 → 1.40, RS 98+ from 0.60 → 2.00, ROC tail from 0.00 → 0.40-0.20, DTF tails lifted similarly). v1.1 captures ~+550% mean return vs Scheme C's +650%.

### Phase 2: Investigate Layer 2 (sequence overlay)
v1.0/v1.1 sequence overlay used win-rate-based penalties. Backtest comparison (Layer 1 only vs Layer 1 + Layer 2) showed **Layer 2 made things worse** — return dropped from +491% (Layer 1 only) to +215% (Layer 1 + Layer 2 v1.1).

Diagnosis: penalties like `ich→rs first-2` (PEN4 -2.0) had been chosen for low win rate (-12pp Δ) but the pattern's MEAN return is +88%. Heavy-tail wins were being excluded.

### Phase 3: Path C — rebuild Layer 2 around mean returns
Re-derived sequence findings using mean fwd return (not win rate) as the primary metric. Generated `audit_sequence_total_return.py` showing patterns ranked by mean. Patterns flagged as "bad" by win rate (`ich→rs`, `ich→dtf`, etc.) actually had high mean returns. Path C's Layer 2:
- Maximum bonus/penalty magnitude capped at ±0.5
- Bonuses for patterns with high mean return (heavy-tail captures)
- Penalties only for patterns with truly low/negative mean

Path C result: +530% mean (cap 12, threshold 7.5), Sharpe 3.18 — better than v1.1 but still below Scheme C.

### Phase 4: User asked about path-dependency
Initial path-dep tests showed Scheme I+ with absurd 100-265% std vs Scheme C's 10-15% std. Looked like a major issue.

**Diagnosis: artifact of test setup.** The `compute_path_start_dates` function used score data range (2023-05-15 → 2026-04-30) for path candidates, then `--start 2025-05-01` filter applied to simulation. So path-dep was measuring variance across pre-regime + regime starts, not within-regime. Custom test using 10 fixed within-regime start dates showed actual variance is ~15-25% std for both schemes — much closer.

### Phase 5: Position cap variations + coarsening
Tested Scheme C and Path C at caps 12, 15, 20. Tested coarse-bucketed Path C (round to 0.5 increments) at all caps. Tested Path C coarse at multiple thresholds.

**Findings:**
- Increasing cap HURTS all schemes by 30-50% on mean return (extra slots get lower-quality entries)
- Coarsening Path C cuts variance in half (23% → 11.5% std) with only ~7pp return cost
- Path C coarse @ 7.0 is best Path C variant (+550% / Sharpe 3.13) but still below Scheme C @ 9.0

### Phase 6: Final comparison and conclusion

| Best of each scheme | Cap | Thr | Mean Return | Path Std | Sharpe |
|---|---|---|---|---|---|
| **Scheme C (LIVE)** | **12** | **9.0** | **+651.6%** | **10.2%** | **3.28** |
| Scheme C | 12 | 8.0 | +666.2% | 21.2% | 3.31 |
| Path C coarse | 12 | 7.0 | +549.9% | 21.1% | 3.13 |
| Path C | 12 | 7.5 | +529.9% | 23.3% | 3.18 |
| Layer 1 only | 12 | 7.0 | +491.3% | 109% (full-range)* | 3.15 |

*Layer 1 only path-dep was tested with the original (full-range) path candidates, so std is not directly comparable.

Scheme C @ 9.0 wins on combined metric. Investigation concluded.

---

## Why Scheme C wins

**Three insights from the investigation:**

1. **Coarseness is a feature, not a bug.** Scheme C's 0.6/1.2/1.8/2.4/3.0 buckets create many ties at score 9.9 each day. Alphabetical tie-breaking on those ties is path-stable. Any finer scoring breaks ties and increases path-dependency.

2. **Empirical curves were optimized for the wrong metric.** Mean win rate ≠ portfolio return in heavy-tail regimes. The "RS dip at 96-98" finding was real on average, but the MEAN return at 96-100 (driven by mega-winners) is excellent. Penalizing those zones excluded the regime's biggest contributors.

3. **Layer 2 sequence overlay is a poor architecture for this regime.** Heavy-tail dynamics dominate total returns. Adjustment magnitudes that meaningfully shift the threshold gate also disrupt path-stable tie-breaking. There's no win available.

---

## Production state

All commits up to and including 2026-04-30 are pushed. No commits made during this session. Local working tree state:

### Modified (uncommitted, but production-safe)
- `indicators.py` — added `score_ticker_v2()` and bucket tables. Existing `score_ticker()` unchanged. Production calls `score_ticker()` so v2 has no effect.
- `sizing_comparison_backtest.py` — added `--score-source` and `--scheme-label` flags. Default behavior unchanged (sqlite source).
- `audit_sequence_v2.py`, `audit_sequence_overlap.py` — added `--start-date` flag for regime filtering. Default behavior unchanged.

### New files (uncommitted)
- `sequence_overlay.py` — Layer 2 framework
- `compute_scheme_i_plus_scores.py` — pre-compute script
- `audit_indicator_curves.py`
- `audit_indicator_interactions_full.py`
- `audit_signal_score_distribution.py`
- `audit_sequence_total_return.py`
- `_test_caps_and_bucketing.py`
- `_diag_pathdep.py`
- `_compute_layer1_only.py`

### Generated artifacts in `backtest_results/` (gitignored)
- `scheme_i_plus_scores.parquet` (v1.0)
- `scheme_i_plus_scores_v11.parquet` (v1.1)
- `scheme_i_plus_layer1_only.parquet`
- `scheme_i_plus_v2_pathC.parquet` (Path C with full bucketing)
- `scheme_i_plus_v2_pathC_coarse05.parquet` (coarse-bucketed)
- Various `.txt` outputs from audits
- Various `.log` files from backtest runs

### Documentation
- `docs/SCHEME_I_PLUS_PROPOSAL.md` — final proposal with conclusion appended; status changed to "INVESTIGATION CONCLUDED"
- `docs/SCORING_INVESTIGATION_2026-05-01.md` — **this file**

---

## Recommendations for the next thread

1. **No production deployment.** Confirmed.

2. **Optional cleanup.** The investigation generated 9 audit/diagnostic scripts and 5+ intermediate parquets. These could be:
   - Kept entirely (they're useful for future regime shifts)
   - Moved to a `docs/scoring_investigation/` subdirectory
   - Or just left in the working tree (uncommitted)
   
   Recommend: leave in working tree, add a comment to `.gitignore` if desired. The scripts are small and self-documenting.

3. **Periodic re-evaluation.** Schedule a quarterly re-run of the audit scripts to detect regime shifts:
   ```bash
   python3 audit_indicator_curves.py --start-date <last-quarter>
   python3 audit_sequence_total_return.py --start-date <last-quarter>
   python3 audit_indicator_interactions_full.py --start-date <last-quarter>
   ```
   Significant changes in the empirical curves (e.g., RS shape, ATR direction flip, DTF acceleration sign) would signal that recalibration is worth attempting again.

4. **Key warning signs for re-evaluation:**
   - ATR direction reverses (low ATR becomes good in the audits) — historical pattern
   - DTF acceleration starts winning (currently decelerating wins in pooled data)
   - Mean win rate at score≥9.0 drops to ~55-60% (from current 74%)
   - Live trading performance degrades materially

5. **If a future re-evaluation justifies recalibration:** the v2 implementation is ready to use:
   - Edit `INDICATOR_WEIGHTS_V2` and bucket tables in `indicators.py`
   - Re-run `compute_scheme_i_plus_scores.py` to regenerate scores
   - Backtest via `sizing_comparison_backtest.py --score-source parquet:<path>`
   - The `--start-date` infrastructure is ready for re-bucketing on the new regime data

6. **What was NOT tried (could be future work):**
   - **Variable position sizing by score** — higher-conviction signals get larger positions. Major architectural change but could capture more upside.
   - **Threshold-conditional weights** — different scoring weights at score 9.0 vs 9.5+ entry zones (entry-context optimization).
   - **Negative weights** — selling short on highly-bad-pattern signals. Major strategy change but could exploit our PEN1-PEN5 findings.
   - **Joint scoring (J1, J2)** — pairwise interactions like ICH×ATR. Deferred from v1; data showed they would add minimal lift.

---

## Things to know

- **Auto mode** was active during most of this session. The user steered the major design decisions at each phase (locked Path B/C choices, requested re-evaluation when path-dep results looked off, requested cap variation tests).
- **Path-dep test gotcha.** When testing on a parquet score source, `compute_path_start_dates` uses the parquet's full date range (2023-2026), which includes pre-regime dates that distort path-dep numbers. Custom diagnostics use fixed within-regime dates (April-June 2025) for clean comparison.
- **Heavy-tail regime is a fundamental constraint.** Any scoring system optimized for "mean win rate" or "Δ above population" will under-weight heavy-tail captures. Optimization for "mean return per pattern" or "total portfolio CAGR" is the right framing — but our experiments showed even the right framing doesn't beat Scheme C in the current regime.
- **Position cap 12 is near-optimal** in current regime. Higher cap dilutes returns; lower cap was not tested but the trend (12 > 15 > 20) suggests cap 8-10 might do better. Could be a follow-up experiment.
