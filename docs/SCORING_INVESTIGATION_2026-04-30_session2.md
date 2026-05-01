# Scoring Investigation — Session 2 Handoff

**Date:** 2026-04-30 (evening session, continuation of earlier investigation)
**Status:** Scheme I+ designed and documented; not yet implemented or tested.
**Production state:** UNCHANGED. Scheme C still live; trading running normally.

This doc captures the second 2026-04-30 session that built on the morning's
sequence-finding investigation. Read this AFTER `HANDOFF.md` and the morning's
`SCORING_INVESTIGATION_2026-04-30.md`.

---

## TL;DR

1. **Production unaffected.** No code changes shipped. Scheme C continues to drive paper + live trading.
2. **Major design output:** [docs/SCHEME_I_PLUS_PROPOSAL.md](SCHEME_I_PLUS_PROPOSAL.md) — full draft scoring spec for Scheme I+ (3-layer: indicator scoring + sequence overlay + threshold gates). Awaiting user review on 8 decision points before implementation.
3. **Key reframe**: user clarified they're explicitly optimizing for the **current AI-infrastructure bull regime** and accept regime-specific overfitting. This dramatically changed the analysis methodology and resurrected several findings we had killed.
4. **Many "robust" findings from the morning session were artifacts** of cross-regime filtering. Several "killed" findings (ATR-first, VOL→MOMENTUM) are positive in current regime and should be USED.
5. **Fast-track concern proved partly wrong.** DTF-first wins 95% at score≥9.0 but only 57% at sub-threshold — the base score IS providing real information. Most "exceptional" sequence findings shouldn't fast-track sub-threshold signals.

---

## The investigation arc

### 1. Started with sequence-finding follow-up
- Re-ran sequence analysis with corrected fire thresholds (`rs_fired = pctl≥90` not ≥50; `hl_fired = count≥4` not ≥2). The old loose thresholds had distorted streak-based "first-firer" labels.
- Result: corrected analysis confirmed RS-first underperforms (-9.9pp) in pooled data, but with much smaller N (n=396 vs old 4,364).
- New finding emerged: MOMENTUM→TREND first-2-type was the cleanest large-N positive (+9.2pp, n=1,383).

### 2. Investigated indicator scoring confounding (per user concern)
- Ran `audit_indicator_curves.py` with empirical bucketing of all 7 indicators.
- Discovered widespread miscalibration: ATR scoring was inverted, ROC needed cap above 50%, Ichimoku 2/3 vs 3/3 deserved different scores, RS had non-monotonic shape with dip at 94-98.

### 3. User raised the interaction-confounding question
- Built `audit_indicator_interactions_full.py` using gradient-boosted regression + partial dependence + SHAP interaction values.
- **Findings (pooled data):** Model R² out-of-sample was -0.13 (negative). 4 of 8 indicators had isolated curves that didn't survive interaction control. Strong interactions exist (H > 0.7 for several pairs).

### 4. User clarified regime-specific framing
- Pivoted to filter all analyses to 2025-Q2+ data (current AI infra bull regime).
- Re-ran `audit_indicator_curves.py` with `--start-date 2025-05-01`. Results dramatically cleaner.
- Re-ran `audit_indicator_interactions_full.py` with same filter. Model R² in-sample +0.30 (much better fit). DTF_63d emerged as top SHAP feature (importance 0.096, far above others).
- **Critical finding:** ATR direction FLIPPED between pooled and current regime. In current regime, HIGH ATR is good (consistent with Scheme C). In pooled, LOW ATR was good. The ATR direction in production is correct *for current regime*.
- Same with DTF acceleration: pooled rewarded DECELERATION, current regime rewards ACCELERATION.

### 5. Re-audited sequence findings under regime-specific lens
- Re-ran `audit_sequence_v2.py --start-date 2025-05-01`.
- Confirmed: 4 previously-"robust" findings still hold (MOMENTUM→TREND, 1-led-Ich, dtf→rs last-2, TREND→MOMENTUM penalty).
- NEW finding: **DTF-first as first-firer wins 95% at score≥9.0** (n=77). Didn't exist in pooled because of structural threshold issues.
- NEW finding: **2-led-Ich is also strong** (+9.2pp), not just 1-led. Sweet spot is 1-or-2 leading.
- Rediscovered: ATR-first works in current regime (+5.1pp), VOL→MOMENTUM works (+4.4pp). Should NOT be killed.

### 6. User asked: how to express exceptional/catastrophic patterns in final score?
- Built `audit_signal_score_distribution.py` to test fast-track viability per pattern.
- **Surprise finding**: most "good" sequence patterns DON'T retain edge at sub-threshold. DTF-first sub-threshold signals win only 57% (vs 74% baseline). Base score IS providing critical information.
- Identified 4 categories of patterns: universal-good (rare, fast-track candidates), high-score-only-good (tie-breaker only), high-score-traps (filter at high score), universal-bad (hard filter).

### 7. Drafted Scheme I+ proposal
- Comprehensive 3-layer design: indicator scoring + sequence overlay + threshold gates.
- 4 hard filters, 4 fast-track patterns, 5 additive bonuses, 1 penalty, 2 joint-scoring rules.
- Awaiting user review on 8 specific decision points before implementation.

---

## Key empirical findings (current regime, 2025-Q2+)

### Indicator scoring
- **RS** is non-monotonic: peak at 88-94 percentile (+15-18pp), dip at 96-98 (+1.7pp), modest recovery at 98+ (+4.6pp). The 96-98 zone needs a reduced score (not full 3.0).
- **Ichimoku** has dramatic tier separation: 0/3 = -21pp, 1/3 = -8pp, 2/3 = +1pp, 3/3 = +11pp. Current Scheme C treats 2/3 same as 3/3 (overweights 2/3).
- **ROC** sweet spot is 15-50% (+10pp). Above 75%: catastrophic (-13pp at ≥100%). Needs cap.
- **ATR** rewards HIGH percentile in current regime (+5.5pp at 90-95). Reverses pooled finding.
- **DTF_63d** is the single most predictive indicator per SHAP (importance 0.096). Peak at 90-95 percentile.
- **DTF_126d** peaks at 85-90 percentile (+15.5pp).
- **CMF** has clean monotonic curve (+12.5pp at ≥0.30) but didn't survive interaction control. Re-include at LOW weight (1.0).
- **HL** has positive isolated curve but interaction-confounded. Keep at low weight (0.5).

### Sequence overlay (regime-tuned)

**Hard filters (skip even if score qualifies):**
- 3 indicators leading Ich (-21pp, n=137)
- ROC → DTF first-2 (-26pp, n=85)
- ICH → DTF first-2 (-19pp, n=211)
- ICH → RS first-2 (-12pp, n=144)

**Fast-track candidates (lower threshold to ~8.5):**
- ATR is first-firer (+6.6pp sub-thresh)
- ROC → ICH first-2 (+7.0pp sub-thresh)
- ICH → CMF first-2 (+7.9pp sub-thresh)
- RS → ATR last-2 (+6.7pp sub-thresh; small N)

**Additive bonuses (tie-breakers):**
- MOMENTUM → TREND first-2 type (+0.5)
- DTF first-firer (+0.5)
- 1-or-2 led Ich (+0.5)
- dtf → rs last-2 (+0.3)
- All-3 firing combo (+0.5 extra; 95.7% win, n=46)

**Penalty:** TREND → MOMENTUM type (N1, n=935): -0.5

**Joint scoring:**
- ICH 3/3 + ATR ≥ 90: +0.5
- DTF 126d ≥ 80 + DTF 63d ≥ 80: +0.5

---

## Local state (uncommitted)

### Scripts created tonight (all in /Users/toddbruschwein/Claude-Workspace/breakout-tracker/)
| Script | Purpose |
|---|---|
| `audit_sequence_v2.py` | Sequence analysis with corrected fire thresholds; supports `--start-date` |
| `audit_sequence_v2_regime.py` | Per-regime stability test of top sequence findings |
| `audit_sequence_overlap.py` | Co-occurrence + marginal lift of P1/P5/P6; supports `--start-date` |
| `audit_sequence_robustness.py` | Time-period stability + score-conditional + bootstrap CIs |
| `audit_sequence_regime.py` | Per-regime baseline-relative analysis with horizon comparison |
| `audit_lastfirer_score_regime.py` | Last-firer freshness + score-streak per regime |
| `audit_indicator_curves.py` | Fine-grained empirical bucket curves for all 7 indicators; supports `--start-date` |
| `audit_indicator_interactions_full.py` | GBM + Partial Dependence + SHAP interaction analysis; supports `--start-date` and `--cv-split-date` |
| `audit_signal_score_distribution.py` | Score distribution per sequence pattern; supports `--start-date` |

### Output files in `backtest_results/` (gitignored)
- `audit_sequence_v2.txt` — pooled sequence findings
- `audit_sequence_v2_regime_filtered.txt` — 2025+ sequence findings
- `audit_sequence_v2_regime.txt` — per-regime stability check
- `audit_sequence_overlap.txt` — pooled overlap
- `audit_sequence_overlap_regime.txt` — 2025+ overlap
- `audit_sequence_regime.txt` — earlier per-regime regime audit
- `audit_sequence_robustness.txt` — robustness with time-period split + bootstrap
- `audit_lastfirer_score_regime.txt` — score-streak findings per regime
- `audit_indicator_curves.txt` — pooled indicator curves
- `audit_indicator_curves_regime.txt` — 2025+ indicator curves
- `audit_indicator_interactions_full.txt` — pooled interaction audit
- `audit_indicator_interactions_full_regime.txt` — 2025+ interaction audit
- `audit_signal_score_distribution.txt` — score distribution per pattern

### Documentation
- `docs/SCHEME_I_PLUS_PROPOSAL.md` — **the deliverable**. 3-layer scoring spec with all bucket tables, filter logic, threshold rules, backtest plan, and 8 decision points.

### Dependencies installed tonight
- `shap 0.49.1` (via `python3 -m pip install shap`). Pulled in numba 0.60, llvmlite 0.43, cloudpickle, slicer.
- `tqdm` got installed as a side-effect into `~/Library/Python/3.9/bin/` which isn't on PATH. Harmless.

### Production code: UNCHANGED
- `indicators.py`, `trade_executor.py`, `ticker_config.yaml` all unchanged from this morning's commit `1c91335`.
- No commits made tonight.

---

## Where to pick up in the next thread

**Recommended next step:** User reviews `docs/SCHEME_I_PLUS_PROPOSAL.md` and answers the 8 decision points at the bottom. Then implementation begins.

**Implementation order once decisions are locked:**
1. Build `score_ticker_v2()` in `indicators.py` (Layer 1 indicator scoring) alongside existing `score_ticker()`. Don't replace.
2. Build `sequence_filter.py` or extend `trade_executor.py` (Layer 2 overlay). Keep enable/disable flag.
3. Update `trade_executor.py` to call new scoring path with `SCHEME=v2` flag (env var or yaml).
4. Run `sizing_comparison_backtest.py` with three configs:
   - Baseline: Scheme C @ 9.0
   - Scheme I (Layer 1 only) @ proposed threshold
   - Scheme I+ (Layer 1 + Layer 2) @ proposed threshold + fast-track
5. Compare metrics, decide on production deployment.

**If user wants to revisit any analysis before implementing:**
- All audit scripts support `--start-date 2025-05-01` for regime filtering
- `audit_indicator_curves_regime.txt` is the source of truth for proposed indicator scoring
- `audit_signal_score_distribution.txt` is the source of truth for fast-track / filter decisions
- `audit_sequence_v2_regime_filtered.txt` is the source of truth for sequence findings in current regime

**Open question worth thinking about:** the proposal includes joint scoring (J1, J2). These add complexity. Could ship Scheme I+ v1 without them and add later if simple version underperforms. Discussed in decision point #6.

---

## Things to know

- **Tonight's work was extensive but fully reversible.** No commits, no production changes. The proposal is in a docs file the user can edit.
- **The user explicitly accepts regime-specific overfitting.** Don't reintroduce cross-regime stability requirements without checking — that filter killed several findings that are real in current regime.
- **The model R² out-of-sample is still slightly negative even within regime** (-0.05 in test). This means Scheme I+ should be expected to MODESTLY improve on Scheme C, not dramatically transform performance. Set expectations accordingly.
- **GitHub Actions cron continues running on schedule.** Daily-trade-execution at 4:30 PM ET keeps trading Scheme C config. Don't be alarmed by automated commits coming in overnight.
