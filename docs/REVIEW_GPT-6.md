# Alpha Scanner skeptical quant review — GPT-6

Review date: 2026-10-13 (America/Los_Angeles). Repository reviewed: `/Users/toddbruschwein/Claude-Workspace/breakout-tracker`, HEAD `c149bc9522eae1b6bb6d148dd9ab660933c254b6`. Scope: all 13 substantive entries in `docs/DECISIONS.md`, `docs/HANDOFF.md`, their implementation, saved audit/sweep artifacts, and selected reruns. No production code, configuration, trading state, or database was intentionally changed; no trading commands, commits, or pushes were run.

## Overall assessment

**The evidence supports conservative decisions to retain existing rules and collect observations. It does not establish that Scheme C, its weights, or its portfolio parameters are robust optima.** Several advertised performance comparisons are reproducible as saved outputs, but the simulator and calibration dataset have material defects. The most serious are future information in cross-sectional calibration inputs, prematurely booked next-open transactions, and a stop model that cannot validate live GTC-stop behavior. Increasing the fetch from 2y to 3y and requesting 20 starts addresses neither these defects nor the retrospective universe.

The operational fixes on May 13 and May 19 have substantially better direct evidence than the strategy optimizations. The September minimum-position change is arithmetically justified, although its return impact has not been established.

## Findings that affect multiple decisions

### 1. Critical: audit RS pre-pass can see the future for short-history tickers

`audit_build_dataset.py:110–123` slices **each ticker** with `full_df.iloc[:idx + 1]`, where `idx` belongs to SPY's longer history. IPOs, gaps, and shorter histories do not share SPY's row numbers. For a later listing, the slice ends after the target date, potentially at the ticker's entire fetched history. There is no as-of-date eligibility gate in this pre-pass. Its outputs feed both peer RS ranks and Dual-TF percentiles (`:366–389`), including the peers used to rank otherwise properly aligned established tickers.

Reproduction: a benchmark with 800 business-day rows and a ticker starting 300 rows later, evaluated at benchmark row 400, used ticker bars through **2025-09-08** to score a target date of **2024-07-15**. Only 101 ticker rows existed as of that target. This is actual feature look-ahead, not merely optimization on historical labels. `build_row` separately aligns the ticker by date, but it receives the already contaminated cross-sectional inputs. Its `method="nearest"` at `:169` is also capable of selecting a future bar across a gap.

The current audit artifact contains 176 tickers; ARM, SNDK, and CRWV have different historical availability. This establishes a relevant failure mode, not a measured estimate of how many past trades were changed. A complete as-of rebuild is required before treating audit-derived weights, Scheme M buckets, or sequence bonuses as validated.

### 2. High: next-open trades are booked into the previous day's close

`sizing_comparison_backtest.py:536–562` immediately realizes tomorrow's sell proceeds; `:600–613,842–864` immediately creates tomorrow's purchases; `:867–883` then marks the mutated portfolio at **today's** close. Later candidates are sized with this mixed-time equity. Tomorrow's sale cash is already available, and tomorrow's holdings are already marked before their entry date. `entry_mode_backtest.py:236–272,367–402` shares the same chronology problem.

Reproduction: $100,000, today's close $100, next open $200, full allocation. The original simulator reports **$50,000 equity today**, although the order has not filled and the account should still hold $100,000 cash. It recovers to $100,000 on the fill day. This creates fictitious drawdown and daily returns; the sizing interaction can change portfolio composition and terminal returns. Bias direction is not reliably positive or negative. Sharpe and drawdown comparisons are not trustworthy until transactions and snapshots have a coherent event clock.

### 3. High: fixed/trailing stop experiments do not model broker stops

The simulator updates the trailing high-water mark with today's close before comparing today's low (`:500–501,529–540`). It therefore can test an earlier low against a stop tightened using a later close. Both fixed and trailing breaches sell at the **following day's open**, rather than at a same-session stop execution or gap-through opening fill. Score exits take precedence over stop breaches, although the stop could already have fired before the score was known.

Reproduction: entry $100, fixed stop $80, session low $70, following open $100. The recorded stop exit has **0% loss**. Conversely, a next-open decline can produce a loss well beyond 20%. Consequently, “fixed-20 eliminates whipsaws” and “trailing is catastrophic” are conclusions about this implementation, not validated comparisons of live stop orders. A nominal 20% stop is not a guaranteed maximum loss.

The live limit-entry pipeline attaches stops only at its next daily run (`trade_executor.py:390–428,1147–1159`), after the fill session. The handoff acknowledges this exposure but gives no evidence that a 3% entry-price bound makes the holding-period risk acceptable. Entry slippage and subsequent price loss are different risks.

### 4. High: missing scores become fabricated daily persistence

`build_daily_scores` (`sizing_comparison_backtest.py:192–208`) carries every last ticker score forward indefinitely. A single qualifying observation can satisfy three “prior days,” and a removed/failed ticker can retain a stale score. In contrast, production `check_persistence` (`trade_executor.py:639–679`) requires three distinct prior DB rows, deliberately allowing calendar gaps. These are different rules.

Read-only DB inspection found 61,286 Scheme C rows on 340 distinct dates, 2025-03-25 through 2026-10-13. Spring 2025 is sparse: March has one date; April five; May four; June four. Daily coverage largely starts in July. Historical DB-driven threshold/sizing sweeps therefore cannot be called production-equivalent. The current DB's pre-April-24 rows have Scheme C-style contributions (DTF 2.5, no CMF), so they are also a retrospectively rescored dataset rather than an immutable record of original decisions.

`get_price` (`:224–243`) selects the nearest bar within three calendar days, in either direction. An independent missing-bar reproduction returned Wednesday's $300 opening price for Tuesday. This can introduce future execution/valuation data and hides failures. NaN Open/Low values are not explicitly rejected either.

### 5. High: “3y / 20 paths” is sensitivity analysis, not independent validation

`compute_path_start_dates` (`:1593–1622`) begins 14 calendar days after the first score, moves by a week, and stops at 40% of the score span. Twenty paths typically share nearly all subsequent observations, the same terminal regime, the same fitted scores, and most mega-winners. Their dispersion is not a confidence interval for strategy alpha. Unequal holding horizons also confound total-return dispersion with time invested; deterministic alphabetical ties can make several paths converge.

The helper does not receive `--start` or `--end`, so changing the primary window need not change the path window. Requesting 20 does not guarantee 20: availability and the 40% bound control the actual count. A business-day fixture with scores from May 2023 to April 2026 produced **12** starts with prices beginning May 2024, and 20 with earlier prices. “Capped at ~10” is an example, not a fixed algorithmic cap.

More decisively, `_run_with_path_dep` (`:2568–2602`) reports the primary run's Sharpe and return, and uses the paths only for return dispersion. **Increasing `--path-starts` alone cannot change that primary Sharpe.** The May 11 explanation jointly attributes a Sharpe reversal to longer price history and more starts without separating their effects. Changing a rolling price fetch also changes the primary calendar, warm-up, and flat-cash prefix. Scheme C's original +598% / Sharpe 2.21 log actually reports a primary calendar starting **2024-04-24**, despite score data beginning March 2025 (`audit_portfolio_schemeC_pathdep.log:25–37`). Its 13-month trading exposure and nearly two-year metric calendar should not be conflated.

### 6. High: universe selection, overlapping labels, and repeated experimentation

The current hand-picked universe has **195 tickers, of which 111 are AI/Tech**. Universe membership was expanded on April 18, May 19, and May 24 (`ticker_config.yaml` git history; handoff's universe section). The original April audit reports 174 tickers, the later saved artifact 176, and current live scoring includes an additional comparison benchmark. These are not one fixed historical universe. Backfilling today's selected tickers into older dates introduces hindsight selection; there is no supplied point-in-time inclusion/exclusion ledger, comprehensive delisting-return treatment, or evidence that failed candidates remained represented. This does not prove every selected name was chosen using returns, but the tests cannot rule out that bias.

The same 2025–26 AI-heavy right tail repeatedly informs indicator inclusion, weights, thresholds, caps, exit choices, and Scheme M sequence rules. A 12-month slice nested inside a 3-year sample is not an independent confirmation. `sequence_overlay.py:108–150` explicitly calibrates bonuses to mean returns in the May-2025-onward regime; evaluating those bonuses in that regime is in-sample model selection.

`audit_analyze.py:66–77` calls its CI routine a “block bootstrap,” but it resamples individual rows independently. Daily ticker observations share market/sector shocks and overlapping 63-day outcomes. The conditioning in `analysis_A1` is only RS fired/not fired, not full score, sector, time, or matched exposure. Thus the reported “incremental edge” is an association conditional on a coarse RS gate, not an identified marginal effect of changing a weight. Global top-K signal matching in `audit_compare_weights.py:87–99` is also a retrospective diagnostic, not an executable daily selection rule.

As a diagnostic on the **stored, potentially contaminated** artifact, I resampled 63-date circular blocks, preserving all tickers within each sampled date (2,000 draws, seed 42). Results are not corrected alpha estimates:

| Indicator, conditional on RS fired | Stored mean effect | Diagnostic 95% date-block interval |
|---|---:|---:|
| CMF | −2.21 pp | −7.34 to +1.76 pp |
| Dual-TF RS | +5.32 pp | −1.86 to +12.54 pp |
| Higher Lows | −0.69 pp | −3.47 to +2.00 pp |

All include zero. Dual-TF's corresponding year-slice effect was **−5.98 pp in 2023**, +8.93 in 2024, +7.13 in 2025, and +20.47 in the available mature-label 2026 subset. This contradicts an assumption of stable, regime-independent weight evidence. These are descriptive subsets, not independent tests; 2026 labels are right-censored and calendar slices can overlap in outcome windows.

Multiple-testing risk increases with the many inspected variants; a selected best Sharpe is not an unbiased performance estimate. See the primary methodological paper [Bailey et al., The Probability of Backtest Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf). I did not compute a formal probability of overfitting or a deflated Sharpe because a complete experiment ledger and clean return series are unavailable.

### 7. Execution and observation portfolios are materially different

The sizing simulator uses fractional shares and a $100k account, lacks the live 5% per-slot commitment cap, 3% DAY limit, whole-share flex, $250 minimum, four-entry daily cap, and broker tradeability exclusions. Production excludes crypto `-USD` and futures (`trade_executor.py:199–200,315–328`); the backtester does not apply that gate. These discrepancies matter at $5–6k and cannot be dismissed by calling sizing “8.3%” in both systems.

`entry_mode_backtest.py:74–104` still defaults to entry 8.5. It fills limits only if next **Open** is within the limit; a DAY order can also fill after an above-limit open if intraday prices subsequently cross it. The four-way experiment combines floor and entry mode, uses current regular-session Close rather than the live extended-hours sizing price, and does not implement the live per-slot formula. A budget checked at sizing price is not guaranteed to remain under 95% after fills up to 3% higher. The live +1-share flex can exceed its per-slot allocation. Therefore the handoff's claim of a guaranteed, tightly validated floor is stronger than either implementation supports.

Scheme M shadow and spillover fill at the same close used to compute signals, after that close is known (`shadow_m.py:284–321`; `spillover_tracker.py:227–259`). Their stops test Close only, not intraday low or actual broker fills. Spillover has no capital or position cap. Its gross P&L cannot be compared directly with a funded parent's return to demonstrate that rotation would help; the alternative must model which parent holding is sold, equal capital, subsequent cash constraints, and common fills.

## Assessment of each decision

### 2026-04-16 — Entry 9.5 → 8.5 with three-day persistence

**Evidence:** The decision and later commit `3f4884c` are consistent with correcting the legacy simulator and preferring a broader threshold. The exact initial seven-threshold sweep is not separately preserved among the inspected saved logs; I did not independently reproduce its 1.98 Sharpe. Commit `853d52f` places an 8.5/p=3 deployment on April 14, so April 16 appears to summarize or reconfirm an already implemented change.

**Overfitting/look-ahead:** Threshold and persistence were selected after examining the same sample. Persistence on carried-forward spring scores does not mean three independent observations. The result does not establish 8.5 as optimal.

**Method adequacy/verdict:** Inadequate as confirmation: narrow effective trading period, correlated starts, retrospective universe, simulator defects. Broadening participation may be a reasonable provisional choice; treat this entry as historical and superseded, not independent evidence supporting today's 9.0 threshold.

### 2026-04-16 — Cap 12, equity sizing 8.3%, fixed 20% stop, skip when full

**Evidence:** `3f4884c` confirms the parameter sweep rationale. Nearby caps and stops gave similar Sharpe (fixed 20 versus 15: 2.09 versus 2.07 in its message). Lower path dispersion at 12 supports a local stability preference, not a universal optimum. Later saved logs support poorer trim/swap results in the chosen window; the $595k is a scenario difference, not necessarily attributable solely to cutting winners.

**Overfitting/look-ahead:** Cap, size, stop, and rotation decisions were selected jointly and sequentially from the same right tail. Trailing's close-before-low and next-open stop modeling undermine the strongest stop conclusion. “0% whipsaw” is a realized diagnostic; it is not a future guarantee.

**Method adequacy/verdict:** Skip-when-full and a broad cap plateau are defensible provisional choices. Exact 12/20% optimality is unproved; live-stop behavior and $5k whole-share implementation require their own simulation. More starts do not compensate for absent failed/delisted names or regime diversity.

### 2026-04-24 — Scheme C weights and entry 9.0

**Evidence:** Saved audit reports do show CMF negative after the RS gate, DTF positive, and HL close to zero. The entry-sweep rows in `audit_portfolio_baseline_pathdep.log:177–187` and `audit_portfolio_schemeC_pathdep.log:177–187` reproduce the cited **+438.1% / 1.97** and **+598.0% / 2.21**, including win rates and drawdowns. Deployment commit is identifiable as `73e72c4`, rather than merely “~2026-04-24.” The headline four-strategy path tables in those files use different cap/threshold configurations from the final sweep row; their best/worst paths must not be substituted for the claimed 12-cap comparison. Non-overlap for the exact five sweep paths is asserted in the commit, while the saved sweep rows preserve dispersion/range rather than all individual returns.

**Overfitting/look-ahead:** Actual ordinal feature leakage affects the weight-calibration chain. Nested 3y/12mo analysis and weight/threshold search reuse labels. A negative conditional CMF association does not prove removing it causes improvement. The date-block diagnostic weakens claims of reliable incremental edge.

**Method adequacy/verdict:** The decision has real in-sample supporting evidence, but no clean external confirmation. Five nearby starts and a selected universe cannot establish the +160 pp lift as durable. Do not infer confidence from non-overlapping, heavily correlated path distributions. Retain provisionally pending as-of rebuild and untouched validation; do not advertise proven superiority.

### 2026-04-28 — Launch live with $5k and four-entry daily cap

**Evidence:** The workflow and mode-aware trade logs confirm a limited live rollout. Two-plus weeks of paper operation, however, did not establish operational cleanliness: the May 13 held-for-orders bug was latent from launch. Paper and live share code/data failure modes. The handoff's current ~$6k equity is not independently verifiable from order-submission logs alone.

**Overfitting/look-ahead:** Small capital is an exposure choice, not statistical validation. Prospective broker fills can provide out-of-sample operational evidence if actual fills, rejections, partials, costs, and equity are reconciled. Submission-price logs cannot substitute for it.

**Method adequacy/verdict:** Reasonable as a bounded experiment, unsupported as proof of strategy readiness. None of the cited historical starts or universe audits establish live-fill accuracy. The phased cap is still active; preserving it without a readout is not a completed rollout assessment.

### 2026-04-30 → 2026-05-01 — Keep Scheme C after Scheme I+ investigation

**Evidence:** `caps_bucketing_test.log` reproduces Scheme C path mean **651.6%, std 10.2%, primary Sharpe 3.28**, Path C **529.9%, std 23.3%, primary Sharpe 3.18**, and coarse Path C **522.6%, std 11.5%, primary Sharpe 3.02**. Returns/std are path summaries; Sharpe is the separate primary run. Coarsening reduces measured dispersion, but that does not identify alphabetic ordering as the cause of superior expected returns. The cap increases reduced Scheme C return by about 21% and 43% at caps 15 and 20; “all schemes 30–50%” overstates the log (Path C cap 15 falls about 9%).

**Overfitting/look-ahead:** Bucket curves, sequence rules, regimes, and thresholds were revised using the same winners, then compared within the fitted regime. Naming AXTI/BW/IREN after seeing their returns is a mechanistic explanation, not prospective validation. Alphabetical capacity allocation embeds an arbitrary portfolio choice; low dispersion across starts need not imply low sensitivity to that choice.

**Method adequacy/verdict:** Keeping the incumbent is more defensible than promoting the searched alternative. The experiment demonstrates selected-sample underperformance, not that Scheme C's coarseness intrinsically produces alpha. Both share contaminated calibration inputs, identical market history, and a chosen universe. See the tie-priority reruns below.

### 2026-05-01 — Deploy Path C/Scheme M shadow at 7.5 / 4.5

**Evidence:** The shadow code and cited sweep logs support an observation-only deployment. They contradict “all four parameters ... empirically-validated optima” if interpreted as optimizing one agreed metric: persistence 1 has Sharpe **3.23** versus 3.17 at persistence 3; entry 7.75 has **3.23** versus 3.17 at 7.5; exit 5.5 has higher path return **565.5%** versus 550.0% at 4.5, while exit 4.0 has higher primary Sharpe **3.22** versus 3.17. Choosing 4.5 can be a stability tradeoff, but that objective must be stated.

The cited +12.4% trade median comes from the exit=5.0 row in `pathc_exit_threshold_sweep.log`; the eventual exit=4.5 row has **+2.9%**. It does not establish “3x better signals” for the deployed configuration. `_test_pathc_exit_threshold.py` also includes terminal liquidations in trade statistics, unlike the main metrics' closed-trade win rate.

**Overfitting/look-ahead:** Extensive in-sample parameter and pattern search; same-close shadow fills are unattainable after observing the close. Fine score differences are sensitive to calibration noise. Shadow observations are prospective signals, but their fills and stops remain hypothetical.

**Method adequacy/verdict:** Deploying a shadow is justified as evidence collection; its “validated optima” and superior trade-quality claims are not. Ten starts concentrated around April–June 2025 do not test independent regimes. A common execution model is needed before comparing shadow with live.

### 2026-05-01 — Living documentation plus append-only decisions

**Evidence:** The documentation structure exists and git retains history. This is a reasonable organizational decision; quantitative evidence is not required.

**Overfitting/look-ahead:** No direct trading look-ahead. Reproducibility risk remains: `backtest_results/` is ignored, and overwritten local parquets/logs are not reconstructed merely by checking out code. Older audit reports describe 116,409 rows through January 2026 while the current dataset has 127,290 rows through April 2026.

**Method adequacy/verdict:** Backtest paths/universe are not criteria for organizing docs. For the promised evidence trail, every numerical claim needs an immutable score/price snapshot, source and universe versions, exact command, actual path dates, and experiment history. Living documentation alone does not deliver that.

### 2026-05-03 — Rename Path C to Scheme M

**Evidence:** Current functions, JSON names, and DB table implement the rename; old decision entries appropriately retain old names. Some stale references remain: the handoff names `score_ticker_v2`, and `sequence_overlay.py` references a removed proposal document. More materially, current `ticker_scores_m` has **21,540 rows, May 4–October 13**, not the claimed preserved 127k historical backfill. The historical Scheme M parquet does exist; absence from today's DB does not prove the rename migration lost it.

**Overfitting/look-ahead:** Renaming has no new fitting risk. It can obscure which version produced which evidence if I+, Path C, and M are treated as interchangeable without score hashes and configuration dates.

**Method adequacy/verdict:** Sensible naming decision; no backtest required for the rename itself. Historical validation and live shadow must be identified separately; changing the label does not make fitted-history performance out of sample.

### 2026-05-05 — Keep exit <5 and reject gap-gated swaps

**Evidence:** The recorded table supports keeping 5 over more aggressive immediate exits in that sample; exit 4 is close in return and slightly ahead in Sharpe, so it does not demonstrate a sharp optimum at 5. The code implements the stated gap/persistence sweep. Separate original May 5 raw sweep logs were not located among the inspected saved artifacts; fresh reruns below are sensitivity checks, not exact reproductions of the May 5 snapshot. The claim that swaps truncate valuable optionality is plausible and consistent with trade examples, not a causal proof.

**Overfitting/look-ahead:** Rejecting an alternative after it disappoints is less aggressive than adopting a selected winner, but the explanation still privileges ex-post mega-winners. Carry-forward persistence and the transaction/stop defects affect both baseline and variants, without guaranteeing equal bias.

**Method adequacy/verdict:** Evidence supports **no demonstrated reason to change**, not “all rotation is inferior.” Revalidate exit/swap rankings with a clean event model, fixed dates, independent periods, and realistic capital. Spillover alone cannot settle rotation because it adds unconstrained gross exposure at same-close fills.

### 2026-05-11 — Reject tighter exits with persistence; require 3y/20 starts

**Evidence:** The code contains the six-config sweep and supports the baseline rules. The reported Scheme C +0.01 Sharpe improvement is too small to justify a change given model and sampling error. The reported Scheme M sign change across windows supports **lack of robustness**. It does not prove “measurement artifact,” “statistical noise,” or equivalence: no paired, dependence-aware uncertainty test or predeclared objective is supplied. A +227 pp return difference alongside −0.06 Sharpe needs a stated objective, not automatic dismissal.

**Overfitting/look-ahead:** Reusing fitted Scheme M history and selecting the favorable regime creates substantial bias. The extra starts are correlated robustness checks; they do not remove calibration look-ahead. Primary Sharpe cannot depend on path count in this implementation, so the attribution to undersampled starts is unestablished.

**Method adequacy/verdict:** Retaining current exits is prudent. The new minimum bar improves coverage but is **necessary at most, never sufficient**. Use explicit calendar dates, fresh warm-up history, real scoring coverage, coherent fills/stops, logged actual paths, and untouched periods. The same chosen universe remains an unresolved bias in every 3y/20-path comparison.

### 2026-05-13 — Cancel stops before sells; remove phantom records

**Evidence:** Commit `61d3e87` documents the broker rejection and removes two live and five paper sell records. Current `execute_exits` does cancel before selling and skips logging when no order ID is returned. This directly supports the repair. I inspected the commit and local logs; I did not independently retrieve workflow run 25764401497 from GitHub.

**Overfitting/look-ahead:** No parameter-fitting issue. Remaining measurement problem: an accepted order is not a filled order. Sells are logged at the observed price/date, and loss exits recorded before actual fills. Two seconds is a delay, not confirmation that cancellation completed. A sell rejected after stop cancellation can leave a position unprotected until the next run. `execute_entries` also logs buys even when `order_id` is None; unfilled accepted limits are logged at sizing price rather than execution price.

**Method adequacy/verdict:** Supported operational fix, incomplete execution/accounting lifecycle. Path starts, price period, and survivorship cannot validate order-state handling. Reconcile confirmed broker fills and stop coverage, and assess partial fills/rejections asynchronously before using these logs as empirical return evidence. No trading or reconciliation writes were performed in this review.

### 2026-05-19 — Preserve existing scores, exclude stale bars, abort if >5% missing

**Evidence:** Independently extracted DB snapshots from git: May 15 trade-exec commit `3c1cf05` has **LUNR 7.1 / IRDM 4.5** on May 15; May 18 backfill commit `9e8a58b` has **LUNR 9.6 / IRDM 4.5**. This confirms the specific overwrite and the unchanged control. Current DB still has LUNR 9.6. `slice_data_to_date`, `date_already_scored`, and the coverage gate implement the described protection (`backfill_subsector.py:51–101,218–259`).

**Overfitting/look-ahead:** Exact as-of slicing reduces stale-data mislabeling. No evidence establishes 5% as an optimal cutoff. Fetch failures and tickers below minimum bars are absent from both the missing list and denominator, so “≤5% missing” does not guarantee near-complete configured-universe coverage. A missing benchmark can also escape the intended count when its whole fetch failed. Skipping a date when **any** row exists permanently freezes partially populated dates.

**Method adequacy/verdict:** Strong support for preserving authoritative scores and rejecting stale target-date bars. The chosen completeness rule needs eligibility/provenance semantics, not a trading-return sweep. “Small and decaying” correctly describes immediate persistence exposure, but the corrupted row remains a permanent historical research input; any rerun using it must identify or quarantine it in a research copy. More paths do not cure corrupted observations.

### 2026-09-30 — Lower live minimum position $300 → $250

**Evidence:** `74651f3` and the live workflow confirm the change. $1,065 divided across four open slots is **$266.25**, below $300 and above $250. This supports clearing that pre-share-rounding gate. It does not prove both intended orders fill or that eventual allocation is at least $250: whole-share quantities, price flex, and limit outcomes intervene. At $6,000, $250 is 4.17% of equity.

**Overfitting/look-ahead:** A practical response to a sizing constraint, not a tested alpha improvement. Choosing a minimum just below observed near-misses can be scenario-specific; the threshold affects subsequent portfolio paths and concentrations.

**Method adequacy/verdict:** Arithmetically reasonable operational adjustment. The $100k fractional-share simulator and the different entry-mode cash-floor implementation cannot validate a $5–6k whole-share minimum. Test realistic account sizes, share prices, four-entry limits, floor/flex interactions, and fills under the frozen strategy. The decision changes practical behavior despite the handoff saying the live config is “unchanged since April 24.”

## Rerun results and limits

These are **reruns of the unchanged, defective simulator**, not corrected estimates of investable returns. I used fresh adjusted Yahoo prices, fixed end **2026-04-30**, and stored daily-score parquets for C/M; no scores were extrapolated past their final April 30 observation. All configurations use cap 12, sizing 1/12 = 8.3333% (a small difference from CLI 8.3%), three prior simulated days, fixed 20% stop, no trim, and $100k. Swap uses gap 3 / prior persistence 3 / legacy victim gate and seven-day protection. The stored score universe is 176 tickers, with current-universe price availability for every one; this does not repair its historical membership bias or contaminated calibration.

**Twenty starts, 2023-05-30 through 2023-10-11, same April-2026 end:**

| Configuration | Mean path return | Path return std, pp | Mean of path Sharpes |
|---|---:|---:|---:|
| C entry 9, exit 5 / 1 day | +1320.5% | 92.9 | 1.712 |
| C entry 9, exit 4 / 1 day | +1742.5% | 105.6 | 1.867 |
| C entry 9, exit 7 / 5 days | +1127.6% | 72.5 | 1.631 |
| C exit 5 + gap-3 swap | +815.4% | 99.6 | 1.484 |
| M entry 7.5, exit 4.5 / 1 day | +1464.0% | 84.1 | 1.928 |
| M entry 7.5, exit 6 / 3 days | +1789.5% | 130.8 | 1.933 |

Unlike the CLI, this table explicitly computes the mean Sharpe across paths; that is still a descriptive summary, not an independent-sample estimate. In these selected reruns, swaps again disappoint and exit 4 beats exit 5, which makes an exact optimum claim for exit 5 untenable. Neither ranking merits changing production given the shared defects.

**Scheme M primary-window sensitivity, same April-2026 end:**

| Start | Baseline return | Variant return | Baseline Sharpe | Variant Sharpe | Delta Sharpe |
|---|---:|---:|---:|---:|---:|
| 2023-05-15 | +1840.7% | +2040.2% | 1.940 | 1.901 | -0.039 |
| 2024-05-01 | +1175.0% | +1626.2% | 2.132 | 2.277 | +0.146 |
| 2025-01-02 | +302.7% | +374.8% | 1.841 | 2.049 | +0.208 |
| 2025-05-01 | +483.4% | +523.9% | 3.046 | 3.125 | +0.079 |

The earliest primary window yields a mildly negative Sharpe difference; the 20-path mean difference is +0.004. This distinguishes calendar sensitivity from path-count effects and does not establish statistical equivalence. Current data/configuration do not exactly reproduce the historical May-11 figures. The original dated vendor snapshot and exact commands were not preserved, so this is an independent sensitivity check with all inputs identified, not a claim to have disproved the original arithmetic.

**Tie sensitivity:** C entry 9 / exit 5, 2025-05-01 through 2026-04-30. I randomized a fixed ticker-priority permutation at seeds 0–19, applied only when scores tie; unequal-score ranking is unchanged and returns never inform priority. Alphabetical priority returns **+544.4%, Sharpe 3.105**. The 20 arbitrary priorities range **+357.4% to +568.3%**, mean +454.6%, std 79.1 pp, with Sharpe 2.769–3.124. The result does not prove alphabetical ordering was selected to overfit, but does refute an inference that small start-date dispersion makes arbitrary capacity allocation unimportant.

**Missing-score sensitivity:** Using the current production score DB and prices through **2026-10-13**, C 9/5 gives:

| Start | Carry-forward return / Sharpe | Exact-row return / Sharpe |
|---|---:|---:|
| 2025-03-25 | +561.8% / 2.138 | +357.5% / 1.831 |
| 2025-07-01 | +328.1% / 1.973 | +357.5% / 2.028 |
| 2026-05-04 | -1.7% / 0.159 | +1.4% / 0.274 |

The exact-row variant supplies only observed snapshots to the same simulator. Its existing missing-score behavior can itself trigger exits or block entry, so **it is not a corrected production replay**; it bounds the sensitivity to removing imputed high scores, not the economic value of a preferred missingness policy. The identical terminal exact-row returns at March/July starts but different Sharpe illustrate how adding a flat-cash prefix changes Sharpe without changing terminal equity. Neither variant uses production's most-recent-three-distinct-row persistence. The May–October check is later observational data, but the current DB was rescored/backfilled, the universe changed, and this simulator does not reproduce broker fills; do not call its return a prospective live-performance measurement.

For scale, the stored daily C parquet gives max drawdown **−51.3%** from the May-2023 primary start versus **−22.8%** from May-2025. This is not a trustworthy loss forecast, but the chosen favorable-period ~20% drawdown is not a defensible universal risk claim even within the existing simulator.


## What would constitute adequate confirmation

1. Rebuild research scores with exact as-of-date peer eligibility, no future/nearest-forward bars, sufficient prehistory, explicit missingness, and versioned universe membership. Include delisted/failed securities and their terminal returns, or narrowly qualify conclusions to a frozen prospective watchlist.
2. Use an event-clock simulator: known-close signals queue next-session orders; book actual modeled fills on their execution date; account for intraday DAY limits, gap-through stops, costs, whole shares, small capital, floors/flex, entry limits, and tradeability. Keep a transparent uncertainty range where daily bars cannot resolve event order.
3. Freeze parameters and an objective before validation. Use chronological walk-forward training/validation/test splits, purge overlapping forward-label windows (at least the label horizon around boundaries), and reserve an untouched period. Compare with equal-capital SPY/SMH and simple daily momentum baselines using consistent cash/exposure accounting.
4. Report paired rule differences, calendar-block uncertainty, turnover/cost sensitivity, concentration/top-winner contribution, fixed-seed tie-order sensitivity, and actual distinct starts. Use fixed-horizon and rolling windows across genuinely different market periods; do not count overlapping paths as independent trials.
5. Maintain an experiment ledger and immutable prices/scores/config hashes. Confirm prospective performance with actual broker fill and equity records; evaluate shadow/spillover on the same capital and fill rules before reopening rotation decisions.

These are review recommendations, not implemented changes. No alternative parameter choice is recommended from the defective historical tests.

## Reproduction record

Scratch evidence and scripts are in `/Users/toddbruschwein/Documents/Codex/2026-10-13/read-docs-handoff-md-and-docs/work/review/`: `probes.py`, `probes.json`, `fetch_prices.py`, `price_manifest.json`, `prices.pkl`, `rerun.py`, `rerun.json`, `audit_checks.py`, `audit_checks.json`, and `history_check.json`. Python 3.9 with pandas 2.3.3, numpy 2.0.2, and yfinance 1.2.0 was used. Imports suppressed bytecode writes; databases were opened read-only; historical snapshots were extracted to scratch, with immutable SQLite access. Yahoo downloads used explicit start 2023-01-01 and exclusive end 2026-10-14, adjusted OHLC, and a scratch timezone cache. All 197 requested current symbols/benchmarks returned data; all 176 audit tickers have a price series in this cache. Fresh vendor data may contain revisions relative to April/May snapshots.

Reruns call the existing simulator with explicit configurations, rather than its misleading default four-strategy comparison. Its original `get_price` was memoized in memory to accelerate repeated paths without changing lookup semantics; production source was untouched. Prices and all input hashes are recorded in the scratch JSON files. The diagnostic date-block bootstrap uses the stored audit, conditional on RS fired, circular 63-date blocks, seed 42, 2,000 draws, and pooled fired-minus-not-fired means. It illustrates dependence sensitivity; it is not a multiple-testing-adjusted or leakage-free confidence interval.

SHA-256 fingerprints of reviewed input artifacts:

- `backtest_results/audit_dataset.parquet`: `872c791618b27de94f86e5418307f79776fb6d7f1478ed6af08b938355b88066`
- `backtest_results/scheme_c_variant_base.parquet`: `111b21b634491e6a33b49d83db5f5886211085e2f7e289136e0342a6e2258fbe`
- `backtest_results/scheme_m_scores.parquet`: `df125b31f61577992c60d74c6c8d9043464d0eefe0dcca89729c055cd074ccc5`
- `breakout_tracker.db`: `6d9858f498a714b4fdc6eaaeeb72ce2e232bb2ef77c24575085268fadfbabbe4`
- Scratch adjusted-price cache: `8641b6feaab0c3595f13b01843e2d3b5e223962e316597ef1277b1967365aa7a`
