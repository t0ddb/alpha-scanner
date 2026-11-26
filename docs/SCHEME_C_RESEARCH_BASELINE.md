# Corrected Scheme C research baseline

Study date: **2026-10-18**. Research only; no commits or pushes. Production
trading code, workflows, configuration, databases, and state files were not
changed. This report qualifies historical backtest claims; it does not change
the trading strategy or recommend deployment changes.

## Result

The preserved October 13 review reproduces exactly: **144 of 144 metrics**
match across its four fixed windows and 20 path starts. With the same 20 starts
and April 30, 2026 endpoint, the corrected, constrained baseline has mean return
**+613.0% at $100k** and **+588.9% at $5k**, versus **+1320.5%** in the old
simulator. Mean path Sharpe falls from **1.712** to **1.425 / 1.403**.

These remain retrospective results on a selected watchlist. **Universe and
survivorship bias are unresolved.** Correcting feature and execution chronology
does not establish investable alpha, validate the selected weights, or make
overlapping starts independent evidence. The earliest one-year window produces
only +8.8% at $100k and −1.2% at $5k; favorable later returns dominate the full
sample.

## Scope and frozen rules

The corrected entry point is `research/scheme_c/run.py`, backed by
`research/scheme_c/baseline.py`. The generic root simulators remain explicitly
**legacy** experiment engines. They have not been generalized into a corrected
implementation of every historical trim, swap, trailing-stop, or Scheme M
variant. Do not use their outputs as corrected Scheme C validation.

The root audit builder now aligns peers, features, and forward-label horizons
to exact benchmark dates. The root sizing helpers no longer substitute nearest
prices or carry scores forward, and SQLite loading is read-only. Its CLI warns
that its old transaction chronology is still unsuitable. The corrected engine
implements Scheme C only; the archived pre-edit engine supplies the controls.

| Rule | Frozen value |
|---|---|
| Scheme C weights | RS 3; Ichimoku 2; Dual-TF 2.5; ROC 1.5; Higher Lows 0.5; ATR 0.5; CMF 0 |
| Indicator definitions | Existing RS/HL gradients; 63-session primary RS; 21/63/126 Dual-TF; existing Ichimoku, ROC and ATR formulas |
| Score precision | Production's one decimal; original research parquet used two decimals |
| Entry / persistence | Current score ≥9; three distinct prior observed scores ≥9, strictly before signal date |
| Exit | Observed current score <5; no score means no score exit |
| Stop | Fixed 20% from actual modeled fill; no trailing stop |
| Capacity / sizing | 12 positions; **0.083** of closing equity; no swaps or trimming |
| Whole-share flex | Existing +1-share rule, with 0.012 equity tolerance |
| Minimum / daily cap | Current live $250 estimate-time minimum and four accepted entries per run |
| Cash / limit | Existing 5% estimate-time floor formula; 3% DAY buy limits |
| Entry circuit breaker | Halt new entries if close equity falls more than 10% from previous close |
| Priority | Descending score, then alphabetical ticker |

Current live execution settings are applied throughout every historical window;
this is not a replay of when each operational setting was introduced. Both
$100k and $5k use the same current live constraints. The $100k case is not the
paper configuration, whose YAML minimum is $500. No thresholds, weights, stops,
capacity, or sizing parameters were swept or optimized.

The previous review used 1/12 = 0.0833333 sizing. The corrected engine uses the
actual YAML value 0.083. That small distinction, precision correction, and
other grouped changes are disclosed rather than described as a single-bug
ablation.

## Inputs and historical scores

The 176-symbol universe is frozen from the original score parquet, allowing a
comparison with the review. It is **not** a point-in-time membership ledger or
the current 195-name universe. SPY supplies the session calendar; SMH is used
only as a performance benchmark. Other downloaded symbols are not silently
added to the scoring cross-section.

The prior review's prices, score/audit parquets, scripts/results, original
simulator, scoring code, and configuration were copied before editing. Its
adjusted price cache begins January 2023, which is insufficient for a properly
warmed-up May 2023 score rebuild. A separate public Yahoo capture supplies
January 2022–October 13, 2026 prices, adjusted close, splits, and dividends for
all 197 requested symbols. Neither capture overwrites the other. On the old
scores and old engine, the new price capture changes mean path return by less
than 0.00001 percentage point; vendor differences do not explain the result.

Rebuild rules:

1. Use each target SPY session explicitly. No benchmark ordinal is reused as a
   ticker ordinal, and no nearest-date lookup is allowed.
2. For each RS horizon, require positive, finite prices on every session in
   that ticker's horizon, including exact start/end dates. IPO peers enter only
   when their own as-of history supports that horizon. No future peer data can
   affect another ticker's rank.
3. Use the existing audit's 261-observation scoring warm-up and require 127
   complete recent sessions for all active indicators. This research eligibility
   rule is explicit; it is not a claim to reproduce production's fetch history.
   Crypto peers use the same equity-session horizons rather than mixing weekend
   counts with SPY. Active indicator formulas and thresholds are unchanged.
4. Preserve missingness. Do not create score rows for ineligible/missing data.
   Zero scores remain valid observations. Persistence uses the most recent three
   actual prior rows, including real history before each portfolio start, and
   intentionally permits calendar gaps. Today's score is an additional test.
5. Produce feature/score rows without forward labels. The repaired audit builder
   keeps labels separate and aligns them to benchmark sessions as well.

The output contains **151,294 rows, 175 scored tickers**, April 3, 2023–October
13, 2026. SSNLF never meets complete-history eligibility in this capture and is
listed explicitly rather than silently counted as covered. The saved coverage
and peer-count tables expose eligibility over time.

Against the original 127,290 score rows: 126,547 pair exactly by date/ticker;
743 old rows lack a valid rebuilt counterpart. Among paired rows, 57,989 scores
change, 1,898 cross the entry gate differently, and 4,204 cross the exit gate
differently. One-versus-two-decimal rounding alone changes 48,540 of the full
rebuilt rows. These counts include rounding, calendar/eligibility, peer repair,
and vendor effects; they do not measure leakage alone.

## Event clock and execution assumptions

Each session follows this sequence:

1. Apply effective-date splits to held shares, entry basis per share, stops,
   and pending buy quantities/limits. Accrue dividends earned by prior holders.
2. Fill previously queued market exits at the current open. Their stops were
   canceled after the preceding close; no duplicate sale is possible.
3. Execute existing gap-through stops at the current open, even below the stop.
4. Fill eligible opening buys. Existing intraday stops fill at their fixed stop
   when Low crosses it. Above-limit opening buys can subsequently fill at the
   limit when Low touches it. Unfilled DAY buys expire that session.
5. Mark actual holdings at that session's exact close, adding actual cash and
   dividend receivables. Pending buys are not holdings; pending sales are not
   cash. Missing/invalid held bars abort the run rather than inventing a mark or
   assuming an unknown stop survived.
6. Evaluate close signals and queue next-session orders. Attach stops after the
   entry fill session, matching the existing daily pipeline. An earlier low on
   that session cannot trigger a stop that did not exist. A newly filled holding
   below its stop at close queues the existing close-stop fallback market exit.

The modeled signal batch is **23:59 America/New_York**, after all same-labeled
equity, crypto, and futures daily bars are assumed final and before the next
equity session. This timing matters even for excluded instruments because they
remain scoring peers. A same-date crypto/futures final bar may not yet exist at
the live workflow's scheduled 16:35 run. Daily bars cannot reproduce that
unfinished cross-section; this baseline does not claim to do so. The batch time,
availability assumption, and regular-session Close proxy are saved in settings.

No forced terminal sale is invented. Open holdings are marked, and final-close
orders remain pending outside the window. Closed-trade statistics exclude open
holdings. Equity includes unrealized gains/losses. Daily returns include the
initial cash anchor; Sharpe uses the review's 4% annual risk-free convention,
252 sessions/year and population standard deviation.

The existing sizing formula estimates projected exit proceeds without booking
them. The estimate-time $250 gate occurs **before** integer rounding; a filled
position can consequently be smaller than $250. The +1-share flex can exceed
the per-slot budget while respecting the existing estimated aggregate floor.
Neither rule is silently tightened. All newly submitted constrained entries
are whole shares. Split entitlements may subsequently be fractional.

Daily bars cannot establish the following, so the model makes explicit choices:

| Unknown | Baseline assumption and limitation |
|---|---|
| After-hours sizing quote / actual run time | Use signal-day regular-session Close. No historical Alpaca latest-trade or run-time tape is available. |
| Broker acceptance and buying power | Reserve full limit notional plus modeled friction from cash actually available at submission. Do not spend projected sales. A separate sale-funded scenario accepts projected proceeds but still books sales only at next-open execution and rejects unaffordable fills. Neither reconstructs Alpaca's margin/settlement engine. |
| Limit queue / liquidity | Full fill at Open if Open ≤ limit; otherwise full fill at limit if Low ≤ limit. Positive daily volume required; no partial fills or queue position modeled. Opening-only fills provide an alternative scenario, not a mathematical bound. |
| Within-session event ordering | Existing stops precede intraday buy touches; opening buys precede intraday stops. Baseline reservations prevent unobserved intraday sales from funding those buys. Entry-session stops are inactive throughout. |
| Stop slippage / costs | Stop fill at min(Open, stop) for a gap, otherwise at stop. Deduct a 5-basis-point one-way friction proxy from cash on every fill; also run 0 and 25 bp. This is not a claim about broker commissions or a guarantee of executable stop prices. |
| Split-adjusted vendor data | Undo recorded subsequent splits in Yahoo's non-dividend-adjusted bars to reconstruct historical share denominations. Apply those splits to holdings/orders on their effective dates. Price/action revisions and omitted actions remain possible. Signal ratios use adjusted bars. |
| Dividend payment dates | Accrue a receivable on ex-date; include it in equity but never recycle it into spendable cash during the run. Actual payment dates and withholding are unavailable. |
| Historical tradeability | Enforce existing crypto `-USD` and GC=F/SI=F exclusions. Other securities are assumed tradeable when valid bars exist; no historical Alpaca asset-status registry was supplied. A blocked-symbol input is supported. |
| Halts / operating failures | Invalid held bars fail loudly. Manual kill switch assumed off; daily-loss breaker modeled. No CI outage, rejected stop attachment, cancellation race, discretionary forced trade, deposits, or withdrawals modeled. |

The 5% cash floor is an **order-sizing estimate**, not a guaranteed daily cash
ratio: prices move, flex changes estimates, limits can fill above sizing price,
and friction consumes cash. In the earliest $100k fixed run there are 158 closes
below 5% cash, despite no negative cash days. No new floor-enforcement trading
rule was introduced. A nominal 20% stop likewise does not cap portfolio loss.

## Matched-window comparisons

All four fixed windows end **2026-04-30**. Returns are cumulative percentages;
drawdown is peak-to-trough percent. The previous-review controls are reproduced
from the original cached inputs, not transcribed from rounded report numbers.

| Start | Old return | Corrected $100k return | Old Sharpe | Corrected Sharpe | Old max DD | Corrected max DD | Corrected $5k return |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2023-05-15 | 1559.8% | 792.3% | 1.700 | 1.480 | −51.3% | −49.3% | 709.5% |
| 2024-05-01 | 1219.7% | 714.3% | 2.021 | 1.826 | −51.2% | −49.0% | 697.0% |
| 2025-01-02 | 324.2% | 222.2% | 1.916 | 1.678 | −47.6% | −42.2% | 244.5% |
| 2025-05-01 | 544.4% | 391.5% | 3.105 | 2.807 | −22.8% | −20.3% | 378.7% |

The April deployment headline **+598% / Sharpe 2.21** is a different historical
experiment with a misleading flat-cash prefix and unavailable original vendor
snapshot/command provenance. This study does not claim to reproduce or directly
replace that number. It replaces the October review's explicitly reproducible
comparisons with corrected estimates on matched dates.

## Twenty path starts and grouped corrections

The exact starts are retained in `settings.json`: May 30, June 6/13/20/27,
July 5/12/19/26, August 2/9/16/23/30, September 6/13/20/27, and October 4/11,
**2023**. Every path ends April 30, 2026. All 20 are distinct and executed.
They overlap heavily and have unequal holding horizons; dispersion is a
descriptive sensitivity measure, **not a confidence interval**.

| Stage | Mean return | Return std, pp | Mean path Sharpe |
|---|---:|---:|---:|
| Preserved review engine, scores and prices | 1320.5% | 92.9 | 1.712 |
| Same old engine/scores, supplemental adjusted prices | 1320.5% | 92.9 | 1.712 |
| Correct event clock, old scores, fractional market diagnostic | 1279.8% | 94.9 | 1.697 |
| Correct event clock, rebuilt scores, fractional market diagnostic | 1577.3% | 108.4 | 1.795 |
| Full execution baseline, $100k | 613.0% | 44.7 | 1.425 |
| Full execution baseline, $5k | 588.9% | 45.3 | 1.403 |

The diagnostic stages also repair observation persistence, missing-score exits,
stop chronology, and warm-up access, and use 0.083 sizing. They omit the full
execution constraints and tradeability exclusions and are not tradable results.
Corrections do **not** uniformly reduce returns: rebuilt scores improve this
particular fractional-market path mean while execution constraints materially
reduce it. Do not assign the full difference to look-ahead alone.

### Follow-up attribution: scores interact with execution

The original sequence changes scores before adding execution constraints. To
measure the score effect under the full execution model as well, a follow-up
run completes the missing comparison: **original scores with full execution**.
It uses the archived transformed bars, identical $100k settings, all four fixed
windows and all 20 path starts. No strategy settings or inputs were selected or
changed. A rebuilt-score control reproduces the first fixed-window baseline
metrics exactly before the additional comparisons run.

| Mean 20-path return, corrected event clock | Original scores | Rebuilt scores | Rebuild effect, pp |
|---|---:|---:|---:|
| Fractional-market diagnostic | 1279.8% | 1577.3% | +297.4 |
| Full execution, $100k | 939.5% | 613.0% | −326.5 |
| Execution-package effect, pp | −340.3 | −964.3 | |

One valid sequential accounting is **1320.5 → 1279.8 → 939.5 → 613.0%**:
the grouped clock/persistence/sizing corrections contribute −40.7 percentage
points, full execution on the old scores contributes −340.3 points, and the
score rebuild under full execution contributes −326.5 points. The net decline
is **707.6 percentage points** (differences calculated before rounding).

Changing scores first instead gives +297.4 points for scores and −964.3 points
for the execution package. The **−624.0-point interaction** means neither
ordering supplies a unique additive causal allocation. Changed scores alter
entry/exit timing and priority; cash, whole shares, capacity, order limits and
later compounding then change the portfolio path. It would be incorrect to
conclude that rebuilding scores helps the constrained strategy merely because
it helps the fractional diagnostic. The rebuild also groups date alignment,
peer eligibility, valid history and production precision, so its effect is not
an isolated estimate of the value of removing look-ahead.

The execution package includes tradeability exclusions, reconstructed share
denominations/corporate actions, whole-share sizing/flex/minimums, cash and
funding rules, daily entry/halt limits, DAY limit fills and friction. These
comparisons do not identify individual constraints' contributions. The zero-cost
scenario improves the baseline by only 12.5 points, so the modeled 5-bp friction
alone does not explain the large decline. Sensitivities below are conditional
scenarios, not components that can be added together.

The follow-up command is
`python3 -B -m research.scheme_c.attribution --output research/scheme_c/results/attribution_reproduction`.
The completed `results/attribution_v1/` capture retains settings, input/source
hashes, source, metrics, all 24 transaction/equity ledgers, and an output manifest.
`results/attribution_v1_checks.json` records independent reconciliation of the
additional ledgers. The original baseline captures remain unchanged.

The corrected $100k path returns range **527.3–682.3%**; worst path drawdown is
**−49.3%**. At $5k, the return range is **490.9–663.5%**, with worst drawdown
**−55.4%**. Nearby starts do not make that loss exposure disappear.

## Daily-bar and cost sensitivity

Only execution assumptions change here; the strategy parameters stay fixed.
These scenarios are not independent trials, probability intervals, or guaranteed
best/worst cases.

| $100k scenario | Mean 20-path return | Mean Sharpe | May-2025 fixed-window return |
|---|---:|---:|---:|
| Baseline: intraday limit touches, cash at submission, 5 bp | 613.0% | 1.425 | 391.5% |
| Zero friction | 625.5% | 1.436 | 386.8% |
| 25 bp per fill | 567.8% | 1.385 | 385.0% |
| Opening-only limit fills | 653.0% | 1.464 | 429.5% |
| Allow projected next-open sales to fund acceptance | 681.1% | 1.491 | 454.3% |

Opening-only fills at $5k give mean +655.2%, Sharpe 1.453, versus baseline
+588.9% / 1.403. Skipping a fill can improve later allocation, so a more
restrictive fill assumption need not lower realized return. Costs can likewise
change integer sizing and future allocation; the May-2025 zero-cost result is
slightly below the 5-bp result. This is path sensitivity, not evidence that fees
create an economic edge.

## Separate calendar windows and benchmarks

These roughly one-year windows reduce horizon confounding. They still share the
selected universe and historically selected strategy. SPY/SMH are fully funded,
frictionless adjusted-close buy-and-hold comparisons from first close to final
close; they are useful context, not exposure-matched alpha estimates.

| Window | Scheme C $100k | Scheme C $5k | $100k Sharpe | $100k max DD | SPY | SMH |
|---|---:|---:|---:|---:|---:|---:|
| 2023-05-15–2024-05-14 | 8.8% | −1.2% | 0.297 | −28.5% | 28.5% | 81.1% |
| 2024-05-15–2025-05-14 | 65.8% | 68.4% | 1.010 | −49.0% | 12.3% | 6.5% |
| 2025-05-15–2026-05-14 | 503.3% | 516.5% | 3.064 | −20.7% | 28.2% | 134.8% |
| 2026-05-04–2026-10-13 | 8.7% | 4.3% | 0.545 | −29.8% | 8.9% | 16.0% |

The last row is a later observational extension, overlaps the preceding row,
and uses rebuilt historical data; it is **not actual live performance or an
untouched prospective test**. The small-account result also demonstrates that
whole-share constraints can change both the return and the portfolio path.

The earliest $100k full-window net gain is also concentrated: AXTI contributes
$392,033 of $792,348 net gain (including its terminal unrealized gain). AXTI,
QUBT, LITE, PSIX, and IREN together contribute **72.2%** of that net gain. This
is a description of this selected sample, not a reason to choose those names
prospectively. The contribution ledger reconciles to total account gain.

## Verification and reproduction

For a fresh checkout, first restore the versioned archive and install the pinned
research dependencies using [the research README](../research/scheme_c/README.md).
Git retains selected summaries; the archive supplies every input and full ledger.
The November 7 publication adds six packaging tests and portable current-checkout
integrity checks, without changing this study's preserved source/results.

Run from the repository root:

```sh
python3 -B -m unittest discover -s tests -v
python3 -B -m research.scheme_c.run --output research/scheme_c/results/reproduction
python3 -B -m research.scheme_c.verify research/scheme_c/results/reproduction
```

The final completed study is `research/scheme_c/results/baseline_v2/`:

- `settings.json`: frozen rules, modeling assumptions, exact windows/starts,
  environment versions, universe, and input fingerprints, written before returns.
- `scores.parquet`, `coverage.csv`, `peer_counts.csv`, `paired_scores.parquet`,
  `score_comparison.json`: rebuilt history, eligibility, and row comparisons.
- `signal_prices.parquet`, `execution_prices.parquet`: exact transformed bars.
- `metrics.json`, `path_summary.json`, `benchmarks.json`: all results.
- **272 compressed run files**: full daily equity/holdings, orders, fills,
  expirations/rejections, stops, corporate actions, fees, and closed trades.
- `legacy_reproduction.json`: all 144 old-review metrics match.
- `source/` and `manifest.json`: executed source snapshots and output SHA-256s.
- `production_integrity.json`: 148 protected original root/workflow files checked;
  zero changes. Only the two explicitly repaired research files are exempt.

The final **37 regression tests** cover short-history peer look-ahead, nearest
future bars, benchmark-aligned labels, future-data invariance, production-formula
parity, fabricated persistence, gaps/duplicates/NaNs, pre-start history, signal
versus fill dates, stop gaps and precedence, delayed stop attachment, intraday
limits and expiration, no terminal phantom trades, whole shares/flex/minimums,
four-entry/cap limits, real cash reservations, daily-loss halts, fees, splits,
dividends, and actual bounded path starts.

Independent ledger reconciliation verifies **224 corrected-model runs,
141,772 daily snapshots, and 105,670 fills**, with zero negative-cash days or
position-cap violations, exact-date marks, and every queued fill strictly after
its signal. The 48 deliberately defective legacy controls are excluded from
those accounting assertions. Verification records and the final test source/log
are retained separately in `results/verification/`; the completed study is not
overwritten. Research dependency versions are pinned in
`research/scheme_c/requirements.txt`.

`baseline_v1/` is retained as the first completed capture. The final v2 capture
also freezes the clarified late-evening source-bar availability assumption,
the vendor-timezone alignment regression, and final source/tests. No strategy
parameter changed between these captures. **All 278 compared quantitative
artifacts are byte-identical**, including every one of the 272 run ledgers,
aggregate metrics, scores, and transformed price tables. The comparison is
recorded in `results/verification/reproduction_checks.json`.

## Unresolved limitations and interpretation

There is no historical inclusion/exclusion ledger, comprehensive failed/delisted
security set, or terminal-delisting-return source. Freezing the 176 symbols
prevents this rerun from silently changing membership; **it does not remove
hindsight selection or survivorship bias**. The universe was assembled and the
weights/thresholds selected after substantial knowledge of the measured period.
Missing failed securities cannot be repaired by more path starts, larger price
history, or the new event clock.

Adjusted prices and corporate actions are revised vendor observations, not
archived point-in-time quotes. Historical broker tradeability, after-hours prices,
liquidity, order acceptance, operational outages, and stop coverage are not fully
observed. No formal alpha significance, multiple-testing correction, or independent
validation is claimed. A future validation should preserve an actual prospective
watchlist and broker fills/equity, or use a point-in-time universe with delistings,
before interpreting these returns as deployable evidence.
