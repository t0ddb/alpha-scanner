# Scheme C cash-allocation experiment

Study date: **2026-10-24**. Research only. Live trading, workflows,
configuration and production state are unchanged. No commit or push.

## Finding

Removing the division of available investment budget across all empty slots
increases mean cumulative return in the 20 matched paths at both account sizes.
It also makes maximum drawdown worse in **every one of the 28 paired windows
at each size**. The $5k result has modestly higher mean Sharpe; $100k has
slightly lower mean Sharpe and mixed shorter-window returns. This is an
exposure/allocation tradeoff, not an improvement on all performance measures.
No variant is promoted to live trading.

| Same 20 starts, ending 2026-04-30 | $5k baseline | $5k sequential | $100k baseline | $100k sequential |
|---|---:|---:|---:|---:|
| Mean cumulative return | 588.9% | 738.6% | 613.0% | 673.8% |
| Change in mean return, percentage points | — | +149.7 | — | +60.8 |
| Mean path Sharpe | 1.403 | 1.446 | 1.425 | 1.412 |
| Worst path max drawdown | −55.4% | −57.4% | −49.3% | −54.8% |
| Return standard deviation, percentage points | 45.3 | 66.7 | 44.7 | 57.0 |
| Lowest path return | 490.9% | 592.6% | 527.3% | 551.1% |
| Highest path return | 663.5% | 879.5% | 682.3% | 765.7% |
| Mean daily cash / equity | 23.1% | 18.6% | 21.6% | 19.2% |

At $5k, all 20 paired starts have higher returns and higher Sharpe, but deeper
drawdowns. At $100k, 19/20 have higher returns, only 4/20 have higher Sharpe,
and all have deeper drawdowns. These overlapping paths are start-date
sensitivities, not independent trials or statistical confidence. Changes above
are percentage points, not percentage increases in final wealth. Returns are
cumulative, not annualized.

## The single change

```text
budget = equity × (1 − cash_floor) − estimated_committed

baseline cap   = max(0, budget / max(1, remaining_slots))
sequential cap = max(0, budget)

target = min(projected_cash, equity × 0.083, cap)
```

Only that denominator changes. Candidates retain descending score/alphabetical
priority. The $250 estimate-time minimum, whole-share rounding, existing
+1-share flex with 0.012 tolerance, 12-position cap, four-entry daily cap,
3% DAY limits, 5% floor formula and 5-bp per-fill friction are unchanged.
All scoring, persistence, exits, the 20% stop and its existing delayed
attachment, daily-loss entry halt, tradeability exclusions, corporate actions,
fills and valuation are unchanged. Both accounts use the baseline's current-live
settings; $100k does not switch to paper-account minimums.

Every accepted order still reserves its full limit notional plus friction from
cash already available at submission. Projected sales do not fund acceptance.
An unaffordable order is rejected, not resized to the last dollar. No borrowing,
fractional entries, new sale-funding assumption or market-risk filter is added.

For example, at $6,000 equity, $1,160 cash, $4,840 committed and four empty
slots, the investment budget is $860. Baseline divides it into a $215 cap,
below the $250 minimum. Sequential allocation can use the $498 target before
unchanged rounding/flex and reservation checks. This synthetic example is
tested; it does not imply that every historical skipped entry is recovered.

The 5% floor retains its **estimate-time** meaning, not a guaranteed closing
cash ratio. Prices, limit fills and friction can leave closing cash below 5%.
Mean floor-breach days per path rise from 86.7 to 109.4 at $5k and 134.5 to
159.3 at $100k. All runs retain nonnegative cash.

## Inputs, windows and control

The exact rebuilt scores and transformed execution bars come from
`research/scheme_c/results/baseline_v2/`, using the October 18 data snapshot.
No new prices through October 24 were downloaded. Universe, dates, event-clock
assumptions, benchmark sessions and parameter values are identical between
allocators.

Each allocator runs at $5,000 and $100,000 over the original four fixed windows,
20 path starts and four additional calendar/observational windows: **112 runs**,
including 56 controls and 56 challengers. Exact dates and settings are written
before simulations in `results/allocation_v2/settings.json`.

Separate in-memory instances of the archived engine keep the control independent.
The challenger delegates to its sizing function with a denominator of one;
actual portfolio capacity remains 12. The engine source on disk is unchanged.
**All 56 controls match the original baseline's full ledgers exactly**, including
orders, fills, positions, cash snapshots, counters and metrics.

## Four fixed windows

All end April 30, 2026. Values are cumulative returns.

| Start | $5k baseline | $5k sequential | Difference, pp | $100k baseline | $100k sequential | Difference, pp |
|---|---:|---:|---:|---:|---:|---:|
| 2023-05-15 | 709.5% | 864.1% | +154.7 | 792.3% | 814.4% | +22.0 |
| 2024-05-01 | 697.0% | 829.5% | +132.5 | 714.3% | 813.1% | +98.8 |
| 2025-01-02 | 244.5% | 271.5% | +27.1 | 222.2% | 189.2% | −33.1 |
| 2025-05-01 | 378.7% | 414.3% | +35.7 | 391.5% | 348.0% | −43.6 |

All four $5k fixed-window Sharpes improve slightly; all four $100k Sharpes
decline. Maximum drawdowns deepen in every pair. The earliest fixed window's
drawdown moves from −50.9% to −57.4% at $5k, and −49.3% to −54.7% at $100k.
The higher path mean should not obscure the $100k return declines in the two
shorter fixed windows.

## Separate calendar windows

| Window | $5k baseline | $5k sequential | $100k baseline | $100k sequential |
|---|---:|---:|---:|---:|
| 2023-05-15–2024-05-14 | −1.2% | −6.0% | 8.8% | −2.7% |
| 2024-05-15–2025-05-14 | 68.4% | 85.8% | 65.8% | 101.4% |
| 2025-05-15–2026-05-14 | 516.5% | 651.3% | 503.3% | 480.6% |
| 2026-05-04–2026-10-13 | 4.3% | 5.0% | 8.7% | 5.4% |

The weak earliest year becomes worse at both sizes. The $100k challenger wins
only one of these four comparisons. The last row is a later observational
extension, overlaps the previous row, and is neither live performance nor an
untouched prospective test. Full metrics and paired changes are retained.

## What changed in the portfolios

| Mean over the 20 paths | $5k baseline | $5k sequential | $100k baseline | $100k sequential |
|---|---:|---:|---:|---:|
| Daily invested / equity | 76.8% | 81.4% | 78.3% | 80.7% |
| Number of positions held | 9.38 | 9.09 | 9.68 | 9.39 |
| Buy fills | 260.2 | 245.0 | 266.5 | 257.5 |
| Submission rejections for limit funding | 145.4 | 174.5 | 152.7 | 187.6 |
| Floor/minimum skips | 120.7 | 433.1 | 0.0 | 158.5 |
| Capacity skips | 690.5 | 451.3 | 701.7 | 602.0 |

The variant invests more dollars on average but holds fewer names and makes
fewer purchases. Larger early allocations consume budget, change later funding
and compound into different portfolios. The initial hypothesis that this would
simply clear minimum-size skips is not borne out by the aggregate counters.
Counters describe repeated candidate-day events on different portfolio paths,
not isolated causal effects or unique investment opportunities.

In the earliest fixed window, the top five names contribute 73.4% versus 72.4%
of net gain at $5k, and 72.2% versus 77.2% at $100k. AXTI remains the largest
contributor. At $100k its contribution rises about $31,975, while LITE's falls
about $41,530; other changes leave total gain only $22,048 higher. Contributions
include realized and terminal unrealized gains and reconcile to account gains.
They do not justify selecting those names prospectively.

## Verification and reproduction

For a fresh checkout, first restore the versioned archive and install the pinned
research dependencies using [the research README](../research/scheme_c/README.md).
It supplies the complete parent baseline, inputs, and historical ledgers. The
November 7 publication rerun reproduces all 112 ledgers and three quantitative
summaries; six packaging tests bring the current suite to 51 tests. Original
capture metadata and the 45-test record below remain historical evidence.

```sh
python3 -B -m unittest discover -s tests -v
python3 -B -m research.scheme_c.allocation --output research/scheme_c/results/allocation_reproduction
python3 -B -m research.scheme_c.verify research/scheme_c/results/allocation_reproduction
```

Run from the repository root. The runner is offline, verifies the entire parent
manifest, and refuses an existing destination. `results/allocation_v2/` retains
settings, archived sources, exact score/price tables, 112 compressed ledgers,
metrics, path summaries, paired comparisons, contributions, reproduction and
integrity records, independent verification, and a SHA-256 output manifest.

All **45 tests** pass, including eight allocation-specific cases. Independent
accounting reconciles **112 runs, 65,704 daily snapshots and 47,910 fills**.
Acceptance checks cover **25,400 submitted orders**: actual-cash reservations,
whole shares, sizing tolerance, priority and daily entry limits. No negative
cash or position-cap violations occur. Queued fills follow their signals and
marks use exact dates. **150 protected root/workflow files** are unchanged.

`allocation_v1/` is preserved as incomplete: simulations completed, but summary
export failed on a NumPy integer count before final verification. The only
repair converts those counts to native integers; rules and inputs did not
change. Its 112 ledgers and three complete quantitative summaries are
byte-identical to v2. Logs and final checks are in `results/allocation_checks/`.
The experiment ledger records both attempts.

## Interpretation

At $5k this is a candidate for further validation: the tested start-date paths
improve return and Sharpe, but drawdowns and dispersion increase and the weak
calendar year worsens. At $100k the higher average long-window return comes
with lower average Sharpe, deeper drawdowns and several shorter-window losses
relative to baseline. Neither establishes a clear live improvement under an
agreed return/risk objective.

Universe/survivorship bias remains unresolved: there is no historical membership
ledger or comprehensive failed/delisted security set. All daily-bar assumptions,
delayed stop attachment, revised vendor data, missing liquidity/partial fills
and historical broker-tradeability limits from
[the baseline report](SCHEME_C_RESEARCH_BASELINE.md) remain. These results do
not establish prospective alpha. No further parameter search or production
change was made in response to the outcomes.
