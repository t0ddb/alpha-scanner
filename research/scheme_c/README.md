# Scheme C research baseline

Research only. Nothing here connects to a broker or opens the production database.
The authoritative method and results are in
[the baseline report](../../docs/SCHEME_C_RESEARCH_BASELINE.md).

## Restore the frozen research bundle

Git contains the research code, tests, reports, frozen source/configuration, and
selected summary results. The complete inputs and historical captures are in the
[versioned research archive](https://github.com/t0ddb/alpha-scanner/releases/download/research-scheme-c-2026-11-07/scheme-c-research-2026-11-07.tar.gz)
(about 164 MiB compressed; 913 files). `bundle.json` records the archive SHA-256
and every member's size and SHA-256. It includes the original and superseded
captures, with no production database or live account state.

Download the archive, then run from the repository root:

```sh
python3 -B -m research.scheme_c.restore_bundle --archive /path/to/scheme-c-research-2026-11-07.tar.gz
python3 -m pip install -r research/scheme_c/requirements.txt
```

The restore command validates the entire archive before writing. It preserves
identical existing files and refuses conflicts, unexpected members, traversal,
or symlink destinations. Restore the bundle before running any full study; the
selected summaries in Git alone are insufficient. Provenance paths in archived
metadata identify the original machine, not required locations on a new machine.

From the repository root, after restoring:

```sh
python3 -B -m unittest discover -s tests -v
python3 -B -m research.scheme_c.run --output research/scheme_c/results/reproduction
```

The rerun is **offline** and refuses an existing output directory. It verifies
the preserved input hashes, rebuilds scores, reproduces the old review, runs
the corrected comparisons, writes full transaction/equity ledgers and metrics,
and compares current root files and workflow fingerprints before and after the
run. Pick a new output directory each time. Reruns do not commit, push, or invoke
production workflows.

The final captured study is `results/baseline_v2/`; `results/verification/`
contains the test log, independent ledger checks, and byte-for-byte reproduction
check. `experiment_ledger.json` records the completed captures and the follow-up
score-by-execution attribution. `results/attribution_v1/` completes the missing
original-scores/full-execution comparison; its independent reconciliation is in
`results/attribution_v1_checks.json`. Reproduce it offline with:

```sh
python3 -B -m research.scheme_c.attribution --output research/scheme_c/results/attribution_reproduction
```

`inputs/` contains the original review prices/results/code, original score and
audit parquets, frozen configuration and scoring code, plus the supplemental
2022–2026 vendor capture with corporate actions. `sources.json` identifies the
original locations and hashes. Bulk inputs and complete ledgers are ignored by
Git and restored from the archive; selected small files are explicitly tracked.

`fetch_inputs.py` is the one-time public-data capture, **not a rerun dependency**.
It refuses to replace the vendor capture. A fresh download is a different input
snapshot and must not be represented as an exact reproduction.

The generic root `sizing_comparison_backtest.py` and `entry_mode_backtest.py`
remain legacy experiment engines. Their historical returns are not corrected
performance estimates. The standalone `baseline.py` is the corrected Scheme C
event engine; the archived pre-edit engine is used only as a comparison control.

All shares, orders, stops, cash, fees, dividends, and marks in these outputs are
simulated. The selected historical watchlist and unavailable delisting history
remain material limitations.

The October 24 [cash-allocation experiment](../../docs/SCHEME_C_CASH_ALLOCATION.md)
compares the baseline with one change: use remaining investment budget without
dividing it by every empty slot. It retains every other rule at $5k and $100k.
The completed capture is `results/allocation_v2/` (112 runs; all 56 control
ledgers reproduce exactly). Mean path return increases but drawdowns deepen.
Reproduce offline with a fresh destination:

```sh
python3 -B -m research.scheme_c.allocation --output research/scheme_c/results/allocation_reproduction
```

## Publication verification, 2026-11-07

The research branch starts from the November 7 production revision, preserving
all newer trading-state updates. Publication adds portable integrity checks:
reruns compare files actually present in the current checkout, rather than
requiring October's ignored local backups or matching its old trading state.
The archived studies and their integrity records remain unchanged. Scoring,
simulation, and strategy parameters are unchanged by this packaging update.

All 51 tests pass, including six archive/integrity cases. A fresh baseline rerun
matches all 278 quantitative artifacts byte for byte; the allocation rerun
matches all 112 ledgers and three quantitative summaries. Independent accounting
checks pass again. The full archive restores into an empty directory and a second
restore preserves every file. See `publication_validation.json` for the counts
and hashes. This publication does not adopt the allocation variant for live use
or resolve the universe/survivorship bias documented in the reports.
