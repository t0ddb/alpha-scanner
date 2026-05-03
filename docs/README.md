# docs/ — Documentation map

**Where to start:** `HANDOFF.md` (project orientation, current state).

## Living documents (kept current)

| Doc | Purpose | When to read |
|---|---|---|
| `HANDOFF.md` | Project state. What's deployed, current config, files to know, gotchas. | First, when picking up the project. |
| `DECISIONS.md` | Append-only chronological log of major decisions + rationale + commit references. | When you need to know *why* something is the way it is. |
| `ALPHA_SCANNER_DOCUMENTATION.md` | Technical deep-dive on architecture, scoring engine, state machine. | When you need to understand internals beyond what HANDOFF.md covers. |
| `PATH_C_SCORING.md` | Spec for the Path C shadow scoring scheme (Layer 1 indicator buckets + Layer 2 sequence overlay). | When working on `shadow_pathc.py`, `sequence_overlay.py`, or `compute_scheme_i_plus_scores.py`. |

## Specs (reference; updated when their module changes)

| Doc | What it specs |
|---|---|
| `ALPACA_PAPER_TRADING_SPEC.md` | Original spec for Alpaca paper trading integration (`trade_executor.py`). |
| `MINERVINI_BACKTEST_SPEC.md` | Spec for `minervini_backtest.py` (built but never executed). |
| `PORTFOLIO_BACKTEST_SPEC.md` | Spec for the original `portfolio_backtest.py` (legacy; superseded by `sizing_comparison_backtest.py`). |
| `PORTFOLIO_ANALYSIS_SPEC.md` | Spec for portfolio post-backtest analysis tooling. |
| `QUARTERLY_REVIEW_SPEC.md` | Spec for `quarterly_review.py` (the 7-section health report). |
| `THRESHOLD_OPTIMIZER_SPEC.md` | Spec for the threshold optimizer (built; results consolidated into Scheme C audit). |
| `NEW_INDICATORS_PROPOSAL.md` | Historical proposal document for indicator additions. |

## Documentation philosophy

We use **living docs + append-only decision log**:
- Living docs (`HANDOFF.md`, etc.) reflect CURRENT state. Past versions
  available via `git show <hash>:docs/HANDOFF.md`.
- Decision log (`DECISIONS.md`) captures the WHY across time. Never edit
  past entries — supersede with new entries that reference the old.
- Active investigations get a working doc; on resolution, content distills
  into HANDOFF.md update + DECISIONS.md entry, then the working doc is
  deleted (git history preserves it).

This avoids the "library of stale dated handoffs" failure mode. See
`DECISIONS.md` 2026-05-01 entry for the rationale.

## Images

`images/` directory contains architecture diagrams referenced by
`README.md` (root) and `ALPHA_SCANNER_DOCUMENTATION.md`.
