from __future__ import annotations

"""
audit_rescore_db_with_filter.py — Apply a pct_from_52w_high filter to an
already-rescored DB by zeroing scores where the filter excludes the signal.

Uses the audit parquet to look up pct_from_52w_high per (date, ticker).

Usage:
    python3 audit_rescore_db_with_filter.py \\
        --source breakout_tracker_schemeC.db \\
        --output breakout_tracker_schemeC_52wfilter.db \\
        --max-pct-from-high -0.05      # exclude rows within 5% of 52w high
"""

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

import pandas as pd


def apply_filter(db_path: Path, parquet_path: Path,
                 max_pct: float | None, min_pct: float | None,
                 cap_to: float = 0.0) -> None:
    """Cap scores where filter excludes the signal.

    cap_to: score value to use for filtered rows. 0.0 forces exits too.
            Use 8.4 to gate ENTRIES only (below 8.5/9.0 entry thresholds,
            but above 5.0 exit threshold — held positions stay held).
    """
    print(f"  loading audit parquet from {parquet_path}")
    audit = pd.read_parquet(parquet_path)
    audit["date"] = pd.to_datetime(audit["date"]).dt.strftime("%Y-%m-%d")
    lookup = audit.set_index(["date", "ticker"])["pct_from_52w_high"].to_dict()
    print(f"  {len(lookup):,} (date, ticker) pairs have pct_from_52w_high")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT rowid, date, ticker, score FROM ticker_scores")
    rows = cur.fetchall()
    print(f"  DB has {len(rows):,} ticker_scores rows")

    updates = []
    n_filtered = 0
    n_missing = 0
    for rowid, date, ticker, score in rows:
        pct = lookup.get((date, ticker))
        if pct is None:
            n_missing += 1
            continue
        exclude = False
        if max_pct is not None and pct > max_pct:
            exclude = True
        if min_pct is not None and pct < min_pct:
            exclude = True
        if exclude and score > cap_to:
            updates.append((cap_to, rowid))
            n_filtered += 1

    print(f"  rows excluded by filter (set to score=0): {n_filtered:,}")
    print(f"  rows without parquet coverage (unchanged): {n_missing:,}")

    cur.executemany("UPDATE ticker_scores SET score = ? WHERE rowid = ?", updates)
    conn.commit()

    # Verification
    cur.execute("SELECT SUM(CASE WHEN score >= 8.5 THEN 1 ELSE 0 END), COUNT(*) FROM ticker_scores")
    n_ge_85, total = cur.fetchone()
    cur.execute("SELECT SUM(CASE WHEN score >= 9.0 THEN 1 ELSE 0 END) FROM ticker_scores")
    n_ge_90 = cur.fetchone()[0]
    print(f"\n  Verification:")
    print(f"    score >= 8.5: {n_ge_85:,} of {total:,} ({100 * n_ge_85 / total:.2f}%)")
    print(f"    score >= 9.0: {n_ge_90:,} of {total:,} ({100 * n_ge_90 / total:.2f}%)")
    conn.close()


def main(source: Path, output: Path, parquet: Path,
         max_pct: float | None, min_pct: float | None,
         cap_to: float = 0.0):
    if not source.exists():
        print(f"ERROR: source DB not found at {source}")
        sys.exit(1)

    for suffix in ("", "-shm", "-wal"):
        src = Path(str(source) + suffix)
        dst = Path(str(output) + suffix)
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied {src.name} → {dst.name}")

    apply_filter(output, parquet, max_pct, min_pct, cap_to)
    print(f"\n  Done. DB_PATH={output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, default=Path("backtest_results/audit_dataset.parquet"))
    ap.add_argument("--max-pct-from-high", type=float, default=None,
                    help="exclude rows where pct_from_52w_high > this (e.g. -0.05 = within 5% of high)")
    ap.add_argument("--min-pct-from-high", type=float, default=None,
                    help="exclude rows where pct_from_52w_high < this (e.g. -0.30 = too far below)")
    ap.add_argument("--cap-to", type=float, default=0.0,
                    help="score to set for filtered rows. 0.0 forces exits; "
                         "8.4 gates ENTRIES only (below entry thresholds, above exit threshold).")
    args = ap.parse_args()
    main(args.source, args.output, args.parquet,
         args.max_pct_from_high, args.min_pct_from_high, args.cap_to)
