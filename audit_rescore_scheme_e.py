from __future__ import annotations

"""
audit_rescore_scheme_e.py — Scheme E: non-monotonic RS curve.

Empirical bucket analysis (audit_rs_fine_grained.py) showed RS forward
returns peak at percentile 90-91, dip at 94-97 (negative median, 47%
win rate), and partially recover at 98-99. This is consistent with
"95-99 cohort = stocks that have already run hard but aren't truly
exceptional yet — likely to mean-revert".

Scheme E rewards the actual signal shape: PEAK at RS 90-91, partial
demotion at 94-97, modest recovery at 98-99. Other indicators preserve
Scheme C's binary structure (Ichimoku 2.0, Dual-TF 2.5, etc.) — no
gradient on those, since the prior gradient experiments showed they
hurt portfolio performance.
"""

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path
import pandas as pd

from indicators import gradient_score


# Non-monotonic RS curve based on the median/win-rate analysis
RS_ANCHORS_NONMONOTONIC = [
    (50, 0.50), (60, 1.00), (70, 1.80), (80, 2.40),
    (88, 2.80),
    (90, 3.00),  # peak — best median (+5.6%) and win rate (56%)
    (91, 2.95),
    (92, 2.85),
    (93, 2.75),
    (94, 2.60),  # dip starts: median goes negative
    (95, 2.55),
    (96, 2.55),
    (97, 2.60),
    (98, 2.75),  # recovery
    (99, 2.85),
    (100, 2.90),  # capped — never quite reaches the 90-91 peak
]

HL_ANCHORS = [(2, 0.125), (3, 0.25), (4, 0.375), (5, 0.50)]


def scheme_e_score(row) -> float:
    """Scheme C with non-monotonic RS gradient."""
    s = 0.0
    s += gradient_score(row.get("rs_percentile", 0) or 0, RS_ANCHORS_NONMONOTONIC)
    s += gradient_score(row.get("higher_lows_count", 0) or 0, HL_ANCHORS)
    # Binary Ichimoku (Scheme C)
    if row.get("ichimoku_fired", 0):
        s += 2.0
    # Binary ROC (Scheme C)
    if row.get("roc_fired", 0):
        s += 1.5
    # Binary Dual-TF (Scheme C)
    if row.get("dual_tf_rs_fired", 0):
        s += 2.5
    # Binary ATR (Scheme C)
    if row.get("atr_fired", 0):
        s += 0.5
    # CMF dropped (weight 0)
    return s


def main(source: Path, output: Path, parquet: Path):
    for suffix in ("", "-shm", "-wal"):
        src = Path(str(source) + suffix)
        dst = Path(str(output) + suffix)
        if src.exists():
            shutil.copy2(src, dst)

    print("loading parquet...")
    p = pd.read_parquet(parquet)
    p["date"] = pd.to_datetime(p["date"]).dt.strftime("%Y-%m-%d")
    indexed = p.set_index(["date", "ticker"]).to_dict("index")

    conn = sqlite3.connect(str(output))
    cur = conn.cursor()
    cur.execute("SELECT rowid, date, ticker FROM ticker_scores")
    rows = cur.fetchall()
    print(f"DB rows: {len(rows):,}")

    updates = []
    misses = 0
    for rowid, dt, ticker in rows:
        row = indexed.get((dt, ticker))
        if row is None:
            misses += 1; continue
        score = round(scheme_e_score(row), 2)
        updates.append((score, rowid))

    cur.executemany("UPDATE ticker_scores SET score = ? WHERE rowid = ?", updates)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM ticker_scores")
    total = cur.fetchone()[0]
    cur.execute("SELECT MIN(score), AVG(score), MAX(score), "
                "SUM(CASE WHEN score >= 8.5 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN score >= 9.0 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN score >= 9.5 THEN 1 ELSE 0 END) "
                "FROM ticker_scores")
    mn, avg, mx, n85, n90, n95 = cur.fetchone()
    conn.close()

    print(f"applied {len(updates):,} updates, misses {misses}")
    print(f"score range: {mn:.2f} → {mx:.2f}, mean {avg:.2f}")
    print(f"selectivity: ≥8.5 {100*n85/total:.2f}%  ≥9.0 {100*n90/total:.2f}%  ≥9.5 {100*n95/total:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("breakout_tracker.db"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, default=Path("backtest_results/audit_dataset.parquet"))
    args = ap.parse_args()
    main(args.source, args.output, args.parquet)
