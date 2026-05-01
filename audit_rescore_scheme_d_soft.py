from __future__ import annotations

"""
audit_rescore_scheme_d_soft.py — Scheme D-soft: gradient curves, but
preserve Scheme C's reward levels for "barely firing" stocks. Only the
HIGH end gets modest differentiation. Doesn't penalize cohorts that
historically delivered alpha.
"""

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path
import pandas as pd

from indicators import gradient_score


# Scheme D-soft anchor sets — dialed back from original Scheme D
RS_ANCHORS = [
    (50, 0.50), (60, 1.00), (70, 1.80), (80, 2.40),
    (90, 2.70), (92, 2.80), (95, 2.90), (99, 3.00),
]
DUAL_TF_ANCHORS = [
    # No penalty for barely qualifying (was 1.0 in Scheme D, now 2.0)
    (70, 2.00), (80, 2.20), (90, 2.40), (95, 2.50),
]
ROC_ANCHORS = [
    # Was 0.6 at 5% threshold in Scheme D — now 1.2
    (5, 1.20), (10, 1.30), (15, 1.35), (25, 1.40), (50, 1.50),
]
ATR_ANCHORS = [
    # Was 0.2 at 80th in Scheme D — now 0.4
    (80, 0.40), (85, 0.43), (90, 0.46), (95, 0.50),
]
HL_ANCHORS = [(2, 0.125), (3, 0.25), (4, 0.375), (5, 0.50)]


def ichimoku_pts(fired: bool, score_3: int) -> float:
    """Two-tier Ichimoku: not fired = 0; fired + 2/3 = 1.5; fired + 3/3 = 2.0."""
    if not fired:
        return 0.0
    if score_3 >= 3:
        return 2.0
    return 1.5  # fired but only 2/3 (above + bullish, no tenkan/kijun)


def scheme_d_soft_score(row) -> float:
    s = 0.0
    s += gradient_score(row.get("rs_percentile", 0) or 0, RS_ANCHORS)
    s += gradient_score(row.get("higher_lows_count", 0) or 0, HL_ANCHORS)
    s += ichimoku_pts(bool(row.get("ichimoku_fired", 0)),
                      int(row.get("ichimoku_score", 0) or 0))
    if row.get("roc_fired", 0):
        s += gradient_score(row.get("roc_value", 0) or 0, ROC_ANCHORS)
    if row.get("atr_fired", 0):
        s += gradient_score(row.get("atr_percentile", 0) or 0, ATR_ANCHORS)
    if row.get("dual_tf_rs_fired", 0):
        rs_63 = row.get("rs_63d_pctl", 0) or 0
        rs_21 = row.get("rs_21d_pctl", 0) or 0
        s += gradient_score(max(rs_63, rs_21), DUAL_TF_ANCHORS)
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
        score = round(scheme_d_soft_score(row), 2)
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
