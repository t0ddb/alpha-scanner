from __future__ import annotations

"""
audit_rescore_scheme_d_variant.py — Rescore DB under Scheme D with
parametrized RS anchors. Used to test multiple anchor variants without
modifying indicators.py.

Usage:
    python3 audit_rescore_scheme_d_variant.py --variant mild --output breakout_tracker_schemeD_mild.db
    python3 audit_rescore_scheme_d_variant.py --variant mod --output breakout_tracker_schemeD_mod.db
"""

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
import pandas as pd

from indicators import (
    gradient_score,
    ROC_GRADIENT_ANCHORS,
    ATR_GRADIENT_ANCHORS,
    ICHIMOKU_GRADIENT_ANCHORS,
    DUAL_TF_GRADIENT_ANCHORS,
    HIGHER_LOWS_GRADIENT_ANCHORS,
)


# Three RS anchor variants — keep everything else as Scheme D
RS_VARIANTS = {
    # Original Scheme D — for reference
    "orig": [(50, 0.25), (60, 0.50), (70, 1.00), (80, 1.50),
             (90, 2.00), (95, 2.50), (99, 3.00)],

    # Mild: barely penalize 90 cohort (-5%), tight top spread
    "mild": [(50, 0.25), (60, 0.50), (70, 1.00), (80, 2.00),
             (90, 2.85), (92, 2.88), (95, 2.92), (99, 3.00)],

    # Moderate: 90→2.7 keeps the cohort near max, +0.30 spread to 99
    "mod":  [(50, 0.25), (60, 0.50), (70, 1.00), (80, 2.00),
             (90, 2.70), (92, 2.80), (95, 2.90), (99, 3.00)],

    # Slightly tighter than orig but recover 90 cohort partially
    "med":  [(50, 0.25), (60, 0.50), (70, 1.00), (80, 1.80),
             (90, 2.50), (92, 2.65), (95, 2.80), (99, 3.00)],
}


def scheme_d_score_row(row, rs_anchors):
    s = 0.0
    s += gradient_score(row.get("rs_percentile", 0) or 0, rs_anchors)
    s += gradient_score(row.get("higher_lows_count", 0) or 0, HIGHER_LOWS_GRADIENT_ANCHORS)
    s += gradient_score(row.get("ichimoku_score", 0) or 0, ICHIMOKU_GRADIENT_ANCHORS)
    s += gradient_score(row.get("roc_value", 0) or 0, ROC_GRADIENT_ANCHORS)
    s += gradient_score(row.get("atr_percentile", 0) or 0, ATR_GRADIENT_ANCHORS)

    rs_63 = row.get("rs_63d_pctl", 0) or 0
    rs_21 = row.get("rs_21d_pctl", 0) or 0
    rs_126 = row.get("rs_126d_pctl", 0) or 0
    cond_a = (rs_126 >= 70) and (rs_63 > rs_126)
    cond_b = (rs_63 >= 80) and (rs_21 >= 80)
    if cond_a or cond_b:
        s += gradient_score(max(rs_63, rs_21), DUAL_TF_GRADIENT_ANCHORS)
    return s


def main(source: Path, output: Path, parquet: Path, variant: str):
    if variant not in RS_VARIANTS:
        print(f"ERROR: unknown variant '{variant}'. Choices: {list(RS_VARIANTS.keys())}")
        sys.exit(1)

    rs_anchors = RS_VARIANTS[variant]
    print(f"variant '{variant}': RS anchors = {rs_anchors}")

    for suffix in ("", "-shm", "-wal"):
        src = Path(str(source) + suffix)
        dst = Path(str(output) + suffix)
        if src.exists():
            shutil.copy2(src, dst)

    print(f"loading parquet...")
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
            misses += 1
            continue
        score = round(scheme_d_score_row(row, rs_anchors), 2)
        updates.append((score, rowid))

    cur.executemany("UPDATE ticker_scores SET score = ? WHERE rowid = ?", updates)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM ticker_scores")
    total = cur.fetchone()[0]
    cur.execute("SELECT SUM(CASE WHEN score >= 8.5 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN score >= 9.0 THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN score >= 9.5 THEN 1 ELSE 0 END) "
                "FROM ticker_scores")
    n85, n90, n95 = cur.fetchone()
    conn.close()

    print(f"applied {len(updates):,} updates, misses {misses}")
    print(f"selectivity: ≥8.5 {100*n85/total:.2f}%  ≥9.0 {100*n90/total:.2f}%  ≥9.5 {100*n95/total:.2f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("breakout_tracker.db"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, default=Path("backtest_results/audit_dataset.parquet"))
    ap.add_argument("--variant", required=True, choices=list(RS_VARIANTS.keys()))
    args = ap.parse_args()
    main(args.source, args.output, args.parquet, args.variant)
