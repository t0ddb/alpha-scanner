from __future__ import annotations

"""
audit_rescore_db_scheme_d.py — Rescore production DB under Scheme D.

Scheme D uses gradient curves on continuous indicator values, so we can't
just rescale the per-indicator points stored in signal_weights (those are
Scheme C bucket outputs). Instead, we join each DB row with the audit
parquet (which has the continuous indicator values per (date, ticker))
and recompute the score from scratch.

NEVER touches the production DB — copies first, then rescores the copy.

Usage:
    python3 audit_rescore_db_scheme_d.py \\
        --source breakout_tracker.db \\
        --output breakout_tracker_schemeD.db \\
        --parquet backtest_results/audit_dataset.parquet
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
    RS_GRADIENT_ANCHORS,
    ROC_GRADIENT_ANCHORS,
    ATR_GRADIENT_ANCHORS,
    ICHIMOKU_GRADIENT_ANCHORS,
    DUAL_TF_GRADIENT_ANCHORS,
    HIGHER_LOWS_GRADIENT_ANCHORS,
)


def scheme_d_score_row(row: dict) -> tuple[float, dict]:
    """Compute Scheme D score + per-indicator weight dict for a parquet row."""
    weights = {}

    rs_pts = gradient_score(row.get("rs_percentile", 0) or 0, RS_GRADIENT_ANCHORS)
    if rs_pts > 0: weights["relative_strength"] = round(rs_pts, 3)

    hl_pts = gradient_score(row.get("higher_lows_count", 0) or 0, HIGHER_LOWS_GRADIENT_ANCHORS)
    if hl_pts > 0: weights["higher_lows"] = round(hl_pts, 3)

    ich_pts = gradient_score(row.get("ichimoku_score", 0) or 0, ICHIMOKU_GRADIENT_ANCHORS)
    if ich_pts > 0: weights["ichimoku_cloud"] = round(ich_pts, 3)

    roc_pts = gradient_score(row.get("roc_value", 0) or 0, ROC_GRADIENT_ANCHORS)
    if roc_pts > 0: weights["roc"] = round(roc_pts, 3)

    atr_pts = gradient_score(row.get("atr_percentile", 0) or 0, ATR_GRADIENT_ANCHORS)
    if atr_pts > 0: weights["atr_expansion"] = round(atr_pts, 3)

    rs_63 = row.get("rs_63d_pctl", 0) or 0
    rs_21 = row.get("rs_21d_pctl", 0) or 0
    rs_126 = row.get("rs_126d_pctl", 0) or 0
    cond_a = (rs_126 >= 70) and (rs_63 > rs_126)
    cond_b = (rs_63 >= 80) and (rs_21 >= 80)
    if cond_a or cond_b:
        strength = max(rs_63, rs_21)
        dtf_pts = gradient_score(strength, DUAL_TF_GRADIENT_ANCHORS)
        if dtf_pts > 0:
            weights["dual_tf_rs"] = round(dtf_pts, 3)

    score = round(sum(weights.values()), 2)
    return score, weights


def rescore(db_path: Path, parquet_path: Path) -> dict:
    print(f"  loading parquet: {parquet_path}")
    p = pd.read_parquet(parquet_path)
    p["date"] = pd.to_datetime(p["date"]).dt.strftime("%Y-%m-%d")
    print(f"  parquet: {len(p):,} rows, {p['date'].min()} → {p['date'].max()}")

    # Index parquet by (date, ticker) for fast lookup
    indexed = p.set_index(["date", "ticker"]).to_dict("index")
    print(f"  indexed {len(indexed):,} (date, ticker) cells")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT rowid, date, ticker FROM ticker_scores")
    rows = cur.fetchall()
    print(f"  DB rows to rescore: {len(rows):,}")

    updates = []
    misses = 0
    for rowid, dt, ticker in rows:
        row = indexed.get((dt, ticker))
        if row is None:
            misses += 1
            continue
        score, weights = scheme_d_score_row(row)
        # Reproduce signals list from weights keys (Scheme D weights only > 0 if contributed)
        signals = list(weights.keys())
        updates.append((score, json.dumps(signals), json.dumps(weights), rowid))

    print(f"  rescored: {len(updates):,}  misses (no parquet coverage): {misses:,}")

    cur.executemany(
        "UPDATE ticker_scores SET score = ?, signals = ?, signal_weights = ? WHERE rowid = ?",
        updates,
    )
    conn.commit()

    # Verification
    cur.execute("SELECT MIN(score), AVG(score), MAX(score), "
                "SUM(CASE WHEN score >= 8.55 THEN 1 ELSE 0 END), COUNT(*) "
                "FROM ticker_scores")
    mn, avg, mx, n_ge_855, total = cur.fetchone()
    cur.execute("SELECT SUM(CASE WHEN score >= 9.0 THEN 1 ELSE 0 END) FROM ticker_scores")
    n_ge_90 = cur.fetchone()[0]
    cur.execute("SELECT SUM(CASE WHEN score >= 9.5 THEN 1 ELSE 0 END) FROM ticker_scores")
    n_ge_95 = cur.fetchone()[0]
    conn.close()

    print(f"\n  Verification:")
    print(f"    score range: {mn:.2f} → {mx:.2f}, mean {avg:.2f}")
    print(f"    score ≥ 8.55: {n_ge_855:,} of {total:,} ({100*n_ge_855/total:.2f}%)  "
          f"[matched-selectivity threshold]")
    print(f"    score ≥ 9.0:  {n_ge_90:,} ({100*n_ge_90/total:.2f}%)")
    print(f"    score ≥ 9.5:  {n_ge_95:,} ({100*n_ge_95/total:.2f}%)")

    return {"rescored": len(updates), "misses": misses}


def main(source: Path, output: Path, parquet: Path):
    if not source.exists():
        print(f"ERROR: source DB not found at {source}")
        sys.exit(1)
    if not parquet.exists():
        print(f"ERROR: parquet not found at {parquet}")
        sys.exit(1)

    for suffix in ("", "-shm", "-wal"):
        src = Path(str(source) + suffix)
        dst = Path(str(output) + suffix)
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied {src.name} → {dst.name}")

    rescore(output, parquet)
    print(f"\n  Done. DB_PATH={output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("breakout_tracker.db"))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--parquet", type=Path, default=Path("backtest_results/audit_dataset.parquet"))
    args = ap.parse_args()
    main(args.source, args.output, args.parquet)
