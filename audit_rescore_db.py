from __future__ import annotations

"""
audit_rescore_db.py — Rescore the ticker_scores table with an alternate
weight scheme, writing the result to a COPY of breakout_tracker.db.

NEVER touches the production DB.

Example:
    python3 audit_rescore_db.py --scheme C --output breakout_tracker_schemeC.db
"""

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path


# Current live weights (from indicators.INDICATOR_WEIGHTS)
CURRENT = {
    "relative_strength": 3.0,
    "ichimoku_cloud":    2.0,
    "higher_lows":       1.0,
    "roc":               1.5,
    "cmf":               1.5,
    "dual_tf_rs":        0.5,
    "atr_expansion":     0.5,
}

SCHEMES = {
    "A_baseline": {
        "relative_strength": 3.0, "ichimoku_cloud": 2.0, "higher_lows": 1.0,
        "roc": 1.5, "cmf": 1.5, "dual_tf_rs": 0.5, "atr_expansion": 0.5,
    },
    "C_drop_cmf_hl": {
        # Drop CMF. Reduce Higher Lows to 0.5. Bump Dual-TF RS to 2.5.
        "relative_strength": 3.0, "ichimoku_cloud": 2.0, "higher_lows": 0.5,
        "roc": 1.5, "cmf": 0.0, "dual_tf_rs": 2.5, "atr_expansion": 0.5,
    },
    "F_minimal_rs_dtf": {
        "relative_strength": 4.0, "ichimoku_cloud": 2.0, "higher_lows": 0.0,
        "roc": 2.0, "cmf": 0.0, "dual_tf_rs": 2.0, "atr_expansion": 0.0,
    },
    # Decomposition of Scheme C — each changes only ONE thing vs baseline
    "C1_drop_cmf_only": {
        "relative_strength": 3.0, "ichimoku_cloud": 2.0, "higher_lows": 1.0,
        "roc": 1.5, "cmf": 0.0, "dual_tf_rs": 0.5, "atr_expansion": 0.5,
    },
    "C2_reduce_hl_only": {
        "relative_strength": 3.0, "ichimoku_cloud": 2.0, "higher_lows": 0.5,
        "roc": 1.5, "cmf": 1.5, "dual_tf_rs": 0.5, "atr_expansion": 0.5,
    },
    "C3_boost_dtf_only": {
        "relative_strength": 3.0, "ichimoku_cloud": 2.0, "higher_lows": 1.0,
        "roc": 1.5, "cmf": 1.5, "dual_tf_rs": 2.5, "atr_expansion": 0.5,
    },
}


def rescore(db_path: Path, scheme_weights: dict[str, float]) -> None:
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    # Verify signal_weights column exists
    cur.execute("SELECT COUNT(*) FROM ticker_scores")
    n_total = cur.fetchone()[0]
    print(f"  rows to rescore: {n_total:,}")

    cur.execute("SELECT rowid, signal_weights FROM ticker_scores")
    updates = []
    seen_keys: set[str] = set()

    for rowid, sw_json in cur.fetchall():
        if not sw_json:
            new_score = 0.0
        else:
            try:
                sw = json.loads(sw_json)
            except Exception:
                sw = {}
            new_score = 0.0
            new_weights = {}
            for indicator, pts in sw.items():
                seen_keys.add(indicator)
                cur_w = CURRENT.get(indicator)
                new_w = scheme_weights.get(indicator)
                if cur_w is None or new_w is None:
                    continue
                if cur_w == 0:
                    continue
                scaled = pts * (new_w / cur_w)
                new_score += scaled
                if scaled > 0:
                    new_weights[indicator] = round(scaled, 3)
            updates.append((round(new_score, 1), json.dumps(new_weights), rowid))

    print(f"  indicator keys observed in DB: {sorted(seen_keys)}")

    # Apply in bulk
    cur.executemany(
        "UPDATE ticker_scores SET score = ?, signal_weights = ? WHERE rowid = ?",
        updates,
    )
    conn.commit()
    conn.close()
    print(f"  applied {len(updates):,} updates")


def main(scheme: str, source_db: Path, output_db: Path):
    if scheme not in SCHEMES:
        print(f"ERROR: unknown scheme '{scheme}'. Available: {list(SCHEMES.keys())}")
        sys.exit(1)

    if not source_db.exists():
        print(f"ERROR: source DB not found at {source_db}")
        sys.exit(1)

    # Copy .db, .db-shm, .db-wal if present
    for suffix in ("", "-shm", "-wal"):
        src = Path(str(source_db) + suffix)
        dst = Path(str(output_db) + suffix)
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  copied {src.name} → {dst.name}")

    # Rescore
    rescore(output_db, SCHEMES[scheme])

    # Quick verification
    conn = sqlite3.connect(str(output_db))
    cur = conn.cursor()
    cur.execute("SELECT MIN(score), AVG(score), MAX(score), "
                "SUM(CASE WHEN score >= 8.5 THEN 1 ELSE 0 END) FROM ticker_scores")
    mn, avg, mx, n_ge_85 = cur.fetchone()
    cur.execute("SELECT COUNT(*) FROM ticker_scores")
    total = cur.fetchone()[0]
    conn.close()
    print(f"\n  Verification:")
    print(f"    score range: {mn:.2f} → {mx:.2f}, mean {avg:.2f}")
    print(f"    score >= 8.5: {n_ge_85:,} of {total:,} ({100 * n_ge_85 / total:.2f}%)")
    print(f"\n  Done. Use DB_PATH={output_db} for the backtest.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", required=True, choices=list(SCHEMES.keys()))
    ap.add_argument("--source", default="breakout_tracker.db", type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    main(args.scheme, args.source, args.output)
