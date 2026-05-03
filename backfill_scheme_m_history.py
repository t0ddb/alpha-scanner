from __future__ import annotations

"""
backfill_scheme_m_history.py — One-time population of ticker_scores_m table
from audit_dataset.parquet.

Computes Scheme M scores (Layer 1 + Layer 2 sequence overlay) for the entire
historical window in the parquet, then inserts into ticker_scores_m.

Run once before deploying shadow_m.py. Subsequent days are populated
by the shadow tracker itself.
"""

import argparse
import pandas as pd

import indicators as ind
import sequence_overlay as so
import subsector_store as store


def _row_to_indicators_dict(row) -> dict:
    """Convert a parquet row into the nested-dict format that
    score_ticker_m() expects."""
    return {
        "relative_strength": {
            "rs_percentile": row.get("rs_percentile", 0) or 0,
        },
        "higher_lows": {
            "consecutive_higher_lows": row.get("higher_lows_count", 0) or 0,
            "triggered": (row.get("higher_lows_count", 0) or 0) >= 4,
        },
        "ichimoku_cloud": {
            "above_cloud": (row.get("ichimoku_score", 0) or 0) >= 1,
            "cloud_bullish": (row.get("ichimoku_score", 0) or 0) >= 2,
            "tenkan_above_kijun": (row.get("ichimoku_score", 0) or 0) >= 3,
            "triggered": bool(row.get("ichimoku_fired", 0)),
        },
        "roc": {
            "roc": row.get("roc_value", 0) or 0,
            "triggered": bool(row.get("roc_fired", 0)),
        },
        "cmf": {
            "cmf": row.get("cmf_value", 0) or 0,
            "triggered": bool(row.get("cmf_fired", 0)),
        },
        "atr_expansion": {
            "atr_percentile": row.get("atr_percentile", 0) or 0,
            "triggered": bool(row.get("atr_fired", 0)),
        },
        "dual_tf_rs": {
            "rs_126d_percentile": row.get("rs_126d_pctl", 0) or 0,
            "rs_63d_percentile":  row.get("rs_63d_pctl", 0) or 0,
            "rs_21d_percentile":  row.get("rs_21d_pctl", 0) or 0,
            "triggered": bool(row.get("dual_tf_rs_fired", 0)),
        },
    }


def _row_record(row, fire_flags, layer_1, layer_2, tags) -> dict:
    return {
        "ticker": row["ticker"],
        "score": round(layer_1 + layer_2, 2),
        "layer_1": round(layer_1, 2),
        "layer_2": round(layer_2, 2),
        "sequence_tags": "|".join(tags) if tags else "",
        "fire_rs": fire_flags["rs"],
        "fire_ich": fire_flags["ich"],
        "fire_hl": fire_flags["hl"],
        "fire_cmf": fire_flags["cmf"],
        "fire_roc": fire_flags["roc"],
        "fire_atr": fire_flags["atr"],
        "fire_dtf": fire_flags["dtf"],
        "rs_pctl": row.get("rs_percentile"),
        "hl_count": row.get("higher_lows_count"),
        "ich_score": row.get("ichimoku_score"),
        "roc_value": row.get("roc_value"),
        "cmf_value": row.get("cmf_value"),
        "atr_pctl": row.get("atr_percentile"),
        "dtf_126d_pctl": row.get("rs_126d_pctl"),
        "dtf_63d_pctl": row.get("rs_63d_pctl"),
    }


def main(input_path: str, db_path: str | None = None,
         start_date: str | None = None):
    print(f"loading {input_path}...")
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if start_date:
        sd = pd.Timestamp(start_date)
        df = df[df["date"] >= sd].copy()
        print(f"  filtered to date >= {start_date}: {len(df):,} rows")

    print(f"  total rows: {len(df):,}")
    print(f"  date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  unique tickers: {df['ticker'].nunique()}")

    # ─── Compute Layer 1 + fire flags per row ───────────────────
    print("\ncomputing Layer 1 scores + fire flags...")
    layer_1_scores = []
    all_flags = {lbl: [] for lbl in so.LABELS}
    for _, row in df.iterrows():
        ind_dict = _row_to_indicators_dict(row)
        result = ind.score_ticker_m(ind_dict)
        layer_1_scores.append(result["score"])
        flags = so.fire_flags_m_from_indicators(ind_dict)
        for lbl in so.LABELS:
            all_flags[lbl].append(flags[lbl])
    df["layer_1"] = layer_1_scores
    for lbl in so.LABELS:
        df[f"fire_{lbl}"] = all_flags[lbl]

    # ─── Compute streaks per ticker ─────────────────────────────
    print("computing streaks (vectorized)...")
    for lbl in so.LABELS:
        col = f"fire_{lbl}"
        is_on = df[col]
        groups = (df["ticker"] != df["ticker"].shift()) | (is_on != is_on.shift())
        groups = groups.cumsum()
        cumulative = is_on.groupby(groups).cumsum()
        df[f"streak_{lbl}"] = cumulative.where(is_on == 1, 0)

    # ─── Compute Layer 2 adjustment per row ─────────────────────
    print("computing Layer 2 adjustments...")
    layer_2_adj = []
    layer_2_tags = []
    for _, row in df.iterrows():
        streaks = {lbl: int(row[f"streak_{lbl}"]) for lbl in so.LABELS}
        features = so.compute_sequence_features(streaks)
        adj, tags = so.compute_layer_2_adjustment(features)
        layer_2_adj.append(adj)
        layer_2_tags.append(tags)
    df["layer_2"] = layer_2_adj
    df["seq_tags"] = layer_2_tags

    # ─── Insert into DB grouped by date ─────────────────────────
    print(f"\ninserting into ticker_scores_m...")
    conn = store.init_db(db_path)

    by_date = df.groupby(df["date"].dt.strftime("%Y-%m-%d"))
    n_dates = len(by_date)
    for i, (date_str, group) in enumerate(by_date, 1):
        records = []
        for _, row in group.iterrows():
            fire_flags = {lbl: int(row[f"fire_{lbl}"]) for lbl in so.LABELS}
            records.append(_row_record(
                row, fire_flags, row["layer_1"], row["layer_2"], row["seq_tags"]
            ))
        store.upsert_ticker_scores_m(conn, date_str, records)
        if i % 100 == 0 or i == n_dates:
            print(f"  {i}/{n_dates} dates done...")

    print("\ndone.")
    print(f"  total v2-score rows in DB: {conn.execute('SELECT COUNT(*) FROM ticker_scores_m').fetchone()[0]:,}")
    print(f"  date range in DB: {conn.execute('SELECT MIN(date), MAX(date) FROM ticker_scores_m').fetchone()}")

    conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--db",    default=None,
                    help="DB path (defaults to ALPHA_DB_PATH or breakout_tracker.db)")
    ap.add_argument("--start-date", default=None,
                    help="Backfill from this date forward (YYYY-MM-DD)")
    args = ap.parse_args()
    main(args.input, args.db, args.start_date)
