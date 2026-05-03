from __future__ import annotations

"""
compute_scheme_m_scores.py — Pre-compute Scheme M scores from
audit_dataset.parquet for use by the backtester.

For each (ticker, date) row in the audit dataset:
  1. Compute Layer 1 base score from indicator values (calls score_ticker_m)
  2. Compute v2 fire flags for that day
  3. Walk back through the ticker's recent history to compute streaks
  4. Apply Layer 2 sequence overlay
  5. Output final score = Layer 1 + Layer 2

Output: backtest_results/scheme_m_scores.parquet
Columns: date, ticker, score (final v2 score), layer_1, layer_2, tags
"""

import argparse
import pandas as pd

import indicators as ind
import sequence_overlay as so


def _row_to_indicators_dict(row) -> dict:
    """Convert a parquet row into the nested-dict format that
    score_ticker_m() and fire_flags_m_from_indicators() expect."""
    return {
        "relative_strength": {
            "rs_percentile": row.get("rs_percentile", 0) or 0,
        },
        "higher_lows": {
            "consecutive_higher_lows": row.get("higher_lows_count", 0) or 0,
            "triggered": (row.get("higher_lows_count", 0) or 0) >= 4,  # v2 threshold
        },
        "ichimoku_cloud": {
            # Recompose composite from ichimoku_score (0-3 integer in parquet)
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


def main(input_path: str, output_path: str):
    print(f"loading {input_path}...")
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows  ({df['date'].min().date()} → {df['date'].max().date()})")

    # ─── Step 1: Layer 1 score per row ──────────────────────────
    print("computing Layer 1 base scores...")
    layer_1_scores = []
    for _, row in df.iterrows():
        result = ind.score_ticker_m(_row_to_indicators_dict(row))
        layer_1_scores.append(result["score"])
    df["layer_1"] = layer_1_scores

    # ─── Step 2: v2 fire flags per row ──────────────────────────
    print("computing v2 fire flags...")
    fire_flag_lists: dict[str, list[int]] = {lbl: [] for lbl in so.LABELS}
    for _, row in df.iterrows():
        flags = so.fire_flags_m_from_indicators(_row_to_indicators_dict(row))
        for lbl in so.LABELS:
            fire_flag_lists[lbl].append(flags[lbl])
    for lbl in so.LABELS:
        df[f"fire_m_{lbl}"] = fire_flag_lists[lbl]

    # ─── Step 3: Streaks per ticker per day ─────────────────────
    print("computing per-ticker streaks (vectorized)...")
    for lbl in so.LABELS:
        col = f"fire_m_{lbl}"
        # Per-ticker: streak resets when fire flag = 0
        is_on = df[col]
        # Group: each new run starts when fire flag changes within a ticker
        groups = (df["ticker"] != df["ticker"].shift()) | (is_on != is_on.shift())
        groups = groups.cumsum()
        cumulative = is_on.groupby(groups).cumsum()
        df[f"streak_m_{lbl}"] = cumulative.where(is_on == 1, 0)

    # ─── Step 4: Layer 2 adjustment per row ─────────────────────
    print("computing Layer 2 sequence adjustments...")
    streak_cols = [f"streak_m_{lbl}" for lbl in so.LABELS]
    layer_2_adj = []
    layer_2_tags = []
    for _, row in df.iterrows():
        streaks = {lbl: int(row[f"streak_m_{lbl}"]) for lbl in so.LABELS}
        features = so.compute_sequence_features(streaks)
        adj, tags = so.compute_layer_2_adjustment(features)
        layer_2_adj.append(adj)
        layer_2_tags.append("|".join(tags))
    df["layer_2"] = layer_2_adj
    df["sequence_tags"] = layer_2_tags

    # ─── Step 5: Final v2 score ─────────────────────────────────
    df["score"] = (df["layer_1"] + df["layer_2"]).round(2)

    # ─── Step 6: Write output ───────────────────────────────────
    out_cols = ["date", "ticker", "score", "layer_1", "layer_2", "sequence_tags"]
    out = df[out_cols].copy()
    print(f"\nwriting {output_path}...")
    out.to_parquet(output_path, index=False)
    print(f"done. {len(out):,} rows written.")

    # ─── Summary stats ──────────────────────────────────────────
    print("\nscore distribution:")
    print(out["score"].describe())
    print(f"\nrows at score >= 9.0: {(out['score'] >= 9.0).sum():,}")
    print(f"rows at score >= 9.5: {(out['score'] >= 9.5).sum():,}")
    print(f"rows at score >= 10.0: {(out['score'] >= 10.0).sum():,}")
    print(f"rows at score >= 10.5: {(out['score'] >= 10.5).sum():,}")
    print(f"rows at score < 0:    {(out['score'] < 0).sum():,}")
    print(f"rows where Layer 2 triggered penalty: {(df['layer_2'] < 0).sum():,}")
    print(f"rows where Layer 2 triggered bonus:   {(df['layer_2'] > 0).sum():,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--output", default="backtest_results/scheme_m_scores.parquet")
    args = ap.parse_args()
    main(args.input, args.output)
