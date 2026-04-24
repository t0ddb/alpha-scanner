from __future__ import annotations

"""
audit_compare_weights.py — Compare alternate indicator-weight schemes on
signal quality (forward return, win rate, Spearman ρ) without re-running
the full indicator pipeline.

Works by rescaling the per-indicator points already stored in the audit
parquet:

    new_score = rs_points      * (w_rs      / 3.0)
              + ichimoku_points* (w_ich     / 2.0)
              + higher_lows_pts* (w_hl      / 1.0)
              + roc_points     * (w_roc     / 1.5)
              + cmf_points     * (w_cmf     / 1.5)
              + dual_tf_rs_pts * (w_dtf     / 0.5)
              + atr_points     * (w_atr     / 0.5)

Each scheme's signals are evaluated at:
  (a) score >= 8.5 + 3-day persistence (production-equivalent), and
  (b) top K by score where K = production's signal count (matched selectivity).

Run: python3 audit_compare_weights.py
     python3 audit_compare_weights.py --target fwd_63d_xspy --window 12mo
"""

import argparse
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CURRENT_WEIGHTS = {
    "rs": 3.0, "ichimoku": 2.0, "higher_lows": 1.0,
    "roc": 1.5, "cmf": 1.5, "dual_tf_rs": 0.5, "atr": 0.5,
}

SCHEMES = {
    "A_baseline":         {"rs": 3.0, "ichimoku": 2.0, "higher_lows": 1.0, "roc": 1.5, "cmf": 1.5, "dual_tf_rs": 0.5, "atr": 0.5},
    "B_dtf_boost":        {"rs": 3.0, "ichimoku": 2.0, "higher_lows": 1.0, "roc": 1.5, "cmf": 0.0, "dual_tf_rs": 2.0, "atr": 0.5},
    "C_drop_cmf_hl":      {"rs": 3.0, "ichimoku": 2.0, "higher_lows": 0.5, "roc": 1.5, "cmf": 0.0, "dual_tf_rs": 2.5, "atr": 0.5},
    "D_edge_prop_12mo":   {"rs": 2.7, "ichimoku": 2.7, "higher_lows": 0.0, "roc": 1.3, "cmf": 0.0, "dual_tf_rs": 2.5, "atr": 0.8},
    "E_rs_ichimoku_dtf":  {"rs": 3.0, "ichimoku": 3.0, "higher_lows": 0.0, "roc": 1.5, "cmf": 0.0, "dual_tf_rs": 2.5, "atr": 0.0},
    "F_minimal_rs_dtf":   {"rs": 4.0, "ichimoku": 2.0, "higher_lows": 0.0, "roc": 2.0, "cmf": 0.0, "dual_tf_rs": 2.0, "atr": 0.0},
}


def compute_score(df: pd.DataFrame, w: dict) -> pd.Series:
    cur = CURRENT_WEIGHTS
    return (
        df["rs_points"]          * (w["rs"]          / cur["rs"])
      + df["ichimoku_points"]    * (w["ichimoku"]    / cur["ichimoku"])
      + df["higher_lows_points"] * (w["higher_lows"] / cur["higher_lows"])
      + df["roc_points"]         * (w["roc"]         / cur["roc"])
      + df["cmf_points"]         * (w["cmf"]         / cur["cmf"])
      + df["dual_tf_rs_points"]  * (w["dual_tf_rs"]  / cur["dual_tf_rs"])
      + df["atr_points"]         * (w["atr"]         / cur["atr"])
    )


def apply_persistence(df: pd.DataFrame, threshold: float, score_col: str,
                      n_days: int = 3) -> pd.Series:
    """Return boolean Series — True if score >= threshold for >= n_days prior days for this ticker."""
    df2 = df.sort_values(["ticker", "date"]).copy()
    above = (df2[score_col] >= threshold).astype(int)
    # Rolling sum over last n days PRIOR (shift by 1 to exclude current)
    rolling_prior = above.groupby(df2["ticker"]).shift(1).rolling(n_days, min_periods=n_days).sum()
    persist = (rolling_prior >= n_days)
    return persist.reindex(df.index, fill_value=False)


def evaluate_scheme(df: pd.DataFrame, name: str, weights: dict,
                    target: str, threshold: float = 8.5) -> dict:
    """Return metrics for this weight scheme."""
    score = compute_score(df, weights)
    df2 = df.assign(_score=score)

    # (a) Production-equivalent: score >= threshold + 3-day persistence
    today_fire = df2["_score"] >= threshold
    persist_ok = apply_persistence(df2, threshold, "_score", n_days=3)
    signals_prod = today_fire & persist_ok

    # (b) Matched selectivity: top K rows by score where K = prod signal count
    vals_prod = df2.loc[signals_prod, target].dropna()
    n_prod = int(signals_prod.sum())

    top_k_mask = df2["_score"].rank(ascending=False, method="first") <= n_prod
    vals_topk = df2.loc[top_k_mask, target].dropna()

    # Spearman ρ across all rows
    clean = df2[[target, "_score"]].dropna()
    if len(clean) > 10 and clean["_score"].std() > 0:
        rho, _ = spearmanr(clean["_score"], clean[target])
    else:
        rho = np.nan

    return {
        "scheme": name,
        "weights": weights,
        "total_weight": round(sum(weights.values()), 2),
        "n_prod_signals": n_prod,
        "prod_mean_fwd": float(vals_prod.mean()) if len(vals_prod) > 0 else np.nan,
        "prod_median_fwd": float(vals_prod.median()) if len(vals_prod) > 0 else np.nan,
        "prod_win_rate": float((vals_prod > 0).mean()) if len(vals_prod) > 0 else np.nan,
        "topk_mean_fwd": float(vals_topk.mean()) if len(vals_topk) > 0 else np.nan,
        "topk_win_rate": float((vals_topk > 0).mean()) if len(vals_topk) > 0 else np.nan,
        "spearman_rho": float(rho) if pd.notna(rho) else np.nan,
        "score_mean": float(df2["_score"].mean()),
        "score_std": float(df2["_score"].std()),
        "pct_score_ge_threshold": float(today_fire.mean()),
    }


def main(path: str, target: str, window: Optional[str], threshold: float):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"loaded {len(df):,} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    if window:
        end = df["date"].max()
        if window.endswith("mo"):
            start = end - pd.DateOffset(months=int(window[:-2]))
        elif window.endswith("y"):
            start = end - pd.DateOffset(years=int(window[:-1]))
        df = df[df["date"] >= start]
        print(f"  filtered to {window}: {len(df):,} rows")
    print(f"  target = {target}, threshold = {threshold}")

    results = []
    for name, weights in SCHEMES.items():
        r = evaluate_scheme(df, name, weights, target, threshold)
        results.append(r)

    out = pd.DataFrame(results)

    # Key comparison table
    print("\n" + "=" * 110)
    print("  WEIGHT SCHEME COMPARISON")
    print("=" * 110)
    print()
    cols = ["scheme", "total_weight", "n_prod_signals",
            "prod_mean_fwd", "prod_win_rate",
            "topk_mean_fwd", "topk_win_rate",
            "spearman_rho", "pct_score_ge_threshold"]
    disp = out[cols].copy()
    for c in ("prod_mean_fwd", "topk_mean_fwd"):
        disp[c] = disp[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
    for c in ("prod_win_rate", "topk_win_rate", "pct_score_ge_threshold"):
        disp[c] = disp[c].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    disp["spearman_rho"] = disp["spearman_rho"].map(lambda v: f"{v:+.4f}" if pd.notna(v) else "—")
    print(disp.to_string(index=False))

    # Weights detail
    print()
    print("  SCHEME DETAILS")
    print("  " + "─" * 104)
    ws = pd.DataFrame([{**{"scheme": r["scheme"]}, **r["weights"]} for r in results])
    print(ws.to_string(index=False))

    # Delta vs baseline on the key metric
    baseline = next(r for r in results if r["scheme"] == "A_baseline")
    print()
    print("  DELTA vs A_baseline (prod signals, threshold=8.5 + persistence-3)")
    print("  " + "─" * 104)
    for r in results:
        if r["scheme"] == "A_baseline":
            continue
        d_mean = (r["prod_mean_fwd"] or 0) - (baseline["prod_mean_fwd"] or 0)
        d_wr = (r["prod_win_rate"] or 0) - (baseline["prod_win_rate"] or 0)
        d_n = r["n_prod_signals"] - baseline["n_prod_signals"]
        d_rho = r["spearman_rho"] - baseline["spearman_rho"]
        print(f"    {r['scheme']:<22}  Δmean_fwd={d_mean:+.2%}  "
              f"Δwin_rate={d_wr:+.1%}  Δn_signals={d_n:+}  Δρ={d_rho:+.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    ap.add_argument("--window", default=None)
    ap.add_argument("--threshold", type=float, default=8.5)
    args = ap.parse_args()
    main(args.input, args.target, args.window, args.threshold)
