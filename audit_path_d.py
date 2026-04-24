from __future__ import annotations

"""
audit_path_d.py — Test extensions to Scheme C:
  D.1 MACD as rescue indicator (add to score; test full-weight and conditional)
  D.2 pct_from_52w_high as filter (exclude rows too far below 52w high)

Signal-level comparison against Scheme C baseline using the audit parquet.
If something looks promising, escalate to portfolio backtest via DB rescore.
"""

import argparse
from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


CURRENT_WEIGHTS = {"rs": 3.0, "ichimoku": 2.0, "higher_lows": 1.0,
                   "roc": 1.5, "cmf": 1.5, "dual_tf_rs": 0.5, "atr": 0.5}

SCHEME_C = {"rs": 3.0, "ichimoku": 2.0, "higher_lows": 0.5,
            "roc": 1.5, "cmf": 0.0, "dual_tf_rs": 2.5, "atr": 0.5}


def compute_scheme_c_score(df: pd.DataFrame) -> pd.Series:
    w, c = SCHEME_C, CURRENT_WEIGHTS
    return (
        df["rs_points"]          * (w["rs"]          / c["rs"])
      + df["ichimoku_points"]    * (w["ichimoku"]    / c["ichimoku"])
      + df["higher_lows_points"] * (w["higher_lows"] / c["higher_lows"])
      + df["roc_points"]         * (w["roc"]         / c["roc"])
      + df["cmf_points"]         * (w["cmf"]         / c["cmf"])
      + df["dual_tf_rs_points"]  * (w["dual_tf_rs"]  / c["dual_tf_rs"])
      + df["atr_points"]         * (w["atr"]         / c["atr"])
    )


def apply_persistence(df: pd.DataFrame, score_col: str, threshold: float,
                      n_days: int = 3) -> pd.Series:
    df2 = df.sort_values(["ticker", "date"]).copy()
    above = (df2[score_col] >= threshold).astype(int)
    rolling_prior = above.groupby(df2["ticker"]).shift(1).rolling(n_days, min_periods=n_days).sum()
    persist = (rolling_prior >= n_days).reindex(df.index, fill_value=False)
    return persist


def bootstrap_ci(values: np.ndarray, n_iter: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    n = len(values)
    boots = [values[rng.integers(0, n, n)].mean() for _ in range(n_iter)]
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def eval_variant(name: str, df: pd.DataFrame, score: pd.Series, mask: pd.Series,
                 target: str, threshold: float = 9.0) -> dict:
    """Evaluate a variant at score >= threshold + 3d persistence AND mask."""
    df2 = df.assign(_score=score)
    today_fire = df2["_score"] >= threshold
    persist_ok = apply_persistence(df2, "_score", threshold)
    signals = today_fire & persist_ok & mask
    vals = df2.loc[signals, target].dropna()
    if len(vals) == 0:
        return {"variant": name, "n_signals": 0, "mean_fwd": np.nan,
                "median_fwd": np.nan, "win_rate": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
    lo, hi = bootstrap_ci(vals.values)
    return {
        "variant": name,
        "n_signals": int(signals.sum()),
        "mean_fwd": float(vals.mean()),
        "median_fwd": float(vals.median()),
        "win_rate": float((vals > 0).mean()),
        "ci_lo": lo, "ci_hi": hi,
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

    # Scheme C score
    scheme_c_score = compute_scheme_c_score(df)
    true_mask = pd.Series(True, index=df.index)

    # ─ D.1: MACD as rescue indicator ─
    print("\n" + "=" * 80)
    print("  D.1: MACD rescue indicator (adds to Scheme C score)")
    print("=" * 80)
    print(f"\n  Baseline: Scheme C @ threshold {threshold}\n")

    variants_d1 = []
    variants_d1.append(eval_variant(
        "C_baseline (no MACD)", df, scheme_c_score, true_mask, target, threshold))

    # Variant: add MACD binary with weight 1.0 — always adds if fired
    for w_macd in (0.5, 1.0, 1.5):
        score_v = scheme_c_score + df["macd_fired"].astype(float) * w_macd
        variants_d1.append(eval_variant(
            f"+MACD (w={w_macd}, always)", df, score_v, true_mask, target, threshold))

    # Variant: MACD only when RS is WEAK (rs_fired=0, i.e., rs_pctl<50)
    for w_macd in (1.0, 2.0):
        rs_weak = (df["rs_fired"] == 0).astype(float)
        score_v = scheme_c_score + df["macd_fired"].astype(float) * rs_weak * w_macd
        variants_d1.append(eval_variant(
            f"+MACD (w={w_macd}, only when RS weak)", df, score_v, true_mask, target, threshold))

    d1 = pd.DataFrame(variants_d1)
    _print_results(d1, "D.1 MACD variants")

    # ─ D.2: pct_from_52w_high as filter on Scheme C signals ─
    print("\n" + "=" * 80)
    print("  D.2: pct_from_52w_high as filter (excludes signals below X% from 52w high)")
    print("=" * 80)
    print(f"\n  Baseline: Scheme C @ threshold {threshold}\n")

    variants_d2 = [eval_variant("C_baseline (no filter)", df, scheme_c_score, true_mask, target, threshold)]

    for min_pct_from_high in (-0.50, -0.35, -0.25, -0.15, -0.10, -0.05, -0.02):
        mask = df["pct_from_52w_high"] >= min_pct_from_high
        variants_d2.append(eval_variant(
            f"+filter pct_from_52w >= {min_pct_from_high:+.0%}",
            df, scheme_c_score, mask, target, threshold))

    # Band filters: exclude both tails
    for lo_pct, hi_pct in [(-0.25, -0.05), (-0.20, -0.03), (-0.25, -0.02), (-0.30, -0.05)]:
        mask = (df["pct_from_52w_high"] >= lo_pct) & (df["pct_from_52w_high"] <= hi_pct)
        variants_d2.append(eval_variant(
            f"+band {lo_pct:+.0%}..{hi_pct:+.0%} below 52w high",
            df, scheme_c_score, mask, target, threshold))

    # Exclude stocks within X% of 52w high (opposite direction)
    for hi_pct in (-0.05, -0.03):
        mask = df["pct_from_52w_high"] <= hi_pct
        variants_d2.append(eval_variant(
            f"+filter pct_from_52w <= {hi_pct:+.0%} (not-at-high)",
            df, scheme_c_score, mask, target, threshold))

    d2 = pd.DataFrame(variants_d2)
    _print_results(d2, "D.2 pct_from_52w_high filter variants")

    # ─ D.2 continued: what is the non-linear shape? ─
    print("\n  D.2b: Forward return by pct_from_52w_high bucket (within Scheme C high-conviction)")
    c_hi = (scheme_c_score >= threshold) & apply_persistence(
        df.assign(_score=scheme_c_score), "_score", threshold)
    hi_df = df.loc[c_hi].copy()
    if len(hi_df) > 0:
        edges = [-1.0, -0.50, -0.35, -0.25, -0.15, -0.10, -0.05, -0.02, 0.0, 0.02]
        labels = [f"{edges[i]:+.0%}..{edges[i+1]:+.0%}" for i in range(len(edges)-1)]
        hi_df["bucket"] = pd.cut(hi_df["pct_from_52w_high"], bins=edges, labels=labels, include_lowest=True)
        g = hi_df.groupby("bucket", observed=True)[target].agg(["count", "mean", "median"])
        g["win_rate"] = hi_df.groupby("bucket", observed=True)[target].apply(lambda s: (s.dropna() > 0).mean())
        print()
        print(g.round(4).to_string())


def _print_results(df: pd.DataFrame, title: str) -> None:
    out = df.copy()
    for c in ("mean_fwd", "median_fwd", "ci_lo", "ci_hi"):
        out[c] = out[c].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
    out["win_rate"] = out["win_rate"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    print(f"\n  {title}:")
    print(out.to_string(index=False))

    # Delta vs first (baseline)
    baseline = df.iloc[0]
    print(f"\n  Δ vs {baseline['variant']}:")
    for _, r in df.iloc[1:].iterrows():
        d_mean = (r["mean_fwd"] or 0) - (baseline["mean_fwd"] or 0)
        d_wr = (r["win_rate"] or 0) - (baseline["win_rate"] or 0)
        d_n = r["n_signals"] - baseline["n_signals"]
        print(f"    {r['variant']:<50}  Δmean_fwd={d_mean:+.2%}  Δwin={d_wr:+.1%}  Δn={d_n:+}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    ap.add_argument("--window", default="12mo")
    ap.add_argument("--threshold", type=float, default=9.0,
                    help="score threshold to use (Scheme C optimum ≈ 9.0)")
    args = ap.parse_args()
    main(args.input, args.target, args.window, args.threshold)
