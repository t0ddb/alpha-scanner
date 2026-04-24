from __future__ import annotations

"""
audit_ml_weights.py — Data-driven weight derivation vs current linear weights.

Reads backtest_results/audit_dataset.parquet and fits several models to see
whether data-driven weighting beats the hand-tuned weighted sum.

Experiments:
  M1. Ridge regression on 16 continuous metrics + time-series CV
  M2. LASSO regression (sparse; feature selection)
  M3. Gradient Boosting (non-linear; partial-dependence on score)
  M4. Compare: live-weights linear vs ML-fit linear vs ML non-linear

All evaluated by out-of-sample forward-return correlation (Spearman) on
time-series CV splits.
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


FEATURES = [
    # Current 7 scored
    "rs_percentile",
    "ichimoku_score",
    "higher_lows_count",
    "roc_value",
    "cmf_value",
    "atr_percentile",
    "rs_63d_pctl",   # proxy for dual_tf_rs gate
    # Dropped-but-available
    "ma_50_close_pct",
    "ma_200_close_pct",
    "pct_from_52w_high",
    "rsi_value",
    "macd_hist",
    "adx_value",
    "obv_slope",
    "consolidation",
    "donchian_pct",
]

LIVE_WEIGHTS = {
    "rs_percentile":     3.0 / 100.0,  # gradient by pctl, so effective per-unit weight is small
    "ichimoku_score":    2.0 / 3.0,    # score is 0-3; binary 2pt if full
    "higher_lows_count": 1.0 / 5.0,
    "roc_value":         1.5 / 10.0,   # per 1% ROC
    "cmf_value":         1.5 / 0.1,    # per 0.1 CMF
    "atr_percentile":    0.5 / 100.0,
    "rs_63d_pctl":       0.5 / 100.0,
}


def _oos_cv_spearman(model, X, y, n_splits: int = 5) -> list[float]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    corrs = []
    for tr, te in tscv.split(X):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        if np.std(pred) == 0 or np.std(y[te]) == 0:
            corrs.append(0.0); continue
        r, _ = spearmanr(pred, y[te])
        corrs.append(float(r) if pd.notna(r) else 0.0)
    return corrs


def fit_and_report(df: pd.DataFrame, target: str):
    cols = FEATURES + [target, "date", "score"]
    clean = df[cols].dropna().copy().sort_values("date")
    X = clean[FEATURES].values
    y = clean[target].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Reference 1: live-weights score → Spearman with forward return (OOS CV)
    tscv = TimeSeriesSplit(n_splits=5)
    live_scores = clean["score"].values
    live_corrs = []
    for tr, te in tscv.split(Xs):
        r, _ = spearmanr(live_scores[te], y[te])
        live_corrs.append(float(r) if pd.notna(r) else 0.0)

    print(f"\n[REFERENCE] Live-weighted score (0-10) → target {target}")
    print(f"  OOS Spearman ρ  per fold: {[round(c,3) for c in live_corrs]}")
    print(f"  OOS Spearman ρ  mean    : {np.mean(live_corrs):+.4f}")

    # M1: RidgeCV — linear, all features retained
    print(f"\n[M1] Ridge regression (linear, all features)")
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=tscv)
    ridge_corrs = _oos_cv_spearman(ridge, Xs, y, 5)
    ridge.fit(Xs, y)  # refit on all for coef reporting
    print(f"  OOS Spearman ρ  mean    : {np.mean(ridge_corrs):+.4f}  per-fold: {[round(c,3) for c in ridge_corrs]}")
    coefs = sorted(zip(FEATURES, ridge.coef_), key=lambda x: -abs(x[1]))
    print("  Top coefficients (scaled feature space):")
    for feat, c in coefs[:10]:
        print(f"    {feat:<20}  {c:+.4f}")

    # M2: LassoCV — sparse
    print(f"\n[M2] LASSO regression (sparse feature selection)")
    lasso = LassoCV(cv=tscv, alphas=np.logspace(-6, -1, 100), max_iter=20000)
    lasso_corrs = _oos_cv_spearman(lasso, Xs, y, 5)
    lasso.fit(Xs, y)
    print(f"  OOS Spearman ρ  mean    : {np.mean(lasso_corrs):+.4f}  per-fold: {[round(c,3) for c in lasso_corrs]}")
    lasso_coefs = sorted(zip(FEATURES, lasso.coef_), key=lambda x: -abs(x[1]))
    nonzero = [(f, c) for f, c in lasso_coefs if abs(c) > 1e-6]
    print(f"  α={lasso.alpha_:.5f}, non-zero features: {len(nonzero)}/{len(FEATURES)}")
    for feat, c in nonzero:
        print(f"    {feat:<20}  {c:+.4f}")

    # M3: Gradient boosting — non-linear
    print(f"\n[M3] Gradient Boosting (non-linear)")
    gbr = GradientBoostingRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
    )
    gbr_corrs = _oos_cv_spearman(gbr, X, y, 5)  # unscaled fine for trees
    print(f"  OOS Spearman ρ  mean    : {np.mean(gbr_corrs):+.4f}  per-fold: {[round(c,3) for c in gbr_corrs]}")
    gbr.fit(X, y)
    imp = sorted(zip(FEATURES, gbr.feature_importances_), key=lambda x: -x[1])
    print("  Feature importances:")
    for feat, v in imp[:10]:
        print(f"    {feat:<20}  {v:.4f}")

    # Verdict
    print(f"\n[VERDICT]")
    live_m = np.mean(live_corrs)
    ridge_m = np.mean(ridge_corrs)
    lasso_m = np.mean(lasso_corrs)
    gbr_m = np.mean(gbr_corrs)
    print(f"  live-weighted-sum   ρ = {live_m:+.4f}")
    print(f"  ridge (linear)      ρ = {ridge_m:+.4f}  delta = {ridge_m - live_m:+.4f}")
    print(f"  lasso (sparse lin.) ρ = {lasso_m:+.4f}  delta = {lasso_m - live_m:+.4f}")
    print(f"  gbr (non-linear)    ρ = {gbr_m:+.4f}  delta = {gbr_m - live_m:+.4f}")
    print()
    if max(ridge_m, lasso_m) - live_m < 0.01:
        print("  → Linear ML gives NEGLIGIBLE lift over current hand-tuned weights.")
    else:
        print("  → Linear ML produces a meaningful lift — current weights are sub-optimal.")
    if gbr_m - max(ridge_m, lasso_m) > 0.01:
        print("  → Non-linear interactions exist (gbr > best linear).")
    else:
        print("  → No meaningful non-linear lift — the problem is essentially linear.")


def main(path: str, target: str, window: Optional[str]):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"loaded {len(df):,} rows")

    if window:
        end = df["date"].max()
        if window.endswith("mo"):
            start = end - pd.DateOffset(months=int(window[:-2]))
        elif window.endswith("y"):
            start = end - pd.DateOffset(years=int(window[:-1]))
        else:
            raise ValueError
        df = df[df["date"] >= start]
        print(f"  filtered to {window}: {len(df):,} rows")

    fit_and_report(df, target)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    ap.add_argument("--window", default=None)
    args = ap.parse_args()
    main(args.input, args.target, args.window)
