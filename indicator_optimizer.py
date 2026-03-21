from __future__ import annotations

"""
indicator_optimizer.py — Hybrid LASSO + Forward Stepwise indicator selection.

Tests all 16 indicators (including 8 previously cut) to find the optimal
combination with gradient weights. Uses time-series cross-validation to
prevent data leakage.

Phases:
    1. Collect 16 continuous metrics across 2yr history
    2. LASSO regression (63d + 42d targets) to identify non-zero predictors
    3. Forward stepwise selection with CV to validate and rank
    4. Pearson vs Spearman comparison to determine scaling shapes
    5. Final recommendation table

Usage:
    python3 indicator_optimizer.py
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# scikit-learn
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Project
from config import load_config, get_all_tickers, get_indicator_config, get_ticker_metadata
from data_fetcher import fetch_batch
from indicators import compute_all_indicators
from indicators_expanded import (
    check_rsi_momentum,
    check_macd_crossover,
    check_adx,
    check_obv_trend,
    check_consolidation_tightness,
    check_donchian_breakout,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

FORWARD_WINDOWS = [10, 21, 42, 63]
OUTPUT_DIR = "backtest_results"
N_CV_SPLITS = 5
STEPWISE_MIN_IMPROVEMENT = 0.001  # 0.1% R² improvement threshold (stock returns are noisy)

# All 16 continuous metric columns
ALL_16_METRICS = [
    "rs_percentile",
    "ichimoku_score",
    "higher_lows_count",
    "ma_pct_above_50",
    "ma_pct_above_200",
    "roc_value",
    "cmf_value",
    "pct_from_52w_high",
    "atr_percentile",
    # 7 new
    "rsi_value",
    "macd_histogram",
    "adx_value",
    "obv_slope",
    "consolidation_ratio",
    "donchian_pct_above",
    "volume_ratio",
]

METRIC_LABELS = {
    "rs_percentile":       "Relative Strength (pctl)",
    "ichimoku_score":      "Ichimoku Cloud (score)",
    "higher_lows_count":   "Higher Lows (count)",
    "ma_pct_above_50":     "Price vs 50-SMA (%)",
    "ma_pct_above_200":    "Price vs 200-SMA (%)",
    "roc_value":           "Rate of Change (%)",
    "cmf_value":           "Chaikin Money Flow",
    "pct_from_52w_high":   "Dist from 52w High (%)",
    "atr_percentile":      "ATR Expansion (pctl)",
    "rsi_value":           "RSI Momentum",
    "macd_histogram":      "MACD Histogram",
    "adx_value":           "ADX Trend Strength",
    "obv_slope":           "OBV Slope",
    "consolidation_ratio": "Consolidation Tightness",
    "donchian_pct_above":  "Donchian Breakout (%)",
    "volume_ratio":        "Volume Ratio",
}

# Current binary weights for comparison
CURRENT_WEIGHTS = {
    "rs_percentile":     3.0,
    "ichimoku_score":    2.5,
    "higher_lows_count": 2.0,
    "ma_pct_above_50":   2.0,  # MA alignment weight
    "ma_pct_above_200":  0.0,  # part of MA alignment
    "roc_value":         1.5,
    "cmf_value":         1.0,
    "pct_from_52w_high": 1.0,
    "atr_percentile":    0.5,
    # Previously cut — weight 0
    "rsi_value":           0.0,
    "macd_histogram":      0.0,
    "adx_value":           0.0,
    "obv_slope":           0.0,
    "consolidation_ratio": 0.0,
    "donchian_pct_above":  0.0,
    "volume_ratio":        0.0,
}


# ─────────────────────────────────────────────────────────────
# PHASE 1: EXPANDED METRIC COLLECTION
# ─────────────────────────────────────────────────────────────

def collect_expanded_metrics(
    data: dict[str, pd.DataFrame],
    cfg: dict,
    test_frequency: int = 5,
) -> pd.DataFrame:
    """
    Walk through historical dates and record 16 continuous metric values
    plus forward returns for every ticker on each test date.

    Extends gradient_analysis.collect_continuous_metrics() with 7 new
    metrics from indicators_expanded.py.
    """
    benchmark_ticker = cfg["benchmark"]["ticker"]
    benchmark_df = data.get(benchmark_ticker)
    if benchmark_df is None:
        print("  [ERROR] No benchmark data.")
        return pd.DataFrame()

    max_forward = max(FORWARD_WINDOWS)
    warmup = 220
    total_days = len(benchmark_df)

    if total_days < warmup + max_forward:
        print(f"  [ERROR] Not enough data. Have {total_days}, need {warmup + max_forward}+.")
        return pd.DataFrame()

    start_idx = warmup
    end_idx = total_days - max_forward
    test_indices = list(range(start_idx, end_idx, test_frequency))

    metadata = get_ticker_metadata(cfg)
    ind_cfg = get_indicator_config(cfg)
    rs_period = ind_cfg["relative_strength"]["period"]

    print(f"  Collecting 16 metrics across {len(test_indices)} dates, "
          f"{len(data) - 1} tickers...")
    print(f"  Window: {benchmark_df.index[start_idx].strftime('%Y-%m-%d')} -> "
          f"{benchmark_df.index[end_idx].strftime('%Y-%m-%d')}")
    print()

    all_rows: list[dict] = []

    for count, idx in enumerate(test_indices, 1):
        if count % 10 == 0:
            print(f"  [{count}/{len(test_indices)}] "
                  f"{benchmark_df.index[idx].strftime('%Y-%m-%d')}...")

        bench_slice = benchmark_df.iloc[:idx + 1]

        # --- RS percentile pre-computation ---
        raw_rs: dict[str, float] = {}
        for ticker, full_df in data.items():
            if ticker == benchmark_ticker:
                continue
            df = full_df.iloc[:idx + 1]
            if len(df) < rs_period + 1 or len(bench_slice) < rs_period + 1:
                continue
            stock_ret = (df["Close"].iloc[-1] / df["Close"].iloc[-rs_period - 1]) - 1
            bench_ret = (bench_slice["Close"].iloc[-1] / bench_slice["Close"].iloc[-rs_period - 1]) - 1
            if bench_ret != 0:
                raw_rs[ticker] = stock_ret / bench_ret if bench_ret > 0 else stock_ret - bench_ret
            else:
                raw_rs[ticker] = 0
        all_rs_values = list(raw_rs.values())

        # --- Per-ticker metric collection ---
        for ticker, full_df in data.items():
            if ticker == benchmark_ticker:
                continue

            df = full_df.iloc[:idx + 1]
            if len(df) < warmup:
                continue

            # Existing 8 indicators via compute_all_indicators
            indicators = compute_all_indicators(
                df, bench_slice, cfg, all_rs_values=all_rs_values,
            )

            # Forward returns
            target_date = benchmark_df.index[idx]
            ticker_indices = full_df.index.get_indexer([target_date], method="nearest")
            ticker_idx = ticker_indices[0]

            fwd: dict[int, float | None] = {}
            entry_price = full_df["Close"].iloc[ticker_idx]
            for w in FORWARD_WINDOWS:
                future_idx = ticker_idx + w
                if future_idx < len(full_df):
                    future_price = full_df["Close"].iloc[future_idx]
                    fwd[w] = round((future_price - entry_price) / entry_price, 4)
                else:
                    fwd[w] = None

            # --- Extract existing 9 continuous metrics ---
            rs = indicators.get("relative_strength", {})
            ich = indicators.get("ichimoku_cloud", {})
            hl = indicators.get("higher_lows", {})
            ma = indicators.get("moving_averages", {})
            roc = indicators.get("roc", {})
            cmf = indicators.get("cmf", {})
            n52 = indicators.get("near_52w_high", {})
            atr = indicators.get("atr_expansion", {})

            # Ichimoku ordinal score
            ich_score = (
                int(ich.get("above_cloud", False))
                + int(ich.get("cloud_bullish", False))
                + int(ich.get("tenkan_above_kijun", False))
            )

            # MA distance metrics
            current_close = df["Close"].iloc[-1]
            sma_50 = ma.get("sma_50")
            sma_200 = ma.get("sma_200")
            ma_close = ma.get("current_close", current_close)
            pct_above_50 = ((ma_close - sma_50) / sma_50 * 100) if sma_50 and sma_50 > 0 else None
            pct_above_200 = ((ma_close - sma_200) / sma_200 * 100) if sma_200 and sma_200 > 0 else None

            # --- Extract 7 NEW continuous metrics ---
            rsi_result = check_rsi_momentum(df)
            rsi_val = rsi_result.get("rsi")
            if rsi_val == 0 and len(df) < 15:
                rsi_val = None

            macd_result = check_macd_crossover(df)
            macd_hist = macd_result.get("histogram")
            if macd_hist == 0 and len(df) < 42:
                macd_hist = None

            adx_result = check_adx(df)
            adx_val = adx_result.get("adx")
            if adx_val == 0 and len(df) < 42:
                adx_val = None

            obv_result = check_obv_trend(df)
            obv_sl = obv_result.get("obv_slope")
            if obv_sl == 0 and len(df) < 60:
                obv_sl = None

            consol_result = check_consolidation_tightness(df)
            consol_ratio = consol_result.get("range_ratio")
            if consol_ratio == 0 and len(df) < 51:
                consol_ratio = None

            donch_result = check_donchian_breakout(df)
            donch_pct = donch_result.get("pct_above")
            if donch_pct == 0 and len(df) < 56:
                donch_pct = None
            elif donch_pct is not None:
                donch_pct = donch_pct * 100  # Convert to %

            # Volume ratio (inline)
            if len(df) >= 20 and df["Volume"].iloc[-20:].mean() > 0:
                vol_ratio = df["Volume"].iloc[-1] / df["Volume"].iloc[-20:].mean()
            else:
                vol_ratio = None

            meta = metadata.get(ticker, {})
            row = {
                "date":                benchmark_df.index[idx].strftime("%Y-%m-%d"),
                "ticker":              ticker,
                "sector":              meta.get("sector_name", ""),
                # Existing 9 metrics
                "rs_percentile":       rs.get("rs_percentile"),
                "ichimoku_score":      ich_score,
                "higher_lows_count":   hl.get("consecutive_higher_lows", 0),
                "ma_pct_above_50":     round(pct_above_50, 2) if pct_above_50 is not None else None,
                "ma_pct_above_200":    round(pct_above_200, 2) if pct_above_200 is not None else None,
                "roc_value":           roc.get("roc"),
                "cmf_value":           cmf.get("cmf"),
                "pct_from_52w_high":   n52.get("pct_from_high"),
                "atr_percentile":      atr.get("atr_percentile"),
                # 7 new metrics
                "rsi_value":           rsi_val,
                "macd_histogram":      macd_hist,
                "adx_value":           adx_val,
                "obv_slope":           obv_sl,
                "consolidation_ratio": consol_ratio,
                "donchian_pct_above":  donch_pct if donch_pct is not None else None,
                "volume_ratio":        round(vol_ratio, 4) if vol_ratio is not None else None,
            }

            # Forward returns
            for w in FORWARD_WINDOWS:
                row[f"fwd_{w}d"] = fwd.get(w)

            all_rows.append(row)

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────
# PHASE 2: LASSO REGRESSION WITH TIME-SERIES CV
# ─────────────────────────────────────────────────────────────

def run_lasso_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "fwd_63d",
    n_splits: int = N_CV_SPLITS,
) -> dict:
    """
    Run LassoCV with time-series cross-validation.
    Returns dict with coefficients, non-zero features, and CV R².
    """
    # Prepare data
    cols = feature_cols + [target_col, "date"]
    clean = df[cols].dropna().copy()
    clean = clean.sort_values("date")

    X = clean[feature_cols].values
    y = clean[target_col].values

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # LassoCV with wide alpha range (stock returns are noisy)
    lasso = LassoCV(
        cv=tscv,
        alphas=np.logspace(-6, -1, 100),
        max_iter=10000,
        random_state=42,
    )
    lasso.fit(X_scaled, y)

    # Compute out-of-sample R² across folds
    r2_scores = []
    for train_idx, test_idx in tscv.split(X_scaled):
        from sklearn.linear_model import Lasso
        fold_lasso = Lasso(alpha=lasso.alpha_, max_iter=10000)
        fold_lasso.fit(X_scaled[train_idx], y[train_idx])
        pred = fold_lasso.predict(X_scaled[test_idx])
        ss_res = np.sum((y[test_idx] - pred) ** 2)
        ss_tot = np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_scores.append(r2)

    # Map coefficients to feature names
    coefficients = {}
    for i, col in enumerate(feature_cols):
        coefficients[col] = round(float(lasso.coef_[i]), 6)

    nonzero = [col for col, coef in coefficients.items() if abs(coef) > 1e-6]
    total_abs = sum(abs(c) for c in coefficients.values() if abs(c) > 1e-6)

    importance = {}
    for col in nonzero:
        importance[col] = round(abs(coefficients[col]) / total_abs, 4) if total_abs > 0 else 0

    return {
        "alpha": round(float(lasso.alpha_), 6),
        "r2_cv": round(float(np.mean(r2_scores)), 4),
        "r2_per_fold": [round(r, 4) for r in r2_scores],
        "coefficients": coefficients,
        "nonzero_features": nonzero,
        "importance": importance,
        "target": target_col,
        "n_observations": len(clean),
        "scaler_mean": {col: round(m, 4) for col, m in zip(feature_cols, scaler.mean_)},
        "scaler_std": {col: round(s, 4) for col, s in zip(feature_cols, scaler.scale_)},
    }


def run_ridge_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "fwd_63d",
    n_splits: int = N_CV_SPLITS,
) -> dict:
    """
    Run RidgeCV with time-series CV. Unlike LASSO, Ridge never zeros out
    coefficients — it ranks ALL features by magnitude, which is more
    appropriate when individual signals are weak but collectively useful.
    """
    cols = feature_cols + [target_col, "date"]
    clean = df[cols].dropna().copy()
    clean = clean.sort_values("date")

    X = clean[feature_cols].values
    y = clean[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    ridge = RidgeCV(
        alphas=np.logspace(-3, 3, 50),
        cv=tscv,
    )
    ridge.fit(X_scaled, y)

    # Out-of-sample R²
    r2_scores = []
    for train_idx, test_idx in tscv.split(X_scaled):
        from sklearn.linear_model import Ridge
        fold_ridge = Ridge(alpha=ridge.alpha_)
        fold_ridge.fit(X_scaled[train_idx], y[train_idx])
        pred = fold_ridge.predict(X_scaled[test_idx])
        ss_res = np.sum((y[test_idx] - pred) ** 2)
        ss_tot = np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_scores.append(r2)

    coefficients = {}
    for i, col in enumerate(feature_cols):
        coefficients[col] = round(float(ridge.coef_[i]), 6)

    total_abs = sum(abs(c) for c in coefficients.values())
    importance = {}
    for col, coef in coefficients.items():
        importance[col] = round(abs(coef) / total_abs, 4) if total_abs > 0 else 0

    return {
        "alpha": round(float(ridge.alpha_), 6),
        "r2_cv": round(float(np.mean(r2_scores)), 4),
        "r2_per_fold": [round(r, 4) for r in r2_scores],
        "coefficients": coefficients,
        "importance": importance,
        "target": target_col,
        "n_observations": len(clean),
    }


# ─────────────────────────────────────────────────────────────
# PHASE 3: FORWARD STEPWISE SELECTION WITH CV
# ─────────────────────────────────────────────────────────────

def _cv_r2(X: np.ndarray, y: np.ndarray, tscv: TimeSeriesSplit) -> float:
    """Compute mean out-of-sample R² across time-series CV folds."""
    r2_scores = []
    for train_idx, test_idx in tscv.split(X):
        model = LinearRegression()
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        ss_res = np.sum((y[test_idx] - pred) ** 2)
        ss_tot = np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r2_scores.append(r2)
    return float(np.mean(r2_scores))


def run_forward_stepwise(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "fwd_63d",
    n_splits: int = N_CV_SPLITS,
    min_improvement: float = STEPWISE_MIN_IMPROVEMENT,
) -> dict:
    """
    Forward stepwise selection: greedily add the feature that improves
    out-of-sample R² the most at each step.
    """
    cols = feature_cols + [target_col, "date"]
    clean = df[cols].dropna().copy()
    clean = clean.sort_values("date")

    X_all = clean[feature_cols].values
    y = clean[target_col].values

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Map column names to indices
    col_to_idx = {col: i for i, col in enumerate(feature_cols)}

    selected: list[str] = []
    remaining = list(feature_cols)
    history: list[dict] = []
    current_r2 = 0.0

    while remaining:
        best_feature = None
        best_r2 = current_r2

        for candidate in remaining:
            trial_cols = selected + [candidate]
            trial_indices = [col_to_idx[c] for c in trial_cols]
            X_trial = X_scaled[:, trial_indices]

            trial_r2 = _cv_r2(X_trial, y, tscv)
            if trial_r2 > best_r2:
                best_r2 = trial_r2
                best_feature = candidate

        if best_feature is None or (best_r2 - current_r2) < min_improvement:
            # Record the stop reason
            if best_feature and (best_r2 - current_r2) < min_improvement:
                history.append({
                    "step": len(selected) + 1,
                    "feature": best_feature,
                    "r2": round(best_r2, 6),
                    "improvement": round(best_r2 - current_r2, 6),
                    "stopped": True,
                })
            break

        selected.append(best_feature)
        remaining.remove(best_feature)
        improvement = best_r2 - current_r2
        current_r2 = best_r2

        history.append({
            "step": len(selected),
            "feature": best_feature,
            "r2": round(current_r2, 6),
            "improvement": round(improvement, 6),
            "stopped": False,
        })

    return {
        "selected_features": selected,
        "selection_history": history,
        "final_r2": round(current_r2, 6),
        "target": target_col,
        "n_observations": len(clean),
    }


# ─────────────────────────────────────────────────────────────
# PHASE 4: PEARSON VS SPEARMAN COMPARISON
# ─────────────────────────────────────────────────────────────

def run_correlation_comparison(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "fwd_63d",
) -> pd.DataFrame:
    """
    Compute Pearson and Spearman correlations for each feature vs target.
    Recommend scaling shape based on the difference.
    """
    rows = []
    for col in feature_cols:
        valid = df[[col, target_col]].dropna()
        if len(valid) < 30:
            rows.append({
                "metric": col,
                "pearson_r": None, "pearson_p": None,
                "spearman_r": None, "spearman_p": None,
                "delta": None, "scaling": "insufficient data",
            })
            continue

        x = valid[col].values
        y = valid[target_col].values

        pr, pp = sp_stats.pearsonr(x, y)
        sr, sp = sp_stats.spearmanr(x, y)
        delta = abs(sr) - abs(pr)

        if delta > 0.05:
            scaling = "s-curve"
        elif delta < -0.05:
            scaling = "investigate"
        else:
            scaling = "linear"

        rows.append({
            "metric": col,
            "pearson_r": round(pr, 4),
            "pearson_p": round(pp, 6),
            "spearman_r": round(sr, 4),
            "spearman_p": round(sp, 6),
            "delta": round(delta, 4),
            "scaling": scaling,
        })

    return pd.DataFrame(rows)


def compute_feature_correlation_matrix(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute Spearman correlation matrix between all features."""
    clean = df[feature_cols].dropna()
    return clean.corr(method="spearman")


# ─────────────────────────────────────────────────────────────
# PHASE 5: FINAL RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────

def compute_final_recommendations(
    lasso_63d: dict,
    lasso_42d: dict,
    ridge_63d: dict,
    ridge_42d: dict,
    stepwise: dict,
    correlation_df: pd.DataFrame,
) -> pd.DataFrame:
    """Synthesize all analyses into a final recommendation table."""
    rows = []

    # Build stepwise order lookup
    stepwise_order = {}
    for item in stepwise["selection_history"]:
        if not item.get("stopped", False):
            stepwise_order[item["feature"]] = item["step"]

    for metric in ALL_16_METRICS:
        label = METRIC_LABELS.get(metric, metric)

        lasso_c63 = lasso_63d["coefficients"].get(metric, 0)
        lasso_c42 = lasso_42d["coefficients"].get(metric, 0)
        ridge_c63 = ridge_63d["coefficients"].get(metric, 0)
        ridge_c42 = ridge_42d["coefficients"].get(metric, 0)
        ridge_imp = ridge_63d["importance"].get(metric, 0)
        step_order = stepwise_order.get(metric, None)

        corr_row = correlation_df[correlation_df["metric"] == metric]
        pearson = corr_row["pearson_r"].iloc[0] if len(corr_row) > 0 else None
        spearman = corr_row["spearman_r"].iloc[0] if len(corr_row) > 0 else None
        scaling = corr_row["scaling"].iloc[0] if len(corr_row) > 0 else ""

        # Verdict uses multiple signals:
        # - LASSO: did it survive regularization?
        # - Ridge: how important is it relative to others?
        # - Stepwise: was it selected?
        # - Correlation: does it predict forward returns?
        # - Direction: is the coefficient consistently positive?
        in_lasso_63 = abs(lasso_c63) > 1e-6
        in_lasso_42 = abs(lasso_c42) > 1e-6
        in_stepwise = step_order is not None
        ridge_top_half = ridge_imp >= (1.0 / len(ALL_16_METRICS))  # above average importance
        positive_direction = ridge_c63 > 0 and ridge_c42 > 0
        has_correlation = spearman is not None and spearman > 0.05

        # Composite evidence score
        evidence = 0
        if in_lasso_63: evidence += 3
        if in_lasso_42: evidence += 2
        if in_stepwise: evidence += 3
        if ridge_top_half and positive_direction: evidence += 2
        if has_correlation: evidence += 1

        if evidence >= 6:
            verdict = "STRONG KEEP"
        elif evidence >= 4:
            verdict = "KEEP"
        elif evidence >= 2:
            verdict = "CONDITIONAL"
        else:
            verdict = "DROP"

        rows.append({
            "metric": metric,
            "label": label,
            "lasso_63d_coef": lasso_c63,
            "lasso_42d_coef": lasso_c42,
            "ridge_63d_coef": ridge_c63,
            "ridge_imp": ridge_imp,
            "stepwise_order": step_order,
            "pearson_r": pearson,
            "spearman_r": spearman,
            "scaling": scaling,
            "current_weight": CURRENT_WEIGHTS.get(metric, 0),
            "evidence": evidence,
            "verdict": verdict,
        })

    result = pd.DataFrame(rows)
    # Sort: STRONG KEEP first, then KEEP, CONDITIONAL, DROP
    verdict_order = {"STRONG KEEP": 0, "KEEP": 1, "CONDITIONAL": 2, "DROP": 3}
    result["_sort"] = result["verdict"].map(verdict_order)
    result = result.sort_values(["_sort", "lasso_63d_coef"], ascending=[True, False])
    result = result.drop(columns=["_sort"])

    return result


# ─────────────────────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_lasso_coefficients(lasso_63d: dict, lasso_42d: dict, output_dir: str) -> None:
    """Horizontal bar chart of LASSO coefficients."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, result, title in [
        (axes[0], lasso_63d, "LASSO Coefficients (63-day target)"),
        (axes[1], lasso_42d, "LASSO Coefficients (42-day target)"),
    ]:
        coefs = result["coefficients"]
        labels = [METRIC_LABELS.get(k, k) for k in coefs.keys()]
        values = list(coefs.values())

        colors = ["#2ecc71" if v > 0 else "#e74c3c" if v < 0 else "#bdc3c7" for v in values]

        y_pos = range(len(labels))
        ax.barh(y_pos, values, color=colors, edgecolor="none")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_xlabel("Coefficient (standardized)")

    plt.tight_layout()
    path = os.path.join(output_dir, "optimizer_lasso_coefs.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_stepwise_r2(stepwise: dict, output_dir: str) -> None:
    """Line chart of cumulative R² as features are added."""
    history = stepwise["selection_history"]
    if not history:
        return

    steps = [h["step"] for h in history if not h.get("stopped", False)]
    r2s = [h["r2"] for h in history if not h.get("stopped", False)]
    labels = [METRIC_LABELS.get(h["feature"], h["feature"]) for h in history if not h.get("stopped", False)]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(steps, r2s, "o-", color="#3498db", linewidth=2, markersize=8)

    for i, (x, y, label) in enumerate(zip(steps, r2s, labels)):
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(5, 10), fontsize=8, rotation=25)

    ax.set_xlabel("Step (features added)", fontsize=11)
    ax.set_ylabel("Out-of-sample R²", fontsize=11)
    ax.set_title("Forward Stepwise Selection — Cumulative R²", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "optimizer_stepwise_r2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_correlation_matrix(corr_matrix: pd.DataFrame, output_dir: str) -> None:
    """Heatmap of feature-to-feature correlation matrix."""
    labels = [METRIC_LABELS.get(c, c) for c in corr_matrix.columns]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = corr_matrix.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            weight = "bold" if abs(val) > 0.8 and i != j else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color, fontweight=weight)

    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman Correlation")
    ax.set_title("16-Indicator Correlation Matrix", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "optimizer_correlation_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


def plot_weight_comparison(recs_df: pd.DataFrame, output_dir: str) -> None:
    """Grouped bar chart: current weights vs LASSO coefficients."""
    # Only show indicators that have either a current weight or a LASSO coefficient
    show = recs_df[
        (recs_df["current_weight"] > 0) | (abs(recs_df["lasso_63d_coef"]) > 1e-6)
    ].copy()

    labels = show["label"].values
    current = show["current_weight"].values
    lasso = np.abs(show["lasso_63d_coef"].values)

    # Normalize LASSO to same scale as current weights for visual comparison
    if lasso.max() > 0:
        lasso_scaled = lasso / lasso.max() * current.max() if current.max() > 0 else lasso
    else:
        lasso_scaled = lasso  # all zeros

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x - width / 2, current, width, label="Current Binary Weight", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, lasso_scaled, width, label="LASSO |Coef| (scaled)", color="#e74c3c", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Weight", fontsize=11)
    ax.set_title("Current Weights vs LASSO Coefficients", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "optimizer_weight_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ─────────────────────────────────────────────────────────────
# PRINT FUNCTIONS
# ─────────────────────────────────────────────────────────────

def print_data_summary(df: pd.DataFrame) -> None:
    """Print metric coverage summary."""
    print(f"\n  {'Metric':<30} {'Non-null':>10} {'Null':>8} {'Coverage':>10}")
    print(f"  {'─' * 62}")
    for col in ALL_16_METRICS:
        if col in df.columns:
            nn = df[col].notna().sum()
            na = df[col].isna().sum()
            pct = nn / len(df) * 100
            print(f"  {METRIC_LABELS.get(col, col):<30} {nn:>10} {na:>8} {pct:>9.1f}%")


def print_lasso_summary(lasso_63d: dict, lasso_42d: dict) -> None:
    """Print LASSO results table."""
    print(f"\n{'=' * 100}")
    print(f"  LASSO REGRESSION — TIME-SERIES CROSS-VALIDATION")
    print(f"{'=' * 100}")

    print(f"\n  Alpha (63d): {lasso_63d['alpha']:.6f}  |  "
          f"Alpha (42d): {lasso_42d['alpha']:.6f}")
    print(f"  CV R² (63d): {lasso_63d['r2_cv']:.4f}  |  "
          f"CV R² (42d): {lasso_42d['r2_cv']:.4f}")
    print(f"  R² per fold (63d): {lasso_63d['r2_per_fold']}")
    print(f"  Observations: {lasso_63d['n_observations']}")

    print(f"\n  {'Indicator':<30} {'Coef 63d':>12} {'Coef 42d':>12} "
          f"{'Imp 63d':>10} {'Status':>12}")
    print(f"  {'─' * 80}")

    # Sort by absolute coefficient
    metrics_sorted = sorted(
        ALL_16_METRICS,
        key=lambda m: abs(lasso_63d["coefficients"].get(m, 0)),
        reverse=True,
    )

    for metric in metrics_sorted:
        label = METRIC_LABELS.get(metric, metric)
        c63 = lasso_63d["coefficients"].get(metric, 0)
        c42 = lasso_42d["coefficients"].get(metric, 0)
        imp = lasso_63d["importance"].get(metric, 0)

        in_63 = abs(c63) > 1e-6
        in_42 = abs(c42) > 1e-6

        if in_63 and in_42:
            status = "✅ BOTH"
        elif in_63:
            status = "🔶 63d only"
        elif in_42:
            status = "🔶 42d only"
        else:
            status = "❌ ZEROED"

        print(f"  {label:<30} {c63:>+12.6f} {c42:>+12.6f} "
              f"{imp:>9.1%} {status:>12}")


def print_ridge_summary(ridge_63d: dict, ridge_42d: dict) -> None:
    """Print Ridge results table — all features ranked by importance."""
    print(f"\n\n{'=' * 100}")
    print(f"  RIDGE REGRESSION — ALL FEATURES RANKED (none zeroed out)")
    print(f"{'=' * 100}")

    print(f"\n  Alpha (63d): {ridge_63d['alpha']:.4f}  |  "
          f"Alpha (42d): {ridge_42d['alpha']:.4f}")
    print(f"  CV R² (63d): {ridge_63d['r2_cv']:.4f}  |  "
          f"CV R² (42d): {ridge_42d['r2_cv']:.4f}")
    print(f"  R² per fold (63d): {ridge_63d['r2_per_fold']}")

    print(f"\n  {'Indicator':<30} {'Coef 63d':>12} {'Coef 42d':>12} "
          f"{'Imp 63d':>10} {'Imp 42d':>10} {'Direction':>10}")
    print(f"  {'─' * 88}")

    # Sort by absolute coefficient (63d)
    metrics_sorted = sorted(
        ALL_16_METRICS,
        key=lambda m: abs(ridge_63d["coefficients"].get(m, 0)),
        reverse=True,
    )

    for metric in metrics_sorted:
        label = METRIC_LABELS.get(metric, metric)
        c63 = ridge_63d["coefficients"].get(metric, 0)
        c42 = ridge_42d["coefficients"].get(metric, 0)
        imp63 = ridge_63d["importance"].get(metric, 0)
        imp42 = ridge_42d["importance"].get(metric, 0)

        # Check if sign is consistent across horizons
        if c63 > 0 and c42 > 0:
            direction = "✅ +"
        elif c63 < 0 and c42 < 0:
            direction = "⚠️ −"
        else:
            direction = "❌ mixed"

        print(f"  {label:<30} {c63:>+12.6f} {c42:>+12.6f} "
              f"{imp63:>9.1%} {imp42:>9.1%} {direction:>10}")


def print_stepwise_summary(stepwise: dict) -> None:
    """Print stepwise selection results."""
    print(f"\n\n{'=' * 100}")
    print(f"  FORWARD STEPWISE SELECTION — TIME-SERIES CV")
    print(f"{'=' * 100}")

    print(f"\n  Target: {stepwise['target']}  |  "
          f"Final R²: {stepwise['final_r2']:.4f}  |  "
          f"Observations: {stepwise['n_observations']}")
    print(f"  Min improvement threshold: {STEPWISE_MIN_IMPROVEMENT:.1%}")

    print(f"\n  {'Step':>6} {'Feature Added':<30} {'Cum R²':>10} "
          f"{'Improvement':>14} {'Status':>10}")
    print(f"  {'─' * 75}")

    for item in stepwise["selection_history"]:
        label = METRIC_LABELS.get(item["feature"], item["feature"])
        stopped = item.get("stopped", False)
        status = "⛔ STOP" if stopped else "✅"
        print(f"  {item['step']:>6} {label:<30} {item['r2']:>10.6f} "
              f"{item['improvement']:>+13.6f} {status:>10}")

    print(f"\n  Selected indicators ({len(stepwise['selected_features'])}):")
    for i, feat in enumerate(stepwise["selected_features"], 1):
        print(f"    {i}. {METRIC_LABELS.get(feat, feat)}")


def print_correlation_comparison(corr_df: pd.DataFrame) -> None:
    """Print Pearson vs Spearman comparison table."""
    print(f"\n\n{'=' * 100}")
    print(f"  PEARSON vs SPEARMAN — SCALING SHAPE RECOMMENDATIONS")
    print(f"{'=' * 100}")

    print(f"\n  {'Indicator':<30} {'Pearson r':>10} {'Spearman r':>11} "
          f"{'Delta':>8} {'Scaling':>14}")
    print(f"  {'─' * 78}")

    for _, row in corr_df.iterrows():
        label = METRIC_LABELS.get(row["metric"], row["metric"])
        pr = f"{row['pearson_r']:>+.4f}" if row["pearson_r"] is not None else "N/A"
        sr = f"{row['spearman_r']:>+.4f}" if row["spearman_r"] is not None else "N/A"
        delta = f"{row['delta']:>+.4f}" if row["delta"] is not None else "N/A"

        emoji = {"s-curve": "📈", "linear": "📏", "investigate": "⚠️"}.get(row["scaling"], "")
        print(f"  {label:<30} {pr:>10} {sr:>11} "
              f"{delta:>8} {emoji} {row['scaling']:>12}")


def print_multicollinearity_warnings(corr_matrix: pd.DataFrame) -> None:
    """Print feature pairs with high correlation."""
    print(f"\n\n{'─' * 100}")
    print(f"  MULTICOLLINEARITY WARNINGS (|r| > 0.80)")
    print(f"{'─' * 100}\n")

    warnings_found = False
    cols = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr_matrix.iloc[i, j]
            if abs(val) > 0.80:
                warnings_found = True
                l1 = METRIC_LABELS.get(cols[i], cols[i])
                l2 = METRIC_LABELS.get(cols[j], cols[j])
                print(f"  ⚠ {l1} ↔ {l2}: r = {val:+.3f}")

    if not warnings_found:
        print("  No pairs with |r| > 0.80 found.")


def print_final_recommendations(recs_df: pd.DataFrame) -> None:
    """Print the final synthesis table."""
    print(f"\n\n{'=' * 100}")
    print(f"  FINAL RECOMMENDATIONS")
    print(f"{'=' * 100}")

    print(f"\n  {'Indicator':<28} {'Ridge63d':>10} {'RidgeImp':>9} "
          f"{'LASSO63':>9} {'Step#':>6} {'Spearman':>9} "
          f"{'Scale':>8} {'CurWt':>6} {'Evid':>5} {'Verdict':>14}")
    print(f"  {'─' * 112}")

    for _, row in recs_df.iterrows():
        label = row["label"][:28]
        rc63 = f"{row['ridge_63d_coef']:>+.5f}"
        rimp = f"{row['ridge_imp']:>7.1%}"
        lc63 = f"{row['lasso_63d_coef']:>+.4f}" if abs(row["lasso_63d_coef"]) > 1e-6 else "    0"
        step = f"{row['stepwise_order']:>4.0f}" if row["stepwise_order"] is not None else "   —"
        sr = f"{row['spearman_r']:>+.3f}" if row["spearman_r"] is not None else "  N/A"
        scale = row["scaling"][:6] if row["scaling"] else ""
        cw = f"{row['current_weight']:.1f}"
        evid = f"{row['evidence']:>3}"

        verdict_emoji = {
            "STRONG KEEP": "✅",
            "KEEP": "🔶",
            "CONDITIONAL": "⚠️",
            "DROP": "❌",
        }.get(row["verdict"], "")

        print(f"  {label:<28} {rc63:>10} {rimp:>9} "
              f"{lc63:>9} {step:>6} {sr:>9} "
              f"{scale:>8} {cw:>6} {evid:>5} {verdict_emoji} {row['verdict']:>12}")

    # Summary counts
    counts = recs_df["verdict"].value_counts()
    print(f"\n  Summary: ", end="")
    for verdict in ["STRONG KEEP", "KEEP", "CONDITIONAL", "DROP"]:
        if verdict in counts.index:
            print(f"{verdict}: {counts[verdict]}  ", end="")
    print()


# ─────────────────────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────────────────────

def run_optimization(
    data: dict[str, pd.DataFrame],
    cfg: dict,
) -> dict:
    """Run the complete optimization pipeline."""

    # Phase 1: Expanded metric collection
    print("\n  Phase 1: Collecting 16 expanded metrics...\n")
    events_df = collect_expanded_metrics(data, cfg, test_frequency=5)

    if events_df.empty:
        print("  [ERROR] No events collected.")
        return {}

    print(f"\n  Collected {len(events_df)} observations.")
    print_data_summary(events_df)

    # Phase 2: LASSO + Ridge regression
    print(f"\n\n  Phase 2: Regularized regression with time-series CV...\n")

    print("  Running LASSO with target=fwd_63d...")
    lasso_63d = run_lasso_analysis(events_df, ALL_16_METRICS, target_col="fwd_63d")
    print(f"  → {len(lasso_63d['nonzero_features'])} non-zero features, "
          f"CV R² = {lasso_63d['r2_cv']:.4f}")

    print("  Running LASSO with target=fwd_42d...")
    lasso_42d = run_lasso_analysis(events_df, ALL_16_METRICS, target_col="fwd_42d")
    print(f"  → {len(lasso_42d['nonzero_features'])} non-zero features, "
          f"CV R² = {lasso_42d['r2_cv']:.4f}")

    print("  Running Ridge with target=fwd_63d (all features ranked, none zeroed)...")
    ridge_63d = run_ridge_analysis(events_df, ALL_16_METRICS, target_col="fwd_63d")
    print(f"  → CV R² = {ridge_63d['r2_cv']:.4f}")

    print("  Running Ridge with target=fwd_42d...")
    ridge_42d = run_ridge_analysis(events_df, ALL_16_METRICS, target_col="fwd_42d")
    print(f"  → CV R² = {ridge_42d['r2_cv']:.4f}")

    print_lasso_summary(lasso_63d, lasso_42d)
    print_ridge_summary(ridge_63d, ridge_42d)

    # Phase 3: Forward stepwise selection
    print(f"\n\n  Phase 3: Forward stepwise selection...\n")
    stepwise = run_forward_stepwise(events_df, ALL_16_METRICS, target_col="fwd_63d")
    print_stepwise_summary(stepwise)

    # Phase 4: Pearson vs Spearman comparison
    print(f"\n\n  Phase 4: Correlation analysis...\n")
    corr_df = run_correlation_comparison(events_df, ALL_16_METRICS)
    corr_matrix = compute_feature_correlation_matrix(events_df, ALL_16_METRICS)
    print_correlation_comparison(corr_df)
    print_multicollinearity_warnings(corr_matrix)

    # Phase 5: Final recommendations
    print(f"\n\n  Phase 5: Synthesizing recommendations...\n")
    recs_df = compute_final_recommendations(
        lasso_63d, lasso_42d, ridge_63d, ridge_42d, stepwise, corr_df,
    )
    print_final_recommendations(recs_df)

    # Visualization
    print(f"\n\n  Phase 6: Generating charts...\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_lasso_coefficients(lasso_63d, lasso_42d, OUTPUT_DIR)
    plot_stepwise_r2(stepwise, OUTPUT_DIR)
    plot_correlation_matrix(corr_matrix, OUTPUT_DIR)
    plot_weight_comparison(recs_df, OUTPUT_DIR)

    return {
        "events_df": events_df,
        "lasso_63d": lasso_63d,
        "lasso_42d": lasso_42d,
        "stepwise": stepwise,
        "correlation": corr_df,
        "corr_matrix": corr_matrix,
        "recommendations": recs_df,
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = load_config()

    print("=" * 100)
    print("  INDICATOR OPTIMIZER")
    print("  Hybrid LASSO + Forward Stepwise Selection")
    print("  Testing all 16 indicators with time-series cross-validation")
    print("=" * 100)
    print()
    print("  This will take 10-15 minutes (data fetch + metric collection).")
    print("  LASSO and stepwise phases are fast (< 1 min).")
    print()

    # Fetch data
    print("  Fetching 2 years of data...\n")
    all_tickers = ["SPY"] + get_all_tickers(cfg)
    data = fetch_batch(all_tickers, period="2y", verbose=True)

    if not data:
        print("  [ERROR] No data fetched.")
        exit(1)

    results = run_optimization(data, cfg)

    print(f"\n{'=' * 100}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"  Charts saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 100}\n")
