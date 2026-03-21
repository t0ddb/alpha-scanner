from __future__ import annotations

"""
gradient_analysis.py — Bucketed backtest of continuous indicator metrics.

Tests whether signal STRENGTH (not just presence) correlates with forward
returns.  For each of the 8 indicators, the continuous metric is bucketed
and forward returns are computed per bucket.  Monotonicity tests determine
whether gradient scoring is justified and what shape the gradient should
take (linear, S-curve, or step).

Usage:
    python3 gradient_analysis.py
"""

import os
import pandas as pd
import numpy as np
from scipy import stats as sp_stats
from datetime import datetime

import matplotlib
matplotlib.use("Agg")                       # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import (
    load_config, get_all_tickers, get_indicator_config,
    get_ticker_metadata,
)
from data_fetcher import fetch_batch
from indicators import compute_all_indicators, INDICATOR_WEIGHTS

# =============================================================
# CONSTANTS
# =============================================================
FORWARD_WINDOWS = [10, 21, 42, 63]

BUCKET_DEFINITIONS: dict[str, list[float]] = {
    "rs_percentile":     [0, 20, 40, 60, 70, 80, 90, 100],
    "ichimoku_score":    [0, 1, 2, 3],
    "higher_lows_count": [0, 1, 2, 3, 4, 5, 6],
    "roc_value":         [-20, -10, -5, 0, 2, 5, 10, 15, 20, 50],
    "cmf_value":         [-0.3, -0.15, -0.05, 0, 0.05, 0.10, 0.15, 0.25, 0.5],
    "pct_from_52w_high": [-50, -30, -20, -10, -5, -2, 0, 5],
    "atr_percentile":    [0, 20, 40, 60, 70, 80, 90, 100],
    "ma_pct_above_50":   [-20, -10, -5, 0, 2, 5, 10, 20, 100],
    "ma_pct_above_200":  [-20, -10, -5, 0, 5, 10, 20, 40, 100],
}

# Human-readable labels for printing / chart titles
METRIC_LABELS: dict[str, str] = {
    "rs_percentile":     "Relative Strength (percentile)",
    "ichimoku_score":    "Ichimoku Cloud (sub-condition count)",
    "higher_lows_count": "Higher Lows (consecutive count)",
    "roc_value":         "Rate of Change (%)",
    "cmf_value":         "Chaikin Money Flow",
    "pct_from_52w_high": "Distance from 52-Week High (%)",
    "atr_percentile":    "ATR Expansion (percentile)",
    "ma_pct_above_50":   "Price vs 50-SMA (%)",
    "ma_pct_above_200":  "Price vs 200-SMA (%)",
}

# Current binary trigger thresholds (for chart annotation)
CURRENT_THRESHOLDS: dict[str, float] = {
    "rs_percentile":     75,
    "ichimoku_score":    2,
    "higher_lows_count": 4,
    "roc_value":         5.0,
    "cmf_value":         0.05,
    "pct_from_52w_high": -2.0,
    "atr_percentile":    80,
    "ma_pct_above_50":   0,
    "ma_pct_above_200":  0,
}

# Map each metric back to its indicator weight key
METRIC_TO_WEIGHT_KEY: dict[str, str] = {
    "rs_percentile":     "relative_strength",
    "ichimoku_score":    "ichimoku_cloud",
    "higher_lows_count": "higher_lows",
    "roc_value":         "roc",
    "cmf_value":         "cmf",
    "pct_from_52w_high": "near_52w_high",
    "atr_percentile":    "atr_expansion",
    "ma_pct_above_50":   "moving_averages",
    "ma_pct_above_200":  "moving_averages",
}

OUTPUT_DIR = "backtest_results"


# =============================================================
# STEP 1: Collect continuous metrics + forward returns
# =============================================================
def collect_continuous_metrics(
    data: dict[str, pd.DataFrame],
    cfg: dict,
    test_frequency: int = 5,
) -> pd.DataFrame:
    """
    Walk through historical dates and record continuous metric values
    plus forward returns for every ticker on each test date.

    Mirrors indicator_analysis_full.collect_all_indicator_events() but
    records gradient values instead of booleans.
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

    print(f"  Collecting continuous metrics across {len(test_indices)} dates, "
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

        # --- RS percentile pre-computation (same as indicator_analysis_full) ---
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

            # --- Extract continuous metrics ---
            rs = indicators.get("relative_strength", {})
            ich = indicators.get("ichimoku_cloud", {})
            hl = indicators.get("higher_lows", {})
            ma = indicators.get("moving_averages", {})
            roc = indicators.get("roc", {})
            cmf = indicators.get("cmf", {})
            n52 = indicators.get("near_52w_high", {})
            atr = indicators.get("atr_expansion", {})

            # Ichimoku ordinal score: count of 3 sub-conditions
            ich_score = (
                int(ich.get("above_cloud", False))
                + int(ich.get("cloud_bullish", False))
                + int(ich.get("tenkan_above_kijun", False))
            )

            # Distance above cloud (%)
            cloud_top = ich.get("cloud_top")
            current_close = df["Close"].iloc[-1]
            if cloud_top and cloud_top > 0:
                pct_above_cloud = ((current_close - cloud_top) / cloud_top) * 100
            else:
                pct_above_cloud = None

            # MA distance metrics
            sma_50 = ma.get("sma_50")
            sma_200 = ma.get("sma_200")
            ma_close = ma.get("current_close", current_close)
            pct_above_50 = ((ma_close - sma_50) / sma_50 * 100) if sma_50 and sma_50 > 0 else None
            pct_above_200 = ((ma_close - sma_200) / sma_200 * 100) if sma_200 and sma_200 > 0 else None

            meta = metadata.get(ticker, {})
            row = {
                "date":               benchmark_df.index[idx].strftime("%Y-%m-%d"),
                "ticker":             ticker,
                "sector":             meta.get("sector_name", ""),
                # Continuous metrics
                "rs_percentile":      rs.get("rs_percentile"),
                "ichimoku_score":     ich_score,
                "pct_above_cloud":    round(pct_above_cloud, 2) if pct_above_cloud is not None else None,
                "higher_lows_count":  hl.get("consecutive_higher_lows", 0),
                "ma_pct_above_50":    round(pct_above_50, 2) if pct_above_50 is not None else None,
                "ma_pct_above_200":   round(pct_above_200, 2) if pct_above_200 is not None else None,
                "roc_value":          roc.get("roc"),
                "cmf_value":          cmf.get("cmf"),
                "pct_from_52w_high":  n52.get("pct_from_high"),
                "atr_percentile":     atr.get("atr_percentile"),
            }

            # Forward returns
            for w in FORWARD_WINDOWS:
                row[f"fwd_{w}d"] = fwd.get(w)

            all_rows.append(row)

    return pd.DataFrame(all_rows)


# =============================================================
# STEP 2: Bucket analysis
# =============================================================
def analyze_buckets(
    df: pd.DataFrame,
    metric_col: str,
    buckets: list[float],
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    For a given continuous metric, bucket observations and compute
    forward return stats per bucket.
    """
    if windows is None:
        windows = FORWARD_WINDOWS

    series = df[metric_col].dropna()
    if series.empty:
        return pd.DataFrame()

    # For integer/ordinal metrics (ichimoku_score, higher_lows_count),
    # use exact values; for continuous metrics, use pd.cut.
    is_ordinal = metric_col in ("ichimoku_score", "higher_lows_count")

    results: list[dict] = []

    if is_ordinal:
        for val in buckets:
            mask = df[metric_col] == val
            subset = df[mask]
            row = _compute_bucket_stats(subset, str(int(val)), val, windows)
            results.append(row)
        # Also group the tail: values >= max bucket
        max_b = max(buckets)
        mask = df[metric_col] > max_b
        if mask.sum() > 0:
            subset = df[mask]
            row = _compute_bucket_stats(subset, f"{int(max_b)}+", max_b + 1, windows)
            results.append(row)
    else:
        for i in range(len(buckets) - 1):
            lo, hi = buckets[i], buckets[i + 1]
            if i == len(buckets) - 2:
                mask = (df[metric_col] >= lo) & (df[metric_col] <= hi)
            else:
                mask = (df[metric_col] >= lo) & (df[metric_col] < hi)
            subset = df[mask]
            label = f"{lo} to {hi}"
            midpoint = (lo + hi) / 2
            row = _compute_bucket_stats(subset, label, midpoint, windows)
            results.append(row)

    return pd.DataFrame(results)


def _compute_bucket_stats(
    subset: pd.DataFrame,
    label: str,
    midpoint: float,
    windows: list[int],
) -> dict:
    """Compute count, mean, median, win rate, std for a bucket subset."""
    row: dict = {"bucket": label, "midpoint": midpoint, "N": len(subset)}
    for w in windows:
        col = f"fwd_{w}d"
        valid = subset[col].dropna()
        n = len(valid)
        if n > 0:
            row[f"avg_{w}d"] = round(valid.mean() * 100, 2)
            row[f"med_{w}d"] = round(valid.median() * 100, 2)
            row[f"wr_{w}d"] = round((valid > 0).mean() * 100, 1)
            row[f"std_{w}d"] = round(valid.std() * 100, 2)
            row[f"n_{w}d"] = n
        else:
            row[f"avg_{w}d"] = None
            row[f"med_{w}d"] = None
            row[f"wr_{w}d"] = None
            row[f"std_{w}d"] = None
            row[f"n_{w}d"] = 0
    return row


# =============================================================
# STEP 3: Monotonicity testing
# =============================================================
def test_monotonicity(bucket_df: pd.DataFrame, window: int = 63) -> dict:
    """
    Test whether bucket returns increase monotonically with signal strength.

    Returns:
        {
            "spearman_r": float,      # rank correlation
            "spearman_p": float,      # p-value
            "mono_score": float,      # fraction of adjacent pairs that increase
            "mono_pairs": str,        # e.g. "5/6"
            "best_fit": str,          # "linear", "step", or "s-curve"
            "recommendation": str,    # "gradient", "stepped", "binary"
        }
    """
    avg_col = f"avg_{window}d"
    valid = bucket_df.dropna(subset=["midpoint", avg_col])
    valid = valid[valid[f"n_{window}d"] >= 30]  # require 30+ observations

    if len(valid) < 3:
        return {
            "spearman_r": None, "spearman_p": None,
            "mono_score": None, "mono_pairs": "N/A",
            "best_fit": "insufficient data",
            "recommendation": "binary",
        }

    midpoints = valid["midpoint"].values
    returns = valid[avg_col].values

    # Spearman rank correlation
    r, p = sp_stats.spearmanr(midpoints, returns)

    # Monotonic score: fraction of adjacent pairs with increasing returns
    n_pairs = len(returns) - 1
    n_increasing = sum(
        1 for i in range(n_pairs) if returns[i + 1] > returns[i]
    )
    mono_score = n_increasing / n_pairs if n_pairs > 0 else 0

    # Best-fit detection
    best_fit = _detect_best_fit(midpoints, returns)

    # Decision rules
    if r is not None and r >= 0.80 and mono_score >= 0.75:
        recommendation = "gradient"
    elif r is not None and r >= 0.60 and mono_score >= 0.60:
        recommendation = "stepped"
    else:
        recommendation = "binary"

    return {
        "spearman_r": round(r, 3) if r is not None else None,
        "spearman_p": round(p, 4) if p is not None else None,
        "mono_score": round(mono_score, 2),
        "mono_pairs": f"{n_increasing}/{n_pairs}",
        "best_fit": best_fit,
        "recommendation": recommendation,
    }


def _detect_best_fit(midpoints: np.ndarray, returns: np.ndarray) -> str:
    """
    Compare linear, step, and s-curve fits; return the best.
    Uses R-squared for comparison.
    """
    n = len(midpoints)
    if n < 3:
        return "insufficient data"

    # Normalise midpoints to [0, 1] for fitting
    x = midpoints.astype(float)
    y = returns.astype(float)

    ss_total = np.sum((y - y.mean()) ** 2)
    if ss_total == 0:
        return "flat"

    # --- Linear fit ---
    slope, intercept = np.polyfit(x, y, 1)
    y_pred_lin = slope * x + intercept
    ss_res_lin = np.sum((y - y_pred_lin) ** 2)
    r2_linear = 1 - ss_res_lin / ss_total

    # --- Step fit (find best single cutpoint) ---
    best_r2_step = -999
    for cut_idx in range(1, n):
        y_pred_step = np.where(
            np.arange(n) < cut_idx,
            y[:cut_idx].mean(),
            y[cut_idx:].mean(),
        )
        ss_res = np.sum((y - y_pred_step) ** 2)
        r2 = 1 - ss_res / ss_total
        if r2 > best_r2_step:
            best_r2_step = r2

    # --- S-curve (simple logistic via 3rd-degree polynomial as proxy) ---
    if n >= 4:
        try:
            coeffs = np.polyfit(x, y, 3)
            y_pred_poly = np.polyval(coeffs, x)
            ss_res_poly = np.sum((y - y_pred_poly) ** 2)
            r2_scurve = 1 - ss_res_poly / ss_total
        except Exception:
            r2_scurve = -999
    else:
        r2_scurve = -999

    fits = {
        "linear": r2_linear,
        "step": best_r2_step,
        "s-curve": r2_scurve,
    }

    return max(fits, key=fits.get)


# =============================================================
# STEP 4: Visualization
# =============================================================
def plot_bucket_charts(
    all_bucket_results: dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    """Generate per-indicator bucket charts showing returns by bucket."""
    os.makedirs(output_dir, exist_ok=True)

    for metric, bdf in all_bucket_results.items():
        if bdf.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x_labels = bdf["bucket"].tolist()
        x_pos = np.arange(len(x_labels))

        colors = {10: "#93c5fd", 21: "#60a5fa", 42: "#3b82f6", 63: "#1d4ed8"}

        for w in FORWARD_WINDOWS:
            avg_col = f"avg_{w}d"
            vals = bdf[avg_col].tolist()
            ax.plot(x_pos, vals, marker="o", label=f"{w}-day", color=colors[w], linewidth=2)

        # Current trigger threshold
        threshold = CURRENT_THRESHOLDS.get(metric)
        if threshold is not None:
            # Find x position closest to threshold
            midpoints = bdf["midpoint"].tolist()
            if midpoints:
                closest_idx = min(range(len(midpoints)),
                                  key=lambda i: abs(midpoints[i] - threshold))
                ax.axvline(x=closest_idx, color="red", linestyle="--",
                          alpha=0.6, label=f"Current threshold ({threshold})")

        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
        ax.set_xlabel("Bucket")
        ax.set_ylabel("Mean Forward Return (%)")
        ax.set_title(METRIC_LABELS.get(metric, metric))
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"bucket_{metric}.png"), dpi=150)
        plt.close(fig)

    print(f"  Saved {len(all_bucket_results)} bucket charts to {output_dir}/")


def plot_monotonicity_heatmap(
    mono_results: dict[str, dict[int, dict]],
    output_dir: str,
) -> None:
    """Generate a heatmap of Spearman r across indicators × windows."""
    os.makedirs(output_dir, exist_ok=True)

    metrics = list(mono_results.keys())
    windows = FORWARD_WINDOWS

    data = np.full((len(metrics), len(windows)), np.nan)
    for i, metric in enumerate(metrics):
        for j, w in enumerate(windows):
            r = mono_results[metric].get(w, {}).get("spearman_r")
            if r is not None:
                data[i, j] = r

    fig, ax = plt.subplots(figsize=(8, max(6, len(metrics) * 0.7)))
    im = ax.imshow(data, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f"{w}d" for w in windows])
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=9)

    # Annotate cells
    for i in range(len(metrics)):
        for j in range(len(windows)):
            val = data[i, j]
            if not np.isnan(val):
                color = "white" if abs(val) > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                       fontsize=10, color=color, fontweight="bold")

    ax.set_title("Monotonicity (Spearman r) — Bucket Return vs Signal Strength")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman r")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "monotonicity_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"  Saved monotonicity heatmap to {output_dir}/")


# =============================================================
# STEP 5: Weight recalibration analysis
# =============================================================
def analyze_gradient_weights(
    df: pd.DataFrame,
    all_bucket_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compare binary edge (current) to gradient regression slope.
    Reports whether relative indicator ranking changes.
    """
    rows: list[dict] = []

    for metric, bdf in all_bucket_results.items():
        if bdf.empty or metric in ("ma_pct_above_50", "ma_pct_above_200"):
            # MA has two sub-metrics; handle the weight once via pct_above_200
            if metric == "ma_pct_above_50":
                continue

        weight_key = METRIC_TO_WEIGHT_KEY.get(metric, "")
        current_weight = INDICATOR_WEIGHTS.get(weight_key, 0)

        # Binary edge: compare triggered (above threshold) vs not triggered
        threshold = CURRENT_THRESHOLDS.get(metric)
        avg_col = "avg_63d"

        if threshold is not None and metric in df.columns:
            series = df[metric].dropna()
            fwd = df.loc[series.index, "fwd_63d"].dropna()

            if metric == "pct_from_52w_high":
                fired = df[df[metric] >= threshold]["fwd_63d"].dropna()
                not_fired = df[df[metric] < threshold]["fwd_63d"].dropna()
            elif metric in ("ichimoku_score", "higher_lows_count"):
                fired = df[df[metric] >= threshold]["fwd_63d"].dropna()
                not_fired = df[df[metric] < threshold]["fwd_63d"].dropna()
            else:
                fired = df[df[metric] >= threshold]["fwd_63d"].dropna()
                not_fired = df[df[metric] < threshold]["fwd_63d"].dropna()

            binary_edge = (
                (fired.mean() - not_fired.mean()) * 100
                if len(fired) > 0 and len(not_fired) > 0 else 0
            )
        else:
            binary_edge = 0

        # Gradient slope: regression of midpoint vs avg return
        valid = bdf.dropna(subset=["midpoint", avg_col])
        valid = valid[valid[f"n_63d"] >= 30]
        if len(valid) >= 3:
            slope, _, r_val, _, _ = sp_stats.linregress(
                valid["midpoint"].values, valid[avg_col].values,
            )
            gradient_slope = round(slope, 4)
            r_squared = round(r_val ** 2, 3)
        else:
            gradient_slope = 0
            r_squared = 0

        rows.append({
            "metric": metric,
            "indicator": METRIC_LABELS.get(metric, metric),
            "weight_key": weight_key,
            "current_weight": current_weight,
            "binary_edge_63d": round(binary_edge, 2),
            "gradient_slope": gradient_slope,
            "r_squared": r_squared,
        })

    result_df = pd.DataFrame(rows)

    # Rank by gradient slope (normalised by range) for weight proposal
    if not result_df.empty:
        result_df = result_df.sort_values("binary_edge_63d", ascending=False)

    return result_df


# =============================================================
# STEP 6: Print summary & recommendations
# =============================================================
def print_bucket_table(metric: str, bdf: pd.DataFrame) -> None:
    """Pretty-print a single indicator's bucket analysis."""
    label = METRIC_LABELS.get(metric, metric)
    print(f"\n  {'─' * 95}")
    print(f"  {label}")
    print(f"  {'─' * 95}")

    header = f"  {'Bucket':<16} {'N':>6}"
    for w in FORWARD_WINDOWS:
        header += f"  {'Avg':>7} {'WR':>6}"
    print(header)

    sub = f"  {'':<16} {'':>6}"
    for w in FORWARD_WINDOWS:
        sub += f"  {f'{w}d':>7} {f'{w}d':>6}"
    print(sub)

    for _, row in bdf.iterrows():
        line = f"  {str(row['bucket']):<16} {row['N']:>6}"
        for w in FORWARD_WINDOWS:
            avg = row.get(f"avg_{w}d")
            wr = row.get(f"wr_{w}d")
            avg_s = f"{avg:>+6.1f}%" if avg is not None else f"{'N/A':>7}"
            wr_s = f"{wr:>5.1f}%" if wr is not None else f"{'N/A':>6}"
            line += f"  {avg_s} {wr_s}"
        # Flag sparse buckets
        if row["N"] < 50:
            line += "  ⚠ sparse"
        print(line)


def print_monotonicity_summary(
    mono_results: dict[str, dict[int, dict]],
) -> None:
    """Print the monotonicity summary table."""
    print(f"\n\n{'=' * 95}")
    print(f"  MONOTONICITY ANALYSIS — GRADIENT RECOMMENDATIONS")
    print(f"{'=' * 95}\n")

    header = (f"  {'Indicator':<35} {'Spearman r':>10} {'Mono':>8} "
              f"{'Best Fit':>10} {'Recommendation':>16}")
    print(header)
    print(f"  {'─' * 90}")

    for metric, windows_dict in mono_results.items():
        # Use 63-day window as primary
        result = windows_dict.get(63, {})
        r = result.get("spearman_r")
        mono = result.get("mono_score")
        fit = result.get("best_fit", "")
        rec = result.get("recommendation", "")

        r_s = f"{r:>+.3f}" if r is not None else "N/A"
        mono_s = result.get("mono_pairs", "N/A")

        emoji = {"gradient": "✅", "stepped": "🔶", "binary": "❌"}.get(rec, "")

        label = METRIC_LABELS.get(metric, metric)
        print(f"  {label:<35} {r_s:>10} {mono_s:>8} "
              f"{fit:>10} {emoji} {rec:>14}")


def print_weight_comparison(weight_df: pd.DataFrame) -> None:
    """Print weight recalibration table."""
    print(f"\n\n{'=' * 95}")
    print(f"  WEIGHT COMPARISON — BINARY vs GRADIENT")
    print(f"{'=' * 95}\n")

    print(f"  {'Indicator':<35} {'Binary Edge':>12} {'Cur Wt':>8} "
          f"{'Grad Slope':>12} {'R²':>6}")
    print(f"  {'─' * 78}")

    for _, row in weight_df.iterrows():
        print(f"  {row['indicator']:<35} {row['binary_edge_63d']:>+10.2f}% "
              f"{row['current_weight']:>7.1f} "
              f"{row['gradient_slope']:>12.4f} {row['r_squared']:>5.3f}")


# =============================================================
# ORCHESTRATOR
# =============================================================
def run_full_analysis(
    data: dict[str, pd.DataFrame],
    cfg: dict,
) -> dict:
    """Run the complete gradient analysis pipeline."""

    # 1. Collect continuous metrics
    print("\n  Phase 1: Collecting continuous metrics...\n")
    events_df = collect_continuous_metrics(data, cfg, test_frequency=5)

    if events_df.empty:
        print("  [ERROR] No events collected.")
        return {}

    print(f"\n  Collected {len(events_df)} observations.\n")

    # 2. Bucket analysis for each metric
    print("\n  Phase 2: Bucket analysis...\n")
    all_bucket_results: dict[str, pd.DataFrame] = {}

    for metric, buckets in BUCKET_DEFINITIONS.items():
        if metric not in events_df.columns:
            print(f"  [SKIP] {metric} not in collected data.")
            continue
        bdf = analyze_buckets(events_df, metric, buckets)
        all_bucket_results[metric] = bdf
        print_bucket_table(metric, bdf)

    # 3. Monotonicity testing
    print("\n\n  Phase 3: Monotonicity testing...\n")
    mono_results: dict[str, dict[int, dict]] = {}

    for metric, bdf in all_bucket_results.items():
        mono_results[metric] = {}
        for w in FORWARD_WINDOWS:
            mono_results[metric][w] = test_monotonicity(bdf, window=w)

    print_monotonicity_summary(mono_results)

    # 4. Visualization
    print(f"\n\n  Phase 4: Generating charts...\n")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_bucket_charts(all_bucket_results, OUTPUT_DIR)
    plot_monotonicity_heatmap(mono_results, OUTPUT_DIR)

    # 5. Weight recalibration
    print(f"\n\n  Phase 5: Weight recalibration analysis...\n")
    weight_df = analyze_gradient_weights(events_df, all_bucket_results)
    print_weight_comparison(weight_df)

    return {
        "events_df": events_df,
        "bucket_results": all_bucket_results,
        "monotonicity": mono_results,
        "weights": weight_df,
    }


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    cfg = load_config()

    print("=" * 95)
    print("  GRADIENT SCORING ANALYSIS")
    print("  Testing whether signal STRENGTH correlates with forward returns")
    print("=" * 95)
    print()
    print("  This will take 10-15 minutes (fetching 2yr data + walking test dates).")
    print()

    # Fetch data
    print("  Fetching 2 years of data...\n")
    all_tickers = ["SPY"] + get_all_tickers(cfg)
    data = fetch_batch(all_tickers, period="2y", verbose=True)

    if not data:
        print("  [ERROR] No data fetched.")
        exit(1)

    results = run_full_analysis(data, cfg)

    print(f"\n{'=' * 95}")
    print(f"  ANALYSIS COMPLETE")
    print(f"  Charts saved to: {OUTPUT_DIR}/")
    print(f"{'=' * 95}\n")
