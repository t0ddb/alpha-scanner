from __future__ import annotations

"""
audit_gradient_buckets.py — Bucket analysis on continuous indicator metrics.

Reads backtest_results/audit_dataset.parquet (built by audit_build_dataset.py)
and computes forward-return-per-quantile-bucket for each candidate gradient
indicator. Used to derive gradient breakpoints empirically (the way the
original RS_GRADIENT was derived).

Candidates (currently binary, proposed for gradient):
  - ROC                     value: roc_value          (21-day % change)
  - ATR Expansion           value: atr_percentile     (within 50-day range)
  - Ichimoku Cloud          value: ichimoku_score     (0-3 ordinal)
  - Dual-TF RS              composite of multi-TF percentiles

Already-gradient (for reference / sanity):
  - Relative Strength       value: rs_percentile
  - Higher Lows             value: higher_lows_count

Usage:
    python3 audit_gradient_buckets.py --target fwd_63d_xspy
    python3 audit_gradient_buckets.py --target fwd_63d_xspy --window 12mo
"""

import argparse
import numpy as np
import pandas as pd


def bucket_stats(df: pd.DataFrame, value_col: str, target: str,
                 buckets: list[tuple[float, float]],
                 bucket_labels: list[str] | None = None) -> pd.DataFrame:
    """For each (lo, hi] bucket of `value_col`, report n + mean fwd return + win rate."""
    rows = []
    for i, (lo, hi) in enumerate(buckets):
        label = bucket_labels[i] if bucket_labels else f"({lo:g}, {hi:g}]"
        if i == 0 and lo == -np.inf:
            mask = df[value_col] <= hi
        elif hi == np.inf:
            mask = df[value_col] > lo
        else:
            mask = (df[value_col] > lo) & (df[value_col] <= hi)
        sub = df.loc[mask, target].dropna()
        rows.append({
            "bucket": label,
            "n": int(mask.sum()),
            "mean_fwd": float(sub.mean()) if len(sub) > 0 else np.nan,
            "median_fwd": float(sub.median()) if len(sub) > 0 else np.nan,
            "win_rate": float((sub > 0).mean()) if len(sub) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def quantile_buckets(df: pd.DataFrame, value_col: str,
                     pcts: list[float]) -> list[tuple[float, float]]:
    """Build bucket edges from quantile percentages (0-100)."""
    edges = [df[value_col].quantile(p / 100) for p in pcts]
    edges = sorted(set([-np.inf] + edges + [np.inf]))
    return list(zip(edges[:-1], edges[1:]))


def _print(name: str, df: pd.DataFrame, current_rule: str):
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"  Current scoring: {current_rule}")
    print(f"{'=' * 80}")
    out = df.copy()
    out["mean_fwd"] = out["mean_fwd"].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
    out["median_fwd"] = out["median_fwd"].map(lambda v: f"{v:+.2%}" if pd.notna(v) else "—")
    out["win_rate"] = out["win_rate"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    print(out.to_string(index=False))


def main(path: str, target: str, window: str | None):
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
    print(f"  target: {target}")

    # ─ ROC ─────────────────────────────────────────────────
    roc_buckets = [
        (-np.inf, -10), (-10, -5), (-5, 0), (0, 3), (3, 5),
        (5, 7.5), (7.5, 10), (10, 15), (15, 25), (25, 50), (50, np.inf),
    ]
    roc_labels = [
        "≤ -10%", "(-10%, -5%]", "(-5%, 0%]", "(0%, 3%]", "(3%, 5%]",
        "(5%, 7.5%]", "(7.5%, 10%]", "(10%, 15%]", "(15%, 25%]", "(25%, 50%]", "> 50%",
    ]
    res = bucket_stats(df, "roc_value", target, roc_buckets, roc_labels)
    _print("ROC (21-day % change)", res,
           "binary, > 5% → 1.5 pts")

    # ─ ATR Expansion ──────────────────────────────────────
    atr_buckets = [
        (-np.inf, 50), (50, 60), (60, 70), (70, 80),
        (80, 85), (85, 90), (90, 95), (95, 100),
    ]
    atr_labels = [
        "< 50th", "[50, 60)", "[60, 70)", "[70, 80)",
        "[80, 85)", "[85, 90)", "[90, 95)", "[95, 100]",
    ]
    res = bucket_stats(df, "atr_percentile", target, atr_buckets, atr_labels)
    _print("ATR Expansion (percentile within 50-day range)", res,
           "binary, ≥ 80th → 0.5 pts")

    # ─ Ichimoku Cloud ─────────────────────────────────────
    # ichimoku_score is 0-3 ordinal (above_cloud + cloud_bullish + tenkan_above_kijun)
    ich_buckets = [(-0.5, 0.5), (0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]
    ich_labels = ["0 (none)", "1 of 3", "2 of 3", "3 of 3 (full)"]
    res = bucket_stats(df, "ichimoku_score", target, ich_buckets, ich_labels)
    _print("Ichimoku Cloud (composite 0-3 score)", res,
           "binary, 'above cloud AND bullish' → 2.0 pts (treats 2/3 and 3/3 the same)")

    # ─ Dual-TF RS ─────────────────────────────────────────
    # The current binary uses two compound conditions (cond_a OR cond_b).
    # For gradient, we look at forward returns conditional on the underlying
    # multi-TF percentiles — what dimension matters?

    # 1. "Acceleration strength" — how much more is 63d than 126d, conditional on
    #    126d already strong. Captures the cond_a flavor.
    df = df.copy()
    df["dtf_accel"] = np.where(df["rs_126d_pctl"] >= 70,
                                df["rs_63d_pctl"] - df["rs_126d_pctl"], np.nan)

    accel_buckets = [(-np.inf, -10), (-10, 0), (0, 5), (5, 10), (10, 20), (20, np.inf)]
    accel_labels = ["≤ -10", "(-10, 0]", "(0, 5]", "(5, 10]", "(10, 20]", "> 20"]
    res = bucket_stats(df.dropna(subset=["dtf_accel"]),
                       "dtf_accel", target, accel_buckets, accel_labels)
    _print("Dual-TF — Acceleration strength (63d − 126d, when 126d ≥ 70)", res,
           "binary cond_a: 126d ≥ 70 + 63d > 126d → contributes to 2.5 pts (all-or-nothing)")

    # 2. "Sustained strength" — min(63d, 21d), captures the cond_b flavor.
    df["dtf_sustained"] = np.minimum(df["rs_63d_pctl"], df["rs_21d_pctl"])
    sus_buckets = [(-np.inf, 50), (50, 70), (70, 80), (80, 85), (85, 90), (90, 95), (95, 100)]
    sus_labels = ["< 50th", "[50, 70)", "[70, 80)", "[80, 85)", "[85, 90)", "[90, 95)", "[95, 100]"]
    res = bucket_stats(df, "dtf_sustained", target, sus_buckets, sus_labels)
    _print("Dual-TF — Sustained strength (min of 63d, 21d percentiles)", res,
           "binary cond_b: 63d ≥ 80 AND 21d ≥ 80 → contributes to 2.5 pts (all-or-nothing)")

    # ─ Reference: existing gradients ──────────────────────
    rs_buckets = [(-np.inf, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 95), (95, 99), (99, 100)]
    rs_labels = ["< 50th", "[50, 60)", "[60, 70)", "[70, 80)", "[80, 90)", "[90, 95)", "[95, 99)", "[99, 100]"]
    res = bucket_stats(df, "rs_percentile", target, rs_buckets, rs_labels)
    _print("[REFERENCE] Relative Strength (already gradient: 50→0.6, 60→1.2, 70→1.8, 80→2.4, 90→3.0)",
           res, "")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    ap.add_argument("--window", default=None, help="e.g. 12mo")
    args = ap.parse_args()
    main(args.input, args.target, args.window)
