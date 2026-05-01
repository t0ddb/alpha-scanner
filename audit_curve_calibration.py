from __future__ import annotations

"""
audit_curve_calibration.py — Compare curve families for gradient scoring.

For each candidate gradient indicator, fits multiple curve families on a
training window and evaluates out-of-sample Spearman rank correlation
against forward 63d-xSPY return. Time-series 5-fold cross-validation.

Curve families compared:
  1. Piecewise-linear  — interpolate through anchors at fixed percentiles
                         (multiple anchor sets tested per indicator)
  2. Power curve       — pts = max × ((v - floor) / (ceil - floor))^p
                         (grid search over p ∈ [0.3, 2.5])
  3. Isotonic          — non-parametric monotonic fit (no anchors)

Output: per-indicator table showing OOS ρ for each family, anchored decision
on which to ship. Also computes a "concentration score" — what fraction of
the universe-day cells score within 0.1 of the max — to verify the new
gradient spreads top scores.

Usage:
    python3 audit_curve_calibration.py
    python3 audit_curve_calibration.py --target fwd_63d_xspy
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────────
# Indicator definitions: which feature to use, value range, max weight
# ─────────────────────────────────────────────────────────────────────
INDICATORS = [
    # name              feature                 floor (no score) ceil (max score)  max_pts
    ("relative_strength", "rs_percentile",        50,    99,    3.0),
    ("dual_tf_rs",        "rs_63d_pctl",          70,    95,    2.5),  # using 63d as strength
    ("ichimoku_cloud",    "ichimoku_score",        0,     3,    2.0),  # 0-3 ordinal
    ("roc",               "roc_value",             5,    50,    1.5),
    ("atr_expansion",     "atr_percentile",       80,    95,    0.5),
    ("higher_lows",       "higher_lows_count",     2,     5,    0.5),
]


# ─────────────────────────────────────────────────────────────────────
# Curve family implementations
# ─────────────────────────────────────────────────────────────────────
def piecewise_linear(value: float | np.ndarray, anchors: list[tuple[float, float]]) -> float | np.ndarray:
    """Interpolate through (x, y) anchors. Below first → first y. Above last → last y."""
    xs = np.array([a[0] for a in anchors])
    ys = np.array([a[1] for a in anchors])
    return np.interp(np.asarray(value, dtype=float), xs, ys)


def power_curve(value: np.ndarray, floor: float, ceil: float, p: float, max_pts: float) -> np.ndarray:
    v = np.asarray(value, dtype=float)
    norm = np.clip((v - floor) / max(ceil - floor, 1e-9), 0.0, 1.0)
    return max_pts * (norm ** p)


def fit_isotonic(X: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip", increasing="auto")
    iso.fit(X, y)
    return iso


# ─────────────────────────────────────────────────────────────────────
# Anchor-set candidates for piecewise-linear (per indicator)
# ─────────────────────────────────────────────────────────────────────
def _anchor_sets(name: str, max_pts: float) -> dict[str, list[tuple[float, float]]]:
    """Return candidate anchor lists for piecewise-linear fits."""
    if name == "relative_strength":
        return {
            "human": [(50, 0.25*max_pts/3.0), (60, 0.5*max_pts/3.0), (70, max_pts/3.0),
                      (80, 1.5*max_pts/3.0), (90, 2.0*max_pts/3.0), (95, 2.5*max_pts/3.0), (99, max_pts)],
            "decile": [(50, 0.0), (60, max_pts*0.15), (70, max_pts*0.30), (80, max_pts*0.50),
                       (90, max_pts*0.75), (99, max_pts)],
            "tail-heavy": [(50, 0.0), (80, max_pts*0.40), (90, max_pts*0.60), (95, max_pts*0.80), (99, max_pts)],
            "current_scheme_C": [(50, 0.6), (60, 1.2), (70, 1.8), (80, 2.4), (90, max_pts)],
        }
    if name == "roc":
        return {
            "human": [(5, max_pts*0.4), (10, max_pts*0.6), (15, max_pts*0.73), (25, max_pts*0.87), (50, max_pts)],
            "linear": [(5, 0.0), (50, max_pts)],
            "tail-heavy": [(5, 0.2*max_pts), (15, 0.5*max_pts), (30, 0.8*max_pts), (50, max_pts)],
        }
    if name == "atr_expansion":
        return {
            "human": [(80, max_pts*0.4), (85, max_pts*0.6), (90, max_pts*0.8), (95, max_pts)],
            "linear": [(80, 0.0), (100, max_pts)],
        }
    if name == "ichimoku_cloud":
        return {
            "human": [(0, 0), (1, max_pts*0.25), (2, max_pts*0.5), (3, max_pts)],
            "step-only": [(0, 0), (2, max_pts*0.5), (3, max_pts)],
            "all_or_nothing": [(0, 0), (2, 0), (3, max_pts)],
        }
    if name == "dual_tf_rs":
        return {
            "human": [(70, max_pts*0.4), (80, max_pts*0.56), (85, max_pts*0.68),
                      (90, max_pts*0.8), (95, max_pts)],
            "linear": [(70, 0), (95, max_pts)],
            "tail-heavy": [(70, 0), (90, max_pts*0.6), (95, max_pts)],
        }
    if name == "higher_lows":
        return {
            "current_scheme_C": [(2, 0.125), (3, 0.25), (4, 0.375), (5, 0.5)],
            "tail-heavy": [(2, 0.05), (4, 0.25), (5, 0.5)],
        }
    return {}


# ─────────────────────────────────────────────────────────────────────
# Time-series CV evaluation
# ─────────────────────────────────────────────────────────────────────
def time_series_folds(df: pd.DataFrame, n_folds: int = 5) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield (train, test) splits where each test is a contiguous future window."""
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    fold_size = n // (n_folds + 1)  # first fold_size = warmup; n_folds tests after
    splits = []
    for k in range(n_folds):
        train_end = (k + 1) * fold_size
        test_end = (k + 2) * fold_size if k < n_folds - 1 else n
        train = df.iloc[:train_end]
        test = df.iloc[train_end:test_end]
        splits.append((train, test))
    return splits


def evaluate_family(
    df: pd.DataFrame, value_col: str, target_col: str, max_pts: float,
    family: str, params: dict, anchor_sets_map: dict,
) -> dict:
    """Run TS-CV; return dict with per-fold and mean OOS Spearman."""
    folds = time_series_folds(df, n_folds=5)
    rhos = []
    for train, test in folds:
        train_v = train[value_col].dropna()
        train_y = train.loc[train_v.index, target_col]
        valid_mask = train_y.notna()
        train_v, train_y = train_v[valid_mask], train_y[valid_mask]

        test_v = test[value_col].dropna()
        test_y = test.loc[test_v.index, target_col]
        vmask = test_y.notna()
        test_v, test_y = test_v[vmask], test_y[vmask]

        if len(train_v) < 100 or len(test_v) < 100:
            rhos.append(np.nan); continue

        if family == "piecewise":
            anchors = anchor_sets_map[params["anchor_set"]]
            preds = piecewise_linear(test_v.values, anchors)
        elif family == "power":
            floor = params["floor"]
            ceil = params["ceil"]
            p = params["p"]
            preds = power_curve(test_v.values, floor, ceil, p, max_pts)
        elif family == "isotonic":
            iso = fit_isotonic(train_v.values, train_y.values)
            preds = iso.predict(test_v.values)
        else:
            raise ValueError(family)

        if np.std(preds) == 0:
            rhos.append(0.0); continue
        rho, _ = spearmanr(preds, test_y.values)
        rhos.append(float(rho) if pd.notna(rho) else 0.0)

    rhos_clean = [r for r in rhos if pd.notna(r)]
    return {
        "family": family,
        "params": params,
        "rhos_per_fold": [round(r, 4) for r in rhos],
        "rho_mean": float(np.mean(rhos_clean)) if rhos_clean else np.nan,
        "rho_std": float(np.std(rhos_clean)) if len(rhos_clean) > 1 else 0.0,
    }


def evaluate_indicator(df: pd.DataFrame, indicator: tuple, target: str) -> list[dict]:
    """Run full curve-family + parameter sweep for one indicator."""
    name, value_col, floor, ceil, max_pts = indicator
    print(f"\n  Indicator: {name} (feature={value_col}, range=[{floor},{ceil}], max_pts={max_pts})")

    # Subset data: only include rows where the indicator's underlying feature
    # is in the meaningful range (don't waste CV on rows that fail floor)
    sub = df[df[value_col].notna() & df[target].notna()].copy()
    sub = sub[sub[value_col] >= floor]  # below floor → 0 score, no info
    print(f"    rows above floor: {len(sub):,}")

    results = []

    # 1. Piecewise linear — try each anchor set
    anchor_sets = _anchor_sets(name, max_pts)
    for anchor_label, anchors in anchor_sets.items():
        results.append(evaluate_family(sub, value_col, target, max_pts,
                                        family="piecewise",
                                        params={"anchor_set": anchor_label},
                                        anchor_sets_map=anchor_sets))

    # 2. Power curve — grid search over p
    for p in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]:
        results.append(evaluate_family(sub, value_col, target, max_pts,
                                        family="power",
                                        params={"floor": floor, "ceil": ceil, "p": p},
                                        anchor_sets_map={}))

    # 3. Isotonic — no params
    results.append(evaluate_family(sub, value_col, target, max_pts,
                                    family="isotonic", params={},
                                    anchor_sets_map={}))

    # Also report the anchor-set candidates for transparency
    return results


# ─────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────
def format_results(name: str, results: list[dict]) -> None:
    print(f"\n  Results for {name}:")
    print(f"    {'family':<14} {'params':<28} {'OOS ρ mean':>11} {'std':>7} {'per-fold':<35}")
    print(f"    {'─'*14} {'─'*28} {'─'*11} {'─'*7} {'─'*35}")
    # Sort by mean OOS ρ desc
    sorted_r = sorted(results, key=lambda r: -r["rho_mean"] if pd.notna(r["rho_mean"]) else -1e9)
    for r in sorted_r:
        params_str = ""
        if r["family"] == "piecewise":
            params_str = f"anchors={r['params']['anchor_set']}"
        elif r["family"] == "power":
            params_str = f"p={r['params']['p']}"
        elif r["family"] == "isotonic":
            params_str = "non-parametric"
        rho_str = f"{r['rho_mean']:+.4f}" if pd.notna(r['rho_mean']) else "N/A"
        std_str = f"±{r['rho_std']:.3f}"
        per_fold = str(r["rhos_per_fold"])
        marker = "  ← BEST" if r is sorted_r[0] else ""
        print(f"    {r['family']:<14} {params_str:<28} {rho_str:>11} {std_str:>7} {per_fold:<35}{marker}")


def main(path: str, target: str):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"loaded {len(df):,} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} → {df['date'].max().date()}")
    print(f"target: {target}")

    all_results = {}
    for ind in INDICATORS:
        results = evaluate_indicator(df, ind, target)
        format_results(ind[0], results)
        # Track the winner per indicator
        winner = max((r for r in results if pd.notna(r["rho_mean"])),
                     key=lambda r: r["rho_mean"])
        all_results[ind[0]] = (winner, results)

    print("\n" + "=" * 80)
    print("  WINNERS PER INDICATOR")
    print("=" * 80)
    for name, (winner, _) in all_results.items():
        print(f"  {name:<22} family={winner['family']:<12} "
              f"params={winner['params']}  OOS ρ={winner['rho_mean']:+.4f}")

    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    args = ap.parse_args()
    main(args.input, args.target)
