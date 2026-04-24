from __future__ import annotations

"""
audit_analyze.py — Rigorous analysis of the scoring system.

Reads backtest_results/audit_dataset.parquet (built by audit_build_dataset.py)
and runs:
    A1. Incremental edge per indicator (fired vs not fired, at 63d forward)
    A2. Re-examine DROPPED indicators at current fresh data
    A3. Indicator redundancy / co-firing correlation
    B1. Current weights vs current edge ranking
    B2. Edge-proportional weights (backtest suggestion)
    D1. Score-bucket → forward-return curve (SPY-adjusted)
    D2. Is the 8.5 cliff real?
    D3. Edge decay — first half vs second half of the window
    E.  Simpler-baseline sanity checks (RS-only, N-of-7, price momentum)

Usage:
    python3 audit_analyze.py
    python3 audit_analyze.py --horizon 42     # use 42d forward returns
    python3 audit_analyze.py --window 12mo    # only last 12 months
"""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

from indicators import INDICATOR_WEIGHTS


SCORED_INDICATORS = [
    ("relative_strength", "rs_fired",         "rs_points"),
    ("ichimoku_cloud",    "ichimoku_fired",   "ichimoku_points"),
    ("higher_lows",       "higher_lows_fired","higher_lows_points"),
    ("cmf",               "cmf_fired",        "cmf_points"),
    ("roc",               "roc_fired",        "roc_points"),
    ("dual_tf_rs",        "dual_tf_rs_fired", "dual_tf_rs_points"),
    ("atr_expansion",     "atr_fired",        "atr_points"),
]

DROPPED_INDICATORS = [
    ("moving_averages",   "ma_50_200_fired"),
    ("near_52w_high",     "near_52w_fired"),
    ("rsi_momentum",      "rsi_fired"),
    ("macd_crossover",    "macd_fired"),
    ("adx_trend",         "adx_fired"),
    ("obv_trend",         "obv_fired"),
    ("donchian_breakout", "donchian_fired"),
]

# Trend-structure indicators (new in this audit — not in production)
TREND_STRUCTURE_INDICATORS = [
    ("higher_highs", "higher_highs_fired"),
    ("lower_lows",   "lower_lows_fired"),
    ("lower_highs",  "lower_highs_fired"),
]


# ─────────────────────────────────────────────────────────────
def bootstrap_mean_ci(values: np.ndarray, n_iter: int = 1000, alpha: float = 0.05) -> tuple[float, float]:
    """Return (lo, hi) 95% CI for the mean via block bootstrap."""
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boots = []
    n = len(values)
    for _ in range(n_iter):
        sample = values[rng.integers(0, n, n)]
        boots.append(sample.mean())
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def _mean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) > 0 else float("nan")


# ─────────────────────────────────────────────────────────────
# A1: Incremental edge per scored indicator
# ─────────────────────────────────────────────────────────────
def analysis_A1(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    For each scored indicator, compute:
      - standalone edge: mean(fwd | fired) - mean(fwd | not fired)
      - fire rate
      - incremental edge = standalone edge computed ONLY among rows where
        RS has ALREADY fired (RS >= 50 pctl). This removes the dominant RS
        confound and measures what the indicator adds on top of RS.

    The incremental edge is what weights were originally set by.
    """
    rows = []
    rs_strong = df["rs_fired"] == 1

    for name, fire_col, _ in SCORED_INDICATORS:
        fired = df[fire_col] == 1
        n_fired = int(fired.sum())
        fire_rate = float(fired.mean())

        r_fired = df.loc[fired, target].dropna()
        r_nofired = df.loc[~fired, target].dropna()

        edge = _mean(r_fired) - _mean(r_nofired)
        lo, hi = bootstrap_mean_ci(r_fired.values) if len(r_fired) > 0 else (np.nan, np.nan)
        lo2, hi2 = bootstrap_mean_ci(r_nofired.values) if len(r_nofired) > 0 else (np.nan, np.nan)

        # Incremental edge: among RS-strong rows only
        if name == "relative_strength":
            inc_edge = None
            inc_n = None
        else:
            sub = df.loc[rs_strong]
            sf = sub[fire_col] == 1
            r_f = sub.loc[sf, target].dropna()
            r_nf = sub.loc[~sf, target].dropna()
            inc_edge = _mean(r_f) - _mean(r_nf)
            inc_n = int(sf.sum())

        rows.append({
            "indicator": name,
            "fire_rate": fire_rate,
            "n_fired": n_fired,
            "mean_fwd_fired": _mean(r_fired),
            "mean_fwd_not_fired": _mean(r_nofired),
            "standalone_edge": edge,
            "fired_ci_lo": lo, "fired_ci_hi": hi,
            "incremental_edge_vs_rs_strong": inc_edge,
            "n_fired_within_rs_strong": inc_n,
        })

    return pd.DataFrame(rows).sort_values("incremental_edge_vs_rs_strong", ascending=False,
                                          na_position="first").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# A2: Re-examine dropped indicators
# ─────────────────────────────────────────────────────────────
def analysis_A2(df: pd.DataFrame, target: str,
                indicators: list[tuple[str, str]] = None) -> pd.DataFrame:
    """
    For each provided indicator, compute:
      - standalone edge
      - incremental edge vs rs_strong (what original calibration measured)
      - conditional edge when RS is WEAK (rescue candidate?)
    """
    if indicators is None:
        indicators = DROPPED_INDICATORS
    rs_strong = df["rs_fired"] == 1
    rs_weak = ~rs_strong
    rows = []
    for name, fire_col in indicators:
        if fire_col not in df.columns:
            continue
        fired = df[fire_col] == 1
        r_fired = df.loc[fired, target].dropna()
        r_nofired = df.loc[~fired, target].dropna()
        edge = _mean(r_fired) - _mean(r_nofired)

        # Incremental within RS-strong (original calibration context)
        sub_s = df.loc[rs_strong]
        sf_s = sub_s[fire_col] == 1
        inc_strong = _mean(sub_s.loc[sf_s, target].dropna()) - _mean(sub_s.loc[~sf_s, target].dropna())

        # Incremental within RS-weak (rescue indicator?)
        sub_w = df.loc[rs_weak]
        sf_w = sub_w[fire_col] == 1
        inc_weak = _mean(sub_w.loc[sf_w, target].dropna()) - _mean(sub_w.loc[~sf_w, target].dropna())

        rows.append({
            "dropped_indicator": name,
            "fire_rate": float(fired.mean()),
            "n_fired": int(fired.sum()),
            "standalone_edge": edge,
            "inc_edge_rs_strong": inc_strong,
            "inc_edge_rs_weak": inc_weak,
        })
    return pd.DataFrame(rows).sort_values("inc_edge_rs_strong", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# D3: edge decay (first half vs second half)
# ─────────────────────────────────────────────────────────────
def analysis_D3(df: pd.DataFrame, target: str) -> pd.DataFrame:
    median_date = df["date"].quantile(0.5)
    first = df[df["date"] < median_date]
    second = df[df["date"] >= median_date]
    rows = []
    for name, fire_col, _ in SCORED_INDICATORS:
        rows.append({
            "indicator": name,
            "first_half_edge": (_mean(first.loc[first[fire_col] == 1, target].dropna())
                                - _mean(first.loc[first[fire_col] == 0, target].dropna())),
            "second_half_edge": (_mean(second.loc[second[fire_col] == 1, target].dropna())
                                 - _mean(second.loc[second[fire_col] == 0, target].dropna())),
        })
    out = pd.DataFrame(rows)
    out["delta"] = out["second_half_edge"] - out["first_half_edge"]
    # Add high-score performance by half
    for label, sub in [("first_half", first), ("second_half", second)]:
        mask = sub["score"] >= 8.5
        vals = sub.loc[mask, target].dropna()
        pass  # done via annotations below

    print(f"\n  [D3 context]")
    print(f"    first half:  dates {first['date'].min().date()} → {first['date'].max().date()}")
    print(f"    second half: dates {second['date'].min().date()} → {second['date'].max().date()}")
    for label, sub in [("first", first), ("second", second)]:
        m = sub["score"] >= 8.5
        v = sub.loc[m, target].dropna()
        if len(v) > 0:
            print(f"    {label} half score>=8.5: N={len(v):,}  "
                  f"mean={v.mean():+.2%}  win={float((v>0).mean()):.1%}")
    return out


# ─────────────────────────────────────────────────────────────
# A3: Redundancy / co-firing correlation
# ─────────────────────────────────────────────────────────────
def analysis_A3(df: pd.DataFrame) -> pd.DataFrame:
    fire_cols = [c for _, c, _ in SCORED_INDICATORS]
    return df[fire_cols].astype(int).corr(method="pearson")


# ─────────────────────────────────────────────────────────────
# B1: weight rank vs edge rank
# ─────────────────────────────────────────────────────────────
def analysis_B1(a1_df: pd.DataFrame) -> pd.DataFrame:
    name_to_weight = {
        "relative_strength": 3.0,
        "ichimoku_cloud":    2.0,
        "cmf":               1.5,
        "roc":               1.5,
        "higher_lows":       1.0,
        "dual_tf_rs":        0.5,
        "atr_expansion":     0.5,
    }
    rows = []
    for _, r in a1_df.iterrows():
        name = r["indicator"]
        rows.append({
            "indicator": name,
            "current_weight": name_to_weight[name],
            "standalone_edge": r["standalone_edge"],
            "incremental_edge": r["incremental_edge_vs_rs_strong"],
        })
    out = pd.DataFrame(rows)
    out["weight_rank"] = out["current_weight"].rank(ascending=False, method="min").astype(int)
    # Edge rank: incremental if available, else standalone (for RS)
    edge_for_rank = out["incremental_edge"].fillna(out["standalone_edge"])
    out["edge_rank"] = edge_for_rank.rank(ascending=False, method="min").astype(int)
    out["rank_delta"] = out["edge_rank"] - out["weight_rank"]
    return out.sort_values("weight_rank").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# B2: edge-proportional weights
# ─────────────────────────────────────────────────────────────
def analysis_B2(a1_df: pd.DataFrame) -> pd.DataFrame:
    name_to_weight = {
        "relative_strength": 3.0,
        "ichimoku_cloud":    2.0,
        "cmf":               1.5,
        "roc":               1.5,
        "higher_lows":       1.0,
        "dual_tf_rs":        0.5,
        "atr_expansion":     0.5,
    }
    # Use incremental edge (standalone for RS). Clip negatives to 0.
    edges = []
    for _, r in a1_df.iterrows():
        e = r["incremental_edge_vs_rs_strong"]
        if e is None or pd.isna(e):
            e = r["standalone_edge"]
        edges.append(max(e, 0.0))
    edges = np.array(edges)
    total = edges.sum()
    if total == 0:
        return pd.DataFrame()
    scaled = edges / total * 10.0  # preserve 0-10 max score scale

    rows = []
    for i, (_, r) in enumerate(a1_df.iterrows()):
        name = r["indicator"]
        rows.append({
            "indicator": name,
            "current_weight": name_to_weight[name],
            "edge_proportional_weight": round(float(scaled[i]), 2),
            "delta": round(float(scaled[i]) - name_to_weight[name], 2),
        })
    return pd.DataFrame(rows).sort_values("edge_proportional_weight", ascending=False)


# ─────────────────────────────────────────────────────────────
# D1/D2: score-bucket → forward-return curve
# ─────────────────────────────────────────────────────────────
def analysis_D(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # 0.5-point buckets from 0 to 10
    edges = np.arange(0, 10.5, 0.5)
    labels = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges)-1)]
    df2 = df.copy()
    df2["score_bucket"] = pd.cut(df2["score"], bins=edges, labels=labels, include_lowest=True)
    g = df2.groupby("score_bucket", observed=True)[target].agg(["count", "mean", "median", "std"])
    g["ci_lo"] = np.nan
    g["ci_hi"] = np.nan
    for idx in g.index:
        vals = df2.loc[df2["score_bucket"] == idx, target].dropna().values
        if len(vals) >= 10:
            lo, hi = bootstrap_mean_ci(vals)
            g.loc[idx, "ci_lo"] = lo
            g.loc[idx, "ci_hi"] = hi
    return g.reset_index()


def analysis_D2_cliff(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Test the 8.5 cliff at fine-grained thresholds 7.5 / 8.0 / 8.5 / 9.0 / 9.5."""
    thresholds = [7.5, 8.0, 8.25, 8.5, 8.75, 9.0, 9.5]
    rows = []
    for t in thresholds:
        mask = df["score"] >= t
        vals = df.loc[mask, target].dropna()
        if len(vals) == 0:
            continue
        lo, hi = bootstrap_mean_ci(vals.values)
        rows.append({
            "threshold": t,
            "n": int(mask.sum()),
            "mean_fwd": float(vals.mean()),
            "median_fwd": float(vals.median()),
            "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# E: simpler baselines
# ─────────────────────────────────────────────────────────────
def analysis_E(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Compare our weighted score to simpler baselines at the high-conviction threshold."""
    # Thresholds calibrated to produce ~same # of signals as score >= 8.5
    current_high = df["score"] >= 8.5
    n_current = int(current_high.sum())

    # Pure RS: top N rows by rs_percentile
    rs_rank = df["rs_percentile"].rank(ascending=False, method="first")
    rs_top = rs_rank <= n_current

    # Pure 63d momentum (roc_value is 21d, but rs_63d_pctl is effectively 63d RS)
    mom_rank = df["rs_63d_pctl"].rank(ascending=False, method="first")
    mom_top = mom_rank <= n_current

    # N-of-7 binary count
    fire_cols = [c for _, c, _ in SCORED_INDICATORS]
    df_n = df.copy()
    df_n["n_fired"] = df_n[fire_cols].sum(axis=1)

    # Match # signals
    count_thresholds = sorted(df_n["n_fired"].unique(), reverse=True)
    n_of_7 = None
    for t in count_thresholds:
        if (df_n["n_fired"] >= t).sum() >= n_current:
            n_of_7 = t
            break

    n_of_7_mask = df_n["n_fired"] >= (n_of_7 or 7)

    # Top-N by N-of-7 count, tie-break by rs_percentile
    df_n["sort_key"] = df_n["n_fired"] * 100 + df_n["rs_percentile"].fillna(0) / 100
    n_of_7_top = df_n["sort_key"].rank(ascending=False, method="first") <= n_current

    rows = []
    for name, mask in [
        ("current_score >= 8.5", current_high),
        (f"top-N rs_percentile (N={n_current})", rs_top),
        (f"top-N rs_63d_pctl (N={n_current})", mom_top),
        (f"N-of-7 >= {n_of_7}, top-N tie-break by RS", n_of_7_top),
    ]:
        vals = df.loc[mask, target].dropna()
        if len(vals) == 0:
            continue
        lo, hi = bootstrap_mean_ci(vals.values)
        rows.append({
            "strategy": name,
            "n_signals": int(mask.sum()),
            "mean_fwd": float(vals.mean()),
            "median_fwd": float(vals.median()),
            "win_rate": float((vals > 0).mean()),
            "ci_lo": lo, "ci_hi": hi,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────
# Pretty printers
# ─────────────────────────────────────────────────────────────
def _p(df: pd.DataFrame, title: str, fmt: dict | None = None):
    print(f"\n{'=' * 80}\n  {title}\n{'=' * 80}")
    s = df.copy()
    if fmt:
        for c, f in fmt.items():
            if c in s.columns:
                s[c] = s[c].map(lambda v: f.format(v) if pd.notna(v) else "—")
    print(s.to_string(index=False))


def main(path: str, target: str, window: Optional[str]):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"loaded {len(df):,} rows, {df['ticker'].nunique()} tickers, "
          f"{df['date'].min().date()} → {df['date'].max().date()}")

    if window:
        end = df["date"].max()
        if window.endswith("mo"):
            n = int(window[:-2])
            start = end - pd.DateOffset(months=n)
        elif window.endswith("y"):
            n = int(window[:-1])
            start = end - pd.DateOffset(years=n)
        else:
            raise ValueError("--window must be like '12mo' or '2y'")
        df = df[df["date"] >= start]
        print(f"  filtered to {window}: {len(df):,} rows, "
              f"{df['date'].min().date()} → {df['date'].max().date()}")

    print(f"  target = {target}")

    # A1
    a1 = analysis_A1(df, target)
    _p(a1[[
        "indicator", "fire_rate", "n_fired",
        "mean_fwd_fired", "mean_fwd_not_fired",
        "standalone_edge", "incremental_edge_vs_rs_strong"
    ]], "A1. Per-indicator edge (target = " + target + ")",
       fmt={"fire_rate": "{:.2%}", "mean_fwd_fired": "{:+.2%}",
            "mean_fwd_not_fired": "{:+.2%}", "standalone_edge": "{:+.2%}",
            "incremental_edge_vs_rs_strong": "{:+.2%}"})

    # B1
    b1 = analysis_B1(a1)
    _p(b1, "B1. Current weights vs current edge ranking",
       fmt={"standalone_edge": "{:+.2%}", "incremental_edge": "{:+.2%}"})

    # B2
    b2 = analysis_B2(a1)
    _p(b2, "B2. Edge-proportional weights (informational)")

    # A3
    a3 = analysis_A3(df)
    _p(a3.round(2), "A3. Indicator co-firing correlation matrix")

    # D1
    d1 = analysis_D(df, target)
    _p(d1, "D1. Score-bucket → forward-return curve",
       fmt={"mean": "{:+.2%}", "median": "{:+.2%}", "std": "{:.2%}",
            "ci_lo": "{:+.2%}", "ci_hi": "{:+.2%}"})

    # D2
    d2 = analysis_D2_cliff(df, target)
    _p(d2, "D2. Threshold-cliff test (8.5 anchor)",
       fmt={"mean_fwd": "{:+.2%}", "median_fwd": "{:+.2%}",
            "ci_lo": "{:+.2%}", "ci_hi": "{:+.2%}"})

    # A2 (dropped indicators)
    a2 = analysis_A2(df, target)
    _p(a2, "A2. Dropped indicators — should any be re-included?",
       fmt={"fire_rate": "{:.2%}", "standalone_edge": "{:+.2%}",
            "inc_edge_rs_strong": "{:+.2%}", "inc_edge_rs_weak": "{:+.2%}"})

    # A2b (trend-structure indicators — new, untested)
    if "higher_highs_fired" in df.columns:
        a2b = analysis_A2(df, target, indicators=TREND_STRUCTURE_INDICATORS)
        if not a2b.empty:
            a2b = a2b.rename(columns={"dropped_indicator": "trend_indicator"})
            _p(a2b, "A2b. Trend-structure indicators (Higher Highs / Lower Lows / Lower Highs)",
               fmt={"fire_rate": "{:.2%}", "standalone_edge": "{:+.2%}",
                    "inc_edge_rs_strong": "{:+.2%}", "inc_edge_rs_weak": "{:+.2%}"})

            # Also report redundancy with Higher Lows (already scored)
            print("\n  [A2b] Co-firing with existing Higher Lows (redundancy check)")
            for name, col in TREND_STRUCTURE_INDICATORS:
                if col in df.columns:
                    both = ((df[col] == 1) & (df["higher_lows_fired"] == 1)).sum()
                    n_this = (df[col] == 1).sum()
                    n_hl = (df["higher_lows_fired"] == 1).sum()
                    print(f"    {name:<14}  fires={n_this:>6,}  "
                          f"co-fires with HL={both:>6,}  "
                          f"overlap={both/max(n_this,1):.1%} of {name} / "
                          f"{both/max(n_hl,1):.1%} of HL")

            # High-score conditional: how often does LL fire when score>=8.5?
            print("\n  [A2b] Trend-structure firings within high-conviction (score>=8.5)")
            high = df[df["score"] >= 8.5]
            for name, col in TREND_STRUCTURE_INDICATORS:
                if col in high.columns:
                    n = int((high[col] == 1).sum())
                    vals_fired = high.loc[high[col] == 1, target].dropna()
                    vals_nof = high.loc[high[col] == 0, target].dropna()
                    if len(vals_fired) > 0:
                        edge = float(vals_fired.mean() - vals_nof.mean())
                        print(f"    score>=8.5 AND {name:<14} fires: N={n:>4}  "
                              f"mean_fwd={vals_fired.mean():+.2%}  "
                              f"vs not fired={vals_nof.mean():+.2%}  "
                              f"delta={edge:+.2%}")

    # D3 (edge decay)
    d3 = analysis_D3(df, target)
    _p(d3, "D3. Edge decay — first half vs second half",
       fmt={"first_half_edge": "{:+.2%}", "second_half_edge": "{:+.2%}",
            "delta": "{:+.2%}"})

    # E
    e = analysis_E(df, target)
    _p(e, "E. Simpler baselines (same signal count as score>=8.5)",
       fmt={"mean_fwd": "{:+.2%}", "median_fwd": "{:+.2%}",
            "win_rate": "{:.1%}",
            "ci_lo": "{:+.2%}", "ci_hi": "{:+.2%}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy",
                    choices=["fwd_10d", "fwd_21d", "fwd_42d", "fwd_63d",
                             "fwd_10d_xspy", "fwd_21d_xspy", "fwd_42d_xspy", "fwd_63d_xspy"])
    ap.add_argument("--window", default=None, help="e.g. '12mo' or '2y' — filter to recent slice")
    args = ap.parse_args()
    main(args.input, args.target, args.window)
