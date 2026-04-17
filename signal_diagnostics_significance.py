"""
signal_diagnostics_significance.py — Bootstrap 95% CI around subsector
Spearman ρ (sm10 score vs 63-day forward return).

Purpose: before acting on the subsector decomposition findings, separate
statistically reliable effects from small-sample noise. Final diagnostic
pass on the universe question.

Method:
  - For each subsector, filter to rows with both score_sm10 and
    fwd_ret_63 non-null
  - Bootstrap: 1000 resamples WITH replacement at the (ticker, date)
    observation level, recomputing Spearman ρ each iteration
  - 95% CI = [2.5%, 97.5%] quantiles of the bootstrap distribution
  - Classify: significant positive / negative / inconclusive based on
    whether CI crosses zero
  - Flag wide CIs (>0.4) as "insufficient data"

Caveat: observation-level bootstrap assumes independence of
(ticker, date) rows. Score is autocorrelated across time and
overlapping forward returns induce cross-observation dependence, so
the bootstrap CI is an approximation. It overstates precision somewhat;
treat p<0.01-level findings as real, borderline cases with skepticism.

Usage:
    python signal_diagnostics_significance.py
    python signal_diagnostics_significance.py --n-iter 2000 --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from config import load_config
from data_fetcher import fetch_all
from signal_diagnostics import (
    HORIZONS,
    compute_forward_returns,
    compute_smoothed_scores,
    load_scores,
)
from signal_diagnostics_subsector import parent_group


OUT_DIR = Path(__file__).parent / "signal_diagnostics_out"
WIDE_CI_FLAG = 0.40   # CI width threshold for "insufficient data"


def fast_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """
    Spearman ρ via Pearson-on-ranks (avoids scipy p-value overhead,
    ~5× faster for bootstrap use).
    """
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    sx = np.sqrt(np.sum(rx * rx))
    sy = np.sqrt(np.sum(ry * ry))
    if sx == 0 or sy == 0:
        return np.nan
    return float(np.sum(rx * ry) / (sx * sy))


def bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, n_iter: int, seed: int,
) -> tuple[float, float, float, np.ndarray]:
    """Return (point_est, ci_low, ci_high, bootstrap_rhos)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    point = fast_spearman(x, y)
    rhos = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        rhos[i] = fast_spearman(x[idx], y[idx])
    ci_low, ci_high = np.quantile(rhos[~np.isnan(rhos)], [0.025, 0.975])
    return point, float(ci_low), float(ci_high), rhos


def classify(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "significant positive"
    if ci_high < 0:
        return "significant negative"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=1000,
                    help="Bootstrap iterations (default: 1000)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 82)
    print("  SIGNAL DIAGNOSTICS — STATISTICAL SIGNIFICANCE (bootstrap CIs)")
    print("=" * 82)

    print("\n  Loading data...")
    scores_df = load_scores()
    cfg = load_config()
    price_data = fetch_all(cfg, period="2y", verbose=False)
    scores_df = compute_smoothed_scores(scores_df)
    df = compute_forward_returns(scores_df, price_data, HORIZONS)
    df = df[df["fwd_ret_63"].notna() & df["score_sm10"].notna()].reset_index(drop=True)
    print(f"  {len(df):,} valid (subsector, ticker, date) rows with sm10+h63 "
          f"across {df['subsector'].nunique()} subsectors")

    # ── Bootstrap per subsector ──
    print(f"\n  Bootstrapping Spearman ρ (sm10 vs fwd_ret_63) — "
          f"{args.n_iter} iterations per subsector...")
    rows = []
    for i, (subsector, sub) in enumerate(sorted(df.groupby("subsector")), 1):
        parent = parent_group(sub["sector"].iloc[0])
        n = len(sub)
        if n < 50:
            rows.append({
                "subsector": subsector, "parent": parent, "n": n,
                "rho_point": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "ci_width": np.nan,
                "classification": "insufficient data",
            })
            continue

        x = sub["score_sm10"].values
        y = sub["fwd_ret_63"].values
        point, ci_low, ci_high, _ = bootstrap_spearman(
            x, y, n_iter=args.n_iter, seed=args.seed,
        )
        cls = classify(ci_low, ci_high)
        rows.append({
            "subsector": subsector, "parent": parent, "n": n,
            "rho_point": point, "ci_low": ci_low, "ci_high": ci_high,
            "ci_width": ci_high - ci_low,
            "classification": cls,
        })
        print(f"    [{i:>2d}/31]  {subsector[:42]:<42s}  "
              f"ρ={point:+.3f}  95% CI=[{ci_low:+.3f}, {ci_high:+.3f}]  "
              f"width={ci_high - ci_low:.3f}", flush=True)

    results = pd.DataFrame(rows).sort_values(
        "rho_point", ascending=False, na_position="last",
    ).reset_index(drop=True)

    # ── Print ranked table ──
    print("\n" + "=" * 108)
    print("  RANKED SIGNIFICANCE TABLE (sm10 score × fwd_ret_63, sorted by ρ)")
    print("=" * 108)
    print(f"  95% CI from {args.n_iter}-iteration bootstrap at "
          f"(ticker, date) observation level.  Seed={args.seed}.\n")

    print(f"  {'Rank':>4s}  {'Subsector':<42s}  {'Parent':<7s}  "
          f"{'N':>5s}  {'ρ':>7s}  {'CI low':>8s}  {'CI high':>8s}  "
          f"{'Width':>6s}  {'Classification':<22s}")
    print(f"  {'─' * 106}")

    for i, r in results.iterrows():
        if pd.isna(r["rho_point"]):
            print(f"  {i+1:>4d}  {r['subsector'][:42]:<42s}  "
                  f"{r['parent']:<7s}  {r['n']:>5d}  "
                  f"{'—':>7s}  {'—':>8s}  {'—':>8s}  {'—':>6s}  "
                  f"{r['classification']:<22s}")
            continue
        wide_flag = " [WIDE]" if r["ci_width"] > WIDE_CI_FLAG else ""
        print(f"  {i+1:>4d}  {r['subsector'][:42]:<42s}  "
              f"{r['parent']:<7s}  {r['n']:>5d}  "
              f"{r['rho_point']:>+7.3f}  {r['ci_low']:>+8.3f}  {r['ci_high']:>+8.3f}  "
              f"{r['ci_width']:>6.3f}  {r['classification']:<22s}{wide_flag}")

    # ── Highlighted summary ──
    sig_pos = results[results["classification"] == "significant positive"]
    sig_neg = results[results["classification"] == "significant negative"]
    inconclusive = results[results["classification"] == "inconclusive"]
    wide = results[results["ci_width"] > WIDE_CI_FLAG]

    print("\n" + "=" * 108)
    print("  HIGHLIGHTED FINDINGS")
    print("=" * 108)

    print(f"\n  SIGNIFICANT POSITIVE ({len(sig_pos)} subsectors) — CI entirely above 0:")
    if sig_pos.empty:
        print("    (none)")
    else:
        for _, r in sig_pos.sort_values("rho_point", ascending=False).iterrows():
            w_flag = " [WIDE CI]" if r["ci_width"] > WIDE_CI_FLAG else ""
            print(f"    {r['subsector'][:45]:<45s}  [{r['parent']}]  "
                  f"ρ={r['rho_point']:+.3f}  CI=[{r['ci_low']:+.3f}, "
                  f"{r['ci_high']:+.3f}]  n={r['n']}{w_flag}")

    print(f"\n  SIGNIFICANT NEGATIVE ({len(sig_neg)} subsectors) — CI entirely below 0:")
    if sig_neg.empty:
        print("    (none)")
    else:
        for _, r in sig_neg.sort_values("rho_point").iterrows():
            w_flag = " [WIDE CI]" if r["ci_width"] > WIDE_CI_FLAG else ""
            print(f"    {r['subsector'][:45]:<45s}  [{r['parent']}]  "
                  f"ρ={r['rho_point']:+.3f}  CI=[{r['ci_low']:+.3f}, "
                  f"{r['ci_high']:+.3f}]  n={r['n']}{w_flag}")

    print(f"\n  INCONCLUSIVE ({len(inconclusive)} subsectors) — CI crosses 0:")
    for _, r in inconclusive.sort_values("rho_point", ascending=False).iterrows():
        w_flag = " [WIDE CI]" if r["ci_width"] > WIDE_CI_FLAG else ""
        print(f"    {r['subsector'][:45]:<45s}  [{r['parent']}]  "
              f"ρ={r['rho_point']:+.3f}  CI=[{r['ci_low']:+.3f}, "
              f"{r['ci_high']:+.3f}]  n={r['n']}{w_flag}")

    print(f"\n  INSUFFICIENT DATA (CI width > {WIDE_CI_FLAG}) — "
          f"{len(wide)} subsectors, regardless of classification:")
    if wide.empty:
        print("    (none)")
    else:
        for _, r in wide.sort_values("ci_width", ascending=False).iterrows():
            print(f"    {r['subsector'][:45]:<45s}  [{r['parent']}]  "
                  f"width={r['ci_width']:.3f}  ρ={r['rho_point']:+.3f}  n={r['n']}")

    # ── Key questions answered ──
    print("\n" + "=" * 108)
    print("  KEY QUESTIONS")
    print("=" * 108)

    def find(sub_name: str):
        m = results[results["subsector"].str.contains(sub_name, case=False, na=False)]
        return m.iloc[0] if not m.empty else None

    targets = [
        ("Chips — Networking", "prior point estimate +0.264"),
        ("Industrial Robotics", "prior point estimate −0.507"),
        ("Power Semiconductors", "prior point estimate −0.374"),
        ("Nuclear Reactors", "prior point estimate +0.487"),
        ("Gene Editing", "prior point estimate −0.278"),
    ]
    print()
    for name, ctx in targets:
        r = find(name)
        if r is None:
            continue
        print(f"  {r['subsector']:<42s}  ρ={r['rho_point']:+.3f}  "
              f"CI=[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
              f"→ {r['classification']}   ({ctx})")

    n_any_sig = len(sig_pos) + len(sig_neg)
    print(f"\n  Subsectors with any statistically significant signal: "
          f"{n_any_sig} / {len(results)} ({n_any_sig/len(results)*100:.0f}%)")
    print(f"    - significant positive: {len(sig_pos)}")
    print(f"    - significant negative: {len(sig_neg)}")
    print(f"    - inconclusive (CI crosses 0): {len(inconclusive)}")
    print(f"    - wide CI (> {WIDE_CI_FLAG}) — interpret with caution even if significant: "
          f"{len(wide)}")

    # ── Save CSV ──
    csv_path = OUT_DIR / "subsector_significance.csv"
    results.to_csv(csv_path, index=False)
    print(f"\n  → saved {csv_path}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
