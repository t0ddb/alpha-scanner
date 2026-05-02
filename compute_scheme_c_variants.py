from __future__ import annotations

"""
compute_scheme_c_variants.py — Generate score parquets for isolated
single-fix variants of Scheme C, plus a combined "all-fixes" variant.

Each variant changes exactly ONE thing from Scheme C, holds everything
else identical. This isolates which empirical-finding fixes actually
improve portfolio returns vs which ones hurt or are neutral.

Variants:
  base:       Scheme C as currently implemented (control)
  ich_tier:   Ichimoku tiered (3/3 > 2/3) instead of binary at 2/3
  cmf_add:    Re-include CMF (was dropped to 0)
  roc_cap:    Cap ROC scoring at zero above 75% (catastrophic tail)
  dtf_split:  Split DTF into separate 126d and 63d binary triggers
  rs_dip:     Reduce RS scoring in empirical dip zone (96-98)
  all:        Stack all 5 fixes together

Output: backtest_results/scheme_c_variant_<name>.parquet
Each contains date, ticker, score columns matching the backtester's expectations.
"""

import argparse
import pandas as pd


def base_score(row, *, ich_tier=False, cmf_add=False, roc_cap=False,
               dtf_split=False, rs_dip=False) -> float:
    """Compute Scheme C score with optional single-fix overrides."""
    s = 0.0

    # ─── RS (gradient) ─────────────────────────────────────────
    rs_pctl = row.get("rs_percentile", 0) or 0
    if rs_dip:
        # Modified: reduce scoring in empirical dip zone (96-98)
        if rs_pctl >= 98:
            s += 3.0
        elif rs_pctl >= 96:
            s += 1.5  # DIP — reduced from full 3.0
        elif rs_pctl >= 90:
            s += 3.0
        elif rs_pctl >= 80:
            s += 2.4
        elif rs_pctl >= 70:
            s += 1.8
        elif rs_pctl >= 60:
            s += 1.2
        elif rs_pctl >= 50:
            s += 0.6
    else:
        # Original Scheme C
        for thresh, pts in [(90, 3.0), (80, 2.4), (70, 1.8), (60, 1.2), (50, 0.6)]:
            if rs_pctl >= thresh:
                s += pts; break

    # ─── HL (gradient, unchanged) ──────────────────────────────
    hl = row.get("higher_lows_count", 0) or 0
    for c, pts in [(5, 0.5), (4, 0.375), (3, 0.25), (2, 0.125)]:
        if hl >= c:
            s += pts; break

    # ─── Ichimoku ──────────────────────────────────────────────
    if ich_tier:
        # Modified: tiered scoring of composite (0/3, 1/3, 2/3, 3/3)
        ich_score = row.get("ichimoku_score", 0) or 0
        if ich_score >= 3:
            s += 2.5  # full alignment
        elif ich_score >= 2:
            s += 1.0  # partial — much smaller (was 2.0 in Scheme C)
        # 0/3 and 1/3 get 0
    else:
        # Original Scheme C: binary at 2/3 (above_cloud AND cloud_bullish)
        if row.get("ichimoku_fired", 0):
            s += 2.0

    # ─── ROC ───────────────────────────────────────────────────
    if roc_cap:
        # Modified: cap above 75% to zero (catastrophic tail)
        roc_v = row.get("roc_value", 0) or 0
        if 5 < roc_v <= 75:
            s += 1.5
        # Above 75 OR below 5: zero
    else:
        # Original Scheme C: binary at >5%
        if row.get("roc_fired", 0):
            s += 1.5

    # ─── CMF ──────────────────────────────────────────────────
    if cmf_add:
        # Modified: re-include with simple gradient
        cmf_v = row.get("cmf_value", 0) or 0
        if cmf_v >= 0.30:
            s += 0.5
        elif cmf_v >= 0.05:
            s += 0.25
    # else: original Scheme C drops CMF (0 weight)

    # ─── DTF ──────────────────────────────────────────────────
    if dtf_split:
        # Modified: split 126d and 63d into separate binary triggers
        dtf_126 = row.get("rs_126d_pctl", 0) or 0
        dtf_63 = row.get("rs_63d_pctl", 0) or 0
        if dtf_126 >= 80:
            s += 1.25
        if dtf_63 >= 80:
            s += 1.25
    else:
        # Original Scheme C: composite OR trigger
        if row.get("dual_tf_rs_fired", 0):
            s += 2.5

    # ─── ATR (binary, unchanged) ──────────────────────────────
    if row.get("atr_fired", 0):
        s += 0.5

    return round(s, 2)


VARIANTS = {
    "base":      dict(),
    "ich_tier":  dict(ich_tier=True),
    "cmf_add":   dict(cmf_add=True),
    "roc_cap":   dict(roc_cap=True),
    "dtf_split": dict(dtf_split=True),
    "rs_dip":    dict(rs_dip=True),
    "all":       dict(ich_tier=True, cmf_add=True, roc_cap=True,
                      dtf_split=True, rs_dip=True),
}


# ─── Type-grouping variants (separate scorer needed) ───────────────
def type_grouping_score(row, *, variant: str) -> float:
    """Variants that score based on TREND / MOMENTUM / VOL groupings."""
    # First compute base Scheme C component scores per indicator
    rs_pctl = row.get("rs_percentile", 0) or 0
    rs_pts = 0.0
    for thresh, pts in [(90, 3.0), (80, 2.4), (70, 1.8), (60, 1.2), (50, 0.6)]:
        if rs_pctl >= thresh: rs_pts = pts; break
    hl = row.get("higher_lows_count", 0) or 0
    hl_pts = 0.0
    for c, pts in [(5, 0.5), (4, 0.375), (3, 0.25), (2, 0.125)]:
        if hl >= c: hl_pts = pts; break
    ich_pts = 2.0 if row.get("ichimoku_fired", 0) else 0.0
    roc_pts = 1.5 if row.get("roc_fired", 0) else 0.0
    dtf_pts = 2.5 if row.get("dual_tf_rs_fired", 0) else 0.0
    atr_pts = 0.5 if row.get("atr_fired", 0) else 0.0
    cmf_pts = 0.0  # Scheme C drops CMF

    # Indicator firing flags (any contribution = "fired" for type grouping)
    trend_fired = (rs_pts > 0) or (hl_pts > 0) or (ich_pts > 0)
    mom_fired   = (roc_pts > 0) or (dtf_pts > 0)
    vol_fired   = (atr_pts > 0)  # CMF dropped in C

    # ATR-only VOL is asymmetric. For variants that need both:
    # Use atr OR cmf (cmf binary at 0.05 threshold)
    cmf_v = row.get("cmf_value", 0) or 0
    vol_fired_with_cmf = vol_fired or (cmf_v > 0.05)

    if variant == "type_required":
        # Require at least 1 firing from each type. If not, score = 0.
        if not (trend_fired and mom_fired and vol_fired_with_cmf):
            return 0.0
        # Otherwise, sum Scheme C contributions
        return round(rs_pts + hl_pts + ich_pts + roc_pts + dtf_pts + atr_pts, 2)

    if variant == "type_bonus":
        # Base = Scheme C. Bonus if all 3 types fire.
        base = rs_pts + hl_pts + ich_pts + roc_pts + dtf_pts + atr_pts
        n_types = sum([trend_fired, mom_fired, vol_fired_with_cmf])
        if n_types == 3:
            base += 0.5
        elif n_types == 2:
            base += 0.25
        return round(base, 2)

    if variant == "mom_heavy":
        # Boost MOMENTUM (DTF + ROC) weights at expense of others.
        # New: RS=2.5 (was 3.0), ROC=2.0 (was 1.5), DTF=3.5 (was 2.5)
        rs_pts_new = rs_pts * (2.5 / 3.0)
        roc_pts_new = (2.0 if row.get("roc_fired", 0) else 0.0)
        dtf_pts_new = (3.5 if row.get("dual_tf_rs_fired", 0) else 0.0)
        return round(rs_pts_new + hl_pts + ich_pts + roc_pts_new + dtf_pts_new + atr_pts, 2)

    if variant == "trend_only":
        # Only score TREND indicators. Other types contribute 0.
        return round(rs_pts + hl_pts + ich_pts, 2)

    raise ValueError(f"unknown variant: {variant}")


TYPE_VARIANTS = ["type_required", "type_bonus", "mom_heavy", "trend_only"]


def main(input_path: str, output_dir: str):
    df = pd.read_parquet(input_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows")

    for variant, kwargs in VARIANTS.items():
        print(f"\ncomputing variant: {variant}  kwargs={kwargs}")
        scores = df.apply(lambda r: base_score(r, **kwargs), axis=1)
        out = df[["date", "ticker"]].copy()
        out["score"] = scores
        out_path = f"{output_dir}/scheme_c_variant_{variant}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"  max score: {out['score'].max():.2f}")
        print(f"  rows >= 9.0: {(out['score'] >= 9.0).sum():,}")
        print(f"  wrote {out_path}")

    for variant in TYPE_VARIANTS:
        print(f"\ncomputing type-grouping variant: {variant}")
        scores = df.apply(lambda r: type_grouping_score(r, variant=variant), axis=1)
        out = df[["date", "ticker"]].copy()
        out["score"] = scores
        out_path = f"{output_dir}/scheme_c_variant_{variant}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"  max score: {out['score'].max():.2f}")
        print(f"  rows >= 8.5: {(out['score'] >= 8.5).sum():,}")
        print(f"  rows >= 9.0: {(out['score'] >= 9.0).sum():,}")
        print(f"  rows >= 9.5: {(out['score'] >= 9.5).sum():,}")
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--output-dir", default="backtest_results")
    args = ap.parse_args()
    main(args.input, args.output_dir)
