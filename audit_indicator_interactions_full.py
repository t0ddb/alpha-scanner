from __future__ import annotations

"""
audit_indicator_interactions_full.py — Full interaction audit for Scheme I.

Asks: are the per-indicator scoring curves we proposed in
audit_indicator_curves.py confounded by interactions among indicators?

Method:
  1. Fit HistGradientBoostingRegressor predicting fwd_63d_xspy from all
     7 scored indicators' continuous values + Ich composite + DTF components.
  2. Time-series CV: train pre-2025-Q2, test 2025-Q2+ (out-of-sample).
  3. Compute Partial Dependence Plots per indicator → "true" marginal effect.
  4. Compute Friedman's H-statistic for top pairs → interaction strength.
  5. Compute SHAP values + interaction values → cross-validate H-statistic.
  6. For top 5 interactions, render raw 2D win-rate heatmaps.
  7. Per-indicator 3-way comparison: isolated curve vs PDP vs score-conditional.
  8. Refined Scheme I scoring spec.

Output: backtest_results/audit_indicator_interactions_full.txt
"""

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence
from sklearn.metrics import r2_score
import shap

warnings.filterwarnings("ignore")

# ─── Indicator definitions ────────────────────────────────────────
# (column in parquet, friendly name, max weight in current Scheme C)
FEATURES = [
    ("rs_percentile",    "RS_pctl",       3.0),
    ("higher_lows_count","HL_count",      0.5),
    ("ichimoku_score",   "ICH_composite", 2.0),
    ("roc_value",        "ROC_pct",       1.5),
    ("cmf_value",        "CMF",           0.0),
    ("rs_126d_pctl",     "DTF_126d",      2.5),
    ("rs_63d_pctl",      "DTF_63d",       2.5),
    ("atr_percentile",   "ATR_pctl",      0.5),
]
FEAT_COLS  = [c for c, _, _ in FEATURES]
FEAT_NAMES = [n for _, n, _ in FEATURES]


# ─── Helpers ──────────────────────────────────────────────────────
def scheme_c_score_vec(df: pd.DataFrame) -> pd.Series:
    """Compute Scheme C scores vectorized."""
    rs_pts = pd.cut(df["rs_percentile"].fillna(0),
                    bins=[-1, 50, 60, 70, 80, 90, 1000.001],
                    labels=[0.0, 0.6, 1.2, 1.8, 2.4, 3.0]).astype(float)
    hl_pts = pd.cut(df["higher_lows_count"].fillna(0),
                    bins=[-1, 2, 3, 4, 5, 1000],
                    labels=[0.0, 0.125, 0.25, 0.375, 0.5]).astype(float)
    ich_pts = (df["ichimoku_fired"].fillna(0) > 0).astype(float) * 2.0
    roc_pts = (df["roc_fired"].fillna(0) > 0).astype(float) * 1.5
    dtf_pts = (df["dual_tf_rs_fired"].fillna(0) > 0).astype(float) * 2.5
    atr_pts = (df["atr_fired"].fillna(0) > 0).astype(float) * 0.5
    return rs_pts + hl_pts + ich_pts + roc_pts + dtf_pts + atr_pts


def isolated_bucket_curve(df: pd.DataFrame, col: str, target: str,
                          bins: list, labels: list, baseline: float) -> pd.DataFrame:
    """Bucket-by-value win rate (the original isolated analysis)."""
    sub = pd.DataFrame({"v": df[col].values, "fwd": df[target].values}).dropna()
    sub["bucket"] = pd.cut(sub["v"], bins=bins, labels=labels,
                            right=False, include_lowest=True)
    grp = sub.groupby("bucket", observed=True).agg(
        n=("fwd", "count"),
        win=("fwd", lambda s: (s > 0).mean()),
        med=("fwd", "median"),
    ).reset_index()
    grp["delta"] = grp["win"] - baseline
    return grp


def pdp_curve(model, X: pd.DataFrame, feat_idx: int, grid: np.ndarray) -> np.ndarray:
    """Compute 1D partial dependence at given grid points."""
    pd_result = partial_dependence(
        model, X, features=[feat_idx], grid_resolution=len(grid),
        kind="average", percentiles=(0.0, 1.0))
    # Interpolate result onto our grid
    pd_grid = pd_result["grid_values"][0]
    pd_avg  = pd_result["average"][0]
    return np.interp(grid, pd_grid, pd_avg)


def friedman_h(model, X: pd.DataFrame, feat_i: int, feat_j: int,
               n_grid: int = 10) -> float:
    """
    Friedman's H-statistic for pairwise interaction.
    H² = sum((PD_ij - PD_i - PD_j)²) / sum(PD_ij²)
    Range [0, 1]. Higher = stronger interaction.
    """
    try:
        pd_ij = partial_dependence(model, X, features=[(feat_i, feat_j)],
                                    grid_resolution=n_grid, kind="average")
        pd_i = partial_dependence(model, X, features=[feat_i],
                                   grid_resolution=n_grid, kind="average")
        pd_j = partial_dependence(model, X, features=[feat_j],
                                   grid_resolution=n_grid, kind="average")
        pdp_ij = pd_ij["average"][0]                # (n_grid, n_grid)
        pdp_i  = pd_i["average"][0]                  # (n_grid,)
        pdp_j  = pd_j["average"][0]                  # (n_grid,)
        # Center each (subtract mean)
        pdp_ij_c = pdp_ij - pdp_ij.mean()
        pdp_i_c  = pdp_i  - pdp_i.mean()
        pdp_j_c  = pdp_j  - pdp_j.mean()
        # Interaction = full 2D - sum of marginals
        interaction = pdp_ij_c - pdp_i_c[:, None] - pdp_j_c[None, :]
        num = np.sum(interaction ** 2)
        den = np.sum(pdp_ij_c ** 2)
        if den < 1e-12:
            return 0.0
        return float(np.sqrt(num / den))
    except Exception as e:
        return float("nan")


def heatmap_2d(df: pd.DataFrame, col_x: str, col_y: str, target: str,
               x_bins: list, x_labels: list,
               y_bins: list, y_labels: list, baseline: float, out: list):
    """Print a 2D win-rate heatmap with sample counts."""
    sub = df[[col_x, col_y, target]].dropna()
    sub["x_b"] = pd.cut(sub[col_x], bins=x_bins, labels=x_labels,
                        right=False, include_lowest=True)
    sub["y_b"] = pd.cut(sub[col_y], bins=y_bins, labels=y_labels,
                        right=False, include_lowest=True)

    grp = sub.groupby(["x_b", "y_b"], observed=True).agg(
        n=(target, "count"),
        win=(target, lambda s: (s > 0).mean()),
    ).reset_index()

    # Pivot to matrix
    win_mat = grp.pivot(index="x_b", columns="y_b", values="win")
    n_mat   = grp.pivot(index="x_b", columns="y_b", values="n")

    out.append(f"\n  Win-rate heatmap: {col_x} (rows) × {col_y} (cols)")
    out.append(f"  Cell shows win % | n.  Δ vs baseline {baseline:.1%} colored.")
    # Header
    header = f"  {'':<14}"
    for y_lbl in y_labels:
        if y_lbl in win_mat.columns:
            header += f"{str(y_lbl):>14}"
    out.append(header)

    for x_lbl in x_labels:
        if x_lbl not in win_mat.index:
            continue
        row = f"  {str(x_lbl):<14}"
        for y_lbl in y_labels:
            if y_lbl not in win_mat.columns or pd.isna(win_mat.loc[x_lbl, y_lbl]):
                row += f"{'—':>14}"
                continue
            wr = win_mat.loc[x_lbl, y_lbl]
            n  = int(n_mat.loc[x_lbl, y_lbl])
            delta = wr - baseline
            arrow = "↑↑" if delta > 0.05 else "↑" if delta > 0.02 else \
                    "↓↓" if delta < -0.05 else "↓" if delta < -0.02 else " "
            cell = f"{wr:.0%}|{n:>5,}{arrow}"
            row += f"{cell:>14}"
        out.append(row)


# ─── Main ─────────────────────────────────────────────────────────
def main(path: str, target: str, output: str, start_date: str | None = None,
         cv_split_date: str | None = None):
    out: list[str] = []
    out.append("=" * 90)
    out.append("audit_indicator_interactions_full.py — Scheme I interaction audit")
    out.append("=" * 90)

    print("loading parquet...")
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target] + FEAT_COLS).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if start_date:
        sd = pd.Timestamp(start_date)
        before = len(df)
        df = df[df["date"] >= sd].copy()
        out.append(f"\n*** REGIME-FILTERED to date >= {start_date} ***")
        out.append(f"    {before:,} → {len(df):,} rows after filter")

    out.append(f"\nloaded {len(df):,} rows (after dropping NaN)  "
               f"({df['date'].min().date()} → {df['date'].max().date()})")
    baseline = (df[target] > 0).mean()
    out.append(f"target: {target}  |  baseline win rate: {baseline:.1%}")

    # Scheme C score for score-conditional analysis
    df["scheme_c_score"] = scheme_c_score_vec(df)

    # ─── Time-series train/test split (within-regime if filtered) ─
    if cv_split_date:
        cut = pd.Timestamp(cv_split_date)
    else:
        # Default: split data at 70% point in time
        sorted_dates = df["date"].sort_values()
        cut = sorted_dates.iloc[int(0.7 * len(sorted_dates))]
    train_mask = df["date"] < cut
    test_mask  = df["date"] >= cut
    X_train = df.loc[train_mask, FEAT_COLS].copy(); X_train.columns = FEAT_NAMES
    y_train = df.loc[train_mask, target].values
    X_test  = df.loc[test_mask,  FEAT_COLS].copy(); X_test.columns  = FEAT_NAMES
    y_test  = df.loc[test_mask,  target].values
    out.append(f"\ntrain: {train_mask.sum():,} rows ({df.loc[train_mask, 'date'].min().date()} → {df.loc[train_mask, 'date'].max().date()})")
    out.append(f"test:  {test_mask.sum():,} rows ({df.loc[test_mask, 'date'].min().date()} → {df.loc[test_mask, 'date'].max().date()})")

    # ─── Fit gradient-boosted model ──────────────────────────────
    print("fitting HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_iter=300, max_depth=5, learning_rate=0.05,
        l2_regularization=1.0, min_samples_leaf=200,
        random_state=42)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test  = model.predict(X_test)
    r2_train = r2_score(y_train, pred_train)
    r2_test  = r2_score(y_test,  pred_test)

    out.append("\n" + "=" * 90)
    out.append("SECTION 1 — Model fit (sanity check on overfitting)")
    out.append("=" * 90)
    out.append(f"  R² in-sample (train):     {r2_train:>+7.4f}")
    out.append(f"  R² out-of-sample (test):  {r2_test:>+7.4f}")
    out.append(f"  Mean predicted fwd return (test): {pred_test.mean():>+8.4f}")
    out.append(f"  Mean actual    fwd return (test): {y_test.mean():>+8.4f}")
    if r2_test < 0:
        out.append("  WARNING: negative test R² — model overfits or data is unpredictable.")
    elif r2_test < r2_train * 0.3:
        out.append("  WARNING: large gap between in-sample and out-of-sample R² — overfitting.")
    else:
        out.append("  OK: model generalizes acceptably.")

    out.append(f"\n  Model details: HistGradientBoostingRegressor, max_iter=300, "
               f"max_depth=5, lr=0.05, l2=1.0, min_samples_leaf=200")

    # ─── Per-indicator: isolated curve vs PDP vs score-conditional ──
    print("computing PDP curves for each indicator...")
    out.append("\n" + "=" * 90)
    out.append("SECTION 2 — Per-indicator: ISOLATED curve vs PDP vs SCORE-CONDITIONAL")
    out.append("=" * 90)
    out.append("ISOLATED:  win rate by raw bucket (what we proposed for Scheme I)")
    out.append("PDP:       partial dependence — model's learned effect controlling for")
    out.append("           all other indicators. The 'true' marginal effect.")
    out.append("CONDITIONAL: isolated curve restricted to score-band (≥9.0 = entry candidates)")

    # Use full dataset for isolated curves and CONDITIONAL
    # Use X_train for PDP (model was trained on it)
    bucket_specs = {
        "rs_percentile": (
            [0, 50, 60, 70, 80, 88, 90, 92, 94, 96, 98, 100.0001],
            ["<50","50-60","60-70","70-80","80-88","88-90","90-92","92-94","94-96","96-98","98+"]),
        "higher_lows_count": (
            [0, 1, 2, 3, 4, 5, 100],
            ["0","1","2","3","4","5+"]),
        "ichimoku_score": (
            [0, 1, 2, 3, 4],
            ["0/3","1/3","2/3","3/3"]),
        "roc_value": (
            [-1000, 0, 5, 10, 20, 35, 50, 75, 1000],
            ["≤0","0-5","5-10","10-20","20-35","35-50","50-75","75+"]),
        "cmf_value": (
            [-1, -0.10, -0.05, 0, 0.05, 0.10, 0.20, 1],
            ["≤-0.10","-0.10—-0.05","-0.05—0","0-0.05","0.05-0.10","0.10-0.20","0.20+"]),
        "rs_126d_pctl": (
            [0, 50, 65, 75, 85, 95, 100.0001],
            ["<50","50-65","65-75","75-85","85-95","95+"]),
        "rs_63d_pctl": (
            [0, 50, 60, 70, 80, 90, 100.0001],
            ["<50","50-60","60-70","70-80","80-90","90+"]),
        "atr_percentile": (
            [0, 30, 50, 70, 80, 90, 100.0001],
            ["<30","30-50","50-70","70-80","80-90","90+"]),
    }

    sig_mask = df["scheme_c_score"] >= 9.0
    sig_baseline = (df.loc[sig_mask, target] > 0).mean()

    verdicts = []  # accumulate per-indicator verdicts

    for col, friendly, _ in FEATURES:
        out.append(f"\n──── {friendly} ({col}) ────")

        bins, labels = bucket_specs[col]

        # Isolated bucket curve (full dataset)
        iso = isolated_bucket_curve(df, col, target, bins, labels, baseline)

        # Score-conditional curve
        cond = isolated_bucket_curve(df.loc[sig_mask], col, target, bins, labels,
                                      sig_baseline)

        # PDP — sample center of each bucket
        feat_idx = FEAT_COLS.index(col)
        bucket_mids = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            if hi >= 999:
                bucket_mids.append(lo + 5)  # arbitrary tail anchor
            else:
                bucket_mids.append((lo + hi) / 2)
        pdp_at_mids = pdp_curve(model, X_train, feat_idx, np.array(bucket_mids))

        # Render side-by-side table
        out.append(f"  {'bucket':<14} {'iso n':>7} {'iso win%':>8} {'iso Δ':>7}  "
                   f"{'cond n':>7} {'cond win%':>9} {'cond Δ':>8}  "
                   f"{'PDP Δ':>8}")
        out.append(f"  {'─'*14} {'─'*7} {'─'*8} {'─'*7}  "
                   f"{'─'*7} {'─'*9} {'─'*8}  {'─'*7}")
        # Build dicts for lookup
        iso_d = {str(r["bucket"]): r for _, r in iso.iterrows()}
        cond_d = {str(r["bucket"]): r for _, r in cond.iterrows()}

        # For verdict scoring: compare directional sign of Δ across iso vs PDP
        agreements = 0; disagreements = 0
        pdp_mean = pdp_at_mids.mean()  # center PDP
        for i, lbl in enumerate(labels):
            ir = iso_d.get(lbl)
            cr = cond_d.get(lbl)
            pdp_v = pdp_at_mids[i] - pdp_mean
            iso_str = (f"{int(ir['n']):>7,} {ir['win']:>7.1%} {ir['delta']:>+6.1%}"
                       if ir is not None and ir["n"] >= 50 else
                       f"{'—':>7} {'—':>8} {'—':>7}")
            cond_str = (f"{int(cr['n']):>7,} {cr['win']:>8.1%} {cr['delta']:>+7.1%}"
                        if cr is not None and cr["n"] >= 30 else
                        f"{'—':>7} {'—':>9} {'—':>8}")
            out.append(f"  {lbl:<14} {iso_str}  {cond_str}  {pdp_v:>+8.4f}")

            # Track sign agreement
            if ir is not None and ir["n"] >= 50:
                iso_sign = 1 if ir["delta"] > 0.005 else (-1 if ir["delta"] < -0.005 else 0)
                pdp_sign = 1 if pdp_v > 0.001 else (-1 if pdp_v < -0.001 else 0)
                if iso_sign == 0 or pdp_sign == 0:
                    pass  # neither strong, no count
                elif iso_sign == pdp_sign:
                    agreements += 1
                else:
                    disagreements += 1

        # Verdict
        n_compared = agreements + disagreements
        if n_compared == 0:
            verdict = "INSUFFICIENT DATA"
        elif disagreements == 0:
            verdict = f"TRUST ISOLATED ({agreements}/{n_compared} buckets agree on sign)"
        elif agreements >= 2 * disagreements:
            verdict = f"MOSTLY TRUST ({agreements}/{n_compared} agree)"
        elif disagreements > agreements:
            verdict = f"USE PDP ({disagreements}/{n_compared} disagree — interactions distort isolated)"
        else:
            verdict = f"MIXED ({agreements}/{n_compared} agree)"
        out.append(f"  → VERDICT: {verdict}")
        verdicts.append((friendly, verdict))

    # ─── Friedman's H + SHAP interaction strength ────────────────
    print("computing Friedman H statistic for all pairs...")
    out.append("\n" + "=" * 90)
    out.append("SECTION 3 — Pairwise interaction strength (Friedman's H + SHAP)")
    out.append("=" * 90)
    out.append("H-statistic ∈ [0, 1].  Higher = stronger interaction between the pair.")
    out.append("Computed on a 10×10 grid via partial dependence.\n")

    pair_results = []
    n_feat = len(FEAT_COLS)
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            h = friedman_h(model, X_train, i, j, n_grid=10)
            pair_results.append((FEAT_NAMES[i], FEAT_NAMES[j], h))

    pair_results.sort(key=lambda r: -r[2])
    out.append(f"  {'pair':<28} {'H-stat':>7}")
    out.append(f"  {'─'*28} {'─'*7}")
    for a, b, h in pair_results:
        out.append(f"  {a + ' × ' + b:<28} {h:>7.3f}")

    # ─── SHAP interaction values ─────────────────────────────────
    print("computing SHAP values + interactions (this can take ~1 min)...")
    out.append("\n" + "─" * 90)
    out.append("SHAP analysis (sample of 2,000 rows for tractable compute)")
    out.append("─" * 90)
    sample_idx = np.random.RandomState(42).choice(len(X_train), size=min(2000, len(X_train)),
                                                   replace=False)
    X_shap = X_train.iloc[sample_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    shap_inter  = explainer.shap_interaction_values(X_shap)

    # Mean abs SHAP per feature (importance)
    mean_abs = np.abs(shap_values).mean(axis=0)
    out.append(f"\n  SHAP feature importance (mean |SHAP|):")
    for i, name in enumerate(FEAT_NAMES):
        out.append(f"    {name:<14} {mean_abs[i]:.5f}")

    # Mean abs interaction value per pair
    out.append(f"\n  SHAP interaction strength (mean |interaction value|):")
    out.append(f"  {'pair':<28} {'mean|interact|':>15}")
    inter_results = []
    for i in range(n_feat):
        for j in range(i + 1, n_feat):
            mi = float(np.abs(shap_inter[:, i, j]).mean()) * 2  # off-diagonal counted twice
            inter_results.append((FEAT_NAMES[i], FEAT_NAMES[j], mi))
    inter_results.sort(key=lambda r: -r[2])
    for a, b, mi in inter_results[:15]:
        out.append(f"  {a + ' × ' + b:<28} {mi:>15.6f}")

    # ─── Top interactions: raw 2D heatmaps ───────────────────────
    print("computing 2D win-rate heatmaps for top interactions...")
    out.append("\n" + "=" * 90)
    out.append("SECTION 4 — Raw 2D win-rate heatmaps for top 5 interactions")
    out.append("=" * 90)
    out.append("Cross-validation: do the top interactions detected by the model also")
    out.append("show up in the raw data? Each cell = win % | sample count.")

    # Map FEAT_NAME back to (col, bins, labels)
    name_to_col = {fn: c for c, fn, _ in FEATURES}
    top_pairs = pair_results[:5]
    for a, b, h in top_pairs:
        col_x = name_to_col[a]
        col_y = name_to_col[b]
        bx, lx = bucket_specs[col_x]
        by, ly = bucket_specs[col_y]
        out.append(f"\n  ── {a} × {b}  (Friedman H = {h:.3f}) ──")
        heatmap_2d(df, col_x, col_y, target, bx, lx, by, ly, baseline, out)

    # ─── Final verdicts + draft Scheme I spec ────────────────────
    out.append("\n" + "=" * 90)
    out.append("SECTION 5 — Per-indicator verdicts (collected from Section 2)")
    out.append("=" * 90)
    for friendly, v in verdicts:
        out.append(f"  {friendly:<14} → {v}")

    out.append("\n" + "=" * 90)
    out.append("SECTION 6 — Implications for Scheme I")
    out.append("=" * 90)
    out.append("Read this in conjunction with audit_indicator_curves.txt.\n")
    out.append("Decision rules:")
    out.append("  • TRUST ISOLATED → keep proposed Scheme I curve from audit_indicator_curves.txt")
    out.append("  • USE PDP        → adjust scoring to follow PDP shape, not isolated bucket shape")
    out.append("  • Top SHAP interactions with H > 0.10 → consider joint scoring or")
    out.append("                                          interaction terms in Scheme I")
    out.append("  • If R² test < 0  → indicators are too noisy individually; sequence/")
    out.append("                       combination scoring may be more important than indicator values")

    text = "\n".join(out)
    print(text)
    with open(output, "w") as f:
        f.write(text + "\n")
    print(f"\n[wrote {output}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    ap.add_argument("--output", default="backtest_results/audit_indicator_interactions_full.txt")
    ap.add_argument("--start-date", default=None,
                    help="Filter to date >= this (YYYY-MM-DD). Use 2025-05-01 for AI-bull regime.")
    ap.add_argument("--cv-split-date", default=None,
                    help="Date to split train/test. Defaults to 70% point in time.")
    args = ap.parse_args()
    main(args.input, args.target, args.output, args.start_date, args.cv_split_date)
