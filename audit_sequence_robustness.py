from __future__ import annotations

"""
audit_sequence_robustness.py — Robustness checks on the first-firer finding.

Three orthogonal questions:
  1. TIME STABILITY: does the first-firer pattern hold across sub-periods?
     Split data into year-halves; recompute the first-firer table per window.
  2. SCORE-CONDITIONAL: at higher score thresholds (9.5, 9.7, 9.9), does
     first-firer still predict, or does score subsume it?
  3. BOOTSTRAP CIs: how confident are we in the headline pair numbers?
     Compute 95% bootstrap confidence intervals on win rate for top pairs.
"""

import argparse
import numpy as np
import pandas as pd

FIRE_COLS = [
    ("rs", "rs_fired"), ("ichimoku", "ichimoku_fired"),
    ("higher_lows", "higher_lows_fired"), ("cmf", "cmf_fired"),
    ("roc", "roc_fired"), ("atr", "atr_fired"),
    ("dual_tf_rs", "dual_tf_rs_fired"),
]


def streak_above(s: pd.Series) -> pd.Series:
    is_on = s.astype(int)
    groups = (is_on != is_on.shift()).cumsum()
    streak = is_on.groupby(groups).cumsum()
    return streak.where(is_on == 1, 0)


def scheme_c_score(row) -> float:
    s = 0.0
    rs = row.get("rs_percentile", 0) or 0
    for thresh, pts in [(90, 3.0), (80, 2.4), (70, 1.8), (60, 1.2), (50, 0.6)]:
        if rs >= thresh:
            s += pts; break
    hl = row.get("higher_lows_count", 0) or 0
    for c, pts in [(5, 0.5), (4, 0.375), (3, 0.25), (2, 0.125)]:
        if hl >= c:
            s += pts; break
    if row.get("ichimoku_fired", 0): s += 2.0
    if row.get("roc_fired", 0):      s += 1.5
    if row.get("dual_tf_rs_fired", 0): s += 2.5
    if row.get("atr_fired", 0):      s += 0.5
    return s


def first_firer(row, streak_cols):
    candidates = [(lbl, row[c]) for lbl, c in streak_cols.items() if row[c] > 0]
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[1])[0]


def last_firer(row, streak_cols):
    candidates = [(lbl, row[c]) for lbl, c in streak_cols.items() if row[c] > 0]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[1])[0]


def first_firer_table(sig: pd.DataFrame, target: str, indent: str = "  ") -> None:
    print(f"{indent}{'indicator':<14} {'n':>6} {'mean':>9} {'median':>9} {'win %':>7}")
    print(f"{indent}{'─'*14} {'─'*6} {'─'*9} {'─'*9} {'─'*7}")
    for label in [lbl for lbl, _ in FIRE_COLS]:
        sub = sig[sig["first_firer"] == label]
        if len(sub) == 0:
            continue
        vals = sub[target].dropna()
        if len(vals) < 10:
            print(f"{indent}{label:<14} {len(vals):>6,}  (n<10, skipped)")
            continue
        print(f"{indent}{label:<14} {len(vals):>6,} "
              f"{vals.mean():>+8.2%} {vals.median():>+8.2%} "
              f"{(vals > 0).mean():>6.1%}")


def bootstrap_winrate_ci(values: np.ndarray, n_iter: int = 5000,
                         alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Return (lo, mid, hi) 95% bootstrap CI on win rate."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan, np.nan)
    wins = (values > 0).astype(int)
    samples = rng.choice(wins, size=(n_iter, n), replace=True)
    rates = samples.mean(axis=1)
    return (np.quantile(rates, alpha / 2),
            wins.mean(),
            np.quantile(rates, 1 - alpha / 2))


def main(path: str, target: str):
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"loaded {len(df):,} rows  (date range {df['date'].min().date()} → {df['date'].max().date()})")

    print("computing scores + streaks...")
    df["scheme_c_score"] = df.apply(scheme_c_score, axis=1)
    for label, col in FIRE_COLS:
        df[f"streak_{label}"] = df.groupby("ticker")[col].transform(streak_above)
    streak_cols = {label: f"streak_{label}" for label, _ in FIRE_COLS}

    sig_all = df[df["scheme_c_score"] >= 9.0].copy()
    print(f"signal rows (score >= 9.0): {len(sig_all):,}\n")

    print("computing first/last firers...")
    sig_all["first_firer"] = sig_all.apply(lambda r: first_firer(r, streak_cols), axis=1)
    sig_all["last_firer"]  = sig_all.apply(lambda r: last_firer(r, streak_cols), axis=1)

    # ──────────────────────────────────────────────────────────
    # PART 1: TIME-PERIOD STABILITY
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 1: TIME-PERIOD STABILITY OF FIRST-FIRER FINDING")
    print("=" * 70)
    print("If RS-first underperformance is regime-dependent (e.g., only")
    print("post-2024), the structural story weakens. We split the data")
    print("into 4 sub-periods and re-tabulate.\n")

    # Build 4 quartiles by calendar
    cuts = pd.to_datetime(["2024-01-01", "2024-09-01", "2025-05-01"])
    sig_all["window"] = pd.cut(
        sig_all["date"],
        bins=[pd.Timestamp("2000-01-01"), *cuts, pd.Timestamp("2030-01-01")],
        labels=["pre-2024", "2024H1", "2024H2-2025Q1", "2025Q2+"],
    )

    for w in ["pre-2024", "2024H1", "2024H2-2025Q1", "2025Q2+"]:
        sub = sig_all[sig_all["window"] == w]
        if len(sub) < 50:
            print(f"\n[{w}]  n={len(sub)} — too small, skipped")
            continue
        date_range = f"{sub['date'].min().date()} → {sub['date'].max().date()}"
        print(f"\n[{w}]  n={len(sub):,}  ({date_range})")
        first_firer_table(sub, target, indent="  ")

    # ──────────────────────────────────────────────────────────
    # PART 2: SCORE-CONDITIONAL ROBUSTNESS
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 2: SCORE-CONDITIONAL FIRST-FIRER (does score subsume sequence?)")
    print("=" * 70)
    print("If at score >= 9.9, all sequences win, then sequence is just")
    print("redundant with score. If RS-first still loses at 9.9, the")
    print("finding is structural.\n")

    for thr in [9.0, 9.5, 9.7, 9.9]:
        sub = sig_all[sig_all["scheme_c_score"] >= thr]
        date_range = f"{sub['date'].min().date()} → {sub['date'].max().date()}" if len(sub) else "—"
        print(f"\n[score >= {thr}]  n={len(sub):,}  ({date_range})")
        first_firer_table(sub, target, indent="  ")

    # ──────────────────────────────────────────────────────────
    # PART 3: BOOTSTRAP CIs ON KILLER PAIRS
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 3: BOOTSTRAP 95% CIs ON TOP/BOTTOM PAIRS")
    print("=" * 70)
    print("Are the headline pair win-rates statistically separable from")
    print("the baseline (~50%)? Bootstrap n=5000 resamples per pair.\n")

    overall_winrate = (sig_all[target].dropna() > 0).mean()
    print(f"  Overall baseline win rate (all signals score>=9.0): {overall_winrate:.1%}\n")

    # Recompute pair counts and pull top/bottom by n>=50
    pair_grp = sig_all.groupby(["first_firer", "last_firer"]).size().reset_index(name="n")
    pair_grp = pair_grp[pair_grp["n"] >= 50].sort_values("n", ascending=False)

    print(f"  {'first':<12} {'last':<12} {'n':>5} {'win %':>7} {'95% CI':>20} {'sep?':>6}")
    print(f"  {'─'*12} {'─'*12} {'─'*5} {'─'*7} {'─'*20} {'─'*6}")

    pair_results = []
    for _, r in pair_grp.iterrows():
        sub = sig_all[(sig_all["first_firer"] == r["first_firer"]) &
                      (sig_all["last_firer"] == r["last_firer"])]
        vals = sub[target].dropna().to_numpy()
        lo, mid, hi = bootstrap_winrate_ci(vals)
        # Separable from baseline (~51%): does CI exclude 0.51?
        sep = ""
        if hi < 0.51: sep = "↓↓"
        elif lo > 0.51: sep = "↑↑"
        elif hi < 0.55 and lo < 0.51: sep = "↓"
        elif lo > 0.51 and lo < 0.55: sep = "↑"
        pair_results.append((r["first_firer"], r["last_firer"], r["n"], mid, lo, hi, sep))

    # Sort by win-rate descending so we see best/worst at the ends
    pair_results.sort(key=lambda x: -x[3])
    for first, last, n, mid, lo, hi, sep in pair_results:
        ci_str = f"[{lo:.1%} – {hi:.1%}]"
        print(f"  {first:<12} {last:<12} {n:>5} {mid:>6.1%} {ci_str:>20}  {sep:>4}")

    # ──────────────────────────────────────────────────────────
    # PART 4: FIRST-FIRER × SCORE CROSSTAB (does score correlate with sequence?)
    # ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 4: SCORE DISTRIBUTION BY FIRST-FIRER")
    print("=" * 70)
    print("If certain first-firers are concentrated at certain score levels,")
    print("the first-firer effect could be a score effect in disguise.\n")

    print(f"  {'first-firer':<14} {'n':>6} {'mean score':>11} {'>=9.5 %':>9} {'>=9.9 %':>9}")
    print(f"  {'─'*14} {'─'*6} {'─'*11} {'─'*9} {'─'*9}")
    for label in [lbl for lbl, _ in FIRE_COLS]:
        sub = sig_all[sig_all["first_firer"] == label]
        if len(sub) == 0:
            continue
        n = len(sub)
        mean_s = sub["scheme_c_score"].mean()
        pct_95 = (sub["scheme_c_score"] >= 9.5).mean()
        pct_99 = (sub["scheme_c_score"] >= 9.9).mean()
        print(f"  {label:<14} {n:>6,} {mean_s:>10.2f} "
              f"{pct_95:>8.1%} {pct_99:>8.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    args = ap.parse_args()
    main(args.input, args.target)
