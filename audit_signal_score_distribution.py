from __future__ import annotations

"""
audit_signal_score_distribution.py — For each strong/exceptional/catastrophic
sequence finding, compute the base-score distribution of matching signals.

Answers four critical questions for Scheme I+ design:

  1. For "exceptional" patterns (e.g., DTF-first), are the matching signals
     ALREADY at high base score? If yes, fast-tracking is moot. If no, we'd
     genuinely miss signals without fast-track.

  2. For "catastrophic" patterns (e.g., ROC→DTF), are matching signals
     ALREADY low-score? If yes, the threshold already filters them. If no,
     we have legitimately bad high-score signals that need an explicit filter.

  3. For each pattern, what's the distribution across thresholds (8.5, 9.0,
     9.5)? Quantifies the "would I miss this without override" question.

  4. Sample size and indicator-mean distribution per pattern — to detect
     whether the pattern's apparent lift is just a proxy for one specific
     indicator value (interaction confound).

Output: backtest_results/audit_signal_score_distribution.txt

Filtered to 2025+ regime by default.
"""

import argparse
import numpy as np
import pandas as pd

FIRE_DEFS = [
    ("rs", "rs_fired_v2"), ("ich", "ichimoku_fired"), ("hl", "hl_fired_v2"),
    ("cmf", "cmf_fired"), ("roc", "roc_fired"), ("atr", "atr_fired"),
    ("dtf", "dual_tf_rs_fired"),
]
LABELS = [lbl for lbl, _ in FIRE_DEFS]

INDICATOR_TYPE = {
    "rs": "TREND", "hl": "TREND", "ich": "TREND",
    "roc": "MOMENTUM", "dtf": "MOMENTUM",
    "cmf": "VOL", "atr": "VOL",
}


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


def ordered_firers(row, streak_cols) -> list[str]:
    firing = [(lbl, row[c]) for lbl, c in streak_cols.items() if row[c] > 0]
    firing.sort(key=lambda x: (-x[1], x[0]))
    return [lbl for lbl, _ in firing]


def first_two_distinct_types(types):
    seen = []
    for t in types:
        if not seen or seen[-1] != t:
            seen.append(t)
        if len(seen) == 2: break
    return tuple(seen)


def score_distribution(sub: pd.DataFrame, label: str, mask, all_sigs: pd.DataFrame,
                       target: str, baseline: float, out: list):
    """For a pattern (defined by mask on `sub`), compute and print:
    - n
    - win rate + delta vs baseline
    - score distribution (min/p25/median/p75/max/mean)
    - % at score >= 9.0, >= 9.5, >= 9.875 (max possible non-trivial)
    - mean indicator values (compared to overall)
    - "fast-track lift" — # of pattern signals that DON'T pass current threshold
    """
    cell = sub[mask]
    n = len(cell)
    if n < 20:
        out.append(f"\n  ── {label}  (n={n}, too small) ──")
        return

    vals = cell[target].dropna()
    wr = (vals > 0).mean() if len(vals) > 0 else float("nan")
    delta = wr - baseline if not np.isnan(wr) else float("nan")

    scores = cell["scheme_c_score"]
    pct_ge_85 = (scores >= 8.5).mean()
    pct_ge_90 = (scores >= 9.0).mean()
    pct_ge_95 = (scores >= 9.5).mean()
    n_below_thresh = (scores < 9.0).sum()

    out.append(f"\n  ── {label} ──")
    out.append(f"    n={n:>5,}  win {wr:>5.1%}  Δbase {delta:>+5.1%}")
    out.append(f"    base score: min={scores.min():.2f}  p25={scores.quantile(0.25):.2f}  "
               f"median={scores.median():.2f}  p75={scores.quantile(0.75):.2f}  "
               f"max={scores.max():.2f}  mean={scores.mean():.2f}")
    out.append(f"    qualified at thresh: ≥8.5={pct_ge_85:>5.1%}  ≥9.0={pct_ge_90:>5.1%}  ≥9.5={pct_ge_95:>5.1%}")
    out.append(f"    SUB-THRESHOLD signals (score<9.0): {n_below_thresh:,}  ({100*n_below_thresh/n:.1f}%)"
               f"  ← would need fast-track to enter")

    # Win rate of sub-threshold portion
    sub_thresh = cell[cell["scheme_c_score"] < 9.0]
    if len(sub_thresh) >= 20:
        st_vals = sub_thresh[target].dropna()
        if len(st_vals) > 0:
            st_wr = (st_vals > 0).mean()
            out.append(f"    sub-thresh subset win rate: {st_wr:.1%}  (vs baseline {baseline:.1%}, Δ {st_wr-baseline:+.1%})")

    # Mean indicator values vs overall
    out.append(f"    mean indicator values (pattern vs overall):")
    for col, name in [("rs_percentile", "RS_pctl"),
                      ("higher_lows_count", "HL_count"),
                      ("ichimoku_score", "ICH_3"),
                      ("roc_value", "ROC%"),
                      ("cmf_value", "CMF"),
                      ("rs_126d_pctl", "DTF_126d"),
                      ("rs_63d_pctl", "DTF_63d"),
                      ("atr_percentile", "ATR_pctl")]:
        if col in cell.columns:
            pat_mean = cell[col].mean()
            overall_mean = all_sigs[col].mean()
            diff = pat_mean - overall_mean
            out.append(f"      {name:<10} {pat_mean:>+8.2f}  (overall {overall_mean:>+7.2f},  diff {diff:>+7.2f})")


def main(path: str, target: str, output: str, start_date: str = "2025-05-01"):
    out: list[str] = []
    out.append("=" * 90)
    out.append("audit_signal_score_distribution.py — Score distribution per sequence pattern")
    out.append("=" * 90)

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print("computing scores + streaks...")
    df["scheme_c_score"] = df.apply(scheme_c_score, axis=1)
    df["rs_fired_v2"] = (df["rs_percentile"].fillna(0) >= 90).astype(int)
    df["hl_fired_v2"] = (df["higher_lows_count"].fillna(0) >= 4).astype(int)

    fire_col_for = {lbl: col for lbl, col in FIRE_DEFS}
    for lbl in LABELS:
        df[f"streak_{lbl}"] = df.groupby("ticker")[fire_col_for[lbl]].transform(streak_above)
    streak_cols = {lbl: f"streak_{lbl}" for lbl in LABELS}

    # Apply regime filter to signal rows
    sd = pd.Timestamp(start_date)
    sig = df[df["scheme_c_score"] >= 9.0].copy()
    sig_sub_thresh_only = df[(df["scheme_c_score"] >= 7.5) & (df["scheme_c_score"] < 9.0)].copy()
    all_high_score = df[df["scheme_c_score"] >= 7.5].copy()  # Includes sub-threshold candidates
    all_high_score = all_high_score[all_high_score["date"] >= sd].copy()

    out.append(f"\n*** REGIME-FILTERED to date >= {start_date} ***")
    out.append(f"loaded {len(df):,} rows; analyzing signals at score >= 7.5 in current regime")
    out.append(f"  signals at score >= 7.5: {len(all_high_score):,}")
    out.append(f"  signals at score >= 9.0: {(all_high_score['scheme_c_score'] >= 9.0).sum():,}")

    print("computing sequences...")
    all_high_score["seq"] = all_high_score.apply(lambda r: ordered_firers(r, streak_cols), axis=1)
    all_high_score = all_high_score[all_high_score["seq"].apply(len) > 0].copy()

    all_high_score["first_firer"] = all_high_score["seq"].apply(lambda s: s[0])
    all_high_score["last_firer"]  = all_high_score["seq"].apply(lambda s: s[-1])
    all_high_score["first_two_types"] = all_high_score["seq"].apply(
        lambda s: first_two_distinct_types([INDICATOR_TYPE[i] for i in s]))
    all_high_score["last_two"] = all_high_score["seq"].apply(
        lambda s: f"{s[-2]}→{s[-1]}" if len(s) >= 2 else None)
    all_high_score["first_two_pair"] = all_high_score["seq"].apply(
        lambda s: f"{s[0]}→{s[1]}" if len(s) >= 2 else None)

    def n_led_ich(row):
        ich_streak = row["streak_ich"]
        if ich_streak <= 0: return -1
        return sum(1 for lbl in LABELS
                   if lbl != "ich" and row[f"streak_{lbl}"] > ich_streak)
    all_high_score["n_led_ich"] = all_high_score.apply(n_led_ich, axis=1)

    # Composite flags from prior overlap analysis
    all_high_score["P1_mom_trend"] = (all_high_score["first_two_types"] == ("MOMENTUM", "TREND"))
    all_high_score["P5_one_led"]   = all_high_score["n_led_ich"].isin([1, 2])
    all_high_score["P6_dtf_rs"]    = (all_high_score["last_two"] == "dtf→rs")

    # Baseline win rate within score>=9.0 universe
    sigs_only = all_high_score[all_high_score["scheme_c_score"] >= 9.0]
    baseline = (sigs_only[target].dropna() > 0).mean()
    out.append(f"\n  baseline win rate (score >= 9.0, current regime): {baseline:.1%}\n")

    # ─── EXCEPTIONAL PATTERNS ───────────────────────────────────
    out.append("\n" + "=" * 90)
    out.append("EXCEPTIONAL PATTERNS  (Δ > +15pp from sequence audit)")
    out.append("=" * 90)
    out.append("Question: are these signals ALREADY at high base score (fast-track moot)")
    out.append("or are they spread across base scores (fast-track adds value)?")

    # Restrict examination to score >= 7.5 to capture potential fast-track candidates
    universe = all_high_score

    score_distribution(universe, "DTF-first (firer with longest streak = dtf)",
                       universe["first_firer"] == "dtf",
                       sigs_only, target, baseline, out)

    score_distribution(universe, "All-3 firing (P1 ∧ P5 ∧ P6)",
                       universe["P1_mom_trend"] & universe["P5_one_led"] & universe["P6_dtf_rs"],
                       sigs_only, target, baseline, out)

    score_distribution(universe, "6 indicators leading Ich",
                       universe["n_led_ich"] == 6,
                       sigs_only, target, baseline, out)

    score_distribution(universe, "rs→atr last-2 firers",
                       universe["last_two"] == "rs→atr",
                       sigs_only, target, baseline, out)

    score_distribution(universe, "dtf→ich first-2 firers",
                       universe["first_two_pair"] == "dtf→ich",
                       sigs_only, target, baseline, out)

    # ─── STRONG PATTERNS ────────────────────────────────────────
    out.append("\n" + "=" * 90)
    out.append("STRONG PATTERNS  (Δ +5 to +15pp)")
    out.append("=" * 90)
    out.append("Should these get larger additive bonus, or just modest scoring lift?\n")

    score_distribution(universe, "MOMENTUM→TREND first-2 type",
                       universe["P1_mom_trend"],
                       sigs_only, target, baseline, out)

    score_distribution(universe, "1-or-2 indicators leading Ich",
                       universe["P5_one_led"],
                       sigs_only, target, baseline, out)

    score_distribution(universe, "dtf→rs last-2 firers",
                       universe["P6_dtf_rs"],
                       sigs_only, target, baseline, out)

    score_distribution(universe, "ATR-first",
                       universe["first_firer"] == "atr",
                       sigs_only, target, baseline, out)

    score_distribution(universe, "roc→ich first-2",
                       universe["first_two_pair"] == "roc→ich",
                       sigs_only, target, baseline, out)

    score_distribution(universe, "roc→atr first-2",
                       universe["first_two_pair"] == "roc→atr",
                       sigs_only, target, baseline, out)

    score_distribution(universe, "ich→cmf first-2",
                       universe["first_two_pair"] == "ich→cmf",
                       sigs_only, target, baseline, out)

    # ─── CATASTROPHIC PATTERNS ──────────────────────────────────
    out.append("\n" + "=" * 90)
    out.append("CATASTROPHIC PATTERNS  (Δ < -15pp)")
    out.append("=" * 90)
    out.append("Are these already low-score (threshold filters them) or do high-score")
    out.append("signals fall into these traps (filter would add value)?\n")

    score_distribution(universe, "ROC→DTF first-2",
                       universe["first_two_pair"] == "roc→dtf",
                       sigs_only, target, baseline, out)

    score_distribution(universe, "ICH→DTF first-2",
                       universe["first_two_pair"] == "ich→dtf",
                       sigs_only, target, baseline, out)

    score_distribution(universe, "3 indicators leading Ich",
                       universe["n_led_ich"] == 3,
                       sigs_only, target, baseline, out)

    score_distribution(universe, "ICH→RS first-2",
                       universe["first_two_pair"] == "ich→rs",
                       sigs_only, target, baseline, out)

    # ─── MODERATE NEGATIVE ──────────────────────────────────────
    out.append("\n" + "=" * 90)
    out.append("MODERATE NEGATIVE PATTERNS  (Δ -5 to -15pp)")
    out.append("=" * 90)

    score_distribution(universe, "TREND→MOMENTUM first-2 type (N1)",
                       universe["first_two_types"] == ("TREND", "MOMENTUM"),
                       sigs_only, target, baseline, out)

    # ─── INTERPRETATION SUMMARY ─────────────────────────────────
    out.append("\n" + "=" * 90)
    out.append("INTERPRETATION GUIDE")
    out.append("=" * 90)
    out.append("""
For each pattern above:

  • If most signals (≥80%) already have score ≥ 9.0:
      → Pattern adds little independent info; fast-track unnecessary
      → Use as TIE-BREAKER among qualifying candidates

  • If many signals (≥30%) have 7.5 ≤ score < 9.0 AND the sub-threshold
    subset has high win rate (above baseline):
      → Genuine fast-track value
      → Recommend lowering threshold for this pattern

  • Mean indicator values diverge significantly from overall:
      → Pattern is partially a proxy for high indicator value
      → Apparent lift may be overstated; bonus magnitude should be reduced

  • Catastrophic pattern with most signals at score ≥ 9.5:
      → Hard filter is critical — these are high-base-score traps
""")

    text = "\n".join(out)
    print(text)
    with open(output, "w") as f:
        f.write(text + "\n")
    print(f"\n[wrote {output}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    ap.add_argument("--output", default="backtest_results/audit_signal_score_distribution.txt")
    ap.add_argument("--start-date", default="2025-05-01")
    args = ap.parse_args()
    main(args.input, args.target, args.output, args.start_date)
