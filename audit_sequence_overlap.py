from __future__ import annotations

"""
audit_sequence_overlap.py — Measure overlap among the 3 robust positive
findings from audit_sequence_v2_regime.py:

  P1: MOMENTUM→TREND   (first 2 distinct types)
  P5: 1 indicator leading Ichimoku
  P6: dtf→rs as last-2 firers

Plus the robust negative:
  N1: TREND→MOMENTUM (first 2 distinct types)

Three perspectives:

  A. Pairwise Jaccard overlap (how often do they co-fire?)
  B. 2×2×2 cross-tab on the 3 positives — win rate per combination cell
  C. Marginal lift: what does adding finding N to existing M findings buy?
  D. Negative-pattern interaction: do positives override the TREND→MOM penalty?
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


def jaccard(a: pd.Series, b: pd.Series) -> float:
    """Jaccard similarity between two boolean series."""
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union > 0 else 0.0


def main(path: str, target: str, output: str, start_date: str | None = None):
    out: list[str] = []
    out.append("=" * 78)
    out.append("audit_sequence_overlap.py — Overlap among robust sequence findings")
    out.append("=" * 78)

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

    sig = df[df["scheme_c_score"] >= 9.0].copy()
    if start_date:
        sd = pd.Timestamp(start_date)
        before_sig = len(sig)
        sig = sig[sig["date"] >= sd].copy()
        out.append(f"\n*** REGIME-FILTERED to signal rows with date >= {start_date} ***")
        out.append(f"    {before_sig:,} → {len(sig):,} signals after filter")
        out.append(f"    (streaks computed on full history)")
    print(f"signal rows (score >= 9.0): {len(sig):,}")

    print("computing per-row sequences and finding flags...")
    sig["seq"] = sig.apply(lambda r: ordered_firers(r, streak_cols), axis=1)
    sig = sig[sig["seq"].apply(len) > 0].copy()

    sig["first_two_types"] = sig["seq"].apply(
        lambda s: first_two_distinct_types([INDICATOR_TYPE[i] for i in s]))
    sig["last_two"] = sig["seq"].apply(
        lambda s: f"{s[-2]}→{s[-1]}" if len(s) >= 2 else None)

    def n_led_ich(row):
        ich_streak = row["streak_ich"]
        if ich_streak <= 0: return -1  # Ich not firing
        return sum(1 for lbl in LABELS
                   if lbl != "ich" and row[f"streak_{lbl}"] > ich_streak)
    sig["n_led_ich"] = sig.apply(n_led_ich, axis=1)

    # ── Define finding flags ──
    sig["P1_mom_trend"]  = (sig["first_two_types"] == ("MOMENTUM", "TREND"))
    sig["P5_one_led"]    = (sig["n_led_ich"] == 1)
    sig["P6_dtf_rs"]     = (sig["last_two"] == "dtf→rs")
    sig["N1_trend_mom"]  = (sig["first_two_types"] == ("TREND", "MOMENTUM"))

    base = (sig[target].dropna() > 0).mean()
    n_total = len(sig)

    out.append(f"\nsignal universe: n={n_total:,}, baseline win rate {base:.1%}\n")

    out.append("Finding prevalence (% of all signals where flag fires):")
    for col, name in [("P1_mom_trend", "P1: MOMENTUM→TREND"),
                      ("P5_one_led",    "P5: 1 leading Ich"),
                      ("P6_dtf_rs",     "P6: dtf→rs last-2"),
                      ("N1_trend_mom",  "N1: TREND→MOMENTUM")]:
        n = sig[col].sum()
        out.append(f"  {name:<26} n = {n:>5,}  ({100*n/n_total:.1f}% of signals)")

    # ────────────────────────────────────────────────────────────
    # SECTION A — Pairwise Jaccard overlap
    # ────────────────────────────────────────────────────────────
    out.append("\n" + "=" * 78)
    out.append("SECTION A — Pairwise overlap (Jaccard) and conditional probabilities")
    out.append("=" * 78)
    out.append("Jaccard = |A∩B| / |A∪B|. P(B|A) = |A∩B| / |A|.\n")

    pairs = [("P1_mom_trend", "P5_one_led"),
             ("P1_mom_trend", "P6_dtf_rs"),
             ("P5_one_led",    "P6_dtf_rs"),
             ("P1_mom_trend", "N1_trend_mom"),
             ("P5_one_led",    "N1_trend_mom"),
             ("P6_dtf_rs",     "N1_trend_mom")]
    out.append(f"  {'A':<14} {'B':<14} {'|A|':>5} {'|B|':>5} {'|A∩B|':>7} "
               f"{'Jaccard':>8} {'P(B|A)':>7} {'P(A|B)':>7}")
    out.append(f"  {'─'*14} {'─'*14} {'─'*5} {'─'*5} {'─'*7} {'─'*8} {'─'*7} {'─'*7}")
    for a, b in pairs:
        sa = sig[a].sum(); sb = sig[b].sum()
        sab = (sig[a] & sig[b]).sum()
        jac = jaccard(sig[a], sig[b])
        pba = sab / sa if sa > 0 else 0
        pab = sab / sb if sb > 0 else 0
        out.append(f"  {a:<14} {b:<14} {sa:>5,} {sb:>5,} {sab:>7,} "
                   f"{jac:>7.1%} {pba:>6.1%} {pab:>6.1%}")

    # ────────────────────────────────────────────────────────────
    # SECTION B — 2×2×2 cross-tab of the 3 positives
    # ────────────────────────────────────────────────────────────
    out.append("\n" + "=" * 78)
    out.append("SECTION B — 2×2×2 cross-tab of 3 positives + win rate per cell")
    out.append("=" * 78)
    out.append("Cells sorted by # of findings firing (0 to 3).\n")

    out.append(f"  {'P1':>3} {'P5':>3} {'P6':>3}  {'n':>5}   "
               f"{'win %':>6}   {'Δ base':>7}   {'mean':>9}   {'median':>9}")
    out.append(f"  {'─'*3} {'─'*3} {'─'*3}  {'─'*5}   {'─'*6}   {'─'*7}   "
               f"{'─'*9}   {'─'*9}")

    rows = []
    for p1 in (0, 1):
        for p5 in (0, 1):
            for p6 in (0, 1):
                mask = ((sig["P1_mom_trend"] == bool(p1)) &
                        (sig["P5_one_led"]    == bool(p5)) &
                        (sig["P6_dtf_rs"]     == bool(p6)))
                cell = sig[mask]
                vals = cell[target].dropna()
                if len(vals) == 0:
                    rows.append((p1+p5+p6, p1, p5, p6, len(cell), None, None, None, None))
                    continue
                wr = (vals > 0).mean()
                rows.append((p1+p5+p6, p1, p5, p6, len(cell), wr,
                             wr - base, vals.mean(), vals.median()))
    # Sort: # of findings asc, then by win rate desc within group
    rows.sort(key=lambda r: (r[0], -(r[5] or 0)))
    for n_findings, p1, p5, p6, n, wr, delta, mn, md in rows:
        if wr is None:
            out.append(f"  {p1:>3} {p5:>3} {p6:>3}  {n:>5,}   (n=0)")
            continue
        arrow = ("↑↑" if delta > 0.10 else "↑" if delta > 0.05 else
                 "↓↓" if delta < -0.10 else "↓" if delta < -0.05 else "  ")
        out.append(f"  {p1:>3} {p5:>3} {p6:>3}  {n:>5,}   {wr:>5.1%}   "
                   f"{delta:>+6.1%} {arrow}   {mn:>+8.2%}   {md:>+8.2%}")

    # ────────────────────────────────────────────────────────────
    # SECTION C — Marginal lift of each finding given the others
    # ────────────────────────────────────────────────────────────
    out.append("\n" + "=" * 78)
    out.append("SECTION C — Marginal lift of each finding (controlled comparison)")
    out.append("=" * 78)
    out.append("For each finding F, compare win rate (F=1 ∧ others-fixed) vs")
    out.append("(F=0 ∧ others-fixed). If marginal lift is small after controlling")
    out.append("for other findings, F adds little independent information.\n")

    def marginal(F: str, other_pos: list[str], out: list):
        out.append(f"\n  Marginal lift of {F}:")
        out.append(f"    {'condition (others)':<40} {'F=0 n':>5} {'F=0 wr':>7}  "
                   f"{'F=1 n':>5} {'F=1 wr':>7}  {'Δ':>6}")
        out.append(f"    {'─'*40} {'─'*5} {'─'*7}  {'─'*5} {'─'*7}  {'─'*6}")
        # Iterate over all combinations of the OTHER positives' state (2^k)
        k = len(other_pos)
        for combo in range(2**k):
            mask = pd.Series(True, index=sig.index)
            descr = []
            for i, ofld in enumerate(other_pos):
                v = bool((combo >> i) & 1)
                mask &= (sig[ofld] == v)
                descr.append(f"{ofld.split('_')[0]}={int(v)}")
            sub = sig[mask]
            f0 = sub[~sub[F]]
            f1 = sub[sub[F]]
            f0_vals = f0[target].dropna()
            f1_vals = f1[target].dropna()
            if len(f0_vals) < 20 or len(f1_vals) < 20:
                out.append(f"    {' & '.join(descr):<40} {len(f0_vals):>5} (n<20 in one cell)")
                continue
            wr0 = (f0_vals > 0).mean()
            wr1 = (f1_vals > 0).mean()
            out.append(f"    {' & '.join(descr):<40} {len(f0_vals):>5,} {wr0:>6.1%}  "
                       f"{len(f1_vals):>5,} {wr1:>6.1%}  {wr1-wr0:>+5.1%}")

    marginal("P1_mom_trend", ["P5_one_led", "P6_dtf_rs"], out)
    marginal("P5_one_led",    ["P1_mom_trend", "P6_dtf_rs"], out)
    marginal("P6_dtf_rs",     ["P1_mom_trend", "P5_one_led"], out)

    # ────────────────────────────────────────────────────────────
    # SECTION D — Negative-pattern interaction
    # ────────────────────────────────────────────────────────────
    out.append("\n" + "=" * 78)
    out.append("SECTION D — Does TREND→MOM penalty stand independently of positives?")
    out.append("=" * 78)
    out.append("Mechanically: TREND→MOM and MOMENTUM→TREND (P1) are mutually")
    out.append("exclusive (different first-2-types). But a TREND→MOM signal can")
    out.append("still have P5 or P6 fire. Check whether P5/P6 'rescue' a")
    out.append("TREND→MOM signal.\n")

    out.append(f"  {'condition':<40} {'n':>5}   {'win %':>6}   {'Δ base':>7}")
    out.append(f"  {'─'*40} {'─'*5}   {'─'*6}   {'─'*7}")
    for label, mask in [
        ("ALL signals (baseline)", pd.Series(True, index=sig.index)),
        ("N1 fires (TREND→MOM)", sig["N1_trend_mom"]),
        ("N1 ∧ NOT P5 ∧ NOT P6 (pure penalty)",
            sig["N1_trend_mom"] & ~sig["P5_one_led"] & ~sig["P6_dtf_rs"]),
        ("N1 ∧ P5 (rescued by 1-led-Ich?)",
            sig["N1_trend_mom"] & sig["P5_one_led"]),
        ("N1 ∧ P6 (rescued by dtf→rs?)",
            sig["N1_trend_mom"] & sig["P6_dtf_rs"]),
        ("N1 ∧ (P5 ∨ P6) (rescued by either?)",
            sig["N1_trend_mom"] & (sig["P5_one_led"] | sig["P6_dtf_rs"])),
        ("NOT N1 (everything else)", ~sig["N1_trend_mom"]),
    ]:
        sub = sig[mask]
        vals = sub[target].dropna()
        if len(vals) < 20:
            out.append(f"  {label:<40} {len(vals):>5}  (n<20)")
            continue
        wr = (vals > 0).mean()
        out.append(f"  {label:<40} {len(vals):>5,}   {wr:>5.1%}   {wr-base:>+6.1%}")

    text = "\n".join(out)
    print(text)
    with open(output, "w") as f:
        f.write(text + "\n")
    print(f"\n[wrote {output}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    ap.add_argument("--output", default="backtest_results/audit_sequence_overlap.txt")
    ap.add_argument("--start-date", default=None,
                    help="Filter SIGNAL ROWS to date >= this (YYYY-MM-DD). "
                         "Streaks still computed on full history.")
    args = ap.parse_args()
    main(args.input, args.target, args.output, args.start_date)
