from __future__ import annotations

"""
audit_sequence_v2.py — Re-run sequence analysis with scoring-aligned fire thresholds.

Old definitions in audit_dataset.parquet:
  rs_fired = rs_percentile >= 50  (very loose vs full-weight scoring at >=90)
  higher_lows_fired = count >= 2  (lowest gradient tier; full weight at >=5)

This script defines NEW fire columns:
  rs_fired_v2 = (rs_percentile >= 90)
  hl_fired_v2 = (higher_lows_count >= 4)
  others unchanged from parquet.

Then runs 9 bucketing strategies on the per-(date,ticker) sequence of
indicator firings. See plan file for details on each bucket.

Output: backtest_results/audit_sequence_v2.txt
"""

import argparse
from itertools import combinations
import numpy as np
import pandas as pd

# ─── Indicator definitions ─────────────────────────────────────────
# (label, fire_column_in_parquet) — but rs/hl get overridden below.
FIRE_DEFS = [
    ("rs", "rs_fired"),
    ("ich", "ichimoku_fired"),
    ("hl",  "higher_lows_fired"),
    ("cmf", "cmf_fired"),
    ("roc", "roc_fired"),
    ("atr", "atr_fired"),
    ("dtf", "dual_tf_rs_fired"),
]
LABELS = [lbl for lbl, _ in FIRE_DEFS]

INDICATOR_TYPE = {
    "rs":  "TREND",
    "hl":  "TREND",
    "ich": "TREND",
    "roc": "MOMENTUM",
    "dtf": "MOMENTUM",
    "cmf": "VOL",
    "atr": "VOL",
}


# ─── Helpers ───────────────────────────────────────────────────────
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
    """Return labels of all currently-firing indicators, sorted longest streak
    first (ties broken alphabetically for determinism)."""
    firing = [(lbl, row[c]) for lbl, c in streak_cols.items() if row[c] > 0]
    firing.sort(key=lambda x: (-x[1], x[0]))
    return [lbl for lbl, _ in firing]


def fmt_row(label: str, n: int, vals: pd.Series, base: float, label_w: int = 28) -> str:
    if len(vals) == 0:
        return f"  {label:<{label_w}} {n:>5}  (no fwd-return data)"
    wr = (vals > 0).mean()
    delta = wr - base
    arrow = ("↑↑" if delta > 0.10 else
             "↑"  if delta > 0.05 else
             "↓↓" if delta < -0.10 else
             "↓"  if delta < -0.05 else "  ")
    return (f"  {label:<{label_w}} {n:>5,} {vals.mean():>+8.2%} "
            f"{vals.median():>+8.2%} {wr:>6.1%} {delta:>+7.1%} {arrow}")


def fmt_header(label_w: int = 28) -> str:
    return (f"  {'bucket':<{label_w}} {'n':>5} {'mean':>9} {'median':>9} "
            f"{'win %':>7} {'Δ base':>7}")


def section(title: str) -> str:
    return "\n" + "=" * 78 + f"\n{title}\n" + "=" * 78


# ─── Bucketings ────────────────────────────────────────────────────
def bucket_A_first_last_pairs(sig: pd.DataFrame, target: str, base: float, out: list):
    out.append(section("SECTION 2 — Bucket A: First/Last firer + top pairs (sanity baseline)"))
    out.append("Direct apples-to-apples vs prior audit_indicator_sequence_v2.txt.\n")

    # First-firer
    out.append("  FIRST-FIRER")
    out.append(fmt_header())
    for label in LABELS:
        sub = sig[sig["first_firer"] == label]
        n = len(sub)
        if n == 0: continue
        vals = sub[target].dropna()
        out.append(fmt_row(label, n, vals, base))

    # Last-firer
    out.append("\n  LAST-FIRER")
    out.append(fmt_header())
    for label in LABELS:
        sub = sig[sig["last_firer"] == label]
        n = len(sub)
        if n < 30: continue
        vals = sub[target].dropna()
        out.append(fmt_row(label, n, vals, base))

    # Top (first, last) pairs
    out.append("\n  (FIRST, LAST) PAIRS  (n >= 50, sorted by win rate)")
    out.append(fmt_header(label_w=28))
    pair_rows = []
    pair_grp = sig.groupby(["first_firer", "last_firer"]).size().reset_index(name="n")
    for _, r in pair_grp[pair_grp["n"] >= 50].iterrows():
        sub = sig[(sig["first_firer"] == r["first_firer"]) & (sig["last_firer"] == r["last_firer"])]
        vals = sub[target].dropna()
        if len(vals) == 0: continue
        wr = (vals > 0).mean()
        pair_rows.append((wr, f"{r['first_firer']}→{r['last_firer']}", len(sub), vals))
    for wr, label, n, vals in sorted(pair_rows, key=lambda x: -x[0]):
        out.append(fmt_row(label, n, vals, base))


def bucket_B_first_by_type(sig: pd.DataFrame, target: str, base: float, out: list):
    out.append(section("SECTION 3 — Bucket B: First-firer by TYPE (TREND/MOM/VOL)"))
    out.append("Reduces 7 indicators to 3 categories. Larger N per cell.\n")
    sig = sig.copy()
    sig["first_type"] = sig["first_firer"].map(INDICATOR_TYPE)
    out.append(fmt_header())
    for t in ["TREND", "MOMENTUM", "VOL"]:
        sub = sig[sig["first_type"] == t]
        if len(sub) < 30: continue
        vals = sub[target].dropna()
        out.append(fmt_row(t, len(sub), vals, base))


def bucket_C_first_K(sig: pd.DataFrame, target: str, base: float, out: list):
    out.append(section("SECTION 4 — Bucket C: First-K ordered patterns"))

    # C1: first-2 ordered
    out.append("\n  C1 — Ordered FIRST-2 firers  (n >= 30, sorted by win rate)")
    out.append(fmt_header(label_w=24))
    rows = []
    sub = sig[sig["seq"].apply(lambda s: len(s) >= 2)].copy()
    sub["pat"] = sub["seq"].apply(lambda s: f"{s[0]}→{s[1]}")
    grp = sub.groupby("pat").size().reset_index(name="n")
    for _, r in grp[grp["n"] >= 30].iterrows():
        cell = sub[sub["pat"] == r["pat"]]
        vals = cell[target].dropna()
        if len(vals) == 0: continue
        wr = (vals > 0).mean()
        rows.append((wr, r["pat"], len(cell), vals))
    for wr, label, n, vals in sorted(rows, key=lambda x: -x[0]):
        out.append(fmt_row(label, n, vals, base, label_w=24))

    # C2: first-3 ordered (n>=50)
    out.append("\n  C2 — Ordered FIRST-3 firers  (n >= 50, sorted by win rate)")
    out.append(fmt_header(label_w=24))
    rows = []
    sub = sig[sig["seq"].apply(lambda s: len(s) >= 3)].copy()
    sub["pat"] = sub["seq"].apply(lambda s: f"{s[0]}→{s[1]}→{s[2]}")
    grp = sub.groupby("pat").size().reset_index(name="n")
    for _, r in grp[grp["n"] >= 50].iterrows():
        cell = sub[sub["pat"] == r["pat"]]
        vals = cell[target].dropna()
        if len(vals) == 0: continue
        wr = (vals > 0).mean()
        rows.append((wr, r["pat"], len(cell), vals))
    for wr, label, n, vals in sorted(rows, key=lambda x: -x[0]):
        out.append(fmt_row(label, n, vals, base, label_w=24))


def bucket_D_last_K(sig: pd.DataFrame, target: str, base: float, out: list):
    out.append(section("SECTION 5 — Bucket D: Last-K ordered patterns (mirror of C)"))

    # D1: last-2 ordered
    out.append("\n  D1 — Ordered LAST-2 firers  (n >= 30, sorted by win rate)")
    out.append(fmt_header(label_w=24))
    rows = []
    sub = sig[sig["seq"].apply(lambda s: len(s) >= 2)].copy()
    sub["pat"] = sub["seq"].apply(lambda s: f"{s[-2]}→{s[-1]}")
    grp = sub.groupby("pat").size().reset_index(name="n")
    for _, r in grp[grp["n"] >= 30].iterrows():
        cell = sub[sub["pat"] == r["pat"]]
        vals = cell[target].dropna()
        if len(vals) == 0: continue
        wr = (vals > 0).mean()
        rows.append((wr, r["pat"], len(cell), vals))
    for wr, label, n, vals in sorted(rows, key=lambda x: -x[0]):
        out.append(fmt_row(label, n, vals, base, label_w=24))

    # D2: last-3 ordered (n>=50)
    out.append("\n  D2 — Ordered LAST-3 firers  (n >= 50, sorted by win rate)")
    out.append(fmt_header(label_w=24))
    rows = []
    sub = sig[sig["seq"].apply(lambda s: len(s) >= 3)].copy()
    sub["pat"] = sub["seq"].apply(lambda s: f"{s[-3]}→{s[-2]}→{s[-1]}")
    grp = sub.groupby("pat").size().reset_index(name="n")
    for _, r in grp[grp["n"] >= 50].iterrows():
        cell = sub[sub["pat"] == r["pat"]]
        vals = cell[target].dropna()
        if len(vals) == 0: continue
        wr = (vals > 0).mean()
        rows.append((wr, r["pat"], len(cell), vals))
    for wr, label, n, vals in sorted(rows, key=lambda x: -x[0]):
        out.append(fmt_row(label, n, vals, base, label_w=24))


def bucket_E_presence(sig: pd.DataFrame, target: str, base: float, out: list):
    out.append(section("SECTION 6 — Bucket E: Presence-only subsets (no ordering)"))
    out.append("Which indicators are firing AT ALL? Tests if order matters vs presence alone.")
    out.append("Subsets are encoded as sorted tuples for grouping.\n")

    sig = sig.copy()
    sig["presence_set"] = sig["seq"].apply(lambda s: tuple(sorted(s)))
    grp = sig.groupby("presence_set").size().reset_index(name="n")
    grp = grp[grp["n"] >= 50].sort_values("n", ascending=False)

    out.append("  PRESENCE SUBSETS  (n >= 50, sorted by win rate)")
    out.append(fmt_header(label_w=42))
    rows = []
    for _, r in grp.iterrows():
        cell = sig[sig["presence_set"] == r["presence_set"]]
        vals = cell[target].dropna()
        if len(vals) == 0: continue
        wr = (vals > 0).mean()
        label = "{" + ",".join(r["presence_set"]) + "}"
        rows.append((wr, label, len(cell), vals))
    for wr, label, n, vals in sorted(rows, key=lambda x: -x[0]):
        out.append(fmt_row(label, n, vals, base, label_w=42))


def bucket_F_streak_spread(sig: pd.DataFrame, target: str, base: float, out: list):
    out.append(section("SECTION 7 — Bucket F: Streak-length SPREAD (clustering)"))
    out.append("std(streak lengths) across firing indicators per setup.")
    out.append("Tight cluster = coordinated fresh setup; wide spread = mixed-age setup.\n")

    # Compute per-row spread
    sig = sig.copy()
    streak_cols = [f"streak_{lbl}" for lbl in LABELS]

    def spread(row):
        vals = [row[c] for c in streak_cols if row[c] > 0]
        if len(vals) < 2: return None
        return float(np.std(vals))

    sig["spread"] = sig.apply(spread, axis=1)
    sig = sig.dropna(subset=["spread"])

    # Quintile bucket
    sig["spread_q"] = pd.qcut(sig["spread"], 5,
                              labels=["Q1 tight", "Q2", "Q3", "Q4", "Q5 spread"],
                              duplicates="drop")
    out.append(fmt_header(label_w=18))
    for q in ["Q1 tight", "Q2", "Q3", "Q4", "Q5 spread"]:
        sub = sig[sig["spread_q"] == q]
        if len(sub) < 30: continue
        vals = sub[target].dropna()
        med_spread = sub["spread"].median()
        label = f"{q} (med std={med_spread:.0f}d)"
        out.append(fmt_row(label, len(sub), vals, base, label_w=18))


def bucket_G_anchor_leadlag(sig: pd.DataFrame, target: str, base: float, out: list):
    out.append(section("SECTION 8 — Bucket G: Anchor lead-lag (Ichimoku as anchor)"))
    out.append("For each setup, count indicators whose streak > Ichimoku's streak.")
    out.append("Higher count = more indicators LED Ichimoku (Ich confirmed late).\n")

    sig = sig.copy()
    streak_cols = [(lbl, f"streak_{lbl}") for lbl in LABELS]

    def leads_count(row):
        ich_streak = row["streak_ich"]
        if ich_streak <= 0: return None
        # Count other firing indicators with longer streak than Ich
        return sum(1 for lbl, c in streak_cols
                   if lbl != "ich" and row[c] > ich_streak)

    sig["n_led_ich"] = sig.apply(leads_count, axis=1)
    sig = sig.dropna(subset=["n_led_ich"])

    out.append(fmt_header(label_w=24))
    for n_leads in range(0, 7):
        sub = sig[sig["n_led_ich"] == n_leads]
        if len(sub) < 30: continue
        vals = sub[target].dropna()
        label = f"{n_leads} led Ich"
        out.append(fmt_row(label, len(sub), vals, base, label_w=24))


def bucket_H_named_patterns(sig: pd.DataFrame, target: str, base: float, out: list):
    out.append(section("SECTION 9 — Bucket H: Five named story patterns"))

    sig = sig.copy()
    streak_cols = {lbl: f"streak_{lbl}" for lbl in LABELS}

    def is_classic(row):
        # TREND first, MOMENTUM in middle, VOL last
        seq = row["seq"]
        if len(seq) < 3: return False
        types = [INDICATOR_TYPE[i] for i in seq]
        # First 1/3 = TREND, last 1/3 = VOL, middle = MOMENTUM
        n = len(types)
        first_third = types[:max(1, n // 3)]
        last_third  = types[-max(1, n // 3):]
        return (all(t == "TREND" for t in first_third)
                and any(t == "VOL" for t in last_third))

    def is_vol_led(row):
        # VOL (CMF or ATR) in first 2 firers
        seq = row["seq"]
        if len(seq) < 2: return False
        return INDICATOR_TYPE.get(seq[0]) == "VOL" or INDICATOR_TYPE.get(seq[1]) == "VOL"

    def is_mom_first(row):
        # MOMENTUM (ROC or DTF) is THE single first-firer
        seq = row["seq"]
        if len(seq) < 1: return False
        return INDICATOR_TYPE.get(seq[0]) == "MOMENTUM"

    def is_all_fresh(row):
        firing = [row[c] for lbl, c in streak_cols.items() if row[c] > 0]
        if not firing: return False
        return max(firing) <= 10

    def is_stale_rs_fresh_confirm(row):
        rs_streak = row["streak_rs"]
        if rs_streak < 30: return False
        # >=2 OTHER indicators fired in last 5 days (streak <= 5 AND > 0)
        n_fresh = sum(1 for lbl, c in streak_cols.items()
                      if lbl != "rs" and 0 < row[c] <= 5)
        return n_fresh >= 2

    patterns = [
        ("CLASSIC (TREND→MOM→VOL)",      is_classic),
        ("VOL-LED (VOL in first 2)",     is_vol_led),
        ("MOMENTUM-FIRST (MOM is #1)",   is_mom_first),
        ("ALL-FRESH (max streak ≤ 10d)", is_all_fresh),
        ("STALE-RS + FRESH-CONFIRM",     is_stale_rs_fresh_confirm),
    ]

    out.append(fmt_header(label_w=32))
    for label, fn in patterns:
        mask = sig.apply(fn, axis=1)
        sub = sig[mask]
        if len(sub) < 30:
            out.append(f"  {label:<32} {len(sub):>5}  (n<30, skipped)")
            continue
        vals = sub[target].dropna()
        out.append(fmt_row(label, len(sub), vals, base, label_w=32))


def bucket_I_categorical_ordering(sig: pd.DataFrame, target: str, base: float, out: list):
    out.append(section("SECTION 10 — Bucket I: Categorical sequence ordering (TREND/MOM/VOL)"))
    out.append("First-TYPE → second-TYPE patterns (collapses indicators within type).\n")

    sig = sig.copy()
    sig["type_seq"] = sig["seq"].apply(
        lambda s: [INDICATOR_TYPE[i] for i in s])

    # First-type → second-type (collapsing repeats: take first DISTINCT type)
    def first_two_distinct_types(types):
        seen = []
        for t in types:
            if not seen or seen[-1] != t:
                seen.append(t)
            if len(seen) == 2:
                break
        return tuple(seen)

    sig["first_two_types"] = sig["type_seq"].apply(first_two_distinct_types)

    # All 6 orderings of 3 types based on FULL distinct ordering
    def full_distinct_ordering(types):
        seen = []
        for t in types:
            if t not in seen:
                seen.append(t)
        return tuple(seen)

    sig["type_order"] = sig["type_seq"].apply(full_distinct_ordering)

    out.append("  I1 — First-2 distinct TYPES")
    out.append(fmt_header(label_w=24))
    rows = []
    grp = sig.groupby("first_two_types").size().reset_index(name="n")
    for _, r in grp.iterrows():
        if r["n"] < 30: continue
        cell = sig[sig["first_two_types"] == r["first_two_types"]]
        vals = cell[target].dropna()
        if len(vals) == 0: continue
        wr = (vals > 0).mean()
        label = "→".join(r["first_two_types"])
        rows.append((wr, label, len(cell), vals))
    for wr, label, n, vals in sorted(rows, key=lambda x: -x[0]):
        out.append(fmt_row(label, n, vals, base, label_w=24))

    out.append("\n  I2 — Full TYPE ordering (all 3 types in order)")
    out.append(fmt_header(label_w=28))
    rows = []
    grp = sig.groupby("type_order").size().reset_index(name="n")
    for _, r in grp.iterrows():
        if r["n"] < 30: continue
        cell = sig[sig["type_order"] == r["type_order"]]
        vals = cell[target].dropna()
        if len(vals) == 0: continue
        wr = (vals > 0).mean()
        label = "→".join(r["type_order"]) if len(r["type_order"]) == 3 else "→".join(r["type_order"]) + " (incomplete)"
        rows.append((wr, label, len(cell), vals))
    for wr, label, n, vals in sorted(rows, key=lambda x: -x[0]):
        out.append(fmt_row(label, n, vals, base, label_w=28))


# ─── Main ──────────────────────────────────────────────────────────
def main(path: str, target: str, output: str, start_date: str | None = None):
    out: list[str] = []

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ─ Section 1: header + threshold-change summary ─
    out.append("=" * 78)
    out.append("audit_sequence_v2.py — Sequence analysis with scoring-aligned thresholds")
    out.append("=" * 78)
    out.append(f"input: {path}")
    out.append(f"target: {target}")
    out.append(f"loaded: {len(df):,} rows, {df['date'].min().date()} → {df['date'].max().date()}")
    if start_date:
        out.append(f"\n*** REGIME-FILTERED to signal rows with date >= {start_date} ***")
        out.append(f"    (streaks computed on full history; only analysis rows filtered)")

    # Compute Scheme C scores (unchanged)
    df["scheme_c_score"] = df.apply(scheme_c_score, axis=1)

    # NEW fire definitions (override rs and hl)
    df["rs_fired_v2"] = (df["rs_percentile"].fillna(0) >= 90).astype(int)
    df["hl_fired_v2"] = (df["higher_lows_count"].fillna(0) >= 4).astype(int)

    # Streak columns — use v2 fire columns for rs and hl
    fire_col_for = {
        "rs":  "rs_fired_v2",
        "ich": "ichimoku_fired",
        "hl":  "hl_fired_v2",
        "cmf": "cmf_fired",
        "roc": "roc_fired",
        "atr": "atr_fired",
        "dtf": "dual_tf_rs_fired",
    }
    for lbl in LABELS:
        df[f"streak_{lbl}"] = df.groupby("ticker")[fire_col_for[lbl]].transform(streak_above)
    streak_cols = {lbl: f"streak_{lbl}" for lbl in LABELS}

    # ─ Threshold change summary ─
    out.append("\n" + "─" * 78)
    out.append("SECTION 1 — Threshold change & signal-universe impact")
    out.append("─" * 78)
    n_rs_old = (df["rs_fired"] == 1).sum()
    n_rs_new = (df["rs_fired_v2"] == 1).sum()
    n_hl_old = (df["higher_lows_fired"] == 1).sum()
    n_hl_new = (df["hl_fired_v2"] == 1).sum()
    out.append(f"  rs_fired:  OLD (>=50 pctl): {n_rs_old:>7,}   NEW (>=90 pctl): {n_rs_new:>7,}   "
               f"({100*n_rs_new/n_rs_old:.1f}% retained)")
    out.append(f"  hl_fired:  OLD (count>=2):  {n_hl_old:>7,}   NEW (count>=4):  {n_hl_new:>7,}   "
               f"({100*n_hl_new/n_hl_old:.1f}% retained)")

    sig = df[df["scheme_c_score"] >= 9.0].copy()
    out.append(f"\n  signal rows (score >= 9.0): {len(sig):,}")

    # Apply regime filter to SIGNAL rows only (streaks already computed on full history)
    if start_date:
        sd = pd.Timestamp(start_date)
        before_sig = len(sig)
        sig = sig[sig["date"] >= sd].copy()
        out.append(f"  AFTER regime filter (date >= {start_date}): {len(sig):,} signals "
                   f"({before_sig:,} → {len(sig):,})")

    # Build sequence features
    print("computing sequences (this is the slow part)...")
    sig["seq"] = sig.apply(lambda r: ordered_firers(r, streak_cols), axis=1)
    sig["first_firer"] = sig["seq"].apply(lambda s: s[0] if s else None)
    sig["last_firer"]  = sig["seq"].apply(lambda s: s[-1] if s else None)

    n_with_firing = sig["seq"].apply(len).gt(0).sum()
    out.append(f"  signals with at least 1 currently-firing indicator: {n_with_firing:,}")
    out.append(f"  median # indicators firing per signal: {int(sig['seq'].apply(len).median())}")

    # Distribution of firing-count
    out.append("\n  Firing-count distribution:")
    counts = sig["seq"].apply(len).value_counts().sort_index()
    for k, v in counts.items():
        out.append(f"    {k} indicators firing: {v:>5,}  ({100*v/len(sig):.1f}%)")

    # Sanity-check: distribution of first-firer under new thresholds
    out.append("\n  First-firer distribution under NEW thresholds:")
    ff_dist = sig["first_firer"].value_counts()
    for lbl, n in ff_dist.items():
        out.append(f"    {lbl}: {n:>5,}  ({100*n/len(sig):.1f}%)")
    none_count = sig["first_firer"].isna().sum()
    if none_count:
        out.append(f"    (none firing): {none_count:>5,}  ({100*none_count/len(sig):.1f}%)")

    base = (sig[target].dropna() > 0).mean()
    out.append(f"\n  Overall baseline win rate (score >= 9.0): {base:.1%}")
    out.append(f"  Δ legend: ↑↑ > +10pp, ↑ > +5pp, ↓ < -5pp, ↓↓ < -10pp\n")

    # Drop signals with no firing indicators for downstream buckets
    sig_active = sig[sig["seq"].apply(len) > 0].copy()

    # ─ Run all buckets ─
    print("running bucket A (sanity baseline)...")
    bucket_A_first_last_pairs(sig_active, target, base, out)
    print("running bucket B (first-firer by TYPE)...")
    bucket_B_first_by_type(sig_active, target, base, out)
    print("running bucket C (first-K patterns)...")
    bucket_C_first_K(sig_active, target, base, out)
    print("running bucket D (last-K patterns)...")
    bucket_D_last_K(sig_active, target, base, out)
    print("running bucket E (presence subsets)...")
    bucket_E_presence(sig_active, target, base, out)
    print("running bucket F (streak spread)...")
    bucket_F_streak_spread(sig_active, target, base, out)
    print("running bucket G (anchor lead-lag)...")
    bucket_G_anchor_leadlag(sig_active, target, base, out)
    print("running bucket H (named patterns)...")
    bucket_H_named_patterns(sig_active, target, base, out)
    print("running bucket I (categorical ordering)...")
    bucket_I_categorical_ordering(sig_active, target, base, out)

    text = "\n".join(out)
    print(text)
    with open(output, "w") as f:
        f.write(text + "\n")
    print(f"\n[wrote {output}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    ap.add_argument("--output", default="backtest_results/audit_sequence_v2.txt")
    ap.add_argument("--start-date", default=None,
                    help="Filter SIGNAL ROWS to date >= this (YYYY-MM-DD). "
                         "Streaks still computed on full history.")
    args = ap.parse_args()
    main(args.input, args.target, args.output, args.start_date)
