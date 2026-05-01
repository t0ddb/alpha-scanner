from __future__ import annotations

"""
audit_sequence_v2_regime.py — Regime stability check for the top 7 findings
from audit_sequence_v2.py.

For each finding, splits the data into 4 regime windows (pre-2024, 2024H1,
2024H2-2025Q1, 2025Q2+) and reports win rate + delta vs PER-REGIME baseline.

Robustness criterion: same-sign Δ in at least 3 of 4 regimes.

Findings tested:
  1. MOMENTUM-first → TREND-second (I1)        — expect positive
  2. RS-first (Bucket A first-firer)            — expect negative
  3. ATR-first (Bucket A first-firer)           — expect negative
  4. Streak-spread Q5 (wide) vs Q1 (tight)      — expect Q5 positive
  5. Anchor lead-lag = 1 indicator leading Ich  — expect positive
  6. dtf→rs last-2 firer pattern                — expect positive
  7. cmf→hl first/last firer pair               — expect positive
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

WINDOWS = [
    ("pre-2024",       pd.Timestamp("2000-01-01"), pd.Timestamp("2024-01-01")),
    ("2024H1",         pd.Timestamp("2024-01-01"), pd.Timestamp("2024-09-01")),
    ("2024H2-2025Q1",  pd.Timestamp("2024-09-01"), pd.Timestamp("2025-05-01")),
    ("2025Q2+",        pd.Timestamp("2025-05-01"), pd.Timestamp("2030-01-01")),
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


def ordered_firers(row, streak_cols) -> list[str]:
    firing = [(lbl, row[c]) for lbl, c in streak_cols.items() if row[c] > 0]
    firing.sort(key=lambda x: (-x[1], x[0]))
    return [lbl for lbl, _ in firing]


def first_two_distinct_types(types):
    seen = []
    for t in types:
        if not seen or seen[-1] != t:
            seen.append(t)
        if len(seen) == 2:
            break
    return tuple(seen)


# ─── Regime test helpers ───────────────────────────────────────────
def fmt_arrow(delta: float) -> str:
    if delta > 0.10: return "↑↑"
    if delta > 0.05: return "↑"
    if delta < -0.10: return "↓↓"
    if delta < -0.05: return "↓"
    return "  "


def test_finding(sig: pd.DataFrame, target: str, name: str, mask_fn,
                 expected_sign: str, out: list, indent: str = "    "):
    """Test a finding (defined by a row mask) across all 4 regimes.
    expected_sign: '+' or '-' for what we expect."""
    out.append(f"\n  ── Finding: {name} (expected sign: {expected_sign}) ──")
    out.append(f"{indent}{'window':<16} {'base n':>6} {'base':>6} {'cell n':>6} "
               f"{'cell win':>8} {'Δ':>7}")
    out.append(f"{indent}{'─'*16} {'─'*6} {'─'*6} {'─'*6} {'─'*8} {'─'*7}")

    same_sign_count = 0
    sign_data = []
    for wname, lo, hi in WINDOWS:
        win_sub = sig[(sig["date"] >= lo) & (sig["date"] < hi)]
        if len(win_sub) < 50:
            out.append(f"{indent}{wname:<16} {len(win_sub):>6} (window too small)")
            continue
        base = (win_sub[target].dropna() > 0).mean()

        cell = win_sub[mask_fn(win_sub)]
        cell_vals = cell[target].dropna()
        if len(cell_vals) < 20:
            out.append(f"{indent}{wname:<16} {len(win_sub):>6} {base:>5.1%} "
                       f"{len(cell_vals):>6} (cell n<20)")
            continue
        cell_wr = (cell_vals > 0).mean()
        delta = cell_wr - base
        arrow = fmt_arrow(delta)
        out.append(f"{indent}{wname:<16} {len(win_sub):>6,} {base:>5.1%} "
                   f"{len(cell_vals):>6,} {cell_wr:>7.1%} "
                   f"{delta:>+6.1%} {arrow}")

        if expected_sign == "+" and delta > 0.02:
            same_sign_count += 1
        elif expected_sign == "-" and delta < -0.02:
            same_sign_count += 1
        sign_data.append((wname, delta))

    # Verdict
    n_evaluated = sum(1 for _, d in sign_data if d is not None)
    out.append(f"{indent}→ Same-sign in {same_sign_count}/{n_evaluated} regimes "
               f"(robustness threshold: 3/4)")
    if n_evaluated >= 3:
        if same_sign_count >= 3:
            out.append(f"{indent}  VERDICT: ROBUST")
        elif same_sign_count == n_evaluated and n_evaluated < 4:
            out.append(f"{indent}  VERDICT: ROBUST IN AVAILABLE REGIMES (some windows too small)")
        elif same_sign_count >= 2:
            out.append(f"{indent}  VERDICT: PARTIALLY ROBUST")
        else:
            out.append(f"{indent}  VERDICT: NOT ROBUST")


def main(path: str, target: str, output: str):
    out: list[str] = []
    out.append("=" * 78)
    out.append("audit_sequence_v2_regime.py — regime stability of top sequence findings")
    out.append("=" * 78)

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=[target]).copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    out.append(f"\nloaded {len(df):,} rows  ({df['date'].min().date()} → {df['date'].max().date()})")

    # Scores + new fire columns + streaks
    print("computing scores + corrected fire columns + streaks...")
    df["scheme_c_score"] = df.apply(scheme_c_score, axis=1)
    df["rs_fired_v2"] = (df["rs_percentile"].fillna(0) >= 90).astype(int)
    df["hl_fired_v2"] = (df["higher_lows_count"].fillna(0) >= 4).astype(int)

    fire_col_for = {lbl: col for lbl, col in FIRE_DEFS}
    for lbl in LABELS:
        df[f"streak_{lbl}"] = df.groupby("ticker")[fire_col_for[lbl]].transform(streak_above)
    streak_cols = {lbl: f"streak_{lbl}" for lbl in LABELS}

    sig = df[df["scheme_c_score"] >= 9.0].copy()
    out.append(f"signal rows (score >= 9.0): {len(sig):,}\n")

    print("computing per-row sequences...")
    sig["seq"] = sig.apply(lambda r: ordered_firers(r, streak_cols), axis=1)
    sig = sig[sig["seq"].apply(len) > 0].copy()
    sig["first_firer"] = sig["seq"].apply(lambda s: s[0])
    sig["last_firer"]  = sig["seq"].apply(lambda s: s[-1])

    # Pre-compute per-row features used by tests
    sig["first_two_types"] = sig["seq"].apply(
        lambda s: first_two_distinct_types([INDICATOR_TYPE[i] for i in s]))

    # Spread quintile (computed across whole signal universe to match v2 buckets)
    def spread_fn(row):
        vals = [row[f"streak_{lbl}"] for lbl in LABELS if row[f"streak_{lbl}"] > 0]
        return float(np.std(vals)) if len(vals) >= 2 else None
    sig["spread"] = sig.apply(spread_fn, axis=1)
    spread_valid = sig.dropna(subset=["spread"])
    spread_q_labels = ["Q1 tight", "Q2", "Q3", "Q4", "Q5 spread"]
    sig.loc[spread_valid.index, "spread_q"] = pd.qcut(
        spread_valid["spread"], 5, labels=spread_q_labels, duplicates="drop")

    # Anchor lead-lag (Ichimoku)
    def n_led_ich(row):
        ich_streak = row["streak_ich"]
        if ich_streak <= 0: return None
        return sum(1 for lbl in LABELS
                   if lbl != "ich" and row[f"streak_{lbl}"] > ich_streak)
    sig["n_led_ich"] = sig.apply(n_led_ich, axis=1)

    # Last-2 ordered pattern
    sig["last_two"] = sig["seq"].apply(
        lambda s: f"{s[-2]}→{s[-1]}" if len(s) >= 2 else None)

    out.append("─" * 78)
    out.append("SECTION 1 — Per-regime baseline win rates (sanity)")
    out.append("─" * 78)
    for wname, lo, hi in WINDOWS:
        sub = sig[(sig["date"] >= lo) & (sig["date"] < hi)]
        if len(sub) < 50:
            out.append(f"  {wname:<16}  n={len(sub):>5}  (too small to test)")
            continue
        base = (sub[target].dropna() > 0).mean()
        out.append(f"  {wname:<16}  n={len(sub):>5,}  baseline win rate = {base:>5.1%}")

    out.append("\n" + "=" * 78)
    out.append("SECTION 2 — Regime stability of top 7 findings")
    out.append("=" * 78)

    # ─ Finding 1: MOMENTUM-first → TREND-second
    test_finding(sig, target,
                 "MOMENTUM→TREND (first 2 distinct types)",
                 lambda d: d["first_two_types"] == ("MOMENTUM", "TREND"),
                 expected_sign="+", out=out)

    # ─ Finding 2: RS-first
    test_finding(sig, target,
                 "RS is first-firer",
                 lambda d: d["first_firer"] == "rs",
                 expected_sign="-", out=out)

    # ─ Finding 3: ATR-first
    test_finding(sig, target,
                 "ATR is first-firer",
                 lambda d: d["first_firer"] == "atr",
                 expected_sign="-", out=out)

    # ─ Finding 4a: Streak-spread Q5 (wide) — positive
    test_finding(sig, target,
                 "Streak-spread Q5 (wide spread)",
                 lambda d: d["spread_q"] == "Q5 spread",
                 expected_sign="+", out=out)

    # ─ Finding 4b: Streak-spread Q1 (tight) — negative (or at least not positive)
    test_finding(sig, target,
                 "Streak-spread Q1 (tight cluster)",
                 lambda d: d["spread_q"] == "Q1 tight",
                 expected_sign="-", out=out)

    # ─ Finding 5: 1 indicator leading Ichimoku
    test_finding(sig, target,
                 "Exactly 1 indicator leading Ichimoku",
                 lambda d: d["n_led_ich"] == 1,
                 expected_sign="+", out=out)

    # ─ Finding 6: dtf→rs last-2 firers
    test_finding(sig, target,
                 "dtf→rs as last-2 firers",
                 lambda d: d["last_two"] == "dtf→rs",
                 expected_sign="+", out=out)

    # ─ Finding 7: cmf→hl as (first, last) pair
    test_finding(sig, target,
                 "cmf-first AND hl-last (pair)",
                 lambda d: (d["first_firer"] == "cmf") & (d["last_firer"] == "hl"),
                 expected_sign="+", out=out)

    # Bonus: contrast finding 1 with the inverse
    out.append("\n" + "=" * 78)
    out.append("SECTION 3 — Contrast tests (the OPPOSITE of finding 1)")
    out.append("=" * 78)
    test_finding(sig, target,
                 "TREND→MOMENTUM (inverse of finding 1)",
                 lambda d: d["first_two_types"] == ("TREND", "MOMENTUM"),
                 expected_sign="-", out=out)
    test_finding(sig, target,
                 "VOL→MOMENTUM (worst pattern from pooled)",
                 lambda d: d["first_two_types"] == ("VOL", "MOMENTUM"),
                 expected_sign="-", out=out)

    text = "\n".join(out)
    print(text)
    with open(output, "w") as f:
        f.write(text + "\n")
    print(f"\n[wrote {output}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="backtest_results/audit_dataset.parquet")
    ap.add_argument("--target", default="fwd_63d_xspy")
    ap.add_argument("--output", default="backtest_results/audit_sequence_v2_regime.txt")
    args = ap.parse_args()
    main(args.input, args.target, args.output)
