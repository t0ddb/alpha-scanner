"""
signal_diagnostics_subsector.py — Decompose the score's predictive power
by subsector.

Follow-up to signal_diagnostics.py. The prior run showed a striking
AI/Tech vs Other divergence at the sector level; this script breaks
both halves down to the subsector level to distinguish real structure
from averaging artifacts or small-N noise. Still pure diagnostic: no
execution code, no backtest changes, no proposed changes to
trade_executor.py.

Analyses:
  1. Bucket table per subsector (all 31) — raw + sm10, h=21 and h=63
  2. Per-subsector Spearman ρ (sm10, h=63)
  3. Ranked summary across all subsectors
  4. Temporal stability (first half vs second half of observation window)

Outputs:
  - Printed tables
  - CSVs in signal_diagnostics_out/:
      - subsector_buckets.csv    (every subsector × bucket × smoothing × horizon)
      - subsector_summary.csv    (ranked summary)
      - temporal_split.csv       (first half vs second half)
  - PNGs:
      - subsector_spearman_heatmap.png
      - subsector_rank_bars.png

Usage:
    python signal_diagnostics_subsector.py
    python signal_diagnostics_subsector.py --min-count 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import load_config
from data_fetcher import fetch_all
from signal_diagnostics import (
    BUCKETS,
    HORIZONS,
    SMOOTHING_WINDOWS,
    bucket_score,
    bucket_stats,
    compute_forward_returns,
    compute_smoothed_scores,
    load_scores,
    spearman_table,
)


OUT_DIR = Path(__file__).parent / "signal_diagnostics_out"
LOW_N_THRESHOLD = 100   # flag buckets with fewer observations


# ============================================================
# HELPERS
# ============================================================

def parent_group(sector: str) -> str:
    return "AI/Tech" if isinstance(sector, str) and "AI" in sector else "Other"


def subsector_bucket_stats(
    df: pd.DataFrame, score_col: str, ret_col: str,
) -> pd.DataFrame:
    """Bucket stats WITHOUT a min-count filter (we want to see low-N cells)."""
    sub = df[[score_col, ret_col]].dropna().copy()
    sub["bucket"] = sub[score_col].apply(bucket_score)
    sub = sub.dropna(subset=["bucket"])
    rows = []
    for label, _, _ in BUCKETS:
        b = sub[sub["bucket"] == label]
        if b.empty:
            rows.append({"bucket": label, "n": 0, "mean": np.nan,
                         "median": np.nan, "win_rate": np.nan, "std": np.nan,
                         "low_n": True})
            continue
        r = b[ret_col].values
        rows.append({
            "bucket": label,
            "n": len(b),
            "mean": float(np.mean(r)),
            "median": float(np.median(r)),
            "win_rate": float(np.mean(r > 0) * 100),
            "std": float(np.std(r)),
            "low_n": len(b) < LOW_N_THRESHOLD,
        })
    return pd.DataFrame(rows)


def _fmt_cell(row: pd.Series) -> str:
    if row["n"] == 0 or pd.isna(row["mean"]):
        return f"{'—':>5s} {'—':>6s} {'—':>5s}"
    tag = "*" if row["low_n"] else " "
    return f"{int(row['n']):>5d}{tag}{row['mean']:>+5.1f}% {row['win_rate']:>4.0f}%"


def print_subsector_panel(
    df_sub: pd.DataFrame, subsector: str, parent: str,
    n_tickers: int, total_rows: int,
):
    """One panel per subsector: raw & sm10 at h=21 and h=63."""
    print(f"\n{'─' * 110}")
    print(f"  {subsector}   [parent: {parent}]   tickers={n_tickers}  rows={total_rows}")
    print(f"{'─' * 110}")
    header = f"  {'Bucket':<9s}"
    col_headers = []
    for h in [21, 63]:
        for col_label in ["raw", "sm10"]:
            col_headers.append(f"h={h}d {col_label:<5s}")
    header += " | " + " | ".join(f"{c:<18s}" for c in col_headers)
    print(header)
    print(f"  {'':>9s}  " + " ".join(
        f"{'n':>5s} {'mean':>6s} {'win%':>5s}     " for _ in range(4)
    ))

    panels = {}
    for h in [21, 63]:
        for col_key, col_col in [("raw", "score"), ("sm10", "score_sm10")]:
            panels[(h, col_key)] = subsector_bucket_stats(
                df_sub, col_col, f"fwd_ret_{h}"
            )

    for label, _, _ in BUCKETS:
        line = f"  {label:<9s}"
        for h in [21, 63]:
            for col_key in ["raw", "sm10"]:
                p = panels[(h, col_key)]
                row = p[p["bucket"] == label].iloc[0]
                line += f" | {_fmt_cell(row):<18s}"
        print(line)


def compute_subsector_summary(
    df: pd.DataFrame, subsector: str,
) -> dict:
    """Single row for the ranked summary table."""
    sub = df[df["subsector"] == subsector]
    parent = parent_group(sub["sector"].iloc[0]) if not sub.empty else "?"
    n_rows = len(sub)
    n_tickers = sub["ticker"].nunique()

    # Spearman sm10 @ h=63
    valid = sub[["score_sm10", "fwd_ret_63"]].dropna()
    if len(valid) >= 50:
        rho, p = stats.spearmanr(valid["score_sm10"], valid["fwd_ret_63"])
        rho = float(rho); pval = float(p)
    else:
        rho, pval = np.nan, np.nan

    # Count of 8+ observations under sm10
    n_8plus = int((sub["score_sm10"] >= 8.0).sum())

    # 9+ bucket sm10 h=63
    bucket9 = sub[(sub["score_sm10"] >= 9.0) & sub["fwd_ret_63"].notna()]["fwd_ret_63"]
    if len(bucket9) >= LOW_N_THRESHOLD:
        mean_9 = float(bucket9.mean())
        win_9 = float((bucket9 > 0).mean() * 100)
        n_9 = len(bucket9)
        flag_9 = ""
    elif len(bucket9) > 0:
        mean_9 = float(bucket9.mean())
        win_9 = float((bucket9 > 0).mean() * 100)
        n_9 = len(bucket9)
        flag_9 = "*"
    else:
        mean_9, win_9, n_9, flag_9 = np.nan, np.nan, 0, "—"

    return {
        "subsector": subsector,
        "parent": parent,
        "tickers": n_tickers,
        "total_rows": n_rows,
        "n_8plus": n_8plus,
        "rho_sm10_h63": rho,
        "rho_pvalue": pval,
        "mean_9plus_h63": mean_9,
        "win_9plus_h63": win_9,
        "n_9plus_h63": n_9,
        "low_n_flag": flag_9,
    }


def per_subsector_spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman ρ across smoothings × horizons per subsector (long-form)."""
    rows = []
    for subsector, sub in df.groupby("subsector"):
        for n in SMOOTHING_WINDOWS:
            col = "score" if n == 1 else f"score_sm{n}"
            for h in HORIZONS:
                valid = sub[[col, f"fwd_ret_{h}"]].dropna()
                if len(valid) < 50:
                    rho, pval = np.nan, np.nan
                else:
                    rho, pval = stats.spearmanr(
                        valid[col], valid[f"fwd_ret_{h}"]
                    )
                rows.append({
                    "subsector": subsector,
                    "parent": parent_group(sub["sector"].iloc[0]),
                    "smoothing": n,
                    "horizon": h,
                    "n": len(valid),
                    "rho": float(rho) if not np.isnan(rho) else np.nan,
                    "p": float(pval) if not np.isnan(pval) else np.nan,
                })
    return pd.DataFrame(rows)


# ============================================================
# CHARTS
# ============================================================

def chart_subsector_heatmap(spearman_long: pd.DataFrame,
                              ranked_summary: pd.DataFrame):
    """Heatmap: subsector × horizon (sm10), sorted by 63d ρ."""
    sub_order = ranked_summary["subsector"].tolist()
    sm10 = spearman_long[spearman_long["smoothing"] == 10].copy()
    pivot = sm10.pivot(index="subsector", columns="horizon", values="rho")
    pivot = pivot.reindex(sub_order)

    fig, ax = plt.subplots(figsize=(9, 11))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                   aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c}d" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    # Colour-code parent with text prefix
    parent_map = {r["subsector"]: r["parent"] for _, r in ranked_summary.iterrows()}
    labels = [f"[{parent_map[s][:3]}] {s[:45]}" for s in pivot.index]
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Forward horizon")
    ax.set_title("Spearman ρ (sm10 score vs forward return) — by subsector")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if np.isnan(v):
                txt = "—"
                colour = "gray"
            else:
                txt = f"{v:+.2f}"
                colour = "black" if abs(v) < 0.28 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    color=colour, fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    path = OUT_DIR / "subsector_spearman_heatmap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  → saved {path}")


def chart_subsector_rank(ranked_summary: pd.DataFrame):
    """Horizontal bar chart of sm10 ρ@63d by subsector, coloured by parent."""
    df = ranked_summary.dropna(subset=["rho_sm10_h63"]).copy()
    df = df.sort_values("rho_sm10_h63")   # ascending for horizontal bar
    colors = ["#2563eb" if p == "AI/Tech" else "#f59e0b" for p in df["parent"]]

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(df["subsector"], df["rho_sm10_h63"], color=colors)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Spearman ρ (sm10, 63d)")
    ax.set_title("Subsector rank — score predictive power (sm10, h=63d)\n"
                 "Blue = AI/Tech, Amber = Other")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / "subsector_rank_bars.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  → saved {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    global LOW_N_THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=LOW_N_THRESHOLD,
                    help=f"Low-N flag threshold (default: {LOW_N_THRESHOLD})")
    args = ap.parse_args()
    LOW_N_THRESHOLD = args.min_count

    OUT_DIR.mkdir(exist_ok=True)

    print("=" * 82)
    print("  SIGNAL DIAGNOSTICS — SUBSECTOR DECOMPOSITION")
    print("=" * 82)

    print("\n  Loading scores from SQLite...")
    scores_df = load_scores()
    print(f"  {len(scores_df):,} rows, {scores_df['subsector'].nunique()} subsectors")

    print("\n  Fetching price data (2y)...")
    cfg = load_config()
    price_data = fetch_all(cfg, period="2y", verbose=False)
    print(f"  {len(price_data)} tickers fetched")

    print("\n  Computing smoothed scores + forward returns...")
    scores_df = compute_smoothed_scores(scores_df)
    df = compute_forward_returns(scores_df, price_data, HORIZONS)
    valid = df[[f"fwd_ret_{h}" for h in HORIZONS]].notna().any(axis=1)
    df = df[valid].reset_index(drop=True)
    print(f"  {len(df):,} rows with at least one forward return")

    # Organize subsectors by parent (Analysis 1 & 2)
    subsector_list = sorted(df["subsector"].unique().tolist())
    ai_tech_subs = sorted([
        s for s in subsector_list
        if parent_group(df[df["subsector"] == s]["sector"].iloc[0]) == "AI/Tech"
    ])
    other_subs = [s for s in subsector_list if s not in ai_tech_subs]

    # ── ANALYSIS 2: AI/Tech per-subsector panels ──
    print("\n" + "=" * 110)
    print(f"  ANALYSIS 2 — AI/TECH SUBSECTORS ({len(ai_tech_subs)} subsectors)")
    print("=" * 110)
    print(f"  Legend: n  mean  win%.  Asterisk (*) flags n < {LOW_N_THRESHOLD} — interpret with caution.")
    for sub in ai_tech_subs:
        sub_df = df[df["subsector"] == sub]
        print_subsector_panel(
            sub_df, sub, "AI/Tech",
            n_tickers=sub_df["ticker"].nunique(),
            total_rows=len(sub_df),
        )

    # ── ANALYSIS 1: Other per-subsector panels ──
    print("\n" + "=" * 110)
    print(f"  ANALYSIS 1 — OTHER SUBSECTORS ({len(other_subs)} subsectors)")
    print("=" * 110)
    print(f"  Legend: n  mean  win%.  Asterisk (*) flags n < {LOW_N_THRESHOLD} — interpret with caution.")
    for sub in other_subs:
        sub_df = df[df["subsector"] == sub]
        print_subsector_panel(
            sub_df, sub, "Other",
            n_tickers=sub_df["ticker"].nunique(),
            total_rows=len(sub_df),
        )

    # ── ANALYSIS 3: Ranked summary ──
    print("\n" + "=" * 110)
    print("  ANALYSIS 3 — RANKED SUMMARY (sorted by sm10 ρ at 63d)")
    print("=" * 110)
    summary_rows = [compute_subsector_summary(df, s) for s in subsector_list]
    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(
        "rho_sm10_h63", ascending=False, na_position="last"
    ).reset_index(drop=True)

    print(f"\n  {'Rank':>4s}  {'Subsector':<42s}  {'Parent':<7s}  "
          f"{'Tix':>3s}  {'Rows':>5s}  {'n8+':>5s}  "
          f"{'ρ(sm10,63d)':>11s}  {'9+ mean':>8s}  {'9+ win%':>7s}  {'9+ n':>6s}")
    print(f"  {'─' * 108}")
    for i, r in summary.iterrows():
        rho_str = (f"{r['rho_sm10_h63']:+.3f}"
                   if not pd.isna(r["rho_sm10_h63"]) else "   —   ")
        if pd.isna(r["mean_9plus_h63"]) or r["n_9plus_h63"] == 0:
            mean_9 = "   —  "
            win_9 = "   —  "
            flag = "—"
        else:
            mean_9 = f"{r['mean_9plus_h63']:+6.1f}%"
            win_9 = f"{r['win_9plus_h63']:>5.0f}%"
            flag = "*" if r["n_9plus_h63"] < LOW_N_THRESHOLD else " "
        print(f"  {i+1:>4d}  {r['subsector'][:42]:<42s}  "
              f"{r['parent']:<7s}  {r['tickers']:>3d}  {r['total_rows']:>5d}  "
              f"{r['n_8plus']:>5d}  {rho_str:>11s}  "
              f"{mean_9:>8s}  {win_9:>7s}  {int(r['n_9plus_h63']):>4d}{flag}")

    summary.to_csv(OUT_DIR / "subsector_summary.csv", index=False)
    print(f"\n  → saved {OUT_DIR / 'subsector_summary.csv'}")

    # Full long-form Spearman matrix (also saved as CSV)
    sp_long = per_subsector_spearman_table(df)
    sp_long.to_csv(OUT_DIR / "subsector_spearman_long.csv", index=False)

    # Full long-form bucket table (every subsector × bucket × smoothing × horizon)
    bucket_rows = []
    for sub in subsector_list:
        sub_df = df[df["subsector"] == sub]
        parent = parent_group(sub_df["sector"].iloc[0])
        for n in SMOOTHING_WINDOWS:
            score_col = "score" if n == 1 else f"score_sm{n}"
            for h in HORIZONS:
                p = subsector_bucket_stats(sub_df, score_col, f"fwd_ret_{h}")
                for _, r in p.iterrows():
                    bucket_rows.append({
                        "subsector": sub, "parent": parent,
                        "smoothing": n, "horizon": h,
                        "bucket": r["bucket"], "n": r["n"],
                        "mean": r["mean"], "median": r["median"],
                        "win_rate": r["win_rate"], "std": r["std"],
                        "low_n": r["low_n"],
                    })
    pd.DataFrame(bucket_rows).to_csv(
        OUT_DIR / "subsector_buckets.csv", index=False
    )
    print(f"  → saved {OUT_DIR / 'subsector_buckets.csv'}")

    # ── ANALYSIS 4: Temporal split ──
    print("\n" + "=" * 110)
    print("  ANALYSIS 4 — TEMPORAL STABILITY (first half vs second half)")
    print("=" * 110)
    dates = sorted(df["date"].unique())
    mid = dates[len(dates) // 2]
    mid_str = pd.Timestamp(mid).strftime("%Y-%m-%d")
    first_half = df[df["date"] < mid]
    second_half = df[df["date"] >= mid]

    print(f"\n  Split date: {mid_str}")
    print(f"    First half:  {first_half['date'].min().date()} → "
          f"{pd.Timestamp(mid).date() - pd.Timedelta(days=1)}  "
          f"({len(first_half):,} rows)")
    print(f"    Second half: {pd.Timestamp(mid).date()} → "
          f"{second_half['date'].max().date()}  "
          f"({len(second_half):,} rows)")

    temporal_rows = []
    print(f"\n  {'Horizon':<8s}  {'Half':<6s}  {'n':>7s}  "
          f"{'ρ(sm10)':>8s}  {'p-value':>9s}  {'ρ(raw)':>8s}")
    print(f"  {'─' * 60}")
    for h in HORIZONS:
        for label, sub in [("First", first_half), ("Second", second_half)]:
            for col_key, col in [("sm10", "score_sm10"), ("raw", "score")]:
                valid_d = sub[[col, f"fwd_ret_{h}"]].dropna()
                if len(valid_d) < 50:
                    rho, pval = np.nan, np.nan
                else:
                    rho, pval = stats.spearmanr(valid_d[col], valid_d[f"fwd_ret_{h}"])
                temporal_rows.append({
                    "horizon": h, "half": label, "kind": col_key,
                    "n": len(valid_d),
                    "rho": float(rho) if not np.isnan(rho) else np.nan,
                    "p_value": float(pval) if not np.isnan(pval) else np.nan,
                })

    # Print as one row per (horizon × half), sm10 + raw side by side
    for h in HORIZONS:
        for half in ["First", "Second"]:
            rho_sm = next((r for r in temporal_rows
                           if r["horizon"] == h and r["half"] == half
                           and r["kind"] == "sm10"), None)
            rho_raw = next((r for r in temporal_rows
                            if r["horizon"] == h and r["half"] == half
                            and r["kind"] == "raw"), None)
            if not rho_sm:
                continue
            rsm = f"{rho_sm['rho']:+.3f}" if not pd.isna(rho_sm['rho']) else "   —   "
            rraw = f"{rho_raw['rho']:+.3f}" if not pd.isna(rho_raw['rho']) else "   —   "
            psm = f"{rho_sm['p_value']:.2e}" if not pd.isna(rho_sm['p_value']) else "   —   "
            print(f"  h={h:>2d}d     {half:<6s}  {rho_sm['n']:>7d}  "
                  f"{rsm:>8s}  {psm:>9s}  {rraw:>8s}")

    pd.DataFrame(temporal_rows).to_csv(OUT_DIR / "temporal_split.csv", index=False)
    print(f"\n  → saved {OUT_DIR / 'temporal_split.csv'}")

    # ── Charts ──
    print("\n" + "=" * 82)
    print("  SAVING CHARTS")
    print("=" * 82)
    chart_subsector_heatmap(sp_long, summary)
    chart_subsector_rank(summary)

    print("\n  Done.")


if __name__ == "__main__":
    main()
