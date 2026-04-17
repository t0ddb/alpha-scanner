"""
signal_diagnostics.py — Does the composite score predict forward returns?

Pure diagnostic: no execution code, no backtest, no changes to live
trade logic. Reads historical scores from SQLite, pulls price data
from yfinance, and computes four analyses:

  1. Forward return by score bucket (raw + 3/5/10/20-day rolling means;
     forward horizons 7/21/63 trading days)
  2. Raw vs smoothed comparison via Spearman rank correlation
  3. Score autocorrelation at lags 1/3/5/10/20
  4. Sector stratification — does the signal work universe-wide?

Outputs:
  - Printed tables on stdout
  - 2-3 PNG charts saved to ./signal_diagnostics_out/

Usage:
    python signal_diagnostics.py
    python signal_diagnostics.py --min-count 30   # bucket min to include
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from config import load_config
from data_fetcher import fetch_all


DB_PATH = Path(__file__).parent / "breakout_tracker.db"
OUT_DIR = Path(__file__).parent / "signal_diagnostics_out"

SMOOTHING_WINDOWS = [1, 3, 5, 10, 20]   # 1 = raw score
HORIZONS = [7, 21, 63]                   # forward trading days
BUCKETS = [
    ("<5.0",   -np.inf, 5.0),
    ("5-6",    5.0,     6.0),
    ("6-7",    6.0,     7.0),
    ("7-8",    7.0,     8.0),
    ("8-8.5",  8.0,     8.5),
    ("8.5-9",  8.5,     9.0),
    ("9+",     9.0,     np.inf),
]


# ============================================================
# DATA
# ============================================================

def load_scores() -> pd.DataFrame:
    """Load (date, ticker, score, sector, subsector) from DB."""
    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql_query(
        "SELECT date, ticker, score, sector, subsector FROM ticker_scores",
        conn,
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


def compute_smoothed_scores(scores_df: pd.DataFrame) -> pd.DataFrame:
    """Add score_smN columns for each N in SMOOTHING_WINDOWS."""
    out = scores_df.copy()
    for n in SMOOTHING_WINDOWS:
        col = f"score_sm{n}"
        if n == 1:
            out[col] = out["score"]
        else:
            # min_periods=n → only emit once we have full window
            out[col] = (
                out.groupby("ticker")["score"]
                   .transform(lambda s: s.rolling(n, min_periods=n).mean())
            )
    return out


def compute_forward_returns(
    scores_df: pd.DataFrame,
    price_data: dict,
    horizons: list[int],
) -> pd.DataFrame:
    """
    Attach fwd_ret_{H} columns for each H in horizons. Forward return
    is computed from the ticker's Close on date D to its Close on the
    H-th trading day after D, using the ticker's own price series (not
    calendar days — that handles weekends/holidays correctly).
    """
    out = scores_df.copy()
    for h in horizons:
        out[f"fwd_ret_{h}"] = np.nan

    # Build per-ticker close-price series indexed by date
    price_series: dict[str, pd.Series] = {}
    for t, df in price_data.items():
        if df is None or df.empty:
            continue
        idx = df.index.tz_localize(None) if df.index.tz else df.index
        s = pd.Series(df["Close"].values, index=pd.DatetimeIndex(idx).normalize())
        s = s[~s.index.duplicated(keep="last")].sort_index()
        price_series[t] = s

    for ticker, group in out.groupby("ticker"):
        s = price_series.get(ticker)
        if s is None or s.empty:
            continue
        # Map each score date to its positional index in the price series
        positions = s.index.get_indexer(group["date"].dt.normalize())
        for h in horizons:
            fwd_pos = positions + h
            fwd_vals = np.full(len(group), np.nan)
            valid = (positions >= 0) & (fwd_pos < len(s)) & (fwd_pos >= 0)
            if valid.any():
                p_now = s.values[positions[valid]]
                p_fwd = s.values[fwd_pos[valid]]
                with np.errstate(divide="ignore", invalid="ignore"):
                    r = np.where(p_now > 0, (p_fwd / p_now - 1.0) * 100.0, np.nan)
                fwd_vals[valid] = r
            out.loc[group.index, f"fwd_ret_{h}"] = fwd_vals

    return out


# ============================================================
# ANALYSIS — 1. BUCKET TABLE
# ============================================================

def bucket_score(value: float) -> str | None:
    if pd.isna(value):
        return None
    for label, lo, hi in BUCKETS:
        if lo <= value < hi:
            return label
    return None


def bucket_stats(df: pd.DataFrame, score_col: str, ret_col: str,
                 min_count: int = 1) -> pd.DataFrame:
    """Aggregate forward return by score bucket."""
    sub = df[[score_col, ret_col]].dropna().copy()
    sub["bucket"] = sub[score_col].apply(bucket_score)
    sub = sub.dropna(subset=["bucket"])
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for label, lo, hi in BUCKETS:
        b = sub[sub["bucket"] == label]
        if len(b) < min_count:
            rows.append({
                "bucket": label, "n": len(b),
                "mean": np.nan, "median": np.nan,
                "win_rate": np.nan, "std": np.nan,
            })
            continue
        rvals = b[ret_col].values
        rows.append({
            "bucket": label,
            "n": len(b),
            "mean": float(np.mean(rvals)),
            "median": float(np.median(rvals)),
            "win_rate": float(np.mean(rvals > 0) * 100),
            "std": float(np.std(rvals)),
        })
    return pd.DataFrame(rows)


def print_bucket_panel(df: pd.DataFrame, horizon: int, title_prefix: str,
                       min_count: int):
    """One panel per forward horizon: all smoothing windows as columns."""
    print(f"\n  {title_prefix} · Forward horizon = {horizon} trading days")
    print(f"  {'─' * 110}")
    header = f"  {'Bucket':<9s}"
    for n in SMOOTHING_WINDOWS:
        tag = "raw" if n == 1 else f"sm{n}"
        header += f" | {tag:<18s}"
    print(header)
    print(f"  {' ':<9s}" + " | ".join(
        f"{'n':>5s} {'mean':>6s} {'win%':>5s}" for _ in SMOOTHING_WINDOWS
    ).rjust(0))

    panels: dict[int, pd.DataFrame] = {}
    for n in SMOOTHING_WINDOWS:
        col = "score" if n == 1 else f"score_sm{n}"
        panels[n] = bucket_stats(df, col, f"fwd_ret_{horizon}", min_count=min_count)

    for label, _, _ in BUCKETS:
        line = f"  {label:<9s}"
        for n in SMOOTHING_WINDOWS:
            p = panels[n]
            row = p[p["bucket"] == label]
            if row.empty or pd.isna(row.iloc[0]["mean"]):
                line += f" | {'—':>5s} {'—':>6s} {'—':>5s}      "
            else:
                r = row.iloc[0]
                line += (f" | {int(r['n']):>5d} {r['mean']:>+5.1f}% "
                         f"{r['win_rate']:>4.0f}%      ")
        print(line)


# ============================================================
# ANALYSIS — 2. RANK CORRELATION
# ============================================================

def spearman_table(df: pd.DataFrame) -> pd.DataFrame:
    """Spearman rho(score_col, fwd_ret_H) for each combination."""
    rows = []
    for n in SMOOTHING_WINDOWS:
        col = "score" if n == 1 else f"score_sm{n}"
        for h in HORIZONS:
            ret_col = f"fwd_ret_{h}"
            sub = df[[col, ret_col]].dropna()
            if len(sub) < 50:
                rows.append({"smoothing": n, "horizon": h, "n": len(sub),
                             "rho": np.nan, "p_value": np.nan})
                continue
            rho, p = stats.spearmanr(sub[col].values, sub[ret_col].values)
            rows.append({"smoothing": n, "horizon": h, "n": len(sub),
                         "rho": float(rho), "p_value": float(p)})
    return pd.DataFrame(rows)


def print_spearman_table(tbl: pd.DataFrame, title: str):
    print(f"\n  {title}")
    print(f"  {'─' * 82}")
    pivot_rho = tbl.pivot(index="smoothing", columns="horizon", values="rho")
    pivot_n = tbl.pivot(index="smoothing", columns="horizon", values="n")
    print(f"  {'Smoothing':<10s}" + "".join(f" | h={h:<2d} rho     n" for h in HORIZONS))
    for n in SMOOTHING_WINDOWS:
        tag = "raw" if n == 1 else f"sm{n}"
        row_parts = [f"  {tag:<10s}"]
        for h in HORIZONS:
            rho = pivot_rho.loc[n, h]
            n_obs = pivot_n.loc[n, h]
            row_parts.append(f"   {rho:>+6.3f}  {int(n_obs):>5d}")
        print("".join(row_parts))


# ============================================================
# ANALYSIS — 3. AUTOCORRELATION
# ============================================================

def score_autocorrelation(scores_df: pd.DataFrame,
                           lags: list[int]) -> pd.DataFrame:
    """Mean (across tickers) of Pearson corr(score_t, score_{t-lag})."""
    rows = []
    for lag in lags:
        tickwise = []
        for ticker, group in scores_df.groupby("ticker"):
            s = group["score"].values
            if len(s) < lag + 10:
                continue
            x = s[lag:]
            y = s[:-lag]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            r = np.corrcoef(x, y)[0, 1]
            if np.isfinite(r):
                tickwise.append(r)
        if tickwise:
            rows.append({
                "lag": lag,
                "tickers": len(tickwise),
                "mean_corr": float(np.mean(tickwise)),
                "median_corr": float(np.median(tickwise)),
                "std_corr": float(np.std(tickwise)),
            })
        else:
            rows.append({"lag": lag, "tickers": 0, "mean_corr": np.nan,
                         "median_corr": np.nan, "std_corr": np.nan})
    return pd.DataFrame(rows)


def print_autocorr(tbl: pd.DataFrame):
    print("\n  SCORE AUTOCORRELATION (Pearson corr score_t vs score_{t-lag})")
    print(f"  {'─' * 72}")
    print(f"  {'Lag (days)':>12s}  {'Tickers':>8s}  {'Mean ρ':>8s}  "
          f"{'Median ρ':>10s}  {'Std ρ':>8s}")
    for _, r in tbl.iterrows():
        print(f"  {int(r['lag']):>12d}  {int(r['tickers']):>8d}  "
              f"{r['mean_corr']:>+7.3f}  {r['median_corr']:>+9.3f}  "
              f"{r['std_corr']:>+7.3f}")


# ============================================================
# ANALYSIS — 4. SECTOR STRATIFICATION
# ============================================================

def stratify_sectors(df: pd.DataFrame, min_count: int):
    """Re-run bucket + Spearman analysis for 'AI/Tech' vs 'Other'."""
    df = df.copy()
    # DB stores the display name ("AI & Tech Capex Cycle"), not the yaml key.
    df["sector_group"] = np.where(
        df["sector"].str.contains("AI", case=False, na=False),
        "AI/Tech",
        "Other",
    )

    for grp, sub in df.groupby("sector_group"):
        print(f"\n{'═' * 110}")
        print(f"  SECTOR STRATUM: {grp}  "
              f"(rows={len(sub):,d}, tickers={sub['ticker'].nunique()})")
        print(f"{'═' * 110}")
        print_bucket_panel(sub, horizon=21, title_prefix="Bucket table", min_count=min_count)
        sp = spearman_table(sub)
        print_spearman_table(sp, "Spearman ρ (score vs forward return)")


# ============================================================
# CHARTS
# ============================================================

def chart_bucket_means(df: pd.DataFrame, horizon: int, min_count: int):
    """Grouped bar chart: bucket on x-axis, one bar per smoothing window."""
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(BUCKETS))
    width = 0.15
    for i, n in enumerate(SMOOTHING_WINDOWS):
        col = "score" if n == 1 else f"score_sm{n}"
        p = bucket_stats(df, col, f"fwd_ret_{horizon}", min_count=min_count)
        means = []
        for label, _, _ in BUCKETS:
            row = p[p["bucket"] == label]
            means.append(row.iloc[0]["mean"] if not row.empty else np.nan)
        tag = "raw" if n == 1 else f"sm{n}"
        ax.bar(x + (i - 2) * width, means, width, label=tag)
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in BUCKETS])
    ax.set_ylabel(f"Avg forward return over {horizon} days (%)")
    ax.set_xlabel("Score bucket")
    ax.set_title(f"Forward return by score bucket — {horizon}-day horizon")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(title="Smoothing", loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = OUT_DIR / f"bucket_means_h{horizon}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  → saved {path}")


def chart_spearman_heatmap(tbl: pd.DataFrame):
    OUT_DIR.mkdir(exist_ok=True)
    pivot = tbl.pivot(index="smoothing", columns="horizon", values="rho")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{h}d" for h in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(["raw" if n == 1 else f"sm{n}" for n in pivot.index])
    ax.set_xlabel("Forward horizon")
    ax.set_ylabel("Smoothing window")
    ax.set_title("Spearman ρ (score vs forward return)")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                    color="black" if abs(v) < 0.15 else "white", fontsize=11)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = OUT_DIR / "spearman_heatmap.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  → saved {path}")


def chart_autocorrelation(tbl: pd.DataFrame):
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(tbl["lag"], tbl["mean_corr"], "o-", label="mean ρ")
    ax.plot(tbl["lag"], tbl["median_corr"], "s--", label="median ρ")
    ax.set_xlabel("Lag (trading days)")
    ax.set_ylabel("Autocorrelation ρ")
    ax.set_title("Score autocorrelation decay")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    path = OUT_DIR / "autocorrelation.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  → saved {path}")


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-count", type=int, default=30,
                    help="Minimum obs per bucket to include (default: 30)")
    args = ap.parse_args()

    print("=" * 82)
    print("  SIGNAL DIAGNOSTICS — Does score predict forward returns?")
    print("=" * 82)

    print("\n  Loading historical scores from SQLite...")
    scores_df = load_scores()
    n_obs = len(scores_df)
    n_tickers = scores_df["ticker"].nunique()
    print(f"  {n_obs:,} (ticker, date) rows; {n_tickers} tickers; "
          f"{scores_df['date'].min().date()} → {scores_df['date'].max().date()}")

    print("\n  Fetching price data (2y) for forward returns...")
    cfg = load_config()
    price_data = fetch_all(cfg, period="2y", verbose=False)
    print(f"  {len(price_data)} tickers fetched")

    print("\n  Computing rolling-average scores...")
    scores_df = compute_smoothed_scores(scores_df)

    print("\n  Computing forward returns...")
    df = compute_forward_returns(scores_df, price_data, HORIZONS)

    # Drop rows where no horizon has a forward return (leaks at end)
    # Keep otherwise — analyses handle NaNs per column.
    valid = df[[f"fwd_ret_{h}" for h in HORIZONS]].notna().any(axis=1)
    print(f"  {valid.sum():,} rows with at least one forward return "
          f"({len(df) - valid.sum():,} dropped — end-of-window leaks)")
    df = df[valid].reset_index(drop=True)

    # ── 1. Bucket tables ──
    print("\n" + "=" * 110)
    print("  1. FORWARD RETURN BY SCORE BUCKET")
    print("=" * 110)
    print(f"  (each cell = n / mean-return / win-rate; empty = <{args.min_count} obs)")
    for h in HORIZONS:
        print_bucket_panel(df, horizon=h, title_prefix="Universe",
                           min_count=args.min_count)

    # ── 2. Rank correlation ──
    print("\n" + "=" * 110)
    print("  2. RAW VS SMOOTHED — Spearman rank correlation")
    print("=" * 110)
    sp = spearman_table(df)
    print_spearman_table(sp, "Spearman ρ(score, forward return) — universe")

    # Delta raw vs sm10
    print("\n  Δ improvement from smoothing (sm10 − raw):")
    raw = sp[sp["smoothing"] == 1].set_index("horizon")["rho"]
    sm10 = sp[sp["smoothing"] == 10].set_index("horizon")["rho"]
    for h in HORIZONS:
        print(f"    h={h:>2d}d:  raw ρ={raw[h]:+.3f}  →  sm10 ρ={sm10[h]:+.3f}  "
              f"(Δ = {sm10[h] - raw[h]:+.3f})")

    # ── 3. Autocorrelation ──
    print("\n" + "=" * 110)
    print("  3. SCORE AUTOCORRELATION")
    print("=" * 110)
    ac = score_autocorrelation(scores_df, [1, 3, 5, 10, 20])
    print_autocorr(ac)

    # ── 4. Sector stratification ──
    print("\n" + "=" * 110)
    print("  4. SECTOR STRATIFICATION (AI/Tech vs Other)")
    print("=" * 110)
    stratify_sectors(df, min_count=args.min_count)

    # ── Charts ──
    print("\n" + "=" * 82)
    print("  SAVING CHARTS")
    print("=" * 82)
    chart_bucket_means(df, horizon=21, min_count=args.min_count)
    chart_spearman_heatmap(sp)
    chart_autocorrelation(ac)

    print("\n  Done.")


if __name__ == "__main__":
    main()
