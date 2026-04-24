from __future__ import annotations

"""
audit_build_dataset.py — Build the unified scoring-audit dataset.

For every (date, ticker) in a rolling 3+ year window, computes:
  • All indicator firings (binary) and continuous values
  • The current weighted score using live INDICATOR_WEIGHTS
  • Forward returns at 10/21/42/63 days (raw + SPY-adjusted)

Output: backtest_results/audit_dataset.parquet

Usage:
    python3 audit_build_dataset.py --years 3 --frequency 1
    python3 audit_build_dataset.py --years 3 --frequency 5 --quick   # faster iteration
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import load_config, get_all_tickers, get_indicator_config, get_ticker_metadata
from data_fetcher import fetch_batch
from indicators import (
    compute_all_indicators,
    score_ticker,
    RS_GRADIENT,
    HIGHER_LOWS_GRADIENT,
    INDICATOR_WEIGHTS,
)
from indicators_expanded import (
    check_rsi_momentum,
    check_macd_crossover,
    check_adx,
    check_obv_trend,
    check_consolidation_tightness,
    check_donchian_breakout,
)


# ─────────────────────────────────────────────────────────────
# Trend-structure helpers (parallel to check_higher_lows)
# ─────────────────────────────────────────────────────────────
def _period_extrema(df: pd.DataFrame, num_periods: int = 4, period_length: int = 5,
                    kind: str = "low") -> list[float] | None:
    """Return a list of per-period extrema (oldest→newest), or None if too few rows."""
    total_needed = num_periods * period_length
    if len(df) < total_needed + period_length:
        return None
    vals = []
    col = "Low" if kind == "low" else "High"
    agg = (lambda s: s.min()) if kind == "low" else (lambda s: s.max())
    for i in range(num_periods + 1):
        end_idx = len(df) - (i * period_length)
        start_idx = end_idx - period_length
        if start_idx < 0:
            break
        vals.append(float(agg(df[col].iloc[start_idx:end_idx])))
    vals.reverse()
    return vals


def _consec_count(vals: list[float], direction: str) -> int:
    """Count consecutive increases/decreases in `vals`."""
    cmp = (lambda a, b: b > a) if direction == "up" else (lambda a, b: b < a)
    consec = 0
    for i in range(1, len(vals)):
        if cmp(vals[i - 1], vals[i]):
            consec += 1
        else:
            consec = 0
    return consec


def check_higher_highs(df, num_periods=4, period_length=5):
    vals = _period_extrema(df, num_periods, period_length, kind="high")
    if vals is None:
        return {"triggered": False, "count": 0}
    c = _consec_count(vals, "up")
    return {"triggered": c >= num_periods, "count": c}


def check_lower_lows(df, num_periods=4, period_length=5):
    vals = _period_extrema(df, num_periods, period_length, kind="low")
    if vals is None:
        return {"triggered": False, "count": 0}
    c = _consec_count(vals, "down")
    return {"triggered": c >= num_periods, "count": c}


def check_lower_highs(df, num_periods=4, period_length=5):
    vals = _period_extrema(df, num_periods, period_length, kind="high")
    if vals is None:
        return {"triggered": False, "count": 0}
    c = _consec_count(vals, "down")
    return {"triggered": c >= num_periods, "count": c}


FORWARD_WINDOWS = [10, 21, 42, 63]
WARMUP_DAYS = 260  # enough for 200-day SMA + 30d slope + buffer
OUTPUT_PATH = "backtest_results/audit_dataset.parquet"


# ─────────────────────────────────────────────────────────────
# RS percentile pre-pass across all tickers per date
# ─────────────────────────────────────────────────────────────
def _raw_rs_by_period(data: dict, bench_df: pd.DataFrame, idx: int, periods: list[int]) -> dict:
    """Return {period: {ticker: raw_rs_value}} for all tickers at `idx`."""
    out = {p: {} for p in periods}
    bench_slice = bench_df.iloc[: idx + 1]
    for ticker, full_df in data.items():
        df = full_df.iloc[: idx + 1]
        for p in periods:
            if len(df) < p + 1 or len(bench_slice) < p + 1:
                continue
            sr = df["Close"].iloc[-1] / df["Close"].iloc[-p - 1] - 1
            br = bench_slice["Close"].iloc[-1] / bench_slice["Close"].iloc[-p - 1] - 1
            out[p][ticker] = (sr / br if br > 0 else sr - br) if br != 0 else 0
    return out


def _pctl(val, all_vals):
    if not all_vals:
        return 0.0
    return sum(1 for v in all_vals if v <= val) / len(all_vals) * 100


# ─────────────────────────────────────────────────────────────
# Per-ticker per-date row builder
# ─────────────────────────────────────────────────────────────
def _forward_return(full_df: pd.DataFrame, idx: int, w: int) -> float | None:
    """Forward return from day idx to idx+w, using Close."""
    if idx + w >= len(full_df):
        return None
    entry = full_df["Close"].iloc[idx]
    future = full_df["Close"].iloc[idx + w]
    if entry <= 0:
        return None
    return float((future - entry) / entry)


def _bench_forward_return(bench_df: pd.DataFrame, idx: int, w: int) -> float | None:
    return _forward_return(bench_df, idx, w)


def _score_from_indicators(ind: dict) -> dict:
    """Return weighted score result dict using live INDICATOR_WEIGHTS."""
    return score_ticker(ind)


def build_row(
    ticker: str,
    full_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    bench_idx: int,
    cfg: dict,
    all_rs_values: list[float],
    multi_tf_pctls: dict,
    metadata: dict,
) -> dict | None:
    """Build a single (ticker, date) row. Returns None if insufficient data."""
    # Align ticker's idx to bench date
    target_date = bench_df.index[bench_idx]
    ticker_idx = full_df.index.get_indexer([target_date], method="nearest")[0]
    if ticker_idx < WARMUP_DAYS:
        return None

    df = full_df.iloc[: ticker_idx + 1]

    # Core 7 scored indicators + display-only (MA, near-52w)
    try:
        inds = compute_all_indicators(
            df, bench_df.iloc[: bench_idx + 1], cfg,
            all_rs_values=all_rs_values,
            multi_tf_rs=multi_tf_pctls,
        )
    except Exception:
        return None

    # Extra dropped indicators (continuous values for analysis)
    rsi_r = check_rsi_momentum(df)
    macd_r = check_macd_crossover(df)
    adx_r = check_adx(df)
    obv_r = check_obv_trend(df)
    cons_r = check_consolidation_tightness(df)
    donch_r = check_donchian_breakout(df)
    # Trend-structure indicators (new, untested)
    hh_r = check_higher_highs(df)
    ll_r = check_lower_lows(df)
    lh_r = check_lower_highs(df)

    # Current weighted score
    score_res = _score_from_indicators(inds)

    # Forward returns (raw + SPY-adjusted)
    fwd_raw = {w: _forward_return(full_df, ticker_idx, w) for w in FORWARD_WINDOWS}
    fwd_spy = {w: _bench_forward_return(bench_df, bench_idx, w) for w in FORWARD_WINDOWS}

    rs = inds.get("relative_strength", {})
    ich = inds.get("ichimoku_cloud", {})
    hl = inds.get("higher_lows", {})
    ma = inds.get("moving_averages", {})
    roc = inds.get("roc", {})
    cmf = inds.get("cmf", {})
    n52 = inds.get("near_52w_high", {})
    atr = inds.get("atr_expansion", {})
    dtf = inds.get("dual_tf_rs", {})

    meta = metadata.get(ticker, {})
    row = {
        "date":        target_date.strftime("%Y-%m-%d"),
        "ticker":      ticker,
        "sector":      meta.get("sector_name", ""),
        "subsector":   meta.get("subsector_name", ""),
        "score":       score_res["score"],

        # ─ Binary firings (current 7 scored) ─
        "rs_fired":          int(rs.get("rs_percentile", 0) >= 50),
        "ichimoku_fired":    int(ich.get("triggered", False)),
        "higher_lows_fired": int(hl.get("consecutive_higher_lows", 0) >= 2),
        "roc_fired":         int(roc.get("triggered", False)),
        "cmf_fired":         int(cmf.get("triggered", False)),
        "dual_tf_rs_fired":  int(dtf.get("triggered", False)),
        "atr_fired":         int(atr.get("triggered", False)),

        # ─ Points contributed (captures gradient for RS and HL) ─
        "rs_points":         score_res["signal_weights"].get("relative_strength", 0.0),
        "ichimoku_points":   score_res["signal_weights"].get("ichimoku_cloud", 0.0),
        "higher_lows_points":score_res["signal_weights"].get("higher_lows", 0.0),
        "roc_points":        score_res["signal_weights"].get("roc", 0.0),
        "cmf_points":        score_res["signal_weights"].get("cmf", 0.0),
        "dual_tf_rs_points": score_res["signal_weights"].get("dual_tf_rs", 0.0),
        "atr_points":        score_res["signal_weights"].get("atr_expansion", 0.0),

        # ─ Continuous metrics (current 7) ─
        "rs_percentile":     rs.get("rs_percentile", 0),
        "ichimoku_score":    (
            int(ich.get("above_cloud", False))
            + int(ich.get("cloud_bullish", False))
            + int(ich.get("tenkan_above_kijun", False))
        ),
        "higher_lows_count": hl.get("consecutive_higher_lows", 0),
        "roc_value":         roc.get("roc", 0),
        "cmf_value":         cmf.get("cmf", 0),
        "atr_percentile":    atr.get("atr_percentile", 0),
        "rs_21d_pctl":       dtf.get("rs_21d_percentile", 0),
        "rs_63d_pctl":       dtf.get("rs_63d_percentile", 0),
        "rs_126d_pctl":      dtf.get("rs_126d_percentile", 0),

        # ─ Dropped indicators (binary + continuous) ─
        "ma_50_200_fired":   int(ma.get("triggered", False)),
        "ma_50_close_pct":   (
            (ma.get("current_close", 0) - ma.get("sma_50", 0))
            / ma.get("sma_50", 1) * 100 if ma.get("sma_50") else 0
        ),
        "ma_200_close_pct":  (
            (ma.get("current_close", 0) - ma.get("sma_200", 0))
            / ma.get("sma_200", 1) * 100 if ma.get("sma_200") else 0
        ),
        "near_52w_fired":    int(n52.get("triggered", False)),
        "pct_from_52w_high": n52.get("pct_from_high", 0),
        "rsi_value":         rsi_r.get("rsi", 0),
        "rsi_fired":         int(rsi_r.get("triggered", False)),
        "macd_hist":         macd_r.get("histogram", 0),
        "macd_fired":        int(macd_r.get("triggered", False)),
        "adx_value":         adx_r.get("adx", 0),
        "adx_fired":         int(adx_r.get("triggered", False)),
        "obv_slope":         obv_r.get("obv_slope", 0),
        "obv_fired":         int(obv_r.get("triggered", False)),
        "consolidation":     cons_r.get("range_ratio", 0),
        "donchian_pct":      donch_r.get("pct_above", 0),
        "donchian_fired":    int(donch_r.get("triggered", False)),
        # Trend-structure (untested)
        "higher_highs_count": hh_r.get("count", 0),
        "higher_highs_fired": int(hh_r.get("triggered", False)),
        "lower_lows_count":   ll_r.get("count", 0),
        "lower_lows_fired":   int(ll_r.get("triggered", False)),
        "lower_highs_count":  lh_r.get("count", 0),
        "lower_highs_fired":  int(lh_r.get("triggered", False)),
    }
    for w in FORWARD_WINDOWS:
        row[f"fwd_{w}d"] = fwd_raw[w]
        spy = fwd_spy[w]
        raw = fwd_raw[w]
        row[f"fwd_{w}d_xspy"] = (raw - spy) if (raw is not None and spy is not None) else None
    return row


# ─────────────────────────────────────────────────────────────
# MAIN BUILDER
# ─────────────────────────────────────────────────────────────
def build(years: int, frequency: int, output_path: str, quick: bool = False) -> None:
    cfg = load_config()
    ind_cfg = get_indicator_config(cfg)
    metadata = get_ticker_metadata(cfg)

    benchmark_ticker = cfg["benchmark"]["ticker"]
    all_tickers = [benchmark_ticker] + get_all_tickers(cfg)

    # Fetch enough history: target_years + warmup (1 year) + max_forward (63d)
    period_str = f"{years + 1}y"
    print(f"[build] fetching {len(all_tickers)} tickers, period={period_str} ...")
    t0 = time.time()
    data = fetch_batch(all_tickers, period=period_str, verbose=False)
    print(f"[build] fetched {len(data)}/{len(all_tickers)} in {time.time()-t0:.1f}s")

    bench_df = data.get(benchmark_ticker)
    if bench_df is None or len(bench_df) < WARMUP_DAYS + max(FORWARD_WINDOWS):
        raise RuntimeError("Benchmark SPY data insufficient")
    data_nobench = {t: df for t, df in data.items() if t != benchmark_ticker}

    max_fwd = max(FORWARD_WINDOWS)
    total_days = len(bench_df)
    start_idx = WARMUP_DAYS
    end_idx = total_days - max_fwd
    # Limit to `years` of dates
    target_dates = years * 252
    if end_idx - start_idx > target_dates:
        start_idx = end_idx - target_dates

    test_indices = list(range(start_idx, end_idx, frequency))
    print(f"[build] {len(test_indices)} date snapshots from "
          f"{bench_df.index[start_idx].strftime('%Y-%m-%d')} to "
          f"{bench_df.index[end_idx].strftime('%Y-%m-%d')}")
    print(f"[build] {len(data_nobench)} tickers × {len(test_indices)} dates "
          f"= up to {len(data_nobench)*len(test_indices):,} rows")

    rs_periods = [ind_cfg["relative_strength"]["period"], 21, 63, 126]
    rs_periods = sorted(set(rs_periods))

    all_rows: list[dict] = []
    t_start = time.time()

    for c, idx in enumerate(test_indices, 1):
        raw_rs = _raw_rs_by_period(data_nobench, bench_df, idx, rs_periods)
        primary_period = ind_cfg["relative_strength"]["period"]
        all_rs_primary = list(raw_rs[primary_period].values())

        for ticker, full_df in data_nobench.items():
            # Per-ticker multi-TF percentiles (needed for dual_tf_rs)
            mtf = {}
            for p in (21, 63, 126):
                vals = raw_rs.get(p, {})
                v = vals.get(ticker)
                if v is not None:
                    mtf[f"rs_{p}d_pctl"] = _pctl(v, list(vals.values()))
            row = build_row(
                ticker, full_df, bench_df, idx, cfg,
                all_rs_values=all_rs_primary,
                multi_tf_pctls=mtf,
                metadata=metadata,
            )
            if row is not None:
                all_rows.append(row)

        if c % 10 == 0 or c == len(test_indices):
            elapsed = time.time() - t_start
            rate = c / elapsed
            eta = (len(test_indices) - c) / rate if rate > 0 else 0
            print(f"[build] {c}/{len(test_indices)}  {bench_df.index[idx].strftime('%Y-%m-%d')}  "
                  f"rows={len(all_rows):,}  elapsed={elapsed:.0f}s  eta={eta:.0f}s",
                  flush=True)

        if quick and c >= 20:
            print("[build] --quick: stopping after 20 dates")
            break

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"[build] wrote {len(df):,} rows to {output_path}")
    print(f"[build] total time: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--frequency", type=int, default=1)
    ap.add_argument("--output", default=OUTPUT_PATH)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    build(years=args.years, frequency=args.frequency,
          output_path=args.output, quick=args.quick)
