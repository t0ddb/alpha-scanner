from __future__ import annotations

"""
data_fetcher.py — Fetches OHLCV data for all tickers via yfinance.

Usage:
    from data_fetcher import fetch_all, fetch_ticker, fetch_sector

    # Fetch 1 year of data for all tickers
    data = fetch_all(cfg, period="1y")

    # Fetch a single ticker
    df = fetch_ticker("NVDA", period="2y")

    # Fetch all tickers in a sector
    data = fetch_sector(cfg, "metals", period="1y")

Returns:
    dict[str, pd.DataFrame] — keyed by ticker symbol, each DataFrame has
    columns: Open, High, Low, Close, Volume with a DatetimeIndex.
"""

import time
import yfinance as yf
import pandas as pd
from config import load_config, get_all_tickers, get_tickers_by_sector, get_ticker_metadata


# -----------------------------------------------------------
# Single ticker fetch
# -----------------------------------------------------------
def fetch_ticker(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    start: str = None,
    end: str = None,
) -> pd.DataFrame | None:
    """
    Fetch OHLCV data for a single ticker.

    Args:
        ticker:   Symbol (e.g., "NVDA", "BTC-USD", "GC=F")
        period:   yfinance period string ("1y", "2y", "5y", "max")
                  Ignored if start/end are provided.
        interval: Data interval ("1d", "1wk", "1mo")
        start:    Start date string "YYYY-MM-DD" (optional)
        end:      End date string "YYYY-MM-DD" (optional)

    Returns:
        DataFrame with OHLCV columns, or None if fetch fails.
    """
    try:
        t = yf.Ticker(ticker)

        if start and end:
            df = t.history(start=start, end=end, interval=interval)
        elif start:
            df = t.history(start=start, interval=interval)
        else:
            df = t.history(period=period, interval=interval)

        if df.empty:
            print(f"  [WARN] No data returned for {ticker}")
            return None

        # Keep only the columns we need
        keep_cols = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in keep_cols if c in df.columns]]

        # Drop any rows where Close is NaN
        df = df.dropna(subset=["Close"])

        return df

    except Exception as e:
        print(f"  [ERROR] Failed to fetch {ticker}: {e}")
        return None


# -----------------------------------------------------------
# Batch fetch with progress reporting
# -----------------------------------------------------------
def fetch_batch(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
    start: str = None,
    end: str = None,
    delay: float = 0.1,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for a list of tickers.

    Args:
        tickers:  List of ticker symbols
        period:   yfinance period string
        interval: Data interval
        start:    Start date (optional)
        end:      End date (optional)
        delay:    Seconds to pause between requests (rate limiting)
        verbose:  Print progress if True

    Returns:
        dict mapping ticker -> DataFrame (failed tickers are excluded)
    """
    results = {}
    total = len(tickers)
    failed = []

    if verbose:
        print(f"\nFetching {total} tickers...")
        print(f"  Period: {period} | Interval: {interval}")
        if start:
            print(f"  Date range: {start} -> {end or 'now'}")
        print()

    for i, ticker in enumerate(tickers, 1):
        if verbose:
            print(f"  [{i}/{total}] {ticker}...", end=" ", flush=True)

        df = fetch_ticker(ticker, period=period, interval=interval,
                          start=start, end=end)

        if df is not None:
            results[ticker] = df
            if verbose:
                print(f"OK ({len(df)} rows, {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})")
        else:
            failed.append(ticker)
            if verbose:
                print("FAILED")

        # Small delay to avoid hitting rate limits
        if i < total:
            time.sleep(delay)

    if verbose:
        print(f"\nDone: {len(results)}/{total} succeeded")
        if failed:
            print(f"Failed: {', '.join(failed)}")

    # Fail loud on catastrophic fetch failure (e.g., yfinance blocked,
    # network down, dependency incompatibility silently breaking parsing).
    # Without this, callers get an empty dict and downstream code shows
    # zero scores / empty tables with no indication of what went wrong.
    if total > 0 and len(results) == 0:
        raise RuntimeError(
            f"Data fetch failed: 0 of {total} tickers returned data. "
            f"yfinance may be blocked, rate-limited, or incompatible with "
            f"the current pandas/numpy versions. Check logs for per-ticker errors."
        )

    return results


# -----------------------------------------------------------
# High-level convenience functions
# -----------------------------------------------------------
def fetch_all(
    cfg: dict,
    period: str = "1y",
    interval: str = "1d",
    start: str = None,
    end: str = None,
    include_benchmark: bool = True,
    delay: float = 0.1,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for every ticker in the config, plus the benchmark.
    """
    tickers = get_all_tickers(cfg)
    if include_benchmark:
        benchmark = cfg["benchmark"]["ticker"]
        if benchmark not in tickers:
            tickers.insert(0, benchmark)

    return fetch_batch(tickers, period=period, interval=interval,
                       start=start, end=end, delay=delay, verbose=verbose)


def fetch_sector(
    cfg: dict,
    sector_key: str,
    period: str = "1y",
    interval: str = "1d",
    start: str = None,
    end: str = None,
    include_benchmark: bool = True,
    delay: float = 0.1,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all tickers in a specific sector.
    """
    tickers = get_tickers_by_sector(cfg, sector_key)
    if include_benchmark:
        benchmark = cfg["benchmark"]["ticker"]
        if benchmark not in tickers:
            tickers.insert(0, benchmark)

    return fetch_batch(tickers, period=period, interval=interval,
                       start=start, end=end, delay=delay, verbose=verbose)


# -----------------------------------------------------------
# Data quality summary
# -----------------------------------------------------------
def data_summary(data: dict[str, pd.DataFrame], cfg: dict = None) -> pd.DataFrame:
    """
    Return a summary DataFrame showing date range, row count,
    and latest close for each fetched ticker.
    """
    rows = []
    metadata = get_ticker_metadata(cfg) if cfg else {}

    for ticker, df in sorted(data.items()):
        meta = metadata.get(ticker, {})
        rows.append({
            "ticker": ticker,
            "name": meta.get("name", ""),
            "sector": meta.get("sector_name", ""),
            "subsector": meta.get("subsector_name", ""),
            "rows": len(df),
            "start": df.index[0].strftime("%Y-%m-%d"),
            "end": df.index[-1].strftime("%Y-%m-%d"),
            "latest_close": round(df["Close"].iloc[-1], 2),
        })

    return pd.DataFrame(rows)


# -----------------------------------------------------------
# Quick test: fetch a small sample and print summary
# -----------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()

    # Test with a small sample: 1 from each asset class + benchmark
    test_tickers = ["SPY", "NVDA", "GLD", "NEM", "BTC-USD", "FET-USD"]
    print("=" * 60)
    print("  DATA FETCHER — SMOKE TEST")
    print("=" * 60)

    data = fetch_batch(test_tickers, period="6mo", verbose=True)

    if data:
        print("\n" + "=" * 60)
        print("  SUMMARY")
        print("=" * 60)
        summary = data_summary(data, cfg)
        print(summary.to_string(index=False))
