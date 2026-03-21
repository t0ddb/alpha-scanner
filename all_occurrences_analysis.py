from __future__ import annotations

"""
all_occurrences_analysis.py — Collect ALL occurrences where a ticker scores
8+ or 9+ on the new 10-point scale, then compare forward returns.
"""

import pandas as pd
import numpy as np
from config import load_config, get_all_tickers, get_ticker_metadata
from data_fetcher import fetch_batch
from indicators import score_all, MAX_SCORE

FORWARD_WINDOWS = [10, 21, 42, 63]
THRESHOLDS = [8.0, 9.0]
WARMUP = 260


def run_all_occurrences():
    cfg = load_config()
    metadata = get_ticker_metadata(cfg)

    print("=" * 100)
    print("  ALL-OCCURRENCES ANALYSIS: Forward returns every time a ticker scores 8+ or 9+")
    print("=" * 100)
    print(f"\n  Thresholds: {THRESHOLDS}")
    print(f"  Forward windows: {FORWARD_WINDOWS} trading days\n")

    # Fetch data
    print("  Fetching 3 years of data...\n")
    all_tickers = ["SPY"] + get_all_tickers(cfg)
    data = fetch_batch(all_tickers, period="3y", verbose=True)

    if not data:
        print("  [ERROR] No data fetched.")
        return

    benchmark = data.get("SPY")
    if benchmark is None:
        print("  [ERROR] No SPY data.")
        return

    dates = benchmark.index
    max_fwd = max(FORWARD_WINDOWS)

    start_idx = WARMUP
    end_idx = len(dates) - max_fwd
    test_indices = list(range(start_idx, end_idx, 5))

    print(f"\n  Walking {len(test_indices)} test dates from {dates[start_idx].date()} to {dates[end_idx].date()}")
    print(f"  Tickers: {len(data) - 1}\n")

    # Collect all events
    all_events = []  # list of dicts

    for step, idx in enumerate(test_indices):
        if (step + 1) % 10 == 0:
            print(f"  [{step+1}/{len(test_indices)}] {dates[idx].date()}...")

        sliced = {}
        for ticker, df in data.items():
            mask = df.index <= dates[idx]
            sub = df[mask]
            if len(sub) >= WARMUP:
                sliced[ticker] = sub

        if "SPY" not in sliced:
            continue

        results = score_all(sliced, cfg)

        for r in results:
            if r["score"] < min(THRESHOLDS):
                continue

            ticker = r["ticker"]
            ticker_df = data.get(ticker)
            if ticker_df is None:
                continue

            current_date = dates[idx]
            ticker_dates = ticker_df.index
            pos_arr = ticker_dates.get_indexer([current_date], method="ffill")
            pos = pos_arr[0]
            if pos < 0:
                continue

            current_close = ticker_df["Close"].iloc[pos]
            fwd = {}
            for w in FORWARD_WINDOWS:
                fwd_pos = pos + w
                if fwd_pos < len(ticker_df):
                    fwd_close = ticker_df["Close"].iloc[fwd_pos]
                    fwd[w] = (fwd_close / current_close) - 1
                else:
                    fwd[w] = None

            meta = metadata.get(ticker, {})
            all_events.append({
                "ticker": ticker,
                "date": current_date.date(),
                "score": r["score"],
                "signals": r["signals"],
                "signal_weights": r["signal_weights"],
                "close": current_close,
                "fwd_returns": fwd,
                "name": meta.get("name", ""),
                "sector": meta.get("sector_name", ""),
                "subsector": meta.get("subsector_name", ""),
            })

    # Also collect baseline (all observations, any score)
    # We'll compute it from the events we already have by counting total obs
    total_obs_per_date = []
    for step, idx in enumerate(test_indices):
        sliced = {}
        for ticker, df in data.items():
            mask = df.index <= dates[idx]
            sub = df[mask]
            if len(sub) >= WARMUP:
                sliced[ticker] = sub
        # Count tickers that could be scored (exclude SPY)
        total_obs_per_date.append(len(sliced) - 1 if "SPY" in sliced else 0)
    total_observations = sum(total_obs_per_date)

    print(f"\n  Total observations across all dates: {total_observations}")
    print(f"  Events scoring 8+: {sum(1 for e in all_events if e['score'] >= 8.0)}")
    print(f"  Events scoring 9+: {sum(1 for e in all_events if e['score'] >= 9.0)}")

    # ═══════════════════════════════════════════════════════════
    # SECTION 1: Side-by-side comparison of 8+ vs 9+
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}")
    print(f"  SECTION 1: AGGREGATE FORWARD RETURNS — 8+ vs 9+")
    print(f"{'=' * 100}\n")

    for threshold in THRESHOLDS:
        events = [e for e in all_events if e["score"] >= threshold]
        n = len(events)
        selectivity = n / total_observations * 100 if total_observations > 0 else 0

        print(f"  ── Score ≥ {threshold:.0f} — {n} events ({selectivity:.1f}% selectivity) ──\n")
        print(f"  {'Window':>8}  {'N':>6}  {'Avg Return':>12}  {'Median':>10}  {'Win Rate':>10}  {'Avg Winner':>12}  {'Avg Loser':>12}")
        print(f"  {'─' * 80}")

        for w in FORWARD_WINDOWS:
            returns = [e["fwd_returns"][w] for e in events if e["fwd_returns"].get(w) is not None]
            if returns:
                avg = np.mean(returns) * 100
                med = np.median(returns) * 100
                wr = sum(1 for r in returns if r > 0) / len(returns) * 100
                winners = [r * 100 for r in returns if r > 0]
                losers = [r * 100 for r in returns if r <= 0]
                avg_win = np.mean(winners) if winners else 0
                avg_loss = np.mean(losers) if losers else 0
                print(f"  {w:>5}d    {len(returns):>5}  {avg:>+11.2f}%  {med:>+9.2f}%  {wr:>8.1f}%  {avg_win:>+11.2f}%  {avg_loss:>+11.2f}%")
        print()

    # ═══════════════════════════════════════════════════════════
    # SECTION 2: Score bracket breakdown
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  SECTION 2: SCORE BRACKET BREAKDOWN (63-day returns)")
    print(f"{'=' * 100}\n")

    brackets = [(8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.1)]
    print(f"  {'Bracket':<12} {'Events':>8} {'Avg 63d':>10} {'Median 63d':>12} {'Win Rate':>10} {'Selectivity':>12}")
    print(f"  {'─' * 70}")

    for lo, hi in brackets:
        events = [e for e in all_events if lo <= e["score"] < hi]
        rets = [e["fwd_returns"][63] for e in events if e["fwd_returns"].get(63) is not None]
        if rets:
            avg = np.mean(rets) * 100
            med = np.median(rets) * 100
            wr = sum(1 for r in rets if r > 0) / len(rets) * 100
            sel = len(events) / total_observations * 100
            label = f"{lo:.1f}-{hi:.1f}" if hi <= 10 else f"{lo:.1f}+"
            print(f"  {label:<12} {len(events):>8} {avg:>+9.2f}% {med:>+11.2f}% {wr:>9.1f}% {sel:>10.1f}%")

    # ═══════════════════════════════════════════════════════════
    # SECTION 3: By sector
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}")
    print(f"  SECTION 3: BY SECTOR (63-day returns)")
    print(f"{'=' * 100}\n")

    for threshold in THRESHOLDS:
        events = [e for e in all_events if e["score"] >= threshold]
        print(f"  ── Score ≥ {threshold:.0f} ──\n")

        sector_groups = {}
        for e in events:
            sector = e["sector"] or "Unknown"
            if sector not in sector_groups:
                sector_groups[sector] = []
            if e["fwd_returns"].get(63) is not None:
                sector_groups[sector].append(e["fwd_returns"][63])

        print(f"  {'Sector':<40} {'N':>5}  {'Avg 63d':>10}  {'Median':>10}  {'Win Rate':>10}")
        print(f"  {'─' * 80}")
        for sector in sorted(sector_groups, key=lambda s: -np.mean(sector_groups[s]) if sector_groups[s] else 0):
            rets = sector_groups[sector]
            if rets:
                avg = np.mean(rets) * 100
                med = np.median(rets) * 100
                wr = sum(1 for r in rets if r > 0) / len(rets) * 100
                print(f"  {sector:<40} {len(rets):>5}  {avg:>+9.2f}%  {med:>+9.2f}%  {wr:>8.1f}%")
        print()

    # ═══════════════════════════════════════════════════════════
    # SECTION 4: Top and bottom performers (all occurrences)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  SECTION 4: TICKER FREQUENCY — Most 8+ Occurrences")
    print(f"{'=' * 100}\n")

    ticker_counts = {}
    for e in all_events:
        if e["score"] >= 8.0:
            t = e["ticker"]
            if t not in ticker_counts:
                ticker_counts[t] = {"count": 0, "returns_63": [], "name": e["name"], "sector": e["sector"]}
            ticker_counts[t]["count"] += 1
            if e["fwd_returns"].get(63) is not None:
                ticker_counts[t]["returns_63"].append(e["fwd_returns"][63])

    sorted_tickers = sorted(ticker_counts.items(), key=lambda x: -x[1]["count"])

    print(f"  {'Ticker':<10} {'Name':<25} {'Sector':<25} {'Times 8+':>10} {'Avg 63d':>10} {'WR 63d':>8}")
    print(f"  {'─' * 95}")
    for ticker, info in sorted_tickers[:30]:
        rets = info["returns_63"]
        if rets:
            avg = np.mean(rets) * 100
            wr = sum(1 for r in rets if r > 0) / len(rets) * 100
        else:
            avg = 0
            wr = 0
        print(f"  {ticker:<10} {info['name'][:24]:<25} {info['sector'][:24]:<25} {info['count']:>10} {avg:>+9.2f}% {wr:>7.1f}%")

    # ═══════════════════════════════════════════════════════════
    # SECTION 5: Signal combination analysis
    # ═══════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}")
    print(f"  SECTION 5: WHICH SIGNALS APPEAR IN 9+ EVENTS?")
    print(f"{'=' * 100}\n")

    events_9 = [e for e in all_events if e["score"] >= 9.0]
    signal_freq = {}
    signal_rets = {}
    for e in events_9:
        for s in e["signals"]:
            if s not in signal_freq:
                signal_freq[s] = 0
                signal_rets[s] = []
            signal_freq[s] += 1
            if e["fwd_returns"].get(63) is not None:
                signal_rets[s].append(e["fwd_returns"][63])

    n_9 = len(events_9)
    print(f"  {'Signal':<25} {'Frequency':>10} {'% of 9+ events':>15} {'Avg 63d when present':>22}")
    print(f"  {'─' * 75}")
    for s in sorted(signal_freq, key=lambda x: -signal_freq[x]):
        freq = signal_freq[s]
        pct = freq / n_9 * 100 if n_9 > 0 else 0
        avg = np.mean(signal_rets[s]) * 100 if signal_rets[s] else 0
        print(f"  {s:<25} {freq:>10} {pct:>13.1f}% {avg:>+20.2f}%")

    print(f"\n{'=' * 100}")
    print(f"  ANALYSIS COMPLETE")
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    run_all_occurrences()
