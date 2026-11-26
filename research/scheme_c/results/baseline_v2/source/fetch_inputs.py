"""One-time vendor capture. Offline reruns use the resulting immutable pickle.

No config, database, brokerage client, credentials, or production module imports.
Run from repository root: python3 -B research/scheme_c/fetch_inputs.py
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import pickle
import sys
import yfinance as yf

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "inputs/vendor_prices.pkl"

if __name__ == "__main__":
    if DEST.exists():
        sys.exit("Refusing to overwrite vendor snapshot; retain the existing capture.")
    with (ROOT / "inputs/review_prices.pkl").open("rb") as f:
        tickers = sorted(pickle.load(f))
    yf.set_tz_cache_location("/private/tmp/scheme-c-yf-cache")

    def fetch(ticker):
        # Raw Yahoo bars are split-adjusted, but not dividend-adjusted.
        # Actions allow reconstruction of historical share denominations.
        for attempt in range(3):
            try:
                df = yf.Ticker(ticker).history(
                    start="2022-01-01", end="2026-10-14", auto_adjust=False,
                    actions=True, raise_errors=True, timeout=30,
                )
                if not df.empty:
                    df.index = df.index.tz_localize(None).normalize()
                    return ticker, df
            except Exception as exc:
                if attempt == 2:
                    return ticker, str(exc)
        return ticker, "empty response"

    data, errors = {}, {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for i, task in enumerate(as_completed([pool.submit(fetch, t) for t in tickers]), 1):
            ticker, value = task.result()
            if isinstance(value, str):
                errors[ticker] = value
            else:
                data[ticker] = value
            if i % 25 == 0:
                print(f"Captured {i}/{len(tickers)}; failures {len(errors)}", flush=True)
    if errors:
        (ROOT / "inputs/fetch_errors.json").write_text(json.dumps(errors, indent=2))
        sys.exit(f"Missing {len(errors)} tickers; snapshot not finalized.")
    DEST.write_bytes(pickle.dumps(dict(sorted(data.items())), protocol=4))
    manifest = {
        "provider": "Yahoo via yfinance", "yfinance_version": yf.__version__,
        "requested_start": "2022-01-01", "requested_end_exclusive": "2026-10-14",
        "capture_date": "2026-10-18", "auto_adjust": False, "actions": True,
        "sha256": hashlib.sha256(DEST.read_bytes()).hexdigest(),
        "tickers": {t: {"first": str(d.index.min().date()),
                         "last": str(d.index.max().date()), "rows": len(d)}
                    for t, d in sorted(data.items())},
    }
    (ROOT / "inputs/vendor_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Captured {len(data)} symbols to {DEST}", flush=True)
