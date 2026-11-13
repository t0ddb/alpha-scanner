"""Rebuild and rerun the frozen Scheme C study entirely offline.

python3 -B -m research.scheme_c.run --output research/scheme_c/results/baseline_v1
Never overwrites an existing result directory. Does not open production SQLite.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from functools import lru_cache
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import platform
import shutil
import sys

import numpy as np
import pandas as pd

from .baseline import PriceBook, ScoreBook, Settings, price_views, rebuild_scores, simulate
from .protection import snapshot

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INPUTS = HERE / "inputs"
FIXED_STARTS = ["2023-05-15", "2024-05-01", "2025-01-02", "2025-05-01"]
END = "2026-04-30"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def compressed(path, result):
    # Stable gzip headers, no timestamps or source filenames.
    path.write_bytes(gzip.compress(json.dumps(result, sort_keys=True, allow_nan=False).encode(), mtime=0))


def load_legacy():
    # The pre-edit engine and its import dependencies were captured as evidence.
    # Never import the root trading module or load any root database.
    sys.path.insert(0, str(INPUTS))
    spec = importlib.util.spec_from_file_location("scheme_c_frozen_legacy", INPUTS / "legacy_simulator.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module


def legacy_runs(legacy, prices, scores, windows, out, stage):
    days = legacy.get_trading_days(prices)
    daily = legacy.build_daily_scores(scores, days)
    # Memoize exact original semantics (including the old nearest-price bug).
    original = legacy.get_price
    @lru_cache(None)
    def lookup(ticker, day, col):
        return original(prices, ticker, day, col)
    legacy.get_price = lambda data, ticker, day, col="Open": lookup(ticker, pd.Timestamp(day), col)
    cfg = legacy.StrategyConfig("C5", 12, "fixed_pct", 1/12, .05, False, 7,
                                9, 5, legacy.StopLossConfig("fixed", .2), 3)
    results = {}
    try:
        for label, start, end in windows:
            run = legacy.PortfolioSimulator(cfg, daily, prices, days, start_date=start, end_date=end).run()
            m = legacy.compute_metrics(run)
            results[label] = {k: m[k] for k in ["total_return", "sharpe", "max_drawdown", "total_trades", "avg_hold", "win_rate"]}
            results[label].update(start=start, end=end)
            compressed(out / f"{stage}__{label}.json.gz", {
                "metrics": results[label], "settings": asdict(cfg), "history": run.history,
                "trades": [asdict(t) for t in run.trades],
            })
    finally:
        legacy.get_price = original
    return results


def summarize_paths(results):
    paths = [m for key, m in results.items() if key.startswith("path_")]
    return {
        "count": len(paths), "starts": [m["start"] for m in paths],
        "mean_return": float(np.mean([m["total_return"] for m in paths])),
        "std_return_pp": float(np.std([m["total_return"] for m in paths], ddof=0)),
        "min_return": min(m["total_return"] for m in paths),
        "max_return": max(m["total_return"] for m in paths),
        "mean_sharpe": float(np.mean([m["sharpe"] for m in paths])),
        "worst_drawdown": min(m["max_drawdown"] for m in paths),
    }


def main(output):
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose a fresh output directory")
    out.mkdir(parents=True)
    protected_before = snapshot(REPO)
    # Validate captured original inputs, not mutable current production files.
    provenance = json.loads((INPUTS / "sources.json").read_text())
    for name, source in provenance["sources"].items():
        if sha(INPUTS / name) != source["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
    vendor_manifest = json.loads((INPUTS / "vendor_manifest.json").read_text())
    if sha(INPUTS / "vendor_prices.pkl") != vendor_manifest["sha256"]:
        raise ValueError("Vendor price hash mismatch")
    old = pd.read_parquet(INPUTS / "original_scores.parquet")
    old["date"] = pd.to_datetime(old.date)
    universe = sorted(old.ticker.unique())
    previous = json.loads((INPUTS / "review_rerun.json").read_text())
    starts = previous["paths"]["C"]["starts"]
    if len(starts) != 20 or len(set(starts)) != 20:
        raise ValueError("Expected the exact 20 distinct review starts")
    windows = [("fixed_" + start, start, END) for start in FIXED_STARTS]
    windows += [(f"path_{i:02d}", start, END) for i, start in enumerate(starts)]
    later_windows = [
        ("calendar_2023", "2023-05-15", "2024-05-14"),
        ("calendar_2024", "2024-05-15", "2025-05-14"),
        ("calendar_2025", "2025-05-15", "2026-05-14"),
        ("later_2026", "2026-05-04", "2026-10-13"),
    ]
    cfg = Settings()
    bridge = replace(cfg, execution_constraints=False, friction_bps=0.)
    scenarios = {
        "clock_original_scores": (bridge, 100_000., "adjusted", "original"),
        "clock_rebuilt_scores": (bridge, 100_000., "adjusted", "rebuilt"),
        "baseline_100k": (cfg, 100_000., "as_traded", "rebuilt"),
        "baseline_5k": (cfg, 5_000., "as_traded", "rebuilt"),
        "no_cost_100k": (replace(cfg, friction_bps=0.), 100_000., "as_traded", "rebuilt"),
        "cost25bp_100k": (replace(cfg, friction_bps=25.), 100_000., "as_traded", "rebuilt"),
        "open_only_100k": (replace(cfg, limit_model="open_only"), 100_000., "as_traded", "rebuilt"),
        "sale_funded_100k": (replace(cfg, funding="next_open_sales"), 100_000., "as_traded", "rebuilt"),
        "open_only_5k": (replace(cfg, limit_model="open_only"), 5_000., "as_traded", "rebuilt"),
    }
    settings = {
        "study": "Scheme C corrected research baseline v1", "study_date": "2026-10-18",
        "universe": universe, "universe_source": "176 unique tickers in preserved original score parquet",
        "universe_bias": "Retrospective selected watchlist; not point-in-time membership; missing delistings unresolved",
        "score_start": "2023-04-03", "score_end": "2026-10-13", "warmup_observations": 261,
        "event_clock": {
            "signal_batch": "23:59 America/New_York on each benchmark session date",
            "daily_bar_availability": "Assume all same-label equity/crypto/futures bars final by batch time",
            "sizing_and_valuation": "Regular-session equity Close proxy; no extended-hours tape",
            "execution": "Next benchmark session; existing stops act within each session",
            "live_schedule_limit": "Not a reconstruction of unfinished non-equity bars at a 16:35 live run",
        },
        "windows": windows, "additional_windows": later_windows,
        "scenarios": {name: {"settings": asdict(c), "capital": capital, "prices": pv, "scores": sv}
                      for name, (c, capital, pv, sv) in scenarios.items()},
        "legacy_settings": "Captured review: 1/12 sizing; current baseline uses exact YAML .083",
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "inputs": {f.name: sha(f) for f in sorted(INPUTS.iterdir()) if f.is_file()},
    }
    # Written before any returns are calculated; no parameter search.
    dump(out / "settings.json", settings)
    source_dir = out / "source"
    source_dir.mkdir()
    for f in [HERE / "baseline.py", HERE / "run.py", HERE / "fetch_inputs.py", HERE / "protection.py",
              REPO / "tests/test_scheme_c_research.py", REPO / "audit_build_dataset.py",
              REPO / "sizing_comparison_backtest.py"]:
        shutil.copy2(f, source_dir / f.name)
    with (INPUTS / "vendor_prices.pkl").open("rb") as f:
        vendor = pickle.load(f)
    adjusted, raw = price_views(vendor)
    scores, coverage, peers = rebuild_scores(adjusted, universe, "2023-04-03", "2026-10-13")
    scores.to_parquet(out / "scores.parquet", index=False)
    coverage.to_csv(out / "coverage.csv", index=False)
    peers.to_csv(out / "peer_counts.csv", index=False)
    # Preserve the exact transformed execution bars and signal bars as tables.
    for label, data in [("signal_prices", adjusted), ("execution_prices", raw)]:
        frame = pd.concat([d.rename_axis("date").reset_index().assign(ticker=t) for t, d in sorted(data.items())])
        frame.to_parquet(out / f"{label}.parquet", index=False)
    joined = old.merge(scores, on=["date", "ticker"], how="outer", suffixes=("_old", "_new"), indicator=True)
    paired = joined[joined._merge == "both"]
    differences = {
        "rebuilt_rows": len(scores), "rebuilt_tickers": int(scores.ticker.nunique()),
        "original_rows": len(old), "paired_rows": len(paired),
        "original_rows_without_rebuilt_score": int((joined._merge == "left_only").sum()),
        "paired_score_changed": int((abs(paired.score_old - paired.score_new) > 1e-9).sum()),
        "paired_entry_gate_changed": int(((paired.score_old >= 9) != (paired.score_new >= 9)).sum()),
        "paired_exit_gate_changed": int(((paired.score_old < 5) != (paired.score_new < 5)).sum()),
        "rounding_alone_changed_rebuilt_score": int((abs(scores.score - scores.score_legacy_rounding) > 1e-9).sum()),
        "unscorable_tickers": coverage.loc[coverage.score_rows == 0, "ticker"].tolist(),
    }
    dump(out / "score_comparison.json", differences)
    paired[["date", "ticker", "score_old", "score_new"]].to_parquet(out / "paired_scores.parquet", index=False)
    print(f"Rebuilt {len(scores):,} scores; {len(paired):,} paired historical rows", flush=True)
    with (INPUTS / "review_prices.pkl").open("rb") as f:
        cached = pickle.load(f)
    legacy = load_legacy()
    all_results = {}
    for name, data in [("review_legacy", cached), ("matched_legacy", adjusted)]:
        all_results[name] = legacy_runs(legacy, data, old, windows, out, name)
        print(name, summarize_paths(all_results[name]), flush=True)
    # This must reproduce the previous chat, not just resemble its headlines.
    reproduction_errors = []
    for label, actual in all_results["review_legacy"].items():
        expected = (previous["fixed_windows"]["C5"][actual["start"]] if label.startswith("fixed_")
                    else previous["paths"]["C"]["runs"]["C5"]["metrics"][int(label.split("_")[1])])
        for key in ("total_return", "sharpe", "max_drawdown", "total_trades", "avg_hold", "win_rate"):
            if not np.isclose(actual[key], expected[key], rtol=1e-10, atol=1e-10):
                reproduction_errors.append({"window": label, "metric": key, "actual": actual[key], "expected": expected[key]})
    dump(out / "legacy_reproduction.json", {"compared_metrics": 24 * 6, "errors": reproduction_errors})
    if reproduction_errors:
        raise AssertionError("Preserved review rerun failed exact reproduction; inspect artifact")
    books = {"original": ScoreBook(old), "rebuilt": ScoreBook(scores)}
    prices = {"adjusted": PriceBook(adjusted), "as_traded": PriceBook(raw)}
    for name, (conf, capital, pv, sv) in scenarios.items():
        all_results[name] = {}
        additional = later_windows if name.startswith("baseline_") else []
        for label, start, end in windows + additional:
            result = simulate(books[sv], prices[pv], start, end, capital, conf)
            compressed(out / f"{name}__{label}.json.gz", result)
            all_results[name][label] = result["metrics"]
        print(name, summarize_paths(all_results[name]), flush=True)
    path_summary = {name: summarize_paths(results) for name, results in all_results.items()}
    dump(out / "metrics.json", all_results)
    dump(out / "path_summary.json", path_summary)
    # Matched, fully-funded adjusted buy-and-hold benchmarks, first close to last
    # close; risk-free convention and first cash return match baseline metrics.
    benchmarks = {}
    for label, start, end in windows + later_windows:
        benchmarks[label] = {}
        for ticker in ("SPY", "SMH"):
            close = adjusted[ticker].loc[start:end, "Close"]
            values = np.r_[1., close.to_numpy() / close.iloc[0]]
            ret = values[1:] / values[:-1] - 1 - .04 / 252
            benchmarks[label][ticker] = {
                "return_pct": float((values[-1] - 1) * 100),
                "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)),
                "max_drawdown": float((values / np.maximum.accumulate(values) - 1).min() * 100),
            }
    dump(out / "benchmarks.json", benchmarks)
    # Compare this checkout before/after execution. Historical state hashes are
    # retained as evidence, not as required versions of today's trading state.
    protected_after = snapshot(REPO)
    changes = [name for name in sorted(set(protected_before) | set(protected_after))
               if protected_before.get(name) != protected_after.get(name)]
    dump(out / "production_integrity.json", {"checked_files": len(protected_before),
         "changed": changes, "basis": "Current checkout immediately before/after this run"})
    if changes:
        raise AssertionError(f"Unexpected protected-file changes: {changes}")
    dump(out / "manifest.json", {str(f.relative_to(out)): sha(f) for f in sorted(out.rglob("*")) if f.is_file()})
    print(f"Completed offline study at {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
