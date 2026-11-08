"""One-change cash-allocation experiment on the frozen corrected engine.

python3 -B -m research.scheme_c.allocation --output research/scheme_c/results/allocation_v1

Loads separate engine instances from the baseline's archived source. Only the
challenger's sizing function changes: remove division by remaining slots from
the estimate-time floor budget. No production module, broker or database use.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from .run import HERE, INPUTS, REPO, compressed, dump, sha, summarize_paths
from .verify import verify

STUDY = HERE / "results/baseline_v2"


def sequential_sizer(original):
    """Preserve every sizing rule except budget / slots -> budget.

    Passing one to the original denominator preserves its minimum gate, integer
    rounding, +1-share flex, projected-cash check and aggregate 95% bound.
    The engine's actual-cash / limit-price acceptance check remains unchanged.
    """
    def size(equity, projected_cash, committed, slots, price, cfg):
        return original(equity, projected_cash, committed, 1, price, cfg)
    return size


def load_engine(name, allocator, source=None):
    if allocator not in {"baseline", "sequential"}:
        raise ValueError("Unknown allocator")
    source = source or STUDY / "source/baseline.py"
    spec = importlib.util.spec_from_file_location(name, source)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[name] = engine  # dataclasses resolve their defining module here
    spec.loader.exec_module(engine)
    if allocator == "sequential":
        engine.size_order = sequential_sizer(engine.size_order)
    return engine


def diagnostics(result):
    history = result["history"]
    contribution = defaultdict(float)
    for event in result["events"]:
        kind = event["event"]
        if kind == "buy_fill":
            contribution[event["ticker"]] -= event["qty"] * event["price"] + event["fee"]
        elif kind == "sell_fill":
            contribution[event["ticker"]] += event["qty"] * event["exit_price"] - event["fee"]
        elif kind == "dividend_accrued":
            contribution[event["ticker"]] += event["amount"]
    for ticker, holding in history[-1]["holdings"].items():
        contribution[ticker] += holding["qty"] * holding["mark"]
    assert np.isclose(sum(contribution.values()),
                      result["metrics"]["final_equity"] - result["metrics"]["capital"])
    return {
        "mean_cash_pct": float(np.mean([h["cash_fraction"] for h in history]) * 100),
        "mean_invested_pct": float(np.mean([h["invested"] / h["equity"] for h in history]) * 100),
        "mean_positions": float(np.mean([h["positions"] for h in history])),
        "contribution_dollars": dict(sorted(contribution.items(), key=lambda p: -p[1])),
    }


def verify_acceptance(result):
    """Check daily submission cash reservations and unchanged acceptance limits."""
    cfg = result["settings"]
    days = {h["date"]: h for h in result["history"]}
    orders = defaultdict(list)
    for event in result["events"]:
        if event["event"] == "buy_submitted":
            orders[event["date"]].append(event)
    checked = 0
    for day, batch in orders.items():
        assert len(batch) <= cfg["max_entries_per_day"]
        assert sum(o["reserved"] for o in batch) <= days[day]["cash"] + 1e-7
        priorities = [(-o["score"], o["ticker"]) for o in batch]
        assert priorities == sorted(priorities)
        for order in batch:
            assert order["qty"] > 0 and int(order["qty"]) == order["qty"]
            assert order["budget"] <= days[day]["equity"] * (cfg["position_pct"] + cfg["flex_pct"]) + 1e-7
            assert np.isclose(order["reserved"], order["qty"] * order["limit"] *
                              (1 + cfg["friction_bps"] / 10_000))
            checked += 1
    return checked


def protected_snapshot():
    # Hash the same production/research root files and workflow paths protected
    # by the original study, using today's state as the starting point. Reading
    # SQLite bytes to hash them does not open a database connection.
    paths = json.loads((INPUTS / "protected_before.json").read_text())
    return {name: sha(REPO / name) for name in sorted(paths)}


def main(output):
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    manifest = json.loads((STUDY / "manifest.json").read_text())
    for name, digest in manifest.items():
        assert sha(STUDY / name) == digest, f"Changed frozen baseline artifact: {name}"
    previous = json.loads((STUDY / "settings.json").read_text())
    expected = json.loads((STUDY / "metrics.json").read_text())
    windows = previous["windows"] + previous["additional_windows"]
    before = protected_snapshot()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    out.mkdir(parents=True)
    scenarios = {}
    for account in ("5k", "100k"):
        base = previous["scenarios"][f"baseline_{account}"]
        for allocator in ("baseline", "sequential"):
            scenarios[f"{allocator}_{account}"] = {**base, "allocator": allocator}
    settings = {
        "study": "Scheme C single-change cash-allocation comparison",
        "study_date": "2026-10-24", "parent_study": str(STUDY),
        "change": "floor budget / max(1, remaining slots) becomes floor budget",
        "unchanged": "Every strategy parameter, minimum, rounding/flex, priority, cash reservation, fill, stop and accounting rule",
        "hypothesis": "Sequential funding can reduce minimum-size skips and idle cash; return and risk effects are unknown before this run",
        "selection": "One predefined challenger; no parameter search; report every window at both capitals",
        "floor_limit": "Existing estimate-time floor, not a guaranteed closing cash ratio; unchanged flex and limit gaps can produce breaches",
        "funding_limit": "No borrowing or projected-sale funding; an unaffordable limit reservation is still rejected, not resized",
        "windows": windows, "scenarios": scenarios,
        "event_clock": previous["event_clock"], "universe": previous["universe"],
        "universe_bias": previous["universe_bias"], "git_head": head,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "inputs": {name: sha(STUDY / name) for name in
                   ("scores.parquet", "execution_prices.parquet", "signal_prices.parquet",
                    "source/baseline.py", "settings.json", "metrics.json", "manifest.json")},
        "baseline_reproduction": "Require all 56 control ledgers to match the frozen baseline exactly",
    }
    # Freeze settings and source before running any historical comparisons.
    dump(out / "settings.json", settings)
    dump(out / "protected_before.json", before)
    source = out / "source"
    source.mkdir()
    for src in (Path(__file__), HERE / "run.py", HERE / "verify.py",
                STUDY / "source/baseline.py", REPO / "tests/test_scheme_c_allocation.py"):
        shutil.copy2(src, source / src.name)
    for name in ("scores.parquet", "execution_prices.parquet", "signal_prices.parquet"):
        shutil.copy2(STUDY / name, out / name)
    frame = pd.read_parquet(out / "execution_prices.parquet")
    prices = {t: g.drop(columns="ticker").set_index("date") for t, g in frame.groupby("ticker")}
    score_frame = pd.read_parquet(out / "scores.parquet")
    engines = {kind: load_engine("scheme_c_allocation_" + kind, kind, source / "baseline.py")
               for kind in ("baseline", "sequential")}
    books = {kind: (engine.ScoreBook(score_frame), engine.PriceBook(prices))
             for kind, engine in engines.items()}
    metrics, detail, controls = {}, {}, []
    submitted = 0
    for name, scenario in scenarios.items():
        kind = scenario["allocator"]
        engine = engines[kind]
        cfg = engine.Settings(**scenario["settings"])
        metrics[name], detail[name] = {}, {}
        for label, start, end in windows:
            result = engine.simulate(*books[kind], start, end, scenario["capital"], cfg)
            if kind == "baseline":
                assert result["metrics"] == expected[name][label], (name, label)
                frozen = json.loads(gzip.decompress((STUDY / f"{name}__{label}.json.gz").read_bytes()))
                assert result == frozen, f"Control ledger changed: {name}/{label}"
                controls.append(f"{name}/{label}")
            submitted += verify_acceptance(result)
            metrics[name][label] = result["metrics"]
            detail[name][label] = diagnostics(result)
            compressed(out / f"{name}__{label}.json.gz", result)
        print(name, summarize_paths(metrics[name]), flush=True)
    dump(out / "metrics.json", metrics)
    dump(out / "diagnostics.json", detail)
    summaries = {name: summarize_paths(runs) for name, runs in metrics.items()}
    dump(out / "path_summary.json", summaries)
    comparisons = {}
    for account in ("5k", "100k"):
        base, alternative = f"baseline_{account}", f"sequential_{account}"
        paired = {}
        for label, _, _ in windows:
            a, b = metrics[base][label], metrics[alternative][label]
            paired[label] = {key + "_delta": b[key] - a[key] for key in
                             ("total_return", "sharpe", "max_drawdown", "final_equity",
                              "fees", "closed_trades", "cash_floor_breach_days")}
            paired[label].update({key + "_delta": detail[alternative][label][key] - detail[base][label][key]
                                 for key in ("mean_cash_pct", "mean_invested_pct", "mean_positions")})
        paths = [v for k, v in paired.items() if k.startswith("path_")]
        comparisons[account] = {
            "windows": paired,
            "path_mean_return_delta_pp": float(np.mean([p["total_return_delta"] for p in paths])),
            "path_mean_sharpe_delta": float(np.mean([p["sharpe_delta"] for p in paths])),
            "path_mean_cash_delta_pp": float(np.mean([p["mean_cash_pct_delta"] for p in paths])),
            "paths_higher_return": int(sum(p["total_return_delta"] > 0 for p in paths)),
            "paths_higher_sharpe": int(sum(p["sharpe_delta"] > 0 for p in paths)),
            "paths_less_severe_drawdown": int(sum(p["max_drawdown_delta"] > 0 for p in paths)),
        }
    dump(out / "comparison.json", comparisons)
    dump(out / "control_reproduction.json", {"exact_full_ledger_matches": len(controls), "runs": controls})
    after = protected_snapshot()
    assert before == after, "Protected files changed during experiment"
    assert head == subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dump(out / "production_integrity.json", {"checked_files": len(before), "changed_files": [], "git_head_unchanged": True})

    def save_manifest():
        dump(out / "manifest.json", {str(f.relative_to(out)): sha(f) for f in sorted(out.rglob("*"))
                                     if f.is_file() and f.name != "manifest.json"})
    save_manifest()
    print("Reconciling all 112 ledgers independently...", flush=True)
    check = verify(out)
    check["submission_reservations_and_limits_checked"] = submitted
    check["baseline_full_ledger_matches"] = len(controls)
    dump(out / "verification.json", check)
    save_manifest()
    print(json.dumps(check, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
