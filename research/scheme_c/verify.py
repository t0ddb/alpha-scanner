"""Independent ledger reconciliation of a completed offline study.

python3 -B -m research.scheme_c.verify research/scheme_c/results/baseline_v1
Prints a JSON verification record; does not change the study artifacts.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def verify(directory):
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    for name, expected in manifest.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == expected, f"Changed study artifact: {name}"
    settings = json.loads((root / "settings.json").read_text())
    score_tables = {
        "rebuilt": pd.read_parquet(root / "scores.parquet"),
        "original": pd.read_parquet(Path(__file__).parent / "inputs/original_scores.parquet"),
    }
    score_maps = {}
    for name, df in score_tables.items():
        score_maps[name] = {(str(pd.Timestamp(d).date()), t): float(s)
                            for d, t, s in df[["date", "ticker", "score"]].itertuples(index=False, name=None)}
    price_maps = {}
    for name, filename in [("adjusted", "signal_prices.parquet"), ("as_traded", "execution_prices.parquet")]:
        df = pd.read_parquet(root / filename)
        if "split" not in df:
            df["split"] = 1.
        price_maps[name] = {(str(pd.Timestamp(d).date()), t): (float(o), float(h), float(lo), float(c), float(s))
                            for d, t, o, h, lo, c, s in df[["date", "ticker", "Open", "High", "Low", "Close", "split"]].itertuples(index=False, name=None)}
    metrics = json.loads((root / "metrics.json").read_text())
    ledgers, days, transactions = 0, 0, 0
    all_cash_nonnegative, all_caps_respected = True, True
    for name, scenario in settings["scenarios"].items():
        prices = price_maps[scenario["prices"]]
        for label, expected_metrics in metrics[name].items():
            result = json.loads(gzip.decompress((root / f"{name}__{label}.json.gz").read_bytes()))
            assert result["metrics"] == expected_metrics
            cash = float(scenario["capital"])
            receivables = 0.
            positions, orders = {}, {}
            events = defaultdict(list)
            for event in result["events"]:
                events[event["date"]].append(event)
            for snapshot in result["history"]:
                date = snapshot["date"]
                for event in events[date]:
                    kind, ticker = event["event"], event.get("ticker")
                    if kind.endswith("_submitted"):
                        orders[event["id"]] = dict(event)
                        if kind == "buy_submitted":
                            score = score_maps[scenario["scores"]][date, ticker]
                            assert score >= scenario["settings"]["entry_threshold"]
                    elif kind == "split":
                        positions[ticker] *= event["ratio"]
                        assert np.isclose(positions[ticker], event["qty"])
                    elif kind == "dividend_accrued":
                        assert ticker in positions
                        receivables += event["amount"]
                    elif kind == "order_split_adjustment":
                        order = orders[event["order_id"]]
                        order["limit"] = round(order["limit"] / prices[date, ticker][4], 2)
                    elif kind == "buy_fill":
                        order = orders[event["order_id"]]
                        assert order["signal_date"] < date
                        assert ticker not in positions
                        o, h, low, close, split = prices[date, ticker]
                        if event["phase"] == "open":
                            assert np.isclose(event["price"], o)
                        else:
                            assert np.isclose(event["price"], order["limit"])
                            assert low <= event["price"] <= h
                        if scenario["settings"]["execution_constraints"]:
                            assert event["price"] <= order["limit"] + 1e-7
                        positions[ticker] = event["qty"]
                        cash -= event["qty"] * event["price"] + event["fee"]
                        assert np.isclose(cash, event["cash_after"], atol=1e-7)
                        transactions += 1
                    elif kind == "sell_fill":
                        assert ticker in positions
                        assert np.isclose(positions.pop(ticker), event["qty"])
                        if event["order_id"] is not None:
                            assert orders[event["order_id"]]["signal_date"] < date
                        cash += event["qty"] * event["exit_price"] - event["fee"]
                        assert np.isclose(cash, event["cash_after"], atol=1e-7)
                        transactions += 1
                assert set(positions) == set(snapshot["holdings"])
                marked = 0.
                for ticker, qty in positions.items():
                    holding = snapshot["holdings"][ticker]
                    assert holding["mark_date"] == date  # no future or stale marks
                    assert np.isclose(holding["mark"], prices[date, ticker][3])
                    assert np.isclose(qty, holding["qty"])
                    marked += qty * holding["mark"]
                assert np.isclose(cash, snapshot["cash"], atol=1e-7)
                assert np.isclose(receivables, snapshot["dividend_receivable"], atol=1e-7)
                assert np.isclose(marked, snapshot["invested"], atol=1e-7)
                assert np.isclose(cash + marked + receivables, snapshot["equity"], atol=1e-7)
                all_cash_nonnegative &= cash >= -1e-7
                all_caps_respected &= len(positions) <= scenario["settings"]["max_positions"]
                days += 1
            assert np.isclose((result["history"][-1]["equity"] / scenario["capital"] - 1) * 100,
                              result["metrics"]["total_return"])
            ledgers += 1
    assert all_cash_nonnegative and all_caps_respected
    return {"artifact_hashes_verified": len(manifest), "ledgers_reconciled": ledgers,
            "daily_snapshots_reconciled": days, "fills_reconciled": transactions,
            "negative_cash_days": 0, "position_cap_violations": 0,
            "marks_on_exact_valuation_date": True, "queued_fills_strictly_after_signal": True}


if __name__ == "__main__":
    print(json.dumps(verify(sys.argv[1]), indent=2))
