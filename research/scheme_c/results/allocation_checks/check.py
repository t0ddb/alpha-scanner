"""Final artifact and aggregate investment-budget checks; run from repo root."""
from collections import defaultdict
import gzip
import json
from pathlib import Path

from research.scheme_c.run import dump, sha

results = Path("research/scheme_c/results")
study = results / "allocation_v2"
checks = results / "allocation_checks"
settings = json.loads((study / "settings.json").read_text())

verified = {}
for name in ("baseline_v2", "allocation_v2"):
    directory = results / name
    manifest = json.loads((directory / "manifest.json").read_text())
    for relative, digest in manifest.items():
        assert sha(directory / relative) == digest, relative
    verified[name] = len(manifest)

unchanged = []
for old in sorted((results / "allocation_v1").glob("*.json.gz")):
    assert sha(old) == sha(study / old.name), old.name
    unchanged.append(old.name)
for name in ("metrics.json", "diagnostics.json", "path_summary.json"):
    assert sha(results / "allocation_v1" / name) == sha(study / name), name
    unchanged.append(name)

orders_checked = 0
for name, scenario in settings["scenarios"].items():
    account = name.rsplit("_", 1)[1]
    assert scenario["settings"] == settings["scenarios"]["baseline_" + account]["settings"]
    cfg = scenario["settings"]
    for label, _, _ in settings["windows"]:
        result = json.loads(gzip.decompress((study / f"{name}__{label}.json.gz").read_bytes()))
        events = defaultdict(list)
        for event in result["events"]:
            events[event["date"]].append(event)
        for day in result["history"]:
            buys = [e for e in events[day["date"]] if e["event"] == "buy_submitted"]
            if not buys:
                continue
            exits = {e["ticker"] for e in events[day["date"]] if e["event"] == "sell_submitted"}
            proceeds = sum(h["qty"] * h["mark"] for t, h in day["holdings"].items() if t in exits)
            committed = max(0., day["equity"] - day["cash"] - proceeds)
            assert committed + sum(o["budget"] for o in buys) <= day["equity"] * (1 - cfg["cash_floor"]) + 1e-7
            assert sum(o["reserved"] for o in buys) <= day["cash"] + 1e-7
            orders_checked += len(buys)

before = json.loads((study / "protected_before.json").read_text())
for name, digest in before.items():
    assert sha(Path(name)) == digest, name

dump(checks / "final_checks.json", {
    "manifests_verified": verified,
    "v1_v2_identical_quantitative_artifacts": len(unchanged),
    "v1_v2_identical_files": unchanged,
    "identical_parameter_values_across_allocators": True,
    "accepted_orders_within_aggregate_estimated_investment_budget": orders_checked,
    "actual_cash_reservations_verified_again": orders_checked,
    "protected_files_still_unchanged": len(before),
})
print(f"Verified {len(unchanged)} identical artifacts and {orders_checked} order budgets.")
