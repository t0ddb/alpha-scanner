"""Complete the score-by-execution comparison without changing frozen rules.

python3 -B -m research.scheme_c.attribution --output research/scheme_c/results/attribution_v1
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pandas as pd

from .baseline import PriceBook, ScoreBook, Settings, simulate
from .run import HERE, INPUTS, compressed, dump, sha, summarize_paths


def main(output):
    study = HERE / "results/baseline_v2"
    out = Path(output).resolve()
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads((study / "manifest.json").read_text())
    for name in ("execution_prices.parquet", "scores.parquet", "settings.json", "path_summary.json", "metrics.json"):
        assert sha(study / name) == manifest[name], name
    assert sha(HERE / "baseline.py") == sha(study / "source/baseline.py")
    provenance = json.loads((INPUTS / "sources.json").read_text())
    assert sha(INPUTS / "original_scores.parquet") == provenance["sources"]["original_scores.parquet"]["sha256"]
    settings = json.loads((study / "settings.json").read_text())
    conf = Settings(**settings["scenarios"]["baseline_100k"]["settings"])
    capital = settings["scenarios"]["baseline_100k"]["capital"]
    windows = settings["windows"]
    out.mkdir(parents=True)
    dump(out / "settings.json", {
        "purpose": "Measure score-rebuild effect conditional on full execution constraints",
        "baseline_study": str(study), "settings": asdict(conf), "capital": capital,
        "windows": windows, "event_clock": settings["event_clock"],
        "original_scores_sha256": sha(INPUTS / "original_scores.parquet"),
        "execution_prices_sha256": sha(study / "execution_prices.parquet"),
        "baseline_code_sha256": sha(HERE / "baseline.py"),
    })
    shutil.copy2(Path(__file__), out / "attribution.py")
    frame = pd.read_parquet(study / "execution_prices.parquet")
    prices = PriceBook({t: g.drop(columns="ticker").set_index("date") for t, g in frame.groupby("ticker")})
    old = ScoreBook(pd.read_parquet(INPUTS / "original_scores.parquet"))
    # Ensure reading the archived transformed bars leaves the known baseline unchanged.
    label, start, end = windows[0]
    check = simulate(ScoreBook(pd.read_parquet(study / "scores.parquet")), prices, start, end, capital, conf)
    expected = json.loads((study / "metrics.json").read_text())["baseline_100k"][label]
    assert check["metrics"] == expected
    runs = {}
    for label, start, end in windows:
        result = simulate(old, prices, start, end, capital, conf)
        runs[label] = result["metrics"]
        compressed(out / f"original_scores_full_execution__{label}.json.gz", result)
    summary = summarize_paths(runs)
    prior = json.loads((study / "path_summary.json").read_text())
    a = prior["clock_original_scores"]["mean_return"]
    b = prior["clock_rebuilt_scores"]["mean_return"]
    c = summary["mean_return"]
    d = prior["baseline_100k"]["mean_return"]
    result = {
        "original_scores_full_execution": summary,
        "score_effect_fractional_market_pp": b-a,
        "score_effect_full_execution_pp": d-c,
        "execution_package_effect_old_scores_pp": c-a,
        "execution_package_effect_rebuilt_scores_pp": d-b,
        "score_execution_interaction_pp": (d-c)-(b-a),
        "control_reproduction": "First fixed-window baseline metrics match exactly",
        "caution": "Conditional effects on this selected sample, not isolated causal estimates of individual rules; the package includes raw share denominations, dividends, tradeability and sizing/funding constraints.",
    }
    dump(out / "metrics.json", runs)
    dump(out / "attribution.json", result)
    dump(out / "manifest.json", {f.name: sha(f) for f in sorted(out.iterdir()) if f.is_file()})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    main(parser.parse_args().output)
