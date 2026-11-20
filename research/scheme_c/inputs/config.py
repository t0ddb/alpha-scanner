from __future__ import annotations

"""
config.py — Loads and provides access to the watchlist configuration.

Usage:
    from config import load_config, get_all_tickers, get_tickers_by_sector, print_summary

    cfg = load_config()
    tickers = get_all_tickers(cfg)
    ai_tickers = get_tickers_by_sector(cfg, "ai_tech")
"""

import yaml
from pathlib import Path


def load_config(path: str = None) -> dict:
    """Load the YAML config file and return as a dictionary."""
    if path is None:
        path = Path(__file__).parent / "ticker_config.yaml"
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_all_tickers(cfg: dict) -> list[str]:
    """Return a flat list of every ticker symbol in the config."""
    tickers = []
    for sector_key, sector in cfg["sectors"].items():
        for sub_key, sub in sector["subsectors"].items():
            tickers.extend(sub["tickers"].keys())
    return tickers


def get_tickers_by_sector(cfg: dict, sector_key: str) -> list[str]:
    """Return all tickers for a given sector key (e.g., 'ai_tech', 'metals', 'crypto')."""
    sector = cfg["sectors"].get(sector_key)
    if not sector:
        raise ValueError(f"Unknown sector: {sector_key}. Available: {list(cfg['sectors'].keys())}")
    tickers = []
    for sub_key, sub in sector["subsectors"].items():
        tickers.extend(sub["tickers"].keys())
    return tickers


def get_tickers_by_subsector(cfg: dict, sector_key: str, subsector_key: str) -> list[str]:
    """Return tickers for a specific subsector."""
    sector = cfg["sectors"].get(sector_key)
    if not sector:
        raise ValueError(f"Unknown sector: {sector_key}")
    sub = sector["subsectors"].get(subsector_key)
    if not sub:
        raise ValueError(f"Unknown subsector: {subsector_key}. Available: {list(sector['subsectors'].keys())}")
    return list(sub["tickers"].keys())


def get_ticker_metadata(cfg: dict) -> dict:
    """
    Return a dict mapping each ticker to its metadata:
    {
        "NVDA": {
            "name": "NVIDIA",
            "sector": "ai_tech",
            "sector_name": "AI & Tech Capex Cycle",
            "subsector": "chips_compute",
            "subsector_name": "Chips — Compute"
        },
        ...
    }
    """
    metadata = {}
    for sector_key, sector in cfg["sectors"].items():
        for sub_key, sub in sector["subsectors"].items():
            for ticker, name in sub["tickers"].items():
                metadata[ticker] = {
                    "name": name,
                    "sector": sector_key,
                    "sector_name": sector["name"],
                    "subsector": sub_key,
                    "subsector_name": sub["name"],
                }
    return metadata


def get_indicator_config(cfg: dict) -> dict:
    """Return the indicator configuration section."""
    return cfg.get("indicators", {})


def get_scoring_config(cfg: dict) -> dict:
    """Return the scoring configuration section."""
    return cfg.get("scoring", {})


def print_summary(cfg: dict) -> None:
    """Print a readable summary of the watchlist."""
    all_tickers = get_all_tickers(cfg)
    print(f"\n{'='*60}")
    print(f"  BREAKOUT TRACKER — WATCHLIST SUMMARY")
    print(f"{'='*60}")
    print(f"  Benchmark: {cfg['benchmark']['ticker']} ({cfg['benchmark']['name']})")
    print(f"  Total tickers: {len(all_tickers)}")
    print(f"{'='*60}\n")

    for sector_key, sector in cfg["sectors"].items():
        sector_tickers = get_tickers_by_sector(cfg, sector_key)
        print(f"  {sector['name']} ({len(sector_tickers)} tickers)")
        print(f"  {'-'*50}")
        for sub_key, sub in sector["subsectors"].items():
            tickers = list(sub["tickers"].keys())
            print(f"    {sub['name']}: {', '.join(tickers)}")
        print()

    print(f"{'='*60}")
    print(f"  Indicators: {', '.join(cfg['scoring']['indicators'])}")
    print(f"  Max score: {cfg['scoring']['max_score']}")
    print(f"  Email threshold: {cfg['scoring']['email_threshold']}+")
    print(f"{'='*60}\n")


# -----------------------------------------------------------
# Quick test: run this file directly to see the summary
# -----------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    print_summary(cfg)
