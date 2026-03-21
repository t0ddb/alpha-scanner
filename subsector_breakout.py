from __future__ import annotations

"""
subsector_breakout.py — Subsector breakout detection engine.

Aggregates individual stock scores into subsector-level metrics,
computes derived signals (z-scores, acceleration), and runs a
state machine to detect emerging → confirmed → fading breakouts.

Usage:
    from subsector_breakout import run_breakout_detection

    summary = run_breakout_detection(results, cfg)
    # summary = {
    #     "new_breakouts": [...],
    #     "confirmed": [...],
    #     "fading": [...],
    #     "all_states": {...},
    # }
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime

from config import load_config, get_ticker_metadata
from subsector_store import (
    init_db,
    upsert_daily,
    get_history,
    get_breakout_states,
    update_breakout_state,
    get_subsector_list,
)


# =============================================================
# COMPUTE SUBSECTOR METRICS from score_all() results
# =============================================================
def compute_subsector_metrics(
    results: list[dict],
    cfg: dict,
    breadth_threshold: float = 7.0,
) -> list[dict]:
    """
    Aggregate individual ticker scores into subsector-level metrics.

    Args:
        results: Output of score_all() — list of dicts with ticker, score, subsector, etc.
        cfg: Full config dict
        breadth_threshold: Score threshold to count a ticker as "hot"

    Returns:
        List of dicts ready for upsert_daily()
    """
    # Get breakout detection config
    bd_cfg = cfg.get("breakout_detection", {})
    threshold = bd_cfg.get("breadth_threshold", breadth_threshold)

    # Build subsector key mapping from config
    subsector_meta = {}
    for sector_key, sector in cfg["sectors"].items():
        for sub_key, sub in sector["subsectors"].items():
            subsector_meta[sub["name"]] = {
                "subsector_key": sub_key,
                "sector": sector["name"],
                "sector_key": sector_key,
            }

    # Group results by subsector display name
    grouped = {}
    for r in results:
        sub_name = r.get("subsector", "")
        if not sub_name:
            continue
        if sub_name not in grouped:
            grouped[sub_name] = []
        grouped[sub_name].append(r)

    records = []
    for sub_name, tickers in grouped.items():
        meta = subsector_meta.get(sub_name, {})
        scores = [t["score"] for t in tickers]
        hot_tickers = [t for t in tickers if t["score"] >= threshold]

        ticker_scores = {t["ticker"]: t["score"] for t in tickers}

        records.append({
            "subsector": meta.get("subsector_key", sub_name),
            "subsector_name": sub_name,
            "sector": meta.get("sector", ""),
            "ticker_count": len(tickers),
            "avg_score": round(np.mean(scores), 2) if scores else 0,
            "max_score": round(max(scores), 1) if scores else 0,
            "breadth": round(len(hot_tickers) / len(tickers), 3) if tickers else 0,
            "hot_count": len(hot_tickers),
            "ticker_scores": ticker_scores,
        })

    return records


# =============================================================
# COMPUTE DERIVED METRICS from history
# =============================================================
def compute_derived_metrics(
    history_df: pd.DataFrame,
    lookback_days: int = 60,
) -> dict:
    """
    Compute derived metrics from subsector history.

    Args:
        history_df: DataFrame from get_history() for one subsector
        lookback_days: Window for rolling mean/std

    Returns:
        Dict with score_change_5d, breadth_change_5d, z_score, acceleration, etc.
    """
    result = {
        "score_change_5d": np.nan,
        "score_change_10d": np.nan,
        "score_change_21d": np.nan,
        "breadth_change_5d": np.nan,
        "breadth_change_10d": np.nan,
        "breadth_change_21d": np.nan,
        "z_score": np.nan,
        "breadth_z_score": np.nan,
        "score_acceleration": np.nan,
        "has_sufficient_history": False,
    }

    if history_df.empty or len(history_df) < 3:
        return result

    scores = history_df["avg_score"].values
    breadths = history_df["breadth"].values
    current_score = scores[-1]
    current_breadth = breadths[-1]

    # Score changes over different windows
    for window, key in [(5, "5d"), (10, "10d"), (21, "21d")]:
        if len(scores) > window:
            result[f"score_change_{key}"] = round(current_score - scores[-window - 1], 2)
            result[f"breadth_change_{key}"] = round(current_breadth - breadths[-window - 1], 3)

    # Z-scores (need sufficient history)
    if len(scores) >= 10:
        result["has_sufficient_history"] = True

        # Use up to lookback_days for rolling stats
        window = min(len(scores) - 1, lookback_days)
        if window >= 5:
            historical = scores[-window - 1:-1]  # exclude current
            mean_score = np.mean(historical)
            std_score = np.std(historical)

            if std_score > 0:
                result["z_score"] = round((current_score - mean_score) / std_score, 2)
            else:
                result["z_score"] = 0.0

            historical_breadth = breadths[-window - 1:-1]
            mean_breadth = np.mean(historical_breadth)
            std_breadth = np.std(historical_breadth)

            if std_breadth > 0:
                result["breadth_z_score"] = round(
                    (current_breadth - mean_breadth) / std_breadth, 2
                )
            else:
                result["breadth_z_score"] = 0.0

    # Acceleration (2nd derivative of score)
    if len(scores) >= 3:
        # First derivative (velocity): recent change
        v1 = scores[-1] - scores[-2]
        v0 = scores[-2] - scores[-3]
        result["score_acceleration"] = round(v1 - v0, 2)

    return result


# =============================================================
# STATE MACHINE: Detect breakout state transitions
# =============================================================
def detect_breakout_state(
    subsector: str,
    derived: dict,
    current: dict,
    prev_state: dict | None,
    cfg: dict,
    current_date: str = None,
) -> dict:
    """
    Run breakout state machine for one subsector.

    State transitions:
        quiet → emerging:     breadth >= trigger AND (z_score > z_trigger OR cold start)
        emerging → confirmed: consecutive_hot >= confirm_days
        confirmed → fading:   breadth < trigger OR score declining 2+ readings
        fading → quiet:       breadth < 0.3 OR consecutive_cool >= fade_days
        fading → confirmed:   breadth recovers AND acceleration positive

    Args:
        subsector: Subsector key
        derived: Output of compute_derived_metrics()
        current: Current day's subsector metric record
        prev_state: Previous state dict from DB (or None for first run)
        cfg: Full config dict
        current_date: Date string (defaults to today)

    Returns:
        New state dict ready for update_breakout_state()
    """
    bd_cfg = cfg.get("breakout_detection", {})
    breadth_trigger = bd_cfg.get("breadth_trigger", 0.5)
    z_trigger = bd_cfg.get("z_score_trigger", 1.0)
    confirm_days = bd_cfg.get("confirm_days", 3)
    fade_cool_days = bd_cfg.get("fade_cool_days", 5)

    date = current_date or datetime.now().strftime("%Y-%m-%d")
    breadth = current.get("breadth", 0)
    z_score = derived.get("z_score", np.nan)
    acceleration = derived.get("score_acceleration", np.nan)
    has_history = derived.get("has_sufficient_history", False)

    # Is this subsector "hot" today?
    is_hot = breadth >= breadth_trigger

    # Initialize from previous state
    if prev_state:
        status = prev_state.get("status", "quiet")
        consecutive_hot = prev_state.get("consecutive_hot", 0)
        consecutive_cool = prev_state.get("consecutive_cool", 0)
        peak_avg_score = prev_state.get("peak_avg_score", 0)
        peak_breadth = prev_state.get("peak_breadth", 0)
        status_since = prev_state.get("status_since", date)
    else:
        status = "quiet"
        consecutive_hot = 0
        consecutive_cool = 0
        peak_avg_score = 0
        peak_breadth = 0
        status_since = date

    # Update counters
    if is_hot:
        consecutive_hot += 1
        consecutive_cool = 0
    else:
        consecutive_cool += 1
        consecutive_hot = 0

    # Track peaks
    peak_avg_score = max(peak_avg_score, current.get("avg_score", 0))
    peak_breadth = max(peak_breadth, breadth)

    # Determine new status
    new_status = status

    if status == "quiet":
        # quiet → emerging: breadth high AND (z-score elevated OR cold start)
        z_ok = (not has_history) or (not np.isnan(z_score) and z_score > z_trigger)
        acc_ok = np.isnan(acceleration) or acceleration >= 0

        if is_hot and z_ok and acc_ok:
            new_status = "emerging"
            status_since = date
            peak_avg_score = current.get("avg_score", 0)
            peak_breadth = breadth

    elif status == "emerging":
        if consecutive_hot >= confirm_days:
            # emerging → confirmed
            new_status = "confirmed"
            status_since = date
        elif consecutive_cool >= 2:
            # Failed to confirm — back to quiet
            new_status = "quiet"
            status_since = date
            peak_avg_score = 0
            peak_breadth = 0

    elif status == "confirmed":
        # confirmed → fading: breadth drops OR score declining
        score_declining = (
            not np.isnan(derived.get("score_change_5d", np.nan))
            and derived["score_change_5d"] < -1.0
        )

        if not is_hot or score_declining:
            new_status = "fading"
            status_since = date

    elif status == "fading":
        if breadth < 0.3 or consecutive_cool >= fade_cool_days:
            # fading → quiet
            new_status = "quiet"
            status_since = date
            peak_avg_score = 0
            peak_breadth = 0
        elif is_hot and not np.isnan(acceleration) and acceleration > 0:
            # fading → confirmed (recovery)
            new_status = "confirmed"
            status_since = date

    return {
        "subsector": subsector,
        "status": new_status,
        "status_since": status_since,
        "consecutive_hot": consecutive_hot,
        "consecutive_cool": consecutive_cool,
        "peak_avg_score": round(peak_avg_score, 2),
        "peak_breadth": round(peak_breadth, 3),
        "updated_at": date,
    }


# =============================================================
# ORCHESTRATOR: Run full breakout detection pipeline
# =============================================================
def run_breakout_detection(
    results: list[dict],
    cfg: dict,
    date: str = None,
    conn=None,
) -> dict:
    """
    Full breakout detection pipeline — called once per daily run.

    1. Compute today's subsector metrics
    2. Persist to SQLite
    3. Load history per subsector
    4. Compute derived metrics
    5. Run state machine
    6. Persist new states
    7. Return summary

    Args:
        results: Output of score_all()
        cfg: Full config dict
        date: Date string (defaults to today)
        conn: SQLite connection (creates one if None)

    Returns:
        {
            "new_breakouts": [subsector_keys that just went emerging],
            "confirmed": [subsector_keys confirmed],
            "fading": [subsector_keys fading],
            "all_states": {subsector: state_dict},
            "metrics": [subsector metric records],
        }
    """
    bd_cfg = cfg.get("breakout_detection", {})
    lookback = bd_cfg.get("lookback_days", 60)
    date = date or datetime.now().strftime("%Y-%m-%d")

    own_conn = conn is None
    if own_conn:
        conn = init_db()

    try:
        # Step 1: Compute today's subsector metrics
        metrics = compute_subsector_metrics(results, cfg)

        # Step 2: Persist to SQLite
        upsert_daily(conn, date, metrics)

        # Step 3-6: For each subsector, load history, derive, detect, persist
        prev_states = get_breakout_states(conn)

        new_breakouts = []
        confirmed = []
        reaccelerations = []
        fading = []
        all_states = {}
        all_derived = {}

        for record in metrics:
            sub_key = record["subsector"]

            # Load history
            history = get_history(conn, sub_key, days=lookback + 30)

            # Compute derived metrics
            derived = compute_derived_metrics(history, lookback_days=lookback)
            all_derived[sub_key] = derived

            # Get previous state
            prev = prev_states.get(sub_key)

            # Detect state transition
            new_state = detect_breakout_state(
                sub_key, derived, record, prev, cfg, current_date=date
            )

            # Track transitions
            prev_status = prev["status"] if prev else "quiet"
            if new_state["status"] == "emerging" and prev_status == "quiet":
                new_breakouts.append(sub_key)
            if new_state["status"] == "confirmed" and prev_status == "emerging":
                confirmed.append(sub_key)
            if new_state["status"] == "confirmed" and prev_status == "fading":
                reaccelerations.append(sub_key)
            if new_state["status"] == "fading":
                fading.append(sub_key)

            # Persist
            update_breakout_state(conn, sub_key, new_state)
            all_states[sub_key] = new_state

        return {
            "new_breakouts": new_breakouts,
            "confirmed": confirmed,
            "reaccelerations": reaccelerations,
            "fading": fading,
            "all_states": all_states,
            "all_derived": all_derived,
            "metrics": metrics,
            "date": date,
        }

    finally:
        if own_conn:
            conn.close()


# =============================================================
# Utility: Pretty-print breakout summary
# =============================================================
def print_breakout_summary(summary: dict, cfg: dict) -> None:
    """Print a formatted breakout detection summary."""
    # Build subsector name lookup from config
    name_lookup = {}
    for sector_key, sector in cfg["sectors"].items():
        for sub_key, sub in sector["subsectors"].items():
            name_lookup[sub_key] = sub["name"]

    # Display labels for each status
    STATUS_LABELS = {
        "quiet": "Quiet",
        "emerging": "Emerging",
        "confirmed": "Confirmed Breakout",
        "fading": "Fading",
    }

    print(f"\n{'='*80}")
    print(f"  SUBSECTOR BREAKOUT DETECTION — {summary.get('date', 'today')}")
    print(f"{'='*80}\n")

    if summary.get("reaccelerations"):
        print("  ⚡ RE-ACCELERATION (fading → confirmed — highest alpha signal):")
        for sub in summary["reaccelerations"]:
            name = name_lookup.get(sub, sub)
            state = summary["all_states"].get(sub, {})
            print(f"     → {name} (peak score: {state.get('peak_avg_score', 0):.1f}, "
                  f"breadth: {state.get('peak_breadth', 0):.0%})")
        print()

    if summary["confirmed"]:
        print("  🔥 CONFIRMED BREAKOUT (emerging → confirmed — actionable signal):")
        for sub in summary["confirmed"]:
            name = name_lookup.get(sub, sub)
            state = summary["all_states"].get(sub, {})
            print(f"     → {name} (peak score: {state.get('peak_avg_score', 0):.1f}, "
                  f"breadth: {state.get('peak_breadth', 0):.0%})")
        print()

    if summary["new_breakouts"]:
        print("  👀 EMERGING (watching — not yet confirmed):")
        for sub in summary["new_breakouts"]:
            name = name_lookup.get(sub, sub)
            state = summary["all_states"].get(sub, {})
            print(f"     → {name} (breadth: {state.get('peak_breadth', 0):.0%})")
        print()

    if summary["fading"]:
        print("  📉 FADING:")
        for sub in summary["fading"]:
            name = name_lookup.get(sub, sub)
            print(f"     → {name}")
        print()

    quiet_count = sum(
        1 for s in summary["all_states"].values() if s["status"] == "quiet"
    )
    reaccel_count = len(summary.get("reaccelerations", []))
    print(f"  Summary: {len(summary['new_breakouts'])} emerging, "
          f"{len(summary['confirmed'])} confirmed breakout, "
          f"{reaccel_count} revival, "
          f"{len(summary['fading'])} fading, "
          f"{quiet_count} quiet\n")

    # Show all subsector metrics sorted by score
    print(f"  {'Subsector':<35} {'Avg':>5} {'Max':>5} {'Breadth':>8} {'Hot':>4} {'Status':>20}")
    print(f"  {'─'*82}")

    metrics_with_state = []
    for m in summary.get("metrics", []):
        state = summary["all_states"].get(m["subsector"], {})
        m["_status"] = state.get("status", "quiet")
        # Track if this is a revival (confirmed via fading path)
        m["_is_reaccel"] = m["subsector"] in summary.get("reaccelerations", [])
        metrics_with_state.append(m)

    metrics_with_state.sort(key=lambda x: -x["avg_score"])

    status_emoji = {"quiet": "  ", "emerging": "👀", "confirmed": "🔥", "fading": "📉"}

    for m in metrics_with_state:
        status = m["_status"]
        emoji = status_emoji.get(status, "  ")
        label = STATUS_LABELS.get(status, status)
        if status == "confirmed" and m["_is_reaccel"]:
            emoji = "⚡"
            label = "Revival"
        print(f"  {m['subsector_name']:<35} {m['avg_score']:>5.1f} {m['max_score']:>5.1f} "
              f"{m['breadth']:>7.0%} {m['hot_count']:>4} {emoji} {label:>18}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    from data_fetcher import fetch_all
    from indicators import score_all

    cfg = load_config()

    print("=" * 80)
    print("  SUBSECTOR BREAKOUT DETECTION — SMOKE TEST")
    print("=" * 80)
    print("\n  Fetching data...")

    data = fetch_all(cfg, period="1y", verbose=False)
    print(f"  Fetched {len(data)} tickers.")

    print("  Scoring (three-tier)...")
    results = score_all(data, cfg)
    print(f"  Scored {len(results)} tickers.")

    print("  Running breakout detection...")
    summary = run_breakout_detection(results, cfg)
    print_breakout_summary(summary, cfg)
