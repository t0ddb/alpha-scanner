from __future__ import annotations

"""
sequence_overlay.py — Scheme I+ Layer 2 (sequence-pattern adjustments).

Adds positive bonuses and negative penalties to the Layer 1 base score
based on the order in which indicators fired, ending today.

This module exposes pure functions that operate on a `streaks` dict
(indicator label → consecutive-firing streak length, ending today).
The CALLER is responsible for computing streaks from history:
  - For backtesting: compute across the full historical parquet
  - For live trading: query last N days from breakout_tracker.db

Streaks should be in TRADING DAYS. An indicator with streak=5 has been
firing for 5 consecutive trading days ending today. Streak=0 means the
indicator is not currently firing.

The "fired" definitions for v2 streaks (scoring-aligned):
  - RS:  percentile >= 90
  - HL:  consecutive_higher_lows >= 4
  - ICH: above_cloud AND cloud_bullish (the existing trigger)
  - ROC: roc > 5%
  - CMF: cmf > 0.05
  - ATR: percentile >= 80
  - DTF: cond_a OR cond_b (existing trigger)

See docs/SCHEME_I_PLUS_PROPOSAL.md for full design rationale.
"""

# ─── Indicator labels (must match streaks dict keys) ──────────────
LABELS = ["rs", "ich", "hl", "cmf", "roc", "atr", "dtf"]

INDICATOR_TYPE = {
    "rs": "TREND", "hl": "TREND", "ich": "TREND",
    "roc": "MOMENTUM", "dtf": "MOMENTUM",
    "cmf": "VOL", "atr": "VOL",
}


def ordered_firers(streaks: dict[str, int]) -> list[str]:
    """Return labels of currently-firing indicators sorted by streak length
    (longest first; ties broken alphabetically for determinism).
    Indicators with streak=0 are not currently firing and excluded."""
    firing = [(lbl, streaks.get(lbl, 0)) for lbl in LABELS]
    firing = [(lbl, s) for lbl, s in firing if s > 0]
    firing.sort(key=lambda x: (-x[1], x[0]))
    return [lbl for lbl, _ in firing]


def first_two_distinct_types(seq: list[str]) -> tuple:
    """Return the first two distinct indicator TYPES in firing order.
    E.g., for seq=['rs', 'ich', 'roc'], returns ('TREND', 'MOMENTUM')."""
    seen = []
    for indicator in seq:
        t = INDICATOR_TYPE[indicator]
        if not seen or seen[-1] != t:
            seen.append(t)
        if len(seen) == 2:
            break
    return tuple(seen)


def n_indicators_leading_ich(streaks: dict[str, int]) -> int:
    """Count how many non-Ich indicators have streak > Ich's streak.
    Returns -1 if Ich is not currently firing."""
    ich_streak = streaks.get("ich", 0)
    if ich_streak <= 0:
        return -1
    return sum(1 for lbl in LABELS
               if lbl != "ich" and streaks.get(lbl, 0) > ich_streak)


def first_three_pattern(seq: list[str]) -> str | None:
    return f"{seq[0]}→{seq[1]}→{seq[2]}" if len(seq) >= 3 else None


def first_two_distinct_types_str(seq: list[str]) -> str:
    t = first_two_distinct_types(seq)
    return "→".join(t) if len(t) == 2 else ""


def compute_sequence_features(streaks: dict[str, int]) -> dict:
    """Compute all Layer 2 sequence features for a single (ticker, date) row.

    Returns a dict with boolean flags for each pattern that fires.
    """
    seq = ordered_firers(streaks)
    if len(seq) == 0:
        return {"_no_indicators_firing": True}

    f = {}
    f["seq"] = seq
    f["first_firer"] = seq[0]
    f["last_firer"] = seq[-1]
    f["last_two"] = f"{seq[-2]}→{seq[-1]}" if len(seq) >= 2 else None
    f["first_two_pair"] = f"{seq[0]}→{seq[1]}" if len(seq) >= 2 else None
    f["first_three_pat"] = first_three_pattern(seq)
    f["first_two_types_str"] = first_two_distinct_types_str(seq)
    f["n_led_ich"] = n_indicators_leading_ich(streaks)

    return f


# ──────────────────────────────────────────────────────────────────
# PATH C v2 — Mean-return-based Layer 2
# ──────────────────────────────────────────────────────────────────
# Replaces v1.1's win-rate-based bonuses/penalties with magnitudes
# derived from MEAN forward return per pattern (regime-filtered to
# 2025-05-01+). See audit_sequence_total_return.txt for source data.
#
# Key principle: heavy-tail captures (high mean even with lower win
# rate) get rewarded. Penalties only for patterns with truly negative
# or near-zero mean returns.
#
# Magnitude cap: |adjustment| <= 0.5 per pattern. Total Layer 2 range
# typically ±1.5; max stack-up around ±2.5 in pathological cases.

# Patterns sorted by mean-return Δ vs population mean (44.2% in regime).
# Format: pattern_key, magnitude, condition_lambda, tag_string
LAYER_2_RULES = [
    # ─── BONUSES (mean return well above population, n>=50 unless flagged) ──
    # First-3 patterns (highest mean returns)
    ("BONUS", "first_three_pat", "roc→atr→ich", +0.5, "first-3 mean +140% (n=67)"),
    ("BONUS", "first_three_pat", "ich→rs→dtf",  +0.5, "first-3 mean +115% (n=79)"),
    ("BONUS", "first_three_pat", "ich→cmf→dtf", +0.4, "first-3 mean +75% (n=60)"),

    # First-2 pairs
    ("BONUS", "first_two_pair", "roc→atr", +0.5, "first-2 mean +95% (n=116)"),
    ("BONUS", "first_two_pair", "ich→rs",  +0.5, "first-2 mean +88% (n=144)"),
    ("BONUS", "first_two_pair", "atr→ich", +0.4, "first-2 mean +93% (n=66)"),

    # Last-2 pairs
    ("BONUS", "last_two", "dtf→roc", +0.5, "last-2 mean +118% (n=55)"),
    ("BONUS", "last_two", "cmf→rs",  +0.4, "last-2 mean +72% (n=58)"),
    ("BONUS", "last_two", "dtf→rs",  +0.3, "last-2 mean +59% (n=277)"),

    # First-firer
    ("BONUS", "first_firer", "atr", +0.4, "first-firer mean +78% (n=126)"),
    ("BONUS", "first_firer", "dtf", +0.4, "first-firer mean +61%, 95% win (n=77)"),
    ("BONUS", "first_firer", "rs",  +0.3, "first-firer mean +59% (n=138)"),

    # Lead-lag (number of indicators leading Ich)
    ("BONUS", "n_led_ich", 2, +0.5, "2-led-Ich mean +76% (n=218)"),
    ("BONUS", "n_led_ich", 1, +0.3, "1-led-Ich mean +50% (n=546)"),

    # Type pattern
    ("BONUS", "first_two_types_str", "VOL→TREND",     +0.3, "type mean +64% (n=124)"),
    ("BONUS", "first_two_types_str", "MOMENTUM→VOL",  +0.3, "type mean +58% (n=255)"),

    # ─── PENALTIES (mean return well below population) ──
    ("PENALTY", "first_three_pat", "ich→roc→cmf", -0.5, "first-3 mean -5% (n=64) — only neg-mean pattern"),
    ("PENALTY", "first_three_pat", "ich→roc→rs",  -0.4, "first-3 mean +0.3% (n=74)"),
    ("PENALTY", "first_three_pat", "ich→dtf→roc", -0.3, "first-3 mean +12% (n=81)"),

    ("PENALTY", "first_two_pair", "roc→dtf", -0.4, "first-2 mean +9.9% (n=85)"),
    ("PENALTY", "first_two_pair", "roc→cmf", -0.3, "first-2 mean +10.6% (n=93)"),
    ("PENALTY", "first_two_pair", "ich→dtf", -0.2, "first-2 mean +22.5% (n=211)"),

    ("PENALTY", "last_two", "atr→rs",  -0.3, "last-2 mean +6.4% (n=69)"),
    ("PENALTY", "last_two", "atr→ich", -0.3, "last-2 mean +15.5% (n=41)"),
    ("PENALTY", "last_two", "ich→dtf", -0.3, "last-2 mean +15.8% (n=37)"),
    ("PENALTY", "last_two", "ich→hl",  -0.3, "last-2 mean +17.8% (n=53)"),

    ("PENALTY", "n_led_ich", 3, -0.4, "3-led-Ich mean +20.6% (n=137)"),
    ("PENALTY", "n_led_ich", 5, -0.3, "5-led-Ich mean +19.9% (n=78)"),
]


def compute_layer_2_adjustment(features: dict) -> tuple[float, list[str]]:
    """Return (adjustment, list-of-pattern-tags) for the given sequence features.

    Path C: additive bonuses/penalties based on per-pattern mean forward return.
    All magnitudes |adjustment| <= 0.5 per pattern. Cumulative effect typically ±1.5.
    """
    if features.get("_no_indicators_firing"):
        return 0.0, ["no_indicators_firing"]

    adj = 0.0
    tags = []

    for kind, feat_key, expected_value, magnitude, note in LAYER_2_RULES:
        actual = features.get(feat_key)
        if actual == expected_value:
            adj += magnitude
            sign = "+" if magnitude >= 0 else ""
            label = f"{kind}: {feat_key}={expected_value} ({sign}{magnitude}; {note})"
            tags.append(label)

    return round(adj, 2), tags


# ─── Streak computation helpers (for backtester / batch use) ──────
def compute_streaks_from_history(fire_flags_per_day: list[dict[str, int]]) -> dict[str, int]:
    """Given a chronological list of daily fire flags (one dict per trading day,
    most recent LAST), compute the current consecutive-firing streak per indicator.

    Each item in fire_flags_per_day should be a dict like:
      {'rs': 0/1, 'ich': 0/1, 'hl': 0/1, 'cmf': 0/1, 'roc': 0/1, 'atr': 0/1, 'dtf': 0/1}

    Returns: {'rs': N, 'ich': M, ...} where N is the current streak length
    (consecutive 1's ending at the LAST entry).
    """
    streaks = {lbl: 0 for lbl in LABELS}
    for lbl in LABELS:
        # Walk backwards from most recent day; count consecutive 1's
        count = 0
        for day in reversed(fire_flags_per_day):
            if day.get(lbl, 0) == 1:
                count += 1
            else:
                break
        streaks[lbl] = count
    return streaks


def fire_flags_v2_from_indicators(indicators: dict) -> dict[str, int]:
    """Convert a single-day indicators dict (output of compute_all_indicators)
    into the v2 binary fire flags. Used by callers to populate fire_flags_per_day.

    Uses scoring-aligned thresholds (RS >= 90, HL >= 4) per Scheme I+ design.
    """
    flags = {}

    rs = indicators.get("relative_strength", {})
    flags["rs"] = int((rs.get("rs_percentile", 0) or 0) >= 90)

    ich = indicators.get("ichimoku_cloud", {})
    flags["ich"] = int(bool(ich.get("triggered", False)))

    hl = indicators.get("higher_lows", {})
    flags["hl"] = int((hl.get("consecutive_higher_lows", 0) or 0) >= 4)

    cmf = indicators.get("cmf", {})
    flags["cmf"] = int(bool(cmf.get("triggered", False)))

    roc = indicators.get("roc", {})
    flags["roc"] = int(bool(roc.get("triggered", False)))

    atr = indicators.get("atr_expansion", {})
    flags["atr"] = int(bool(atr.get("triggered", False)))

    dtf = indicators.get("dual_tf_rs", {})
    flags["dtf"] = int(bool(dtf.get("triggered", False)))

    return flags
