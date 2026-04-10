"""
wash_sale_tracker.py — Track loss exits and wash sale re-entry events
for tax awareness.

NOTE: As of the current strategy, wash sales are NOT used to block
re-entries. The tracker exists purely to log when a wash sale occurs
so Todd has visibility for tax reporting. Backtesting showed that
re-entering after a loss often produces strong gains (VIAV -21% →
same-day re-entry → +42%), so blocking re-entries would destroy alpha.

Persisted to wash_sale_log.json in the project root.

Schema:
{
    "cooldowns": {
        "VIAV": {
            "exit_date": "2026-03-09",
            "loss_amount": -4200.50,
            "cooldown_until": "2026-04-08"
        }
    },
    "violations": [
        {
            "ticker": "VIAV",
            "loss_exit_date": "2026-03-09",
            "loss_amount": -4200.50,
            "reentry_date": "2026-03-09",
            "days_between": 0
        }
    ]
}
"""

from __future__ import annotations
import json
from datetime import datetime, timedelta, date
from pathlib import Path


WASH_SALE_FILE = Path(__file__).parent / "wash_sale_log.json"
COOLDOWN_DAYS = 30


def _to_date(d) -> date:
    """Coerce a date-like value to a datetime.date."""
    if isinstance(d, date) and not isinstance(d, datetime):
        return d
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, str):
        return datetime.strptime(d, "%Y-%m-%d").date()
    raise ValueError(f"Cannot convert {d!r} to date")


def _empty() -> dict:
    return {"cooldowns": {}, "violations": []}


def _load() -> dict:
    """Load the wash sale log from disk, upgrading old flat format if needed."""
    if not WASH_SALE_FILE.exists():
        return _empty()
    try:
        with open(WASH_SALE_FILE, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _empty()

    if not isinstance(data, dict):
        return _empty()

    # Detect old flat schema: {ticker: {exit_date, loss_amount, cooldown_until}}
    # and upgrade to the new nested schema.
    if "cooldowns" not in data and "violations" not in data:
        upgraded = {"cooldowns": {}, "violations": []}
        for k, v in data.items():
            if isinstance(v, dict) and "cooldown_until" in v:
                upgraded["cooldowns"][k] = v
        return upgraded

    data.setdefault("cooldowns", {})
    data.setdefault("violations", [])
    return data


def _save(data: dict) -> None:
    """Write the wash sale log to disk."""
    with open(WASH_SALE_FILE, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# ─────────────────────────────────────────────────────────────
# Loss exit tracking (cooldown window, used for awareness only)
# ─────────────────────────────────────────────────────────────

def record_loss_exit(ticker: str, exit_date, loss_amount: float) -> None:
    """
    Record a loss exit. Starts a 30-day cooldown window for tax tracking.
    Does NOT block re-entry — that's intentional.
    """
    if loss_amount >= 0:
        return  # not a loss

    exit_d = _to_date(exit_date)
    cooldown_until = exit_d + timedelta(days=COOLDOWN_DAYS)

    data = _load()
    data["cooldowns"][ticker] = {
        "exit_date": exit_d.isoformat(),
        "loss_amount": round(float(loss_amount), 2),
        "cooldown_until": cooldown_until.isoformat(),
    }
    _save(data)


def is_in_cooldown(ticker: str, current_date=None) -> bool:
    """
    Return True if `ticker` is currently within its 30-day cooldown window.
    This is informational only — does NOT block trades.
    """
    if current_date is None:
        current_date = date.today()
    current_d = _to_date(current_date)

    data = _load()
    entry = data["cooldowns"].get(ticker)
    if not entry:
        return False

    cooldown_until = _to_date(entry["cooldown_until"])
    return current_d <= cooldown_until


# Backward-compat alias
def is_blocked(ticker: str, current_date=None) -> bool:
    return is_in_cooldown(ticker, current_date)


def get_cooldowns(current_date=None) -> dict:
    """Return a dict of all tickers currently in cooldown (informational)."""
    if current_date is None:
        current_date = date.today()
    current_d = _to_date(current_date)

    data = _load()
    active = {}
    for ticker, entry in data["cooldowns"].items():
        cooldown_until = _to_date(entry["cooldown_until"])
        if current_d <= cooldown_until:
            active[ticker] = entry
    return active


# Backward-compat alias
def get_blocked_tickers(current_date=None) -> dict:
    return get_cooldowns(current_date)


def cleanup_expired(current_date=None) -> int:
    """Remove cooldown entries whose window has expired. Returns count removed."""
    if current_date is None:
        current_date = date.today()
    current_d = _to_date(current_date)

    data = _load()
    expired = [t for t, e in data["cooldowns"].items()
               if _to_date(e["cooldown_until"]) < current_d]
    for t in expired:
        del data["cooldowns"][t]

    if expired:
        _save(data)
    return len(expired)


def get_cooldown_reason(ticker: str, current_date=None) -> str | None:
    """Human-readable cooldown explanation, or None."""
    if not is_in_cooldown(ticker, current_date):
        return None
    data = _load()
    entry = data["cooldowns"][ticker]
    return f"Wash sale cooldown (until {entry['cooldown_until']})"


# Backward-compat alias
def get_block_reason(ticker: str, current_date=None) -> str | None:
    return get_cooldown_reason(ticker, current_date)


# ─────────────────────────────────────────────────────────────
# Wash sale violations (re-entries within the cooldown window)
# ─────────────────────────────────────────────────────────────

def record_violation(ticker: str, reentry_date) -> dict | None:
    """
    Record a wash sale event: a buy placed while the ticker is in its
    30-day cooldown window. Returns the violation record (or None if
    the ticker is not actually in cooldown).
    """
    reentry_d = _to_date(reentry_date)

    data = _load()
    cooldown = data["cooldowns"].get(ticker)
    if not cooldown:
        return None

    cooldown_until = _to_date(cooldown["cooldown_until"])
    if reentry_d > cooldown_until:
        return None  # not actually a wash sale

    loss_exit_date = _to_date(cooldown["exit_date"])
    days_between = (reentry_d - loss_exit_date).days

    violation = {
        "ticker": ticker,
        "loss_exit_date": loss_exit_date.isoformat(),
        "loss_amount": cooldown["loss_amount"],
        "reentry_date": reentry_d.isoformat(),
        "days_between": days_between,
    }
    data["violations"].append(violation)
    _save(data)
    return violation


def get_violations() -> list[dict]:
    """Return all logged wash sale violations in order."""
    return list(_load()["violations"])


def total_disallowed_loss() -> float:
    """Sum of disallowed losses across all recorded violations."""
    return round(sum(abs(v["loss_amount"]) for v in get_violations()), 2)


if __name__ == "__main__":
    # CLI summary
    cooldowns = get_cooldowns()
    violations = get_violations()

    print("Wash Sale Tracker — Status")
    print("=" * 50)
    print()

    print(f"Active cooldowns: {len(cooldowns)}")
    for t, e in sorted(cooldowns.items()):
        print(f"  {t:<8s}  loss ${e['loss_amount']:>10,.2f}   "
              f"cooldown until {e['cooldown_until']}")

    print()
    print(f"Recorded violations: {len(violations)}")
    for v in violations:
        print(f"  {v['ticker']:<8s}  loss ${v['loss_amount']:>10,.2f}   "
              f"re-entered {v['reentry_date']} ({v['days_between']}d later)")

    if violations:
        print()
        print(f"Total disallowed losses: ${total_disallowed_loss():,.2f}")
        print("(Deferred, not eliminated — added to the cost basis of the replacement shares.)")
