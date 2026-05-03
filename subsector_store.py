from __future__ import annotations

"""
subsector_store.py — SQLite persistence layer for subsector breakout tracking.

Stores daily subsector metrics and breakout state machine state.

Database: breakout_tracker.db (project root)

Usage:
    from subsector_store import init_db, upsert_daily, get_history, get_breakout_states

    conn = init_db()
    upsert_daily(conn, "2025-01-15", records)
    history = get_history(conn, "chips_networking", days=90)
"""

import sqlite3
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta


DB_PATH = Path(__file__).parent / "breakout_tracker.db"


def init_db(db_path: str | Path = None) -> sqlite3.Connection:
    """Create tables if needed and return a connection."""
    path = db_path or DB_PATH
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS subsector_daily (
            date           TEXT NOT NULL,
            subsector      TEXT NOT NULL,
            subsector_name TEXT NOT NULL,
            sector         TEXT NOT NULL,
            ticker_count   INTEGER NOT NULL,
            avg_score      REAL NOT NULL,
            max_score      REAL NOT NULL,
            breadth        REAL NOT NULL,
            hot_count      INTEGER NOT NULL,
            ticker_scores  TEXT NOT NULL,
            PRIMARY KEY (date, subsector)
        );

        CREATE TABLE IF NOT EXISTS subsector_breakout_state (
            subsector         TEXT PRIMARY KEY,
            status            TEXT NOT NULL DEFAULT 'quiet',
            status_since      TEXT,
            consecutive_hot   INTEGER DEFAULT 0,
            consecutive_cool  INTEGER DEFAULT 0,
            peak_avg_score    REAL DEFAULT 0,
            peak_breadth      REAL DEFAULT 0,
            updated_at        TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ticker_scores (
            date           TEXT NOT NULL,
            ticker         TEXT NOT NULL,
            name           TEXT,
            subsector      TEXT,
            sector         TEXT,
            score          REAL NOT NULL,
            signals        TEXT,
            signal_weights TEXT,
            PRIMARY KEY (date, ticker)
        );

        CREATE INDEX IF NOT EXISTS idx_daily_subsector
            ON subsector_daily (subsector, date);

        CREATE INDEX IF NOT EXISTS idx_daily_date
            ON subsector_daily (date);

        CREATE INDEX IF NOT EXISTS idx_ticker_scores_ticker
            ON ticker_scores (ticker, date);

        CREATE INDEX IF NOT EXISTS idx_ticker_scores_score
            ON ticker_scores (score, date);

        -- ─── Scheme M shadow tracking table ────────────────────
        -- Stores per-day Scheme M scores AND fire flags AND raw indicator
        -- values. Fire flags are needed for Layer 2 streak computation.
        -- Raw values are stored so we can re-derive scores or audit later.
        --
        -- Renamed from ticker_scores_v2 on 2026-05-03 (Path C → Scheme M).
        -- Migration handled below if old table is present.
        CREATE TABLE IF NOT EXISTS ticker_scores_m (
            date              TEXT NOT NULL,
            ticker            TEXT NOT NULL,
            score             REAL NOT NULL,        -- final score (Layer 1 + Layer 2)
            layer_1           REAL NOT NULL,        -- 0-10 base
            layer_2           REAL NOT NULL,        -- additive (can be negative)
            sequence_tags     TEXT,                 -- pipe-separated pattern tags
            -- Fire flags (binary 0/1) for streak computation downstream
            fire_rs           INTEGER NOT NULL DEFAULT 0,
            fire_ich          INTEGER NOT NULL DEFAULT 0,
            fire_hl           INTEGER NOT NULL DEFAULT 0,
            fire_cmf          INTEGER NOT NULL DEFAULT 0,
            fire_roc          INTEGER NOT NULL DEFAULT 0,
            fire_atr          INTEGER NOT NULL DEFAULT 0,
            fire_dtf          INTEGER NOT NULL DEFAULT 0,
            -- Raw indicator values (for audit / re-scoring)
            rs_pctl           REAL,
            hl_count          INTEGER,
            ich_score         INTEGER,              -- 0/3, 1/3, 2/3, 3/3 composite
            roc_value         REAL,
            cmf_value         REAL,
            atr_pctl          REAL,
            dtf_126d_pctl     REAL,
            dtf_63d_pctl      REAL,
            PRIMARY KEY (date, ticker)
        );

        CREATE INDEX IF NOT EXISTS idx_ticker_scores_m_ticker
            ON ticker_scores_m (ticker, date);

        CREATE INDEX IF NOT EXISTS idx_ticker_scores_m_score
            ON ticker_scores_m (score, date);
    """)

    # ─── Migration: rename ticker_scores_v2 → ticker_scores_m if present ──
    # Idempotent — runs every init_db() but only does work once.
    has_old = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='ticker_scores_v2'"
    ).fetchone()
    if has_old:
        # Copy data into the new table (which CREATE TABLE IF NOT EXISTS just made).
        # INSERT OR IGNORE handles the case where someone wrote to ticker_scores_m
        # already; old data takes a back seat to anything new.
        conn.execute("""INSERT OR IGNORE INTO ticker_scores_m
                        SELECT * FROM ticker_scores_v2""")
        conn.execute("DROP TABLE ticker_scores_v2")
        # Drop the old indexes (they reference the dropped table)
        conn.execute("DROP INDEX IF EXISTS idx_ticker_scores_v2_ticker")
        conn.execute("DROP INDEX IF EXISTS idx_ticker_scores_v2_score")

    conn.commit()
    return conn


def upsert_ticker_scores_m(conn: sqlite3.Connection, date: str, records: list[dict]) -> None:
    """Insert or replace Scheme M ticker scores + fire flags + raw values.

    Each record dict should contain:
      ticker, score, layer_1, layer_2, sequence_tags,
      fire_rs, fire_ich, fire_hl, fire_cmf, fire_roc, fire_atr, fire_dtf,
      rs_pctl, hl_count, ich_score, roc_value, cmf_value, atr_pctl,
      dtf_126d_pctl, dtf_63d_pctl
    """
    if not records:
        return
    conn.executemany(
        """INSERT OR REPLACE INTO ticker_scores_m (
              date, ticker, score, layer_1, layer_2, sequence_tags,
              fire_rs, fire_ich, fire_hl, fire_cmf, fire_roc, fire_atr, fire_dtf,
              rs_pctl, hl_count, ich_score, roc_value, cmf_value, atr_pctl,
              dtf_126d_pctl, dtf_63d_pctl
           ) VALUES (?, ?, ?, ?, ?, ?,  ?, ?, ?, ?, ?, ?, ?,  ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                date, r["ticker"], r["score"], r["layer_1"], r["layer_2"],
                r.get("sequence_tags", ""),
                r["fire_rs"], r["fire_ich"], r["fire_hl"], r["fire_cmf"],
                r["fire_roc"], r["fire_atr"], r["fire_dtf"],
                r.get("rs_pctl"), r.get("hl_count"), r.get("ich_score"),
                r.get("roc_value"), r.get("cmf_value"), r.get("atr_pctl"),
                r.get("dtf_126d_pctl"), r.get("dtf_63d_pctl"),
            )
            for r in records
        ],
    )
    conn.commit()


def get_fire_flags_history_m(conn: sqlite3.Connection, ticker: str,
                              end_date: str, days: int = 90) -> list[dict]:
    """Get the last N daily fire-flag rows for a ticker, ending at end_date
    (inclusive). Returns list of dicts ordered chronologically (oldest first)."""
    cur = conn.execute(
        """SELECT date, fire_rs, fire_ich, fire_hl, fire_cmf,
                  fire_roc, fire_atr, fire_dtf
           FROM ticker_scores_m
           WHERE ticker = ? AND date <= ?
           ORDER BY date DESC LIMIT ?""",
        (ticker, end_date, days),
    )
    rows = [
        dict(date=r[0], rs=r[1], ich=r[2], hl=r[3], cmf=r[4],
             roc=r[5], atr=r[6], dtf=r[7])
        for r in cur.fetchall()
    ]
    rows.reverse()  # chronological (oldest first)
    return rows


def get_m_scores_for_persistence(conn: sqlite3.Connection, ticker: str,
                                  end_date: str, days: int = 5) -> list[float]:
    """Get the last N Scheme M scores for a ticker, ending BEFORE end_date.
    Used for persistence check. Returns list ordered most-recent first."""
    cur = conn.execute(
        """SELECT score FROM ticker_scores_m
           WHERE ticker = ? AND date < ?
           ORDER BY date DESC LIMIT ?""",
        (ticker, end_date, days),
    )
    return [r[0] for r in cur.fetchall()]


def upsert_daily(conn: sqlite3.Connection, date: str, records: list[dict]) -> None:
    """Insert or replace daily subsector snapshot records."""
    if not records:
        return

    conn.executemany(
        """INSERT OR REPLACE INTO subsector_daily
           (date, subsector, subsector_name, sector, ticker_count,
            avg_score, max_score, breadth, hot_count, ticker_scores)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                date,
                r["subsector"],
                r["subsector_name"],
                r["sector"],
                r["ticker_count"],
                r["avg_score"],
                r["max_score"],
                r["breadth"],
                r["hot_count"],
                json.dumps(r["ticker_scores"]),
            )
            for r in records
        ],
    )
    conn.commit()


def get_history(
    conn: sqlite3.Connection, subsector: str, days: int = 90
) -> pd.DataFrame:
    """Get daily history for one subsector, most recent `days` trading days."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    df = pd.read_sql_query(
        """SELECT * FROM subsector_daily
           WHERE subsector = ? AND date >= ?
           ORDER BY date""",
        conn,
        params=(subsector, cutoff),
    )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_all_history(
    conn: sqlite3.Connection, days: int = 90
) -> pd.DataFrame:
    """Get daily history for ALL subsectors, most recent `days` trading days."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    df = pd.read_sql_query(
        """SELECT * FROM subsector_daily
           WHERE date >= ?
           ORDER BY date, subsector""",
        conn,
        params=(cutoff,),
    )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_history_between(
    conn: sqlite3.Connection,
    subsector: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Get daily history for one subsector between two dates."""
    df = pd.read_sql_query(
        """SELECT * FROM subsector_daily
           WHERE subsector = ? AND date >= ? AND date <= ?
           ORDER BY date""",
        conn,
        params=(subsector, start_date, end_date),
    )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_breakout_states(conn: sqlite3.Connection) -> dict[str, dict]:
    """Get current breakout state for all subsectors."""
    cursor = conn.execute("SELECT * FROM subsector_breakout_state")
    columns = [desc[0] for desc in cursor.description]

    states = {}
    for row in cursor.fetchall():
        record = dict(zip(columns, row))
        states[record["subsector"]] = record

    return states


def update_breakout_state(conn: sqlite3.Connection, subsector: str, state: dict) -> None:
    """Insert or update breakout state for a subsector."""
    conn.execute(
        """INSERT OR REPLACE INTO subsector_breakout_state
           (subsector, status, status_since, consecutive_hot, consecutive_cool,
            peak_avg_score, peak_breadth, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            subsector,
            state["status"],
            state.get("status_since", ""),
            state.get("consecutive_hot", 0),
            state.get("consecutive_cool", 0),
            state.get("peak_avg_score", 0),
            state.get("peak_breadth", 0),
            state.get("updated_at", datetime.now().strftime("%Y-%m-%d")),
        ),
    )
    conn.commit()


def cleanup_old_records(conn: sqlite3.Connection, retention_days: int = 180) -> int:
    """Delete daily records older than retention_days. Returns count deleted."""
    cutoff = (datetime.now() - timedelta(days=retention_days)).strftime("%Y-%m-%d")
    cursor = conn.execute(
        "DELETE FROM subsector_daily WHERE date < ?", (cutoff,)
    )
    conn.commit()
    return cursor.rowcount


def get_latest_date(conn: sqlite3.Connection) -> str | None:
    """Get the most recent date in subsector_daily, or None if empty."""
    cursor = conn.execute("SELECT MAX(date) FROM subsector_daily")
    row = cursor.fetchone()
    return row[0] if row and row[0] else None


def get_subsector_list(conn: sqlite3.Connection) -> list[str]:
    """Get list of all subsectors that have data."""
    cursor = conn.execute("SELECT DISTINCT subsector FROM subsector_daily ORDER BY subsector")
    return [row[0] for row in cursor.fetchall()]


# =============================================================
# TICKER SCORES TABLE
# =============================================================

def upsert_ticker_scores(conn: sqlite3.Connection, date: str, results: list[dict]) -> None:
    """Insert or replace daily ticker score records."""
    if not results:
        return

    conn.executemany(
        """INSERT OR REPLACE INTO ticker_scores
           (date, ticker, name, subsector, sector, score, signals, signal_weights)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                date,
                r["ticker"],
                r.get("name", ""),
                r.get("subsector", ""),
                r.get("sector", ""),
                r["score"],
                json.dumps(r.get("signals", [])),
                json.dumps(r.get("signal_weights", {})),
            )
            for r in results
        ],
    )
    conn.commit()


def get_ticker_history(
    conn: sqlite3.Connection, ticker: str, days: int = 365
) -> pd.DataFrame:
    """Get score history for one ticker."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    df = pd.read_sql_query(
        """SELECT * FROM ticker_scores
           WHERE ticker = ? AND date >= ?
           ORDER BY date""",
        conn,
        params=(ticker, cutoff),
    )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_high_scores(
    conn: sqlite3.Connection, min_score: float = 8.0, days: int = 365
) -> pd.DataFrame:
    """Get all ticker-date pairs where score >= min_score."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    df = pd.read_sql_query(
        """SELECT * FROM ticker_scores
           WHERE score >= ? AND date >= ?
           ORDER BY date, score DESC""",
        conn,
        params=(min_score, cutoff),
    )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def get_scores_on_date(
    conn: sqlite3.Connection, date: str
) -> pd.DataFrame:
    """Get all ticker scores for a specific date."""
    df = pd.read_sql_query(
        """SELECT * FROM ticker_scores
           WHERE date = ?
           ORDER BY score DESC""",
        conn,
        params=(date,),
    )

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df
