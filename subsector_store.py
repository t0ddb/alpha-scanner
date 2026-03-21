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
    """)

    conn.commit()
    return conn


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
