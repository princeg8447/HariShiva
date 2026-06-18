"""
SQLite database layer for HariShiva V2.

The original project stored state in several places: a JSON "learning_data"
file and one JSON file per person. V2 consolidates all of that into a single
SQLite database (data/harishiva.db) so it can be queried, backed up and
migrated consistently.

Each call to get_connection() opens a short-lived connection with WAL mode
enabled, which is safe to use from multiple threads.
"""

import sqlite3
from contextlib import contextmanager

from app import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 0,
    last_seen   TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS person_facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    fact       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS person_preferences (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    item       TEXT NOT NULL,
    liked      INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS person_notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    note       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS person_moods (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    mood       TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id  INTEGER REFERENCES persons(id) ON DELETE SET NULL,
    user_text  TEXT NOT NULL,
    bot_text   TEXT NOT NULL,
    lang       TEXT NOT NULL DEFAULT 'en',
    feedback   TEXT NOT NULL DEFAULT 'neutral',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS learned_facts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    fact       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    original_query  TEXT,
    wrong_response  TEXT,
    correction      TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS deleted_keywords (
    keyword    TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS frequent_topics (
    word  TEXT PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS profile (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS behavior_reports (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def get_connection():
    """Yield a short-lived SQLite connection with sane defaults."""
    config.ensure_directories()
    conn = sqlite3.connect(config.DATABASE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables if they don't already exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)
