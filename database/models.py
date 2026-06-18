"""
Repository-style data access for HariShiva V2.

Each repository wraps a small set of related tables and exposes plain
functions/methods returning dicts or primitives - no ORM, just thin
wrappers around the schema defined in database.database.
"""

from __future__ import annotations

from datetime import datetime

from database.database import get_connection


# ── Persons ──────────────────────────────────────────────────────────────
class PersonRepository:
    @staticmethod
    def get_or_create(name: str) -> dict:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM persons WHERE name = ?", (name,)
            ).fetchone()
            if row:
                return dict(row)
            cur = conn.execute(
                "INSERT INTO persons (name, visit_count) VALUES (?, 0)", (name,)
            )
            return {
                "id": cur.lastrowid,
                "name": name,
                "visit_count": 0,
                "last_seen": None,
            }

    @staticmethod
    def get(name: str) -> dict | None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM persons WHERE name = ?", (name,)
            ).fetchone()
            return dict(row) if row else None

    @staticmethod
    def mark_seen(person_id: int) -> None:
        with get_connection() as conn:
            conn.execute(
                "UPDATE persons SET visit_count = visit_count + 1, "
                "last_seen = ? WHERE id = ?",
                (datetime.now().isoformat(), person_id),
            )

    @staticmethod
    def list_names() -> list[str]:
        with get_connection() as conn:
            rows = conn.execute("SELECT name FROM persons ORDER BY name").fetchall()
            return [r["name"] for r in rows]

    @staticmethod
    def delete(name: str) -> bool:
        with get_connection() as conn:
            cur = conn.execute("DELETE FROM persons WHERE name = ?", (name,))
            return cur.rowcount > 0


# ── Per-person facts / preferences / notes / moods ──────────────────────
class FactRepository:
    @staticmethod
    def add(person_id: int, fact: str) -> bool:
        with get_connection() as conn:
            existing = conn.execute(
                "SELECT 1 FROM person_facts WHERE person_id = ? AND lower(fact) = lower(?)",
                (person_id, fact),
            ).fetchone()
            if existing:
                return False
            conn.execute(
                "INSERT INTO person_facts (person_id, fact) VALUES (?, ?)",
                (person_id, fact),
            )
            return True

    @staticmethod
    def list(person_id: int, limit: int = 10) -> list[str]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT fact FROM person_facts WHERE person_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (person_id, limit),
            ).fetchall()
            return [r["fact"] for r in reversed(rows)]

    @staticmethod
    def forget(person_id: int, keyword: str) -> int:
        with get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM person_facts WHERE person_id = ? AND fact LIKE ?",
                (person_id, f"%{keyword}%"),
            )
            return cur.rowcount


class PreferenceRepository:
    @staticmethod
    def add(person_id: int, item: str, liked: bool = True) -> None:
        with get_connection() as conn:
            existing = conn.execute(
                "SELECT 1 FROM person_preferences WHERE person_id = ? "
                "AND lower(item) = lower(?) AND liked = ?",
                (person_id, item, int(liked)),
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO person_preferences (person_id, item, liked) "
                    "VALUES (?, ?, ?)",
                    (person_id, item, int(liked)),
                )

    @staticmethod
    def list(person_id: int, liked: bool = True, limit: int = 8) -> list[str]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT item FROM person_preferences WHERE person_id = ? "
                "AND liked = ? ORDER BY id DESC LIMIT ?",
                (person_id, int(liked), limit),
            ).fetchall()
            return [r["item"] for r in reversed(rows)]

    @staticmethod
    def forget(person_id: int, keyword: str) -> int:
        with get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM person_preferences WHERE person_id = ? AND item LIKE ?",
                (person_id, f"%{keyword}%"),
            )
            return cur.rowcount


class NoteRepository:
    @staticmethod
    def add(person_id: int, note: str) -> None:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO person_notes (person_id, note) VALUES (?, ?)",
                (person_id, note),
            )

    @staticmethod
    def list(person_id: int, limit: int = 5) -> list[str]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT note FROM person_notes WHERE person_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (person_id, limit),
            ).fetchall()
            return [r["note"] for r in reversed(rows)]


class MoodRepository:
    @staticmethod
    def add(person_id: int, mood: str) -> None:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO person_moods (person_id, mood) VALUES (?, ?)",
                (person_id, mood),
            )
            # keep only the latest 30 per person
            conn.execute(
                "DELETE FROM person_moods WHERE person_id = ? AND id NOT IN ("
                "  SELECT id FROM person_moods WHERE person_id = ? "
                "  ORDER BY id DESC LIMIT 30)",
                (person_id, person_id),
            )

    @staticmethod
    def latest(person_id: int) -> str | None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT mood FROM person_moods WHERE person_id = ? "
                "ORDER BY id DESC LIMIT 1",
                (person_id,),
            ).fetchone()
            return row["mood"] if row else None


# ── Conversations (global + per-person via person_id) ───────────────────
class ConversationRepository:
    @staticmethod
    def add(
        user_text: str,
        bot_text: str,
        lang: str = "en",
        person_id: int | None = None,
    ) -> int:
        with get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO conversations (person_id, user_text, bot_text, lang, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (person_id, user_text, bot_text, lang,
                 datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            )
            return cur.lastrowid

    @staticmethod
    def recent(limit: int = 200, person_id: int | None = None) -> list[dict]:
        with get_connection() as conn:
            if person_id is not None:
                rows = conn.execute(
                    "SELECT * FROM conversations WHERE person_id = ? "
                    "ORDER BY id DESC LIMIT ?",
                    (person_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM conversations ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in reversed(rows)]

    @staticmethod
    def mark_feedback(conversation_id: int, feedback: str) -> None:
        with get_connection() as conn:
            conn.execute(
                "UPDATE conversations SET feedback = ? WHERE id = ?",
                (feedback, conversation_id),
            )

    @staticmethod
    def last_id() -> int | None:
        with get_connection() as conn:
            row = conn.execute("SELECT max(id) AS id FROM conversations").fetchone()
            return row["id"] if row and row["id"] is not None else None


# ── Global learned facts / corrections / tombstones ─────────────────────
class LearnedFactRepository:
    @staticmethod
    def add(fact: str) -> bool:
        if DeletedKeywordRepository.is_tombstoned(fact):
            return False
        with get_connection() as conn:
            existing = conn.execute(
                "SELECT 1 FROM learned_facts WHERE lower(fact) = lower(?)", (fact,)
            ).fetchone()
            if existing:
                return False
            conn.execute("INSERT INTO learned_facts (fact) VALUES (?)", (fact,))
            return True

    @staticmethod
    def list(limit: int = 8) -> list[str]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT fact FROM learned_facts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [r["fact"] for r in reversed(rows)]

    @staticmethod
    def all_with_dates() -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT fact, created_at FROM learned_facts ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]

    @staticmethod
    def forget(keyword: str) -> int:
        kw = keyword.lower().strip()
        with get_connection() as conn:
            cur = conn.execute(
                "DELETE FROM learned_facts WHERE lower(fact) LIKE ?", (f"%{kw}%",)
            )
            removed = cur.rowcount
        DeletedKeywordRepository.add(kw)
        return removed


class CorrectionRepository:
    @staticmethod
    def add(original_query: str, wrong_response: str, correction: str) -> None:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO corrections (original_query, wrong_response, correction) "
                "VALUES (?, ?, ?)",
                (original_query, wrong_response, correction),
            )
            conn.execute(
                "DELETE FROM corrections WHERE id NOT IN ("
                "  SELECT id FROM corrections ORDER BY id DESC LIMIT 50)"
            )

    @staticmethod
    def recent(limit: int = 5) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM corrections ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in reversed(rows)]


class DeletedKeywordRepository:
    @staticmethod
    def add(keyword: str) -> None:
        kw = keyword.lower().strip()
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO deleted_keywords (keyword) VALUES (?)", (kw,)
            )
            conn.execute(
                "DELETE FROM deleted_keywords WHERE keyword NOT IN ("
                "  SELECT keyword FROM deleted_keywords ORDER BY created_at DESC LIMIT 200)"
            )

    @staticmethod
    def is_tombstoned(text: str) -> bool:
        t = text.lower()
        with get_connection() as conn:
            rows = conn.execute("SELECT keyword FROM deleted_keywords").fetchall()
            return any(r["keyword"] in t for r in rows)

    @staticmethod
    def list() -> list[str]:
        with get_connection() as conn:
            rows = conn.execute("SELECT keyword FROM deleted_keywords").fetchall()
            return [r["keyword"] for r in rows]


# ── Frequent topics ───────────────────────────────────────────────────────
class FrequentTopicRepository:
    @staticmethod
    def increment(word: str) -> None:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO frequent_topics (word, count) VALUES (?, 1) "
                "ON CONFLICT(word) DO UPDATE SET count = count + 1",
                (word,),
            )

    @staticmethod
    def top(n: int = 5) -> list[tuple[str, int]]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT word, count FROM frequent_topics ORDER BY count DESC LIMIT ?",
                (n,),
            ).fetchall()
            return [(r["word"], r["count"]) for r in rows]

    @staticmethod
    def prune(max_keys: int = 200, keep: int = 150) -> None:
        with get_connection() as conn:
            total = conn.execute("SELECT count(*) AS c FROM frequent_topics").fetchone()["c"]
            if total > max_keys:
                conn.execute(
                    "DELETE FROM frequent_topics WHERE word NOT IN ("
                    "  SELECT word FROM frequent_topics ORDER BY count DESC LIMIT ?)",
                    (keep,),
                )


# ── Global profile (key/value) ───────────────────────────────────────────
class ProfileRepository:
    @staticmethod
    def get(key: str, default=None):
        with get_connection() as conn:
            row = conn.execute(
                "SELECT value FROM profile WHERE key = ?", (key,)
            ).fetchone()
            return row["value"] if row else default

    @staticmethod
    def set(key: str, value) -> None:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO profile (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(value)),
            )

    @staticmethod
    def get_all() -> dict:
        with get_connection() as conn:
            rows = conn.execute("SELECT key, value FROM profile").fetchall()
            return {r["key"]: r["value"] for r in rows}


# ── Behavior reports ──────────────────────────────────────────────────────
class BehaviorReportRepository:
    @staticmethod
    def save(report_json: str) -> None:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO behavior_reports (report_json) VALUES (?)", (report_json,)
            )
            conn.execute(
                "DELETE FROM behavior_reports WHERE id NOT IN ("
                "  SELECT id FROM behavior_reports ORDER BY id DESC LIMIT 30)"
            )

    @staticmethod
    def latest() -> str | None:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT report_json FROM behavior_reports ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return row["report_json"] if row else None
