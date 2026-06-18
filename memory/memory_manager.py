"""
Memory management for HariShiva V2.

Two kinds of memory, both backed by the SQLite database (database/models.py)
instead of the original's per-person JSON files and a global learning_data.json:

- PersonMemory: facts, preferences, notes, moods and conversation history
  for one specific person.
- AdaptiveLearning: global assistant memory - learned facts, corrections,
  frequent topics, language/response-style preferences and feedback stats.
"""

import json
from datetime import datetime

from database.models import (
    ConversationRepository,
    CorrectionRepository,
    DeletedKeywordRepository,
    FactRepository,
    FrequentTopicRepository,
    LearnedFactRepository,
    MoodRepository,
    NoteRepository,
    PersonRepository,
    PreferenceRepository,
    ProfileRepository,
)


# ── Per-person memory ────────────────────────────────────────────────────
class PersonMemory:
    def __init__(self, name: str):
        self.name = name
        record = PersonRepository.get_or_create(name)
        self.id = record["id"]
        self.visit_count = record["visit_count"]
        self.last_seen = record["last_seen"]

    def on_seen(self) -> None:
        PersonRepository.mark_seen(self.id)
        record = PersonRepository.get(self.name)
        self.visit_count = record["visit_count"]
        self.last_seen = record["last_seen"]

    def add_fact(self, fact: str) -> bool:
        return FactRepository.add(self.id, fact)

    def forget_fact(self, keyword: str) -> int:
        """Remove facts/preferences/dislikes containing keyword. Returns count removed."""
        removed = FactRepository.forget(self.id, keyword)
        removed += PreferenceRepository.forget(self.id, keyword)
        return removed

    def add_preference(self, item: str, like: bool = True) -> None:
        PreferenceRepository.add(self.id, item, like)

    def add_note(self, note: str) -> None:
        NoteRepository.add(self.id, note)

    def add_mood(self, mood: str) -> None:
        MoodRepository.add(self.id, mood)

    def record_conversation(self, user_text: str, bot_reply: str, lang: str = "en") -> None:
        ConversationRepository.add(user_text, bot_reply, lang, person_id=self.id)

    def build_context(self) -> str:
        parts = [
            "=== MEMORY CONTEXT ===",
            f"The user's name is {self.name}. Use this name when talking to them.",
        ]
        v = self.visit_count
        parts.append(f"{self.name} has visited {v} time{'s' if v != 1 else ''}.")

        if self.last_seen:
            try:
                dt = datetime.fromisoformat(self.last_seen)
                parts.append(f"Last seen: {dt.strftime('%d %b %Y at %I:%M %p')}")
            except ValueError:
                pass

        facts = FactRepository.list(self.id, limit=10)
        if facts:
            parts.append("Known facts: " + "; ".join(facts))

        likes = PreferenceRepository.list(self.id, liked=True, limit=8)
        if likes:
            parts.append("Likes: " + ", ".join(likes))

        dislikes = PreferenceRepository.list(self.id, liked=False, limit=4)
        if dislikes:
            parts.append("Dislikes: " + ", ".join(dislikes))

        notes = NoteRepository.list(self.id, limit=5)
        if notes:
            parts.append("Notes: " + "; ".join(notes))

        for c in ConversationRepository.recent(limit=4, person_id=self.id):
            try:
                dt = datetime.fromisoformat(c["created_at"]).strftime("%d %b")
            except ValueError:
                dt = ""
            parts.append(
                f"  [{dt}] {self.name}: {c['user_text'][:60]}  |  Bot: {c['bot_text'][:60]}"
            )

        mood = MoodRepository.latest(self.id)
        if mood:
            parts.append(f"Last mood: {mood}")

        return "\n".join(parts)

    def get_greeting(self) -> str:
        name = self.name
        v = self.visit_count
        days_ago = None
        if self.last_seen:
            try:
                days_ago = (datetime.now() - datetime.fromisoformat(self.last_seen)).days
            except ValueError:
                pass

        if v <= 1:
            return f"Hello {name}! Nice to meet you. I will remember you from now on."
        if days_ago is not None:
            if days_ago == 0:
                return f"Welcome back {name}! Good to see you again today."
            if days_ago == 1:
                return f"Hey {name}! I saw you yesterday. How are you doing?"
            if days_ago <= 7:
                return f"Hey {name}! It's been {days_ago} days. Welcome back!"
            return f"Hey {name}! Long time - {days_ago} days! Great to see you again!"
        return f"Welcome back {name}! Visit number {v}. Great to see you!"

    def get_summary(self) -> str:
        v = self.visit_count
        f = len(FactRepository.list(self.id, limit=1000))
        if self.last_seen:
            try:
                ls_str = datetime.fromisoformat(self.last_seen).strftime("%d %b at %I:%M %p")
            except ValueError:
                ls_str = "unknown"
        else:
            ls_str = "never"
        prefs = PreferenceRepository.list(self.id, liked=True, limit=3)
        prefs_str = ", ".join(prefs) if prefs else "nothing noted"
        return (
            f"I know {self.name}. Visited {v} times. Last seen {ls_str}. "
            f"{f} facts. Likes: {prefs_str}."
        )


_person_memories: dict[str, PersonMemory] = {}


def get_person_memory(name: str) -> PersonMemory:
    if name not in _person_memories:
        _person_memories[name] = PersonMemory(name)
    return _person_memories[name]


# ── Global adaptive learning ─────────────────────────────────────────────
_HINDI_WORDS = {
    "kya", "hai", "nahi", "mujhe", "aur", "karo", "batao",
    "tera", "mera", "tum", "aap", "haan", "theek", "accha",
}


def _get_list(key: str) -> list:
    raw = ProfileRepository.get(key)
    return json.loads(raw) if raw else []


def _set_list(key: str, value: list) -> None:
    ProfileRepository.set(key, json.dumps(value))


class AdaptiveLearning:
    """Global, persistent assistant memory backed by the `profile` table
    plus the learned_facts / corrections / frequent_topics tables."""

    def __init__(self):
        self._last_bot_response = ""
        self._last_user_input = ""
        self._pending_correction = False

    def new_session(self) -> int:
        count = int(ProfileRepository.get("session_count", "0")) + 1
        ProfileRepository.set("session_count", count)
        return count

    def record_conversation(self, user_input: str, bot_response: str, lang: str = "en") -> None:
        ConversationRepository.add(user_input, bot_response, lang)

        # Frequent topics (words longer than 4 chars)
        for word in user_input.lower().split():
            if len(word) > 4:
                FrequentTopicRepository.increment(word)
        FrequentTopicRepository.prune()

        # Language preference - detect Hindi words
        words = user_input.lower().split()
        hi_count = sum(1 for w in words if w in _HINDI_WORDS)
        self._update_lang_preference("hi" if hi_count >= 2 else "en")

        # Auto-extract facts from user input (local pattern matching)
        from memory.memory_extractor import extract_facts_local

        for fact in extract_facts_local(user_input):
            self.learn_fact(fact)

        self._last_bot_response = bot_response
        self._last_user_input = user_input

    def _update_lang_preference(self, lang: str) -> None:
        """Rolling-average update: 80% previous + 20% latest observation."""
        prev = ProfileRepository.get("lang_preference", "en")
        if lang == "hi":
            score = 0.8 * (1 if prev == "hi" else 0) + 0.2
            new_lang = "hi" if score > 0.5 else "en"
        else:
            score = 0.8 * (1 if prev == "en" else 0) + 0.2
            new_lang = "en" if score > 0.5 else "hi"
        ProfileRepository.set("lang_preference", new_lang)

    def mark_last_feedback(self, feedback_type: str) -> None:
        last_id = ConversationRepository.last_id()
        if last_id is None:
            return
        ConversationRepository.mark_feedback(last_id, feedback_type)

        pos = int(ProfileRepository.get("feedback_positive", "0"))
        neg = int(ProfileRepository.get("feedback_negative", "0"))
        if feedback_type == "negative":
            neg += 1
            self._pending_correction = True
        else:
            pos += 1
            self._pending_correction = False
        ProfileRepository.set("feedback_positive", pos)
        ProfileRepository.set("feedback_negative", neg)

        total = pos + neg
        if total > 0:
            ratio = pos / total
            ProfileRepository.set("positive_ratio", round(ratio, 2))
            if ratio < 0.4:
                style = "detailed"
            elif ratio > 0.75:
                style = "short"
            else:
                style = "balanced"
            ProfileRepository.set("response_style", style)

    def learn_correction(self, correct_info: str) -> None:
        CorrectionRepository.add(self._last_user_input, self._last_bot_response, correct_info)
        self._pending_correction = False

    def learn_fact(self, fact: str) -> bool:
        return LearnedFactRepository.add(fact)

    def forget_fact(self, keyword: str) -> int:
        kw = keyword.lower().strip()
        removed = LearnedFactRepository.forget(kw)

        for key in ("preferences", "dislikes"):
            old = _get_list(key)
            new = [x for x in old if kw not in x.lower()]
            removed += len(old) - len(new)
            _set_list(key, new)

        DeletedKeywordRepository.add(kw)
        return removed

    def learn_preference(self, item: str, like: bool = True) -> None:
        key = "preferences" if like else "dislikes"
        items = _get_list(key)
        if item not in items:
            items.append(item)
            _set_list(key, items)

    def set_user_name(self, name: str) -> None:
        ProfileRepository.set("name", name)

    def get_user_name(self) -> str | None:
        return ProfileRepository.get("name")

    def build_learning_context(self) -> str:
        parts = []

        name = ProfileRepository.get("name")
        if name:
            parts.append(f"User name: {name}")

        prefs = _get_list("preferences")
        if prefs:
            parts.append(f"User likes: {', '.join(prefs[:6])}")

        dislikes = _get_list("dislikes")
        if dislikes:
            parts.append(f"User dislikes: {', '.join(dislikes[:4])}")

        top_topics = FrequentTopicRepository.top(5)
        if top_topics:
            parts.append("User ke favorite topics: " + ", ".join(t for t, _ in top_topics))

        facts = LearnedFactRepository.list(limit=8)
        if facts:
            parts.append("Remembered facts: " + "; ".join(facts))

        corrections = CorrectionRepository.recent(limit=5)
        if corrections:
            parts.append("Past mistakes to avoid:")
            for c in corrections:
                parts.append(f"  Q: {c['original_query']} - correct answer: {c['correction']}")

        if self._pending_correction:
            parts.append("NOTE: User marked last answer as WRONG. Be careful and accurate.")

        style = ProfileRepository.get("response_style", "balanced")
        ratio = float(ProfileRepository.get("positive_ratio", "0.5"))
        if style == "short":
            parts.append(f"User prefers short answers (satisfaction rate: {int(ratio * 100)}%).")
        elif style == "detailed":
            parts.append(f"User wants more detailed answers (satisfaction rate: {int(ratio * 100)}%).")

        lang_pref = ProfileRepository.get("lang_preference", "en")
        if lang_pref == "hi":
            parts.append("User mostly speaks Hindi - prefer Hinglish or Hindi replies.")

        return "\n".join(parts)

    def get_learning_summary(self) -> str:
        session_count = int(ProfileRepository.get("session_count", "0"))
        facts_count = len(LearnedFactRepository.list(limit=10_000))
        corrections_count = len(CorrectionRepository.recent(limit=10_000))
        name = ProfileRepository.get("name", "unknown")
        top_topics = FrequentTopicRepository.top(3)
        top_str = ", ".join(t for t, _ in top_topics) if top_topics else "nothing yet"
        ratio = float(ProfileRepository.get("positive_ratio", "0.5"))
        style = ProfileRepository.get("response_style", "balanced")
        lang = ProfileRepository.get("lang_preference", "en")
        return (
            f"I have had {session_count} sessions with you. "
            f"I know {facts_count} facts and fixed {corrections_count} mistakes. "
            f"Your name is {name}. You talk most about: {top_str}. "
            f"Satisfaction rate: {int(ratio * 100)}%. "
            f"I am using {style} responses in {lang} language."
        )


_learner: AdaptiveLearning | None = None


def get_learner() -> AdaptiveLearning:
    global _learner
    if _learner is None:
        _learner = AdaptiveLearning()
    return _learner


def forget_keyword_everywhere(keyword: str) -> int:
    """Forget `keyword` from global learning and every cached PersonMemory.

    Returns the total number of facts/preferences removed.
    """
    removed = get_learner().forget_fact(keyword)
    for pm in _person_memories.values():
        removed += pm.forget_fact(keyword)
    return removed
