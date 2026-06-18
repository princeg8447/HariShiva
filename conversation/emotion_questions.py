"""
Emotion-based check-ins and periodic behaviour analysis for HariShiva V2.

- ``maybe_emotion_feedback`` reacts (with a cooldown) to a detected dominant
  emotion, e.g. asking "are you okay?" if the user looks sad.
- ``generate_daily_behavior_analysis`` / ``behavior_summary_spoken`` /
  ``start_behavior_analysis_loop`` port the original's periodic
  self-reflection report, now backed by the SQLite repositories instead of
  ``learning_data.json``.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from datetime import datetime

from app import config
from database.models import (
    BehaviorReportRepository,
    ConversationRepository,
    FrequentTopicRepository,
    LearnedFactRepository,
    ProfileRepository,
)

# ── Emotion-based check-ins ─────────────────────────────────────────────
_EMOTION_COOLDOWN_SECONDS = 5 * 60
_last_emotion_feedback: dict[str, float] = {}

_EMOTION_MESSAGES = {
    "en": {
        "sad": "You look a bit sad. Is everything okay?",
        "angry": "You seem upset. Do you want to talk about it?",
        "fear": "You look a little worried. Is something wrong?",
        "happy": "You look happy today!",
        "surprise": "You look surprised!",
    },
    "hi": {
        "sad": "Aap thode udaas lag rahe hain. Sab theek hai?",
        "angry": "Aap pareshaan lag rahe hain. Kya baat karna chahenge?",
        "fear": "Aap thode chintit lag rahe hain. Kuch problem hai?",
        "happy": "Aap aaj khush lag rahe hain!",
        "surprise": "Aap hairan lag rahe hain!",
    },
}


def maybe_emotion_feedback(dominant_emotion: str, lang: str = "en") -> str | None:
    """Return a spoken reaction to `dominant_emotion`, or None if it's not
    interesting or was already mentioned recently (cooldown)."""
    emotion = (dominant_emotion or "").lower()
    if emotion not in _EMOTION_MESSAGES["en"]:
        return None

    now = time.time()
    last = _last_emotion_feedback.get(emotion, 0.0)
    if now - last < _EMOTION_COOLDOWN_SECONDS:
        return None

    _last_emotion_feedback[emotion] = now
    table = _EMOTION_MESSAGES.get(lang, _EMOTION_MESSAGES["en"])
    return table.get(emotion)


# ── Behaviour analysis ───────────────────────────────────────────────────
_TOPIC_STOPWORDS = {
    "about", "there", "which", "would", "could", "should", "please",
    "something", "anything", "everything", "mujhe", "tumhe", "aapko",
    "matlab", "chahiye", "sakta", "sakti", "sakte", "karna", "karte",
    "karein", "hota", "hoti", "hote", "abhi", "phir", "dobara", "bahut",
    "thoda", "zyada", "kaise", "kyun", "kyon", "kahan", "kitna", "kitni",
    "yahan", "wahan", "really", "actually", "basically", "wonder",
    "today", "yesterday", "tomorrow",
}


def _satisfaction_metric() -> dict:
    pos = int(ProfileRepository.get("feedback_positive", "0") or 0)
    neg = int(ProfileRepository.get("feedback_negative", "0") or 0)
    total = pos + neg
    ratio = (pos / total) if total else None

    if ratio is None:
        status, label = "unknown", "Not enough feedback yet"
    elif ratio >= 0.7:
        status, label = "good", "User seems satisfied"
    elif ratio >= 0.4:
        status, label = "neutral", "Mixed feedback"
    else:
        status, label = "needs_improvement", "User often unsatisfied"

    return {
        "positive": pos,
        "negative": neg,
        "ratio": ratio,
        "status": status,
        "label": label,
    }


def _language_drift_metric() -> dict:
    convos = ConversationRepository.recent(limit=50)
    if len(convos) < 4:
        return {"trend": "not_enough_data", "old_hindi_ratio": None, "new_hindi_ratio": None}

    half = len(convos) // 2
    old_half, new_half = convos[:half], convos[half:]

    def _hi_ratio(rows: list[dict]) -> float:
        if not rows:
            return 0.0
        return sum(1 for r in rows if r.get("lang") == "hi") / len(rows)

    old_ratio = _hi_ratio(old_half)
    new_ratio = _hi_ratio(new_half)
    delta = new_ratio - old_ratio

    if delta > 0.15:
        trend = "shifting_to_hindi"
    elif delta < -0.15:
        trend = "shifting_to_english"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "old_hindi_ratio": round(old_ratio, 2),
        "new_hindi_ratio": round(new_ratio, 2),
    }


def _fact_speed_metric() -> dict:
    facts = LearnedFactRepository.all_with_dates()
    today = datetime.now().date()
    facts_today = 0
    facts_week = 0
    for f in facts:
        try:
            created = datetime.fromisoformat(f["created_at"]).date()
        except (ValueError, TypeError):
            continue
        delta_days = (today - created).days
        if delta_days == 0:
            facts_today += 1
        if delta_days < 7:
            facts_week += 1

    return {
        "total_facts": len(facts),
        "facts_today": facts_today,
        "facts_this_week": facts_week,
    }


def _topical_focus_metric() -> dict:
    top = FrequentTopicRepository.top(n=10)
    filtered = [(w, c) for w, c in top if w not in _TOPIC_STOPWORDS]
    return {
        "top_topics": filtered[:5],
    }


def generate_daily_behavior_analysis() -> dict:
    """Compute a fresh behaviour report, persist it, and return it."""
    report = {
        "generated_at": datetime.now().isoformat(),
        "satisfaction": _satisfaction_metric(),
        "language_drift": _language_drift_metric(),
        "fact_speed": _fact_speed_metric(),
        "topical_focus": _topical_focus_metric(),
        "session_count": int(ProfileRepository.get("session_count", "0") or 0),
    }
    BehaviorReportRepository.save(json.dumps(report))
    return report


def behavior_summary_spoken(report: dict, lang: str = "en") -> str:
    """Turn `report` into a short (<=40 word) spoken summary."""
    sat = report.get("satisfaction", {})
    drift = report.get("language_drift", {})
    facts = report.get("fact_speed", {})
    topics = report.get("topical_focus", {}).get("top_topics", [])

    if lang == "hi":
        parts = [sat.get("label", "")]
        if facts.get("facts_today"):
            parts.append(f"Aaj {facts['facts_today']} nayi baatein seekhi.")
        if drift.get("trend") == "shifting_to_hindi":
            parts.append("Aap zyada Hindi mein baat kar rahe hain.")
        elif drift.get("trend") == "shifting_to_english":
            parts.append("Aap zyada English mein baat kar rahe hain.")
        if topics:
            top_word = topics[0][0]
            parts.append(f"Aapko '{top_word}' topic mein zyada interest hai.")
    else:
        parts = [sat.get("label", "")]
        if facts.get("facts_today"):
            parts.append(f"I learned {facts['facts_today']} new things about you today.")
        if drift.get("trend") == "shifting_to_hindi":
            parts.append("You've been speaking more Hindi lately.")
        elif drift.get("trend") == "shifting_to_english":
            parts.append("You've been speaking more English lately.")
        if topics:
            top_word = topics[0][0]
            parts.append(f"You seem interested in '{top_word}' lately.")

    summary = " ".join(p for p in parts if p)
    words = summary.split()
    if len(words) > 40:
        summary = " ".join(words[:40])
    return summary


def start_behavior_analysis_loop() -> threading.Thread:
    """Start a daemon thread that periodically refreshes the behaviour report."""

    def _loop():
        time.sleep(90)
        while True:
            try:
                generate_daily_behavior_analysis()
            except Exception:
                pass
            time.sleep(config.BEHAVIOR_INTERVAL_SECONDS)

    t = threading.Thread(target=_loop, daemon=True, name="behavior-analysis")
    t.start()
    return t
