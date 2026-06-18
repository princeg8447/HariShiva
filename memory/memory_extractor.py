"""
Fact extraction for HariShiva V2.

Two extraction strategies, ported from the original:

- extract_facts_local(): zero-cost pattern matching, runs synchronously on
  every turn (used by AdaptiveLearning.record_conversation).
- extract_facts_async(): Groq-based extraction of personal facts from
  natural conversation, runs in a background thread per turn.
"""

import json
import threading

from database.models import DeletedKeywordRepository

_FACT_PATTERNS = [
    ("main ", "User is: "),
    ("i am ", "User is: "),
    ("mera naam ", "User name: "),
    ("my name is ", "User name: "),
    ("i work ", "User work: "),
    ("i study ", "User study: "),
    ("i live ", "User lives: "),
    ("mujhe pasand hai ", "User likes: "),
    ("i like ", "User likes: "),
    ("mujhe pasand nahi ", "User dislikes: "),
    ("i don't like ", "User dislikes: "),
    ("meri age ", "User age: "),
    ("my age is ", "User age: "),
    ("i am from ", "User from: "),
    ("main rehta ", "User lives: "),
]


def extract_facts_local(text: str) -> list[str]:
    """Pattern-based extraction - no API calls. Returns short fact strings."""
    lower = text.lower()
    facts = []
    for trigger, label in _FACT_PATTERNS:
        if trigger in lower:
            rest = lower.split(trigger, 1)[1].strip()
            chunk = " ".join(rest.split()[:5])  # max 5 words
            if len(chunk) > 2:
                fact = label + chunk
                if not DeletedKeywordRepository.is_tombstoned(fact):
                    facts.append(fact)
    return facts


_EXTRACT_SYSTEM_PROMPT = (
    "Extract personal facts about the speaker from their message. "
    "Output ONLY a JSON array of short strings (max 6 words each). "
    'Examples: ["works as software engineer", "lives in Mumbai", '
    '"likes cricket", "age 22"]. '
    "Only include facts explicitly stated (job, location, age, hobbies, "
    "family, studies). If nothing personal is mentioned, output []."
)


def extract_facts_async(person_mem, user_text: str) -> None:
    """Background thread: ask Groq for personal facts and add them to person_mem."""
    if not person_mem:
        return

    def _run():
        try:
            from ai.groq_client import get_client

            client = get_client()
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                max_tokens=80,
            )
            raw = resp.choices[0].message.content.strip().strip("`").strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
            facts = json.loads(raw)
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, str) and 3 < len(fact) < 80:
                        if DeletedKeywordRepository.is_tombstoned(fact):
                            continue
                        person_mem.add_fact(fact)
        except Exception:
            pass  # non-critical, silent fail

    threading.Thread(target=_run, daemon=True).start()
