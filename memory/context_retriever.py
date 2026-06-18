"""
Builds combined textual context for LLM prompts from the global learner
and (optionally) the currently-visible person's memory.
"""

from database.models import ConversationRepository, PersonRepository
from memory.memory_manager import get_learner, get_person_memory


def get_full_context(person_name: str | None = None) -> str:
    """Combine global learning context with a specific person's memory context."""
    parts = []

    learning_ctx = get_learner().build_learning_context()
    if learning_ctx:
        parts.append(learning_ctx)

    if person_name:
        parts.append(get_person_memory(person_name).build_context())

    return "\n\n".join(p for p in parts if p)


def get_recent_conversation(limit: int = 6, person_name: str | None = None) -> list[dict]:
    """Return the most recent conversation turns, optionally scoped to one person."""
    person_id = None
    if person_name:
        record = PersonRepository.get(person_name)
        person_id = record["id"] if record else None

    return ConversationRepository.recent(limit=limit, person_id=person_id)
