"""Smoke test for the AI layer - run manually with the venv python."""

from database.database import init_db
from ai.response_engine import (
    _normalize_search_query,
    generate_response,
    needs_web_search,
    web_search,
)

init_db()

q = "India ka PM kaun hai"
print("normalize:", _normalize_search_query(q))
print("needs_web_search:", needs_web_search(q))

result = web_search("capital of france")
print("web_search('capital of france'):", result[:150])

# Without GROQ_API_KEY this should fail gracefully
print("generate_response:", generate_response("hello", "en"))
