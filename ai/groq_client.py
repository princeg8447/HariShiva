"""
Thin wrapper around the Groq SDK.

The original project hardcoded the API key directly in source. V2 reads it
from the environment (app.config.GROQ_API_KEY / .env) and only creates the
client lazily, on first use.
"""

from groq import Groq

from app import config

_client: Groq | None = None


def get_client() -> Groq:
    global _client
    if _client is None:
        if not config.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to HariShiva_V2/.env"
            )
        _client = Groq(api_key=config.GROQ_API_KEY)
    return _client
