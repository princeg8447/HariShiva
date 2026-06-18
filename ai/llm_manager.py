"""
Small wrapper around chat-completion calls so the rest of the app doesn't
need to know which model/provider is in use.
"""

from ai.groq_client import get_client

MODEL_NAME = "llama-3.1-8b-instant"


def chat_completion(
    messages: list[dict],
    max_tokens: int = 120,
    temperature: float = 0.4,
    model: str = MODEL_NAME,
) -> str:
    """Run a chat completion and return the assistant's reply text."""
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()
