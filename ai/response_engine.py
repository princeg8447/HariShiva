"""
Response engine for HariShiva V2.

Combines:
- a live web-search pipeline (DuckDuckGo Instant Answer + Wikipedia, no API
  key needed) for questions that need up-to-date facts, and
- the main generate_response() entry point that builds the full prompt
  (system instructions + memory context + live data + recent history) and
  calls the LLM via ai.llm_manager.
"""

import re
from datetime import datetime

import requests

from ai.llm_manager import chat_completion
from memory.context_retriever import get_full_context, get_recent_conversation
from memory.memory_manager import get_learner, get_person_memory

_USER_AGENT = {"User-Agent": "HariShivaBot/2.0"}

# ── Web search triggers ──────────────────────────────────────────────────
_WEB_SEARCH_TRIGGERS = re.compile(
    r"\b(who is|who are|kaun hai|kaun hain|kaun the|kaun tha|kaun thi|"
    r"cm of|pm of|ceo of|president of|"
    r"chief minister|prime minister|pradhan mantri|mukhyamantri|mukhya mantri|"
    r"rashtrapati|rajyapal|governor of|"
    r"current|abhi|latest|new|naya|nayi|aaj kal|recently|"
    r"election|result|score|match|news|khabar|winner|"
    r"discovered|launched|released|appointed|died|born|"
    r"capital of|currency of|founder of|invented|"
    r"ipl|world cup|olympic|award|oscar|nobel)\b",
    re.IGNORECASE,
)

# ── Hindi -> English query normalisation ────────────────────────────────
_HINDI_QUERY_MAP = [
    ("pradhan mantri", "prime minister"),
    ("pradhaan mantri", "prime minister"),
    ("mukhyamantri", "chief minister"),
    ("mukhya mantri", "chief minister"),
    ("rashtrapati", "president"),
    ("rajyapal", "governor"),
    ("bharat ke", "india"),
    ("bharat ka", "india"),
    ("bharat ki", "india"),
    ("bharat", "india"),
    ("india ke", "india"),
    ("india ka", "india"),
    ("india ki", "india"),
    ("kaun hain", ""),
    ("kaun hai", ""),
    ("kaun the", ""),
    ("kaun tha", ""),
    ("ke baare mein", ""),
    ("ke bare mein", ""),
    ("abhi", "current"),
    ("aaj kal", "current"),
    ("batao", ""),
    ("bolo", ""),
    ("bata", ""),
]

_QUERY_NOISE = (
    "who is", "who are", "who was", "who were",
    "tell me about", "what is", "i want to know",
    "can you tell", "please tell",
)

_ROLE_TERMS = (
    "prime minister", "chief minister", "president",
    "governor", "minister of", "pm of", "cm of",
)

_ROLE_ARTICLE_MARKERS = (
    " of ", " in ", "list of", "history of",
    "chief minister of", "prime minister of", "governor of",
    "president of", "election", "office (", "ministry of",
    "government of", "cabinet of", "council of", "advisor",
)

_ROLE_BARE_TITLES = (
    "prime minister", "chief minister", "president", "governor",
    "prime ministers", "chief ministers",
)

_FAKE_NAMES = {
    "the", "this", "india", "a", "an", "he", "she",
    "it", "they", "as", "there", "that", "which",
}

_HOLDER_PATTERNS = [
    r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3}),\s+who\s+is\s+the\s+current",
    r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\s+is\s+the\s+\w+(?:\s+and)?\s+current",
    r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\s+is\s+(?:the\s+)?(?:current|incumbent|serving)",
    r"(?:current|incumbent|serving)\s+[\w\s]{0,25}?is\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\s+has\s+been\s+(?:serving\s+as|the)",
    r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\s+(?:became|was\s+(?:elected|appointed|sworn|named))\s+"
    r"(?:as\s+)?(?:the\s+)?(?:prime\s+minister|president|chief\s+minister|governor|cm|pm)",
    r"(?:prime\s+minister|president|chief\s+minister|governor)\s+of\s+\w+\s+is\s+"
    r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})",
    r"([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\s+(?:serves?|served)\s+as\s+(?:the\s+)?"
    r"(?:prime\s+minister|president|chief\s+minister)",
]


def _normalize_search_query(prompt: str) -> str:
    """Translate Hinglish/Hindi political queries to clean English keywords."""
    q = prompt.lower().strip()
    for hindi, eng in _HINDI_QUERY_MAP:
        q = q.replace(hindi, eng)
    q = re.sub(r"\bpm\b", "prime minister", q)
    q = re.sub(r"\bcm\b", "chief minister", q)
    for noise in _QUERY_NOISE:
        q = q.replace(noise, " ")
    q = " ".join(q.split())
    if any(role in q for role in _ROLE_TERMS):
        if "current" not in q and "history" not in q and "list" not in q:
            q = "current " + q
    return q.strip()


def _extract_current_holder(extract: str) -> str:
    """Pull a current officeholder's name out of a Wikipedia role article."""
    for pat in _HOLDER_PATTERNS:
        m = re.search(pat, extract, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            for conn in ("and ", "or ", "the ", "a ", "an ", "by ", "as "):
                if name.lower().startswith(conn):
                    name = name[len(conn):].strip()
                    break
            if name.lower() not in _FAKE_NAMES and len(name) > 3:
                return name
    return ""


def _wiki_fetch_role_extract(title: str) -> str:
    """Fetch the full article body (up to 3000 chars) for role articles."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "prop": "extracts",
                "titles": title, "format": "json", "explaintext": "1",
            },
            timeout=8,
            headers=_USER_AGENT,
        ).json()
        pages = resp.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "").strip()
            if extract:
                return extract[:3000]
    except Exception:
        pass
    return ""


def _wiki_fetch_extract(title: str) -> str:
    """Fetch the first 5 sentences of a Wikipedia article by title."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "prop": "extracts", "exsentences": "5",
                "exintro": "1", "titles": title, "format": "json",
                "explaintext": "1",
            },
            timeout=5,
            headers=_USER_AGENT,
        ).json()
        pages = resp.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract", "").strip()
            if extract:
                return extract[:400]
    except Exception:
        pass
    return ""


def _ddg_search(query: str) -> str:
    """DuckDuckGo Instant Answer API - free, no key."""
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=5,
            headers=_USER_AGENT,
        ).json()

        if r.get("Answer"):
            return r["Answer"][:200]

        abstract = r.get("AbstractText", "")
        if abstract:
            _role_keywords = (
                "head of government", "head of state", "is the office",
                "is a political position", "is the position",
                "executive authority", "is the highest",
            )
            if any(kw in abstract.lower() for kw in _role_keywords):
                holder = _extract_current_holder(abstract)
                if holder:
                    r2 = requests.get(
                        "https://api.duckduckgo.com/",
                        params={"q": holder, "format": "json",
                                "no_html": "1", "skip_disambig": "1"},
                        timeout=5,
                        headers=_USER_AGENT,
                    ).json()
                    person_abstract = r2.get("AbstractText", "")
                    if person_abstract:
                        return person_abstract[:400]
            return abstract[:400]

        topics = r.get("RelatedTopics", [])
        if topics and isinstance(topics[0], dict) and topics[0].get("Text"):
            return topics[0]["Text"][:300]
    except Exception:
        pass
    return ""


def _is_role_article(title: str) -> bool:
    t = title.lower().strip()
    if any(m in t for m in _ROLE_ARTICLE_MARKERS):
        return True
    bare = re.sub(r"\s*\(.*?\)", "", t).strip()
    return bare in _ROLE_BARE_TITLES


def _wiki_search(query: str) -> str:
    """Wikipedia search + extracts API, with current-officeholder resolution."""
    try:
        is_role_q = any(role in query.lower() for role in _ROLE_TERMS)
        wiki_query = (query + " person") if is_role_q else query

        search = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": wiki_query,
                "format": "json", "srlimit": "8",
            },
            timeout=5,
            headers=_USER_AGENT,
        ).json()
        results = search.get("query", {}).get("search", [])
        if not results:
            return ""

        is_who_query = re.search(
            r"\b(who is|who are|kaun hai|kaun hain|current|incumbent|"
            r"cm of|pm of|chief minister|prime minister|governor|president|"
            r"ceo|founder|pradhan mantri|mukhyamantri|rashtrapati)\b",
            query, re.IGNORECASE,
        )

        if is_who_query:
            results = sorted(results, key=lambda r: _is_role_article(r["title"]))

        # Pass 1: person articles
        for r in results[:5]:
            title = r["title"]
            if _is_role_article(title):
                continue
            extract = _wiki_fetch_extract(title)
            if extract:
                return extract

        # Pass 2: mine current holder from role articles
        role_fallback = ""
        for r in results[:5]:
            title = r["title"]
            if _is_role_article(title):
                extract = _wiki_fetch_role_extract(title)
                if not extract:
                    continue
                holder = _extract_current_holder(extract)
                if holder:
                    person_extract = _wiki_fetch_extract(holder)
                    if person_extract:
                        return person_extract
                    ddg_person = _ddg_search(holder)
                    if ddg_person:
                        return ddg_person
                if not role_fallback:
                    role_fallback = extract
            else:
                extract = _wiki_fetch_extract(title)
                if extract:
                    return extract

        if role_fallback:
            return role_fallback
    except Exception:
        pass
    return ""


def web_search(query: str) -> str:
    """Normalise query, try DuckDuckGo, then Wikipedia, then raw query on Wikipedia."""
    clean_q = _normalize_search_query(query)

    result = _ddg_search(clean_q)
    if not result:
        result = _wiki_search(clean_q)
    if not result and clean_q != query:
        result = _wiki_search(query)
    return result


def needs_web_search(prompt: str) -> bool:
    """Return True if this question likely needs live/current data."""
    return bool(_WEB_SEARCH_TRIGGERS.search(prompt))


# ── Main response generation ─────────────────────────────────────────────
_BASE_SYSTEM_PROMPT = (
    "You are HariShiva, a smart warm AI friend and teacher on Raspberry Pi with a LIVE CAMERA. "
    "You CAN see the physical environment through your eyes - NEVER say you cannot see or perceive the world. "
    "HOW TO TALK: Be warm and personal, speak naturally like a real human.\n"
    "1. Answer ONLY what the user asked - nothing else.\n"
    "2. NEVER offer to tell a story or say 'chalo' unless explicitly asked.\n"
    "3. Keep replies under 55 words. Be engaging and natural, not robotic.\n"
    "4. Reply in the SAME language the user used (Hindi/Hinglish/English).\n"
    "5. If you don't know something, say so in one sentence.\n"
    "6. If the user's name is in the MEMORY CONTEXT below, use it confidently. "
    "Never make up a name that is NOT in memory.\n"
    "7. Use TODAY's date above when asked about date, day, or time.\n"
    "8. If LIVE DATA is provided below, use it - it is more accurate than your training.\n"
    "9. NEVER say 'I am a text-based AI' or 'I cannot see' - you have eyes and can see the world."
)


def generate_response(
    prompt: str,
    lang_code: str,
    person_name: str | None = None,
    scene_description: str | None = None,
    city: str = "Delhi",
) -> str:
    """Build the full prompt (system + memory + live data + history) and call the LLM."""
    now = datetime.now()
    base_sys = (
        f"TODAY: {now.strftime('%A, %d %B %Y')}  |  TIME: {now.strftime('%I:%M %p')}  "
        f"|  City: {city}\n" + _BASE_SYSTEM_PROMPT
    )

    if person_name:
        base_sys += f"\n\nCURRENT USER: {person_name}. You know this person - address them by name."

    messages = [{"role": "system", "content": base_sys}]

    ctx_block = get_full_context(person_name)
    if ctx_block:
        messages.append({"role": "system", "content": ctx_block})

    if scene_description:
        messages.append({"role": "system", "content": f"LIVE CAMERA: {scene_description}"})

    if needs_web_search(prompt):
        live = web_search(prompt)
        if live:
            messages.append({
                "role": "system",
                "content": f"LIVE DATA (use this, it is more recent than your training):\n{live}",
            })

    # Only the last 2 turns - prevents inheriting hallucinated context
    for turn in get_recent_conversation(limit=2, person_name=person_name):
        messages.append({"role": "user", "content": turn["user_text"]})
        messages.append({"role": "assistant", "content": turn["bot_text"]})

    if lang_code == "en":
        lang_prefix = "[IMPORTANT: Reply ONLY in English. Do NOT use Hindi or Devanagari.]\n"
    else:
        lang_prefix = "[IMPORTANT: Reply in Hindi or Hinglish. Do NOT reply in English only.]\n"
    messages.append({"role": "user", "content": lang_prefix + prompt})

    try:
        reply = chat_completion(messages, max_tokens=120, temperature=0.4)
    except Exception:
        return "Sorry, I could not connect to Groq right now."

    get_learner().record_conversation(prompt, reply, lang=lang_code)
    if person_name:
        get_person_memory(person_name).record_conversation(prompt, reply, lang=lang_code)

    return reply
