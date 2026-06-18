"""
Small bilingual text helpers shared by the conversation layer.

`ai.response_engine` already builds the LLM system/user prompts; this module
just holds the short canned replies that `offline_conversation` returns
directly without going through the LLM.
"""


def tr(lang: str, en: str, hi: str) -> str:
    """Pick the English or Hindi/Hinglish variant of a canned reply."""
    return en if lang == "en" else hi


def not_understood(lang: str) -> str:
    return tr(
        lang,
        "Sorry, I didn't catch that. Could you say it again?",
        "Maaf kijiye, samajh nahi aaya. Phir se boliye?",
    )


def vision_unavailable(lang: str) -> str:
    return tr(
        lang,
        "My eyes (camera) aren't active right now.",
        "Abhi meri aankhein (camera) active nahi hain.",
    )


def feature_unavailable(lang: str) -> str:
    return tr(
        lang,
        "That feature isn't available right now.",
        "Yeh feature abhi available nahi hai.",
    )


def need_a_name(lang: str) -> str:
    return tr(
        lang,
        "Whose name do you mean?",
        "Kiska naam bol rahe ho?",
    )
