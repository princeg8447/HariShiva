"""
Offline command dispatcher and main conversation loop for HariShiva V2.

`handle_custom_commands` ports the original's `handle_custom_commands` -
a big "is this a command we handle ourselves" dispatcher that runs *before*
falling back to the LLM (`ai.response_engine.generate_response`). It covers
adaptive-learning commands (feedback, corrections, remember/forget,
preferences), person-memory commands, face/vision queries,
weather/date/time, and small-talk shortcuts.

`run_conversation_loop` ports the original `main()` voice loop: wait for the
wake word, listen, dispatch, speak the reply.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime

import requests

from app import config
from ai.response_engine import generate_response
from conversation import prompt_builder
from conversation.emotion_questions import (
    behavior_summary_spoken,
    generate_daily_behavior_analysis,
    start_behavior_analysis_loop,
)
from memory.memory_extractor import extract_facts_async
from memory.memory_manager import forget_keyword_everywhere, get_learner, get_person_memory
from voice import audio_manager
from voice.speech_to_text import (
    calibrate_microphone,
    infer_lang,
    listen,
    listen_for_stop,
    wait_for_wake_word,
)
from voice.text_to_speech import speak

tr = prompt_builder.tr


# ── Active-person state (set by the vision thread when a known face is seen)
_active_person_name: str | None = None
_active_person_lock = threading.Lock()


def set_active_person(name: str | None) -> None:
    global _active_person_name
    with _active_person_lock:
        _active_person_name = name


def get_active_person() -> str | None:
    with _active_person_lock:
        return _active_person_name


# ── Lazy hooks into the (not-yet-loaded) vision layer ────────────────────
def _vision_get_scene_objects() -> list[str]:
    try:
        from vision.face_detector import get_scene_objects
    except (ImportError, ModuleNotFoundError):
        return []
    try:
        return get_scene_objects()
    except Exception:
        return []


def _vision_recognize_face_now() -> str | None:
    """Returns the recognised name, "" if a face was seen but not recognised,
    or None if vision/face recognition isn't available at all."""
    try:
        from vision.face_recognition import recognize_face_now
    except (ImportError, ModuleNotFoundError):
        return None
    try:
        name, _confidence = recognize_face_now()
        return name or ""
    except Exception:
        return None


def _vision_list_known_people() -> list[str] | None:
    try:
        from vision.face_recognition import list_known_people
    except (ImportError, ModuleNotFoundError):
        return None
    try:
        return list_known_people()
    except Exception:
        return None


def _vision_enroll(name: str) -> bool:
    try:
        from vision.enrollment import enrollment_queue
    except (ImportError, ModuleNotFoundError):
        return False
    try:
        enrollment_queue.put(name)
        return True
    except Exception:
        return False


def _vision_delete_person(name: str) -> bool:
    try:
        from vision.enrollment import delete_person
    except (ImportError, ModuleNotFoundError):
        return False
    try:
        return delete_person(name)
    except Exception:
        return False


# ── Weather / date / time ─────────────────────────────────────────────────
_last_city: str | None = None

_CITY_PATTERNS = [
    r'(?:weather|mausam|temperature|temp|barish|rain|aaj)\s+(?:in|of|at|for|about)\s+([a-z][a-z ]{1,20})',
    r'([a-z][a-z ]{1,20})\s+(?:ka|ki|ke|mein|me)\s+(?:mausam|weather|temp|barish)',
    r'([a-z][a-z ]{1,20})\s+(?:weather|temperature|temp|mausam)',
    r'(?:weather|mausam|temp|temperature)\s+([a-z][a-z ]{1,20})',
]
_CITY_STOPWORDS = {
    'tell', 'me', 'what', 'is', 'the', 'in', 'at', 'of', 'for',
    'about', 'today', 'aaj', 'kal', 'batao', 'bata', 'how', 'like',
}


def _extract_city_from_text(text: str) -> str | None:
    """Pull a city name out of a weather/temperature query."""
    t = text.lower().strip()
    for p in _CITY_PATTERNS:
        m = re.search(p, t)
        if m:
            candidate = m.group(1).strip()
            words = [w for w in candidate.split() if w not in _CITY_STOPWORDS]
            if words:
                return ' '.join(words).title()
    return None


def get_weather(city: str | None = None) -> str:
    """Fetch live weather and return a natural-language string."""
    global _last_city
    city = city or _last_city or config.DEFAULT_CITY
    _last_city = city

    if not config.WEATHER_API_KEY:
        return "Weather lookup isn't configured (missing WEATHER_API_KEY)."

    try:
        r = requests.get(
            "http://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={config.WEATHER_API_KEY}&units=metric",
            timeout=6,
        ).json()
        if r.get("cod") != 200:
            return f"Sorry, I couldn't find weather data for {city}."
        main = r["main"]
        temp = round(main["temp"])
        feels = round(main["feels_like"])
        humidity = main["humidity"]
        desc = r["weather"][0]["description"].capitalize()
        wind_kmh = round(r["wind"]["speed"] * 3.6)
        t_min = round(main["temp_min"])
        t_max = round(main["temp_max"])
        time_str = datetime.now().strftime("%I:%M %p")
        return (
            f"{city} mein abhi {temp}°C hai, feels like {feels}°C. "
            f"{desc}. "
            f"Min {t_min}°C, Max {t_max}°C. "
            f"Humidity {humidity}%, Wind {wind_kmh} km/h. "
            f"Time: {time_str}."
        )
    except Exception as e:
        return f"Weather fetch failed: {e}"


def get_current_datetime() -> str:
    return datetime.now().strftime("%A, %d %B %Y — %I:%M %p")


# ── Trigger phrase tables ─────────────────────────────────────────────────
_POSITIVE_FEEDBACK = [
    "sahi hai", "sahi jawab", "bilkul sahi", "thik hai bhai", "perfect",
    "that's right", "thats right", "correct answer", "good answer", "well done",
]
_NEGATIVE_FEEDBACK = [
    "galat hai", "galat jawab", "ghalat hai", "wrong answer",
    "that's wrong", "thats wrong", "not correct", "bakwas",
]
_CORRECTION_TRIGGERS = [
    "actually ", "correct answer is ", "sahi jawab hai ", "asal mein ",
]
_REMEMBER_FACT_TRIGGERS = [
    "yaad rakh ki ", "yaad rakho ki ", "yaad rakhna ki ",
    "remember that ", "remember ki ",
]
_FORGET_TRIGGERS = [
    "bhool ja ", "bhul ja ", "bhool jao ", "forget that ", "forget about ", "forget ",
]
_LIKE_TRIGGERS = [
    "mujhe pasand hai ", "i like ", "mujhe acha lagta hai ", "mujhe achha lagta hai ",
]
_DISLIKE_TRIGGERS = [
    "mujhe pasand nahi ", "i don't like ", "i dont like ",
    "mujhe acha nahi lagta ", "mujhe achha nahi lagta ",
]
_LEARNING_SUMMARY_TRIGGERS = [
    "what have you learned", "tumne kya seekha", "apne baare mein kya seekha", "learning summary",
]
_BEHAVIOR_TRIGGERS = [
    "behavior analysis", "behaviour analysis", "apna analysis batao", "kaisa chal raha hai",
]
_FACTS_LIST_TRIGGERS = [
    "kya yaad hai", "what do you remember", "tumhe kya yaad hai", "what facts do you know",
]
_ENROLL_TRIGGERS = [
    "enroll my face", "remember my face", "mera chehra yaad rakho", "mera face yaad rakho",
]
_FORGET_FACE_PATTERNS = [
    r"forget (?:the )?face of (\w+)",
    r"delete (?:the )?face of (\w+)",
    r"(\w+) ka chehra bhool ja",
    r"(\w+) ka chehra bhula do",
]
_REMEMBER_ABOUT_PATTERNS = [
    r"remember about (\w+)\s*[:\-]\s*(.+)",
    r"(\w+) ke baare mein yaad rakho ki (.+)",
]
_LIKES_PATTERNS = [
    r"(\w+) likes? (.+)",
    r"(\w+) ko (.+) pasand hai",
]
_TELL_ME_ABOUT_PATTERNS = [
    r"(?:tell me about|what do you know about) (\w+)",
    r"(\w+) ke baare mein batao",
]
_REMEMBER_THIS_TRIGGERS = [
    "remember this:", "remember this -", "remember this ",
]
_KNOWN_PEOPLE_TRIGGERS = [
    "who do you know", "kaun kaun ko jaante ho", "known people",
    "kitne logon ko jaante ho", "how many people do you know",
]
_WHO_IS_THAT_TRIGGERS = [
    "who is that", "who am i", "yeh kaun hai", "ye kaun hai", "main kaun hoon",
]
_HOW_MANY_PEOPLE_TRIGGERS = [
    "how many people", "kitne log", "kitne insaan",
]
_SEE_TRIGGERS = [
    "what can you see", "what do you see", "what are you seeing",
    "what you can see", "what you see", "what do you look",
    "what can you look", "what is visible", "describe what you see",
    "describe the scene", "look around", "what's around",
    "kya dikh raha hai", "kya dekh rahe ho", "kya nazar aa raha",
    "camera mein kya hai", "camera mein kya dikh",
    "aankhen kholo", "batao kya dikh raha", "abhi kya dikh raha",
    "what is in front", "saamne kya hai", "aage kya hai",
    "i am asking what you can see", "asking what you can see",
    "क्या दिख रहा", "क्या देख रहे",
    "कैमरे में क्या", "सामने क्या है",
]
_REMEMBER_ME_TRIGGERS = [
    "do you remember me", "mujhe pehchante ho", "mujhe yaad hai", "kya tujhe yaad hai mera",
]
_KNOW_ABOUT_ME_TRIGGERS = [
    "what do you know about me", "mere baare mein kya pata hai", "meri memory batao",
]
_CAP_TRIGGERS = [
    "what can you do", "what do you do", "your capabilities", "your features",
    "help me", "how can you help", "kya kar sakte ho", "kya kar sakti ho",
    "tumhara kaam kya hai", "aap kya kar sakte", "tum kya kar sakte",
    "tell me about yourself", "apne baare mein batao", "introduce yourself",
    "what are you", "aap kaun ho", "tum kaun ho", "tu kya hai",
]
_NAME_INTRO_TRIGGERS = [
    ("my name is ", 1), ("mera naam ", 1), ("mera naam hai ", 1),
    ("main hoon ", 1), ("i am ", 1), ("i'm ", 1),
]
_NAME_FALSE_POSITIVES = {
    "a", "an", "the", "not", "no", "going", "here",
    "coming", "just", "very", "really", "fine", "ok",
    "okay", "sorry", "trying", "glad", "good", "bad",
    # emotions / moods - "I am sad" etc. is not a name introduction
    "sad", "happy", "angry", "upset", "depressed", "tired", "exhausted",
    "sleepy", "bored", "excited", "worried", "anxious", "nervous", "scared",
    "afraid", "frustrated", "annoyed", "lonely", "stressed", "confused",
    "disappointed", "hurt", "embarrassed", "jealous", "proud", "grateful",
    "thankful", "calm", "relaxed", "content", "miserable", "devastated",
    "heartbroken", "furious", "irritated", "overwhelmed", "hopeful",
    "curious", "surprised", "shocked", "cheerful", "joyful", "gloomy",
    "restless", "energetic", "weak", "strong", "hungry", "thirsty", "sick",
    "ill", "unwell", "well", "healthy", "great", "awesome", "amazing",
    "fantastic", "wonderful", "terrible", "awful", "alright", "busy", "free",
    "ready", "done", "alone", "lost", "stuck",
    # present-participle continuations - "I am feeling/going/doing ..."
    "feeling", "doing", "gonna", "about", "working", "studying", "learning",
    "thinking", "wondering", "looking", "talking", "listening", "watching",
    "missing", "loving", "hating", "hoping", "planning", "waiting",
    "leaving", "staying", "playing", "eating", "drinking", "sleeping",
    "walking", "running", "sitting", "standing", "crying", "laughing",
}
# Specific phrasings only - bare words like "time"/"date"/"today" are too
# common in unrelated sentences (e.g. "today I am sad") and would hijack
# them into a date/time announcement instead of a real reply.
_TIME_TRIGGERS = [
    "what time", "what's the time", "whats the time", "time is it",
    "current time", "time now", "samay kya", "kitne baje", "baje hain",
    "baj gaya", "baj gayi",
]
_DATE_TRIGGERS = [
    "what's the date", "whats the date", "what is the date",
    "which date", "what date", "today's date", "todays date", "date today",
    "what day is it", "which day is it", "what's the day", "whats the day",
    "aaj ki tarikh", "aaj ka din", "aaj kaun sa din", "aaj kaunsa din", "aaj konsa din",
    "kaun sa din", "konsa din", "kaunsa din", "din kya hai", "tarikh kya hai",
    "kya tarikh", "kya din", "tarikh batao", "date batao",
]


def handle_custom_commands(text: str, lang: str) -> str | None:
    """Try to handle `text` ourselves. Return a reply string, or None to fall
    back to the LLM (ai.response_engine.generate_response)."""
    lower = text.lower().strip()
    learner = get_learner()
    active_name = get_active_person()
    active_pm = get_person_memory(active_name) if active_name else None

    # ── Feedback on the previous answer ──────────────────────────────────
    if any(t in lower for t in _POSITIVE_FEEDBACK):
        learner.mark_last_feedback("positive")
        return tr(lang, "Glad that helped!", "Accha, khushi hui ki kaam aaya!")

    if any(t in lower for t in _NEGATIVE_FEEDBACK):
        learner.mark_last_feedback("negative")
        return tr(
            lang,
            "Sorry about that. You can tell me the correct answer by saying "
            "'actually ...' and I'll remember it.",
            "Maaf kijiye. Sahi jawab batane ke liye 'actually ...' bol sakte hain, main yaad rakhunga.",
        )

    # ── Corrections ───────────────────────────────────────────────────────
    for trig in _CORRECTION_TRIGGERS:
        if trig in lower:
            correction = text[lower.index(trig) + len(trig):].strip()
            if correction:
                learner.learn_correction(correction)
                return tr(lang, "Got it, thanks for the correction!", "Theek hai, sahi karne ke liye dhanyavaad!")

    # ── Remember a fact ───────────────────────────────────────────────────
    for trig in _REMEMBER_FACT_TRIGGERS:
        if trig in lower:
            fact = text[lower.index(trig) + len(trig):].strip()
            if fact:
                learner.learn_fact(fact)
                if active_pm:
                    active_pm.add_fact(fact)
                return tr(lang, "Okay, I'll remember that.", "Theek hai, yaad rakh liya.")

    # ── Forget something ─────────────────────────────────────────────────
    for trig in _FORGET_TRIGGERS:
        if trig in lower:
            keyword = text[lower.index(trig) + len(trig):].strip()
            if keyword:
                removed = forget_keyword_everywhere(keyword)
                if removed:
                    return tr(lang, "Okay, I've forgotten that.", "Theek hai, woh bhula diya.")
                return tr(lang, "I didn't have that remembered.", "Yeh mujhe yaad nahi tha.")

    # ── Preferences ───────────────────────────────────────────────────────
    for trig in _LIKE_TRIGGERS:
        if trig in lower:
            item = text[lower.index(trig) + len(trig):].strip()
            if item:
                learner.learn_preference(item, like=True)
                if active_pm:
                    active_pm.add_preference(item, like=True)
                return tr(lang, f"Noted - you like {item}.", f"Theek hai, aapko {item} pasand hai, yaad rakh liya.")

    for trig in _DISLIKE_TRIGGERS:
        if trig in lower:
            item = text[lower.index(trig) + len(trig):].strip()
            if item:
                learner.learn_preference(item, like=False)
                if active_pm:
                    active_pm.add_preference(item, like=False)
                return tr(lang, f"Noted - you don't like {item}.", f"Theek hai, aapko {item} pasand nahi, yaad rakh liya.")

    # ── Learning summary ──────────────────────────────────────────────────
    if any(t in lower for t in _LEARNING_SUMMARY_TRIGGERS):
        return learner.get_learning_summary()

    # ── Behaviour analysis ────────────────────────────────────────────────
    if any(t in lower for t in _BEHAVIOR_TRIGGERS):
        report = generate_daily_behavior_analysis()
        return behavior_summary_spoken(report, lang)

    # ── List remembered facts ────────────────────────────────────────────
    if any(t in lower for t in _FACTS_LIST_TRIGGERS):
        from database.models import LearnedFactRepository
        facts = LearnedFactRepository.list(limit=8)
        if not facts:
            return tr(lang, "I don't have any facts stored yet.", "Abhi tak mujhe koi baat yaad nahi hai.")
        joined = "; ".join(facts)
        return tr(lang, f"Here's what I remember: {joined}.", f"Mujhe yeh yaad hai: {joined}.")

    # ── Face enrollment ───────────────────────────────────────────────────
    if any(t in lower for t in _ENROLL_TRIGGERS):
        # Allow "remember my face, my name is X" in one sentence.
        name = None
        for pat, _skip in _NAME_INTRO_TRIGGERS:
            if pat in lower:
                rest = lower.split(pat, 1)[1].strip().split()
                if rest and rest[0] not in _NAME_FALSE_POSITIVES:
                    name = rest[0].capitalize()
                    learner.set_user_name(name)
                    set_active_person(name)
                break
        name = name or active_name or learner.get_user_name()
        if not name:
            return prompt_builder.need_a_name(lang)
        if _vision_enroll(name):
            return tr(lang, f"Okay {name}, look at the camera - I'm learning your face.", f"Theek hai {name}, camera ki taraf dekhiye - main aapka chehra seekh raha hoon.")
        return prompt_builder.feature_unavailable(lang)

    # ── "Enroll <name>" / bare "enroll" ──────────────────────────────────
    m = re.search(r"\b(?:enroll|enrol|register)\b(?:\s+(?:the\s+|face\s+of\s+)?([a-zA-Z]+))?", lower)
    if m:
        _enroll_stop = {
            "my", "me", "face", "mera", "chehra", "please", "the", "a",
            "to", "now", "kar", "karo", "abhi", "this", "him", "her",
        }
        cand = (m.group(1) or "").strip()
        if cand and cand not in _enroll_stop:
            name = cand.capitalize()
        else:
            name = active_name or learner.get_user_name()
        if not name:
            return tr(lang, "Whose face should I enroll? Say enroll, then the name.", "Kiska chehra enroll karun? Boliye - enroll, phir naam.")
        if _vision_enroll(name):
            return tr(lang, f"Okay {name}, please look at the camera - I will capture 20 photos.", f"Theek hai {name}, camera ki taraf dekhiye - main 20 photos lunga.")
        return prompt_builder.feature_unavailable(lang)

    # ── Forget a face ─────────────────────────────────────────────────────
    for pat in _FORGET_FACE_PATTERNS:
        m = re.search(pat, lower)
        if m:
            name = m.group(1).capitalize()
            if _vision_delete_person(name):
                return tr(lang, f"Okay, I've forgotten {name}'s face.", f"Theek hai, {name} ka chehra bhula diya.")
            return tr(lang, f"I don't have a face stored for {name}.", f"{name} ka koi chehra mere paas nahi tha.")

    # ── Person-memory commands ────────────────────────────────────────────
    for pat in _REMEMBER_ABOUT_PATTERNS:
        m = re.search(pat, lower)
        if m:
            name, fact = m.group(1).capitalize(), m.group(2).strip()
            if fact:
                get_person_memory(name).add_fact(fact)
                return tr(lang, f"Got it, I'll remember that about {name}.", f"Theek hai, {name} ke baare mein yaad rakh liya.")

    for pat in _LIKES_PATTERNS:
        m = re.search(pat, lower)
        if m:
            name, item = m.group(1).capitalize(), m.group(2).strip()
            if item:
                get_person_memory(name).add_preference(item, like=True)
                return tr(lang, f"Noted - {name} likes {item}.", f"Theek hai, {name} ko {item} pasand hai.")

    for pat in _TELL_ME_ABOUT_PATTERNS:
        m = re.search(pat, lower)
        if m:
            who = m.group(1)
            # "about me/you" is not a third person - fall through to the
            # about-me handler / LLM instead of creating a person called "Me".
            if who in ("me", "myself", "mujhe", "mere", "mera", "you", "yourself", "tum", "aap"):
                break
            from database.models import PersonRepository
            if PersonRepository.get(who.capitalize()) is None:
                return tr(lang, f"I don't know {who.capitalize()} yet.", f"Main {who.capitalize()} ko abhi tak nahi jaanta.")
            return get_person_memory(who.capitalize()).get_summary()

    if any(t in lower for t in _KNOW_ABOUT_ME_TRIGGERS):
        if active_pm:
            ctx = active_pm.build_context()
            return ctx[:280] + "..." if len(ctx) > 280 else ctx
        return tr(lang, "I don't know you yet. Please tell me your name!", "Main aapko abhi tak nahi jaanta. Apna naam bataiye!")

    for trig in _REMEMBER_THIS_TRIGGERS:
        if trig in lower:
            note = text[lower.index(trig) + len(trig):].strip()
            if note and active_pm:
                active_pm.add_note(note)
                return tr(lang, "Okay, noted.", "Theek hai, note kar liya.")
            if note:
                learner.learn_fact(note)
                return tr(lang, "Okay, noted.", "Theek hai, note kar liya.")

    # ── Known people ──────────────────────────────────────────────────────
    if any(t in lower for t in _KNOWN_PEOPLE_TRIGGERS):
        names = _vision_list_known_people()
        if names is None:
            return prompt_builder.feature_unavailable(lang)
        if not names:
            return tr(lang, "I don't know anyone's face yet.", "Abhi tak mujhe kisi ka bhi chehra nahi pata.")
        joined = ", ".join(names)
        return tr(lang, f"I know {len(names)} people: {joined}.", f"Main {len(names)} logon ko jaanta hoon: {joined}.")

    # ── "Who is that / who am I" ─────────────────────────────────────────
    if any(t in lower for t in _WHO_IS_THAT_TRIGGERS):
        name = _vision_recognize_face_now()
        if name is None:
            return prompt_builder.vision_unavailable(lang)
        if name:
            return tr(lang, f"That's {name}!", f"Yeh {name} hain!")
        return tr(lang, "I see a face but I don't recognise it.", "Mujhe ek chehra dikh raha hai lekin pehchana nahi.")

    # ── How many people ───────────────────────────────────────────────────
    if any(t in lower for t in _HOW_MANY_PEOPLE_TRIGGERS):
        seen = _vision_get_scene_objects()
        faces = [o.replace("person:", "") for o in seen if o.startswith("person:")]
        has_unknown = any(o in ("person", "unknown person") for o in seen)
        total = len(faces) + (1 if has_unknown else 0)

        if total == 0:
            return tr(lang, "I don't see anyone in front of me right now.", "Abhi saamne koi nahi dikh raha.")

        if faces:
            if lang == 'en':
                if total == 1:
                    reply = f"There is 1 person in front of me. They're {faces[0]}."
                else:
                    reply = f"There are {total} people in front of me. "
                    reply += " ".join(f"{name}." for name in faces)
                return reply
            if total == 1:
                return f"Saamne sirf 1 insaan hai. Woh hain {faces[0]}."
            reply = f"Saamne {total} log hain. "
            reply += " ".join(f"{name}." for name in faces)
            return reply

        return tr(
            lang,
            "There is 1 person in front of me but I don't recognise their face.",
            "Saamne 1 insaan hai lekin chehra pehchana nahi.",
        )

    # ── "What can you see" - full scene description ─────────────────────
    if any(t in lower for t in _SEE_TRIGGERS):
        seen = _vision_get_scene_objects()
        if not seen:
            return tr(lang, "I can't see anything right now. My eyes may not be active.", "Mujhe abhi kuch nahi dikh raha. Camera shayad active nahi hai.")

        faces = [o.replace("person:", "") for o in seen if o.startswith("person:")]
        objects = [
            o for o in seen
            if not o.startswith("person:") and not o.startswith("emotion:")
            and o not in ("person", "unknown person")
        ]
        has_unknown = any(o in ("person", "unknown person") for o in seen)

        parts = []
        if faces and lang == 'en':
            if len(faces) == 1:
                parts.append(f"I can see {faces[0]} in front of me.")
            else:
                parts.append(f"I can see {len(faces)} people: {', '.join(faces)}.")
        elif faces:
            if len(faces) == 1:
                parts.append(f"Saamne {faces[0]} dikh rahe hain.")
            else:
                parts.append(f"Saamne {len(faces)} log hain: {', '.join(faces)}.")
        elif has_unknown and lang == 'en':
            parts.append("I can see a person in front of me but I don't recognise their face.")
        elif has_unknown:
            parts.append("Ek insaan saamne dikh raha hai lekin chehra pehchana nahi.")
        else:
            parts.append(tr(lang, "I don't see anyone in front of me right now.", "Abhi saamne koi nahi dikh raha."))

        if objects:
            if lang == 'en':
                obj_list = []
                for o in objects:
                    article = "an" if o[0].lower() in "aeiou" else "a"
                    obj_list.append(f"{article} {o}")
                if len(obj_list) == 1:
                    obj_str = obj_list[0]
                elif len(obj_list) == 2:
                    obj_str = f"{obj_list[0]} and {obj_list[1]}"
                else:
                    obj_str = ", ".join(obj_list[:-1]) + f", and {obj_list[-1]}"
                parts.append(f"In the scene I can also see {obj_str}.")
            else:
                obj_str = ", ".join(objects)
                parts.append(f"Scene mein {obj_str} bhi dikh raha hai.")

        return " ".join(parts)

    # ── "Do you remember me?" ─────────────────────────────────────────────
    if any(t in lower for t in _REMEMBER_ME_TRIGGERS):
        if active_pm:
            from database.models import FactRepository
            facts = FactRepository.list(active_pm.id, limit=3)
            v = active_pm.visit_count
            name = active_pm.name
            if facts:
                sample = "; ".join(facts)
                return f"Yes {name}! You have visited {v} times. I know: {sample}."
            return f"Yes {name}! You have visited {v} times. Tell me more about yourself!"
        return tr(lang, "I don't think we have met before. What is your name?", "Lagta hai hum pehle nahi mile. Aapka naam kya hai?")

    # ── Weather ────────────────────────────────────────────────────────────
    if "weather" in lower or "mausam" in lower or "barish" in lower or "temperature" in lower or "temp" in lower:
        city = _extract_city_from_text(lower)
        return get_weather(city)

    # ── Time / date ───────────────────────────────────────────────────────
    if any(w in lower for w in _TIME_TRIGGERS):
        now = datetime.now()
        return f"Abhi {now.strftime('%I:%M %p')} baj rahe hain. {now.strftime('%A, %d %B %Y')}."

    if any(w in lower for w in _DATE_TRIGGERS):
        now = datetime.now()
        return f"Aaj {now.strftime('%A, %d %B %Y')} hai."

    if "your name" in lower:
        return "My name is HariShiva."

    # ── Capabilities / help ───────────────────────────────────────────────
    if any(t in lower for t in _CAP_TRIGGERS):
        return (
            "Main HariShiva hoon — aapka personal AI assistant. "
            "Main yeh sab kar sakta hoon: "
            "1. Chehra pehchaanta hoon. "
            "2. Objects detect karta hoon apni aankhon se. "
            "3. Mausam batata hoon — kisi bhi city ka. "
            "4. Date aur time batata hoon. "
            "5. Current news aur facts search karta hoon internet se. "
            "6. Aapko yaad rakhta hoon — naam, pasand, aur baatein. "
            "7. Hindi, English, aur Hinglish mein baat karta hoon. "
            "8. GPT se kisi bhi sawaal ka jawaab deta hoon. "
            "Bas 'Hari' bolo aur poochho jo bhi chahiye!"
        )

    # ── Name introduction - activates PersonMemory ───────────────────────
    intro_name = None
    for pat, _skip in _NAME_INTRO_TRIGGERS:
        if pat in lower:
            rest = lower.split(pat, 1)[1].strip().split()
            if rest:
                candidate = rest[0].capitalize()
                if candidate.lower() not in _NAME_FALSE_POSITIVES:
                    intro_name = candidate
                    break

    if intro_name:
        learner.set_user_name(intro_name)
        learner.learn_fact(f"User name is {intro_name}")
        pm = get_person_memory(intro_name)
        pm.on_seen()
        set_active_person(intro_name)

        # New face? Start enrollment automatically so "Hari" -> name ->
        # photo capture happens in one natural flow.
        known = _vision_list_known_people()
        if known is not None and intro_name.lower() not in (k.lower() for k in known):
            if _vision_enroll(intro_name):
                return tr(
                    lang,
                    f"Nice to meet you {intro_name}! Please look at the camera - I will capture 20 photos to remember your face.",
                    f"Aapse milkar khushi hui {intro_name}! Camera ki taraf dekhiye - main aapka chehra yaad rakhne ke liye 20 photos lunga.",
                )
        return pm.get_greeting()

    if "what's my name" in lower or "whats my name" in lower or "mera naam kya hai" in lower:
        if active_pm:
            return f"Your name is {active_pm.name}!"
        saved = learner.get_user_name()
        return f"Your name is {saved}." if saved else tr(lang, "I don't know your name yet. Please tell me!", "Mujhe abhi tak aapka naam pata nahi hai. Bataiye!")

    return None


# ── "Said something before the wake word" handler ────────────────────────
def _handle_unknown_intro(text: str) -> None:
    """Called by wait_for_wake_word for phrases that didn't include the wake
    word - lets someone introduce themselves before saying 'Hari'."""
    lower = text.lower().strip()
    learner = get_learner()
    for pat, _skip in _NAME_INTRO_TRIGGERS:
        if pat in lower:
            rest = lower.split(pat, 1)[1].strip().split()
            if rest:
                candidate = rest[0].capitalize()
                if candidate.lower() not in _NAME_FALSE_POSITIVES:
                    learner.set_user_name(candidate)
                    pm = get_person_memory(candidate)
                    pm.on_seen()
                    set_active_person(candidate)
                    return


# ── Main conversation loop ────────────────────────────────────────────────
def run_conversation_loop() -> None:
    learner = get_learner()
    learner.new_session()

    audio_manager.init_audio()
    if not calibrate_microphone():
        print("[Voice] Microphone calibration failed - continuing anyway.")

    start_behavior_analysis_loop()

    speak("Hari is ready. Say Hari to wake me up.", "en")

    while True:
        try:
            command = wait_for_wake_word(on_text=_handle_unknown_intro)

            if command:
                # Command spoken in the same breath as the wake word.
                text, lang = command, infer_lang(command)
            else:
                speak(tr(config.DEFAULT_LANG, "Yes, I am listening.", "Haan, main sun raha hoon."), config.DEFAULT_LANG)
                text, lang = listen()
            if not text:
                continue
            print(f"[Voice] Heard: '{text}' ({lang})")

            person_name = get_active_person()
            reply = handle_custom_commands(text, lang)

            if reply is None:
                scene_objects = _vision_get_scene_objects()
                scene_description = ", ".join(scene_objects) if scene_objects else None
                reply = generate_response(
                    text, lang, person_name=person_name, scene_description=scene_description,
                )
            else:
                learner.record_conversation(text, reply, lang=lang)
                if person_name:
                    get_person_memory(person_name).record_conversation(text, reply, lang=lang)

            stop_thread = threading.Thread(target=listen_for_stop, daemon=True)
            stop_thread.start()
            speak(reply, lang)

            if person_name:
                extract_facts_async(get_person_memory(person_name), text)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[ConversationLoop] error: {e}")
            time.sleep(1)
