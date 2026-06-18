"""
Speech-to-text for HariShiva V2: microphone calibration, the main listen()
loop (parallel Hindi/English transcription), wake-word detection and
barge-in (interrupting the bot while it's speaking).
"""

import re
import threading
import time

import speech_recognition as sr

from voice import audio_manager

_SR = sr.Recognizer()
_SR.pause_threshold = 0.8  # don't split sentences on natural mid-phrase pauses

# Roman-script Hindi/Hinglish words used to disambiguate which transcription
# (hi-IN vs en-IN) the user actually meant.
HINDI_WORDS = {
    'hai', 'hain', 'ho', 'hua', 'hui', 'hue', 'hoga', 'hogi', 'honge',
    'karo', 'karna', 'karta', 'karti', 'karte', 'karein', 'kiya',
    'kya', 'aur', 'nahi', 'nahin', 'mat', 'mein', 'se', 'ko', 'ka', 'ki',
    'ke', 'yeh', 'ye', 'woh', 'wo', 'kuch', 'bhi', 'toh', 'to', 'aap',
    'main', 'hum', 'tum', 'mera', 'meri', 'tera', 'teri', 'unka', 'unki',
    'aaj', 'kal', 'abhi', 'phir', 'fir', 'pehle', 'baad', 'dost', 'bhai',
    'gaya', 'gayi', 'gaye', 'raha', 'rahi', 'rahe', 'tha', 'thi', 'the',
    'jao', 'jaao', 'jaana', 'aana', 'lega', 'legi', 'lenge', 'dena',
    'chahiye', 'chahta', 'chahti', 'chahte', 'sab', 'kab', 'kyun', 'kyon',
    'kahan', 'kaisa', 'kaisi', 'kaise', 'kitna', 'kitni', 'haan', 'lekin',
    'magar', 'par', 'acha', 'achha', 'theek', 'bilkul', 'zaroor', 'pata',
    'bata', 'bolo', 'batao', 'suno', 'suniye', 'dekho', 'dekhiye', 'bahut',
    'sirf', 'wala', 'wali', 'wale', 'zyada', 'thoda', 'kafi', 'subah',
    'raat', 'din', 'ghar', 'bahar', 'yahan', 'wahan', 'idhar', 'udhar',
    'log', 'baat', 'kaam', 'naam', 'agar', 'jaise', 'saath', 'liye',
    'isliye', 'matlab', 'chaliye', 'lijiye', 'dijiye', 'boliye', 'hariji',
    'samajh', 'lagta', 'lagti', 'laga', 'dikh', 'sun', 'bol', 'chal',
    'ruk', 'jaldi', 'dheere', 'seedha', 'sahi', 'galat', 'alag', 'sath',
    'unse', 'usse', 'mujhe', 'tumhe', 'aapko', 'humko', 'sabko', 'kisiko',
}


def infer_lang(text: str) -> str:
    """Return 'hi' if text looks like Hindi/Hinglish, else 'en'."""
    if re.search(r'[ऀ-ॿ]', text):
        return 'hi'
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return 'hi' if sum(1 for w in words if w in HINDI_WORDS) >= 2 else 'en'


def _is_own_echo(text: str) -> bool:
    """True if `text` is mostly words from our own last TTS sentence.

    Bluetooth speakers keep playing ~1-3s after pygame reports playback done,
    so flag/timestamp guards alone can't stop the mic hearing our own tail.
    """
    spoken = audio_manager.last_tts_text.lower()
    if not spoken:
        return False
    words = re.findall(r'\b\w+\b', text.lower())
    if not words:
        return False
    spoken_words = set(re.findall(r'\b\w+\b', spoken))
    overlap = sum(1 for w in words if w in spoken_words)
    return overlap / len(words) >= 0.7


def calibrate_microphone(max_attempts: int = 10) -> bool:
    """Calibrate ambient noise once at startup. Returns True on success."""
    for attempt in range(max_attempts):
        try:
            with sr.Microphone() as source:
                _SR.adjust_for_ambient_noise(source, duration=1.0)
                return True
        except Exception:
            time.sleep(1)
    return False


def listen() -> tuple[str, str]:
    """Listen once and return (text, lang_code). Returns ("", "") on silence/timeout.

    Runs hi-IN and en-IN Google transcription in parallel and picks whichever
    matches the actual spoken language based on Hindi-word density.
    """
    while audio_manager.is_speaking:
        time.sleep(0.05)

    capture_start = time.time()
    try:
        with sr.Microphone() as source:
            audio = _SR.listen(source, timeout=4, phrase_time_limit=12)
    except sr.WaitTimeoutError:
        return "", ""
    except Exception:
        return "", ""

    # TTS played during our capture window - that's our own voice, not the user.
    if audio_manager.is_speaking or audio_manager.last_tts_end > capture_start:
        return "", ""

    results: dict[str, str] = {}

    def _recog_hi():
        try:
            results['hi'] = _SR.recognize_google(audio, language="hi-IN").strip()
        except Exception:
            results['hi'] = ""

    def _recog_en():
        try:
            results['en'] = _SR.recognize_google(audio, language="en-IN").strip()
        except Exception:
            results['en'] = ""

    t_hi = threading.Thread(target=_recog_hi, daemon=True)
    t_en = threading.Thread(target=_recog_en, daemon=True)
    t_hi.start()
    t_en.start()
    t_hi.join()
    t_en.join()

    hi_text = results.get('hi', '')
    en_text = results.get('en', '')

    if not hi_text and not en_text:
        return "", ""

    def _hindi_word_count(t: str) -> int:
        words = re.findall(r'\b[a-zA-Z]+\b', t.lower())
        deva = len(re.findall(r'[ऀ-ॿ]+', t))
        roman = sum(1 for w in words if w in HINDI_WORDS)
        return deva + roman

    hi_score = _hindi_word_count(hi_text)
    en_score = _hindi_word_count(en_text)

    if en_text and en_score == 0:
        chosen, lang = en_text, 'en'
    elif hi_text and hi_score > 0:
        chosen, lang = hi_text, infer_lang(hi_text)
    elif en_text:
        chosen, lang = en_text, infer_lang(en_text)
    else:
        chosen, lang = hi_text, 'hi'

    if _is_own_echo(chosen):
        print(f"[Voice] Ignored own echo: '{chosen}'")
        return "", ""

    return chosen, lang


def listen_for_stop() -> None:
    """Barge-in: wait for TTS to start, then listen for real speech to interrupt it."""
    time.sleep(1.5)  # let TTS start cleanly before listening for barge-in
    if not audio_manager.is_speaking or audio_manager.stop_speaking:
        return

    barge_sr = sr.Recognizer()
    barge_sr.energy_threshold = 2500  # higher threshold, ignores speaker bleed
    barge_sr.dynamic_energy_threshold = False

    try:
        with sr.Microphone() as source:
            while audio_manager.is_speaking and not audio_manager.stop_speaking:
                try:
                    barge_sr.listen(source, timeout=0.8, phrase_time_limit=2)
                    audio_manager.stop_speaking = True
                    return
                except sr.WaitTimeoutError:
                    continue
                except Exception:
                    return
    except Exception:
        pass


def wait_for_wake_word(wake_words: tuple[str, ...] = ("hari",), on_text=None) -> str:
    """Block until a wake word is heard. Returns any command spoken in the
    same breath after the wake word (e.g. "hari what time is it" -> "what
    time is it"), or "" if the wake word was said alone.

    For every recognized phrase that does NOT contain a wake word,
    `on_text(text)` is called if provided - e.g. to let an unknown person
    introduce themselves before the wake word.
    """
    with sr.Microphone() as source:
        while True:
            try:
                # Don't capture our own TTS (speaker bleed would re-trigger
                # the wake word, e.g. the "Say Hari..." enrollment invite).
                if audio_manager.is_speaking:
                    time.sleep(0.2)
                    continue
                capture_start = time.time()
                audio = _SR.listen(source, timeout=3, phrase_time_limit=10)
                if audio_manager.is_speaking or audio_manager.last_tts_end > capture_start:
                    continue
                text = _SR.recognize_google(audio, language="en-IN").lower()
                if _is_own_echo(text):
                    print(f"[Voice] Ignored own echo: '{text}'")
                    continue
                for w in wake_words:
                    if w in text:
                        remainder = text.split(w, 1)[1].strip(" ,.!?")
                        print(f"[Voice] Wake word heard: '{text}'")
                        return remainder
                if on_text and text.strip():
                    on_text(text)
            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                pass
            except Exception:
                pass
