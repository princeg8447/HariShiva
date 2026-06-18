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
from pathlib import Path
from app import config

# Try importing offline wake-word dependencies
try:
    import pvporcupine
    from pvrecorder import PvRecorder
    _HAS_PORCUPINE = True
except ImportError:
    _HAS_PORCUPINE = False

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
    # ── Try Offline Wake Word Detection (Porcupine) ──────────────────────────
    if _HAS_PORCUPINE and config.PICOVOICE_ACCESS_KEY:
        porcupine = None
        recorder = None
        try:
            # 1. Search for custom .ppn model file
            model_file = None
            if config.WAKE_WORD_MODEL_PATH:
                model_path = Path(config.WAKE_WORD_MODEL_PATH)
                if model_path.exists():
                    model_file = model_path
            else:
                wake_word_dir = Path(config.WAKE_WORD_MODELS_DIR)
                if wake_word_dir.exists():
                    ppn_files = list(wake_word_dir.glob("*.ppn"))
                    # Try to find one matching WAKE_WORD
                    for f in ppn_files:
                        if config.WAKE_WORD.lower() in f.name.lower():
                            model_file = f
                            break
                    # Default fallback to first .ppn if exists
                    if not model_file and ppn_files:
                        model_file = ppn_files[0]

            # 2. Initialize Porcupine
            if model_file:
                print(f"[Voice] Initializing offline wake word using custom model: {model_file.name}")
                porcupine = pvporcupine.create(
                    access_key=config.PICOVOICE_ACCESS_KEY,
                    keyword_paths=[str(model_file)]
                )
            elif config.WAKE_WORD.lower() in pvporcupine.KEYWORDS:
                print(f"[Voice] Initializing offline wake word using built-in keyword: {config.WAKE_WORD}")
                porcupine = pvporcupine.create(
                    access_key=config.PICOVOICE_ACCESS_KEY,
                    keywords=[config.WAKE_WORD.lower()]
                )

            # 3. Stream audio and listen for wake word
            if porcupine:
                recorder = PvRecorder(device_index=-1, frame_length=porcupine.frame_length)
                recorder.start()
                print(f"[Voice] Offline wake word detection active. Listening for '{config.WAKE_WORD}'...")

                while True:
                    # Don't capture speaker output (mute mic/stop detection while speaking)
                    if audio_manager.is_speaking:
                        if recorder.is_recording:
                            recorder.stop()
                        time.sleep(0.2)
                        continue
                    else:
                        if not recorder.is_recording:
                            recorder.start()

                    pcm = recorder.read()
                    keyword_index = porcupine.process(pcm)
                    if keyword_index >= 0:
                        print(f"[Voice] Offline wake word detected: '{config.WAKE_WORD}'")
                        return ""  # Trigger standard prompt listening
        except Exception as e:
            print(f"[Voice] Offline wake word initialization failed: {e}. Falling back to Google.")
        finally:
            if recorder:
                try:
                    recorder.stop()
                    recorder.delete()
                except Exception:
                    pass
            if porcupine:
                try:
                    porcupine.delete()
                except Exception:
                    pass

    # ── Fallback: Original Cloud-based Google Speech Recognition ─────────────
    print("[Voice] Using cloud-based Google wake word engine...")
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
