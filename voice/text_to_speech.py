"""
Text-to-speech for HariShiva V2: tone-aware Edge-TTS voice selection (with
gTTS fallback) and playback through pygame's mixer.
"""

import asyncio
import os
import re
import tempfile
import time

import edge_tts
import pygame
from gtts import gTTS

from voice import audio_manager
from voice.speech_to_text import HINDI_WORDS

_EDGE_VOICE_EN = "en-IN-PrabhatNeural"
_EDGE_VOICE_HI = "hi-IN-MadhurNeural"
_TTS_RATE_EN = "-8%"
_TTS_RATE_HI = "-4%"
_TTS_VOLUME = "+10%"


def _detect_tone(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in ['haha', 'ha ha', 'hehe', 'lol']):
        return 'laugh'
    if text.count('!') >= 2 or any(w in lower for w in ['great', 'amazing', 'wonderful', 'zabardast']):
        return 'cheerful'
    if any(w in lower for w in ['sorry', 'unfortunately', 'sad', 'maafi']):
        return 'empathetic'
    if '?' in text:
        return 'curious'
    return 'neutral'


def _pick_voice(text: str, lang_hint: str) -> tuple[str, str, str]:
    tone = _detect_tone(text)
    if re.search(r'[ऀ-ॿ]', text) or lang_hint == 'hi':
        base_voice, base_rate = _EDGE_VOICE_HI, _TTS_RATE_HI
    elif sum(1 for w in re.findall(r'\b[a-zA-Z]+\b', text.lower()) if w in HINDI_WORDS) >= 2:
        base_voice, base_rate = _EDGE_VOICE_HI, _TTS_RATE_HI
    else:
        base_voice, base_rate = _EDGE_VOICE_EN, _TTS_RATE_EN

    if tone in ('laugh', 'cheerful'):
        return base_voice, '+5%', '+2Hz'
    if tone == 'curious':
        return base_voice, '-3%', '+1Hz'
    if tone == 'empathetic':
        return base_voice, '-10%', '-1Hz'
    return base_voice, base_rate, '+0Hz'


def _preprocess_tts(text: str) -> str:
    text = text.strip()
    for abbr, expanded in (('AI', 'A.I.'), ('API', 'A.P.I.'), ('RPi', 'Raspberry Pi'), ('GPT', 'G.P.T.')):
        text = re.sub(rf'\b{abbr}\b', expanded, text)
    text = text.replace('...', '. ')
    text = re.sub(r' {2,}', ' ', text)
    return text


async def _edge_tts_async(text: str, voice: str, rate: str, pitch: str, path: str) -> None:
    communicate = edge_tts.Communicate(text, voice, rate=rate, volume=_TTS_VOLUME, pitch=pitch)
    await communicate.save(path)


def _generate_speech(text: str, lang: str, path: str) -> None:
    voice, rate, pitch = _pick_voice(text, lang)
    try:
        asyncio.run(_edge_tts_async(text, voice, rate, pitch, path))
    except Exception:
        gtts_lang = 'hi' if voice == _EDGE_VOICE_HI else 'en'
        gTTS(text=text, lang=gtts_lang).save(path)


def speak(text: str, lang: str) -> None:
    """Speak `text` aloud. Blocks until playback finishes (or is interrupted)."""
    if not audio_manager.speak_lock.acquire(blocking=True, timeout=10):
        return

    try:
        audio_manager.is_speaking = True
        audio_manager.stop_speaking = False
        audio_manager.last_tts_text = text

        if not audio_manager.audio_ready():
            return

        text = _preprocess_tts(text)
        tmp_mp3 = os.path.join(tempfile.gettempdir(), f"harishiva_tts_{os.getpid()}.mp3")
        try:
            _generate_speech(text, lang, tmp_mp3)
            pygame.mixer.music.load(tmp_mp3)
            pygame.mixer.music.play()
        except Exception:
            return

        while pygame.mixer.music.get_busy():
            if audio_manager.stop_speaking:
                pygame.mixer.music.stop()
                break
            time.sleep(0.1)

        time.sleep(1.0)  # mic settle time - Bluetooth keeps playing ~1s after pygame finishes
        if os.path.exists(tmp_mp3):
            os.remove(tmp_mp3)
    finally:
        audio_manager.last_tts_end = time.time()
        audio_manager.is_speaking = False
        audio_manager.speak_lock.release()
