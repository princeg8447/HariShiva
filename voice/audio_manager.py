"""
Audio subsystem bring-up and shared speaking-state for HariShiva V2.

Other voice modules (speech_to_text, text_to_speech) read/write the module
attributes `is_speaking` / `stop_speaking` and use `speak_lock` to make sure
only one TTS playback happens at a time and barge-in can interrupt it.
"""

import os
import subprocess
import threading
import time

import pygame

speak_lock = threading.Lock()
is_speaking = False
stop_speaking = False
last_tts_end = 0.0   # time.time() of the last TTS playback end (echo guard)
last_tts_text = ""   # what we last said - used to reject our own echo

_AUDIO_OK = False


def init_audio(logger=None) -> bool:
    """Start PulseAudio if needed and initialise the pygame mixer.

    Returns True if pygame.mixer is usable (TTS playback possible).
    """
    global _AUDIO_OK

    subprocess.run(
        ["pulseaudio", "--start", "--daemonize"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1)

    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
    pygame.init()

    for attempt in range(5):
        try:
            pygame.mixer.init()
            _AUDIO_OK = True
            break
        except Exception:
            if logger:
                logger.warning("[Audio] mixer init attempt %d/5 failed, retrying...", attempt + 1)
            time.sleep(2)

    if not _AUDIO_OK and logger:
        logger.warning("[Audio] pygame.mixer unavailable - TTS disabled")

    return _AUDIO_OK


def audio_ready() -> bool:
    return _AUDIO_OK
