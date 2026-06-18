"""Smoke test for the voice layer - import + pure-function checks only.

Does not touch the microphone or play audio (no display/speaker assumptions
when run headless via the agent).
"""

from voice.speech_to_text import infer_lang
from voice.text_to_speech import _detect_tone, _pick_voice, _preprocess_tts

print("infer_lang('hello there'):", infer_lang("hello there"))
print("infer_lang('aap kaise ho'):", infer_lang("aap kaise ho"))

print("detect_tone('wow great!! amazing'):", _detect_tone("wow great!! amazing"))
print("pick_voice('aap kaise ho', 'hi'):", _pick_voice("aap kaise ho", "hi"))
print("pick_voice('hello there', 'en'):", _pick_voice("hello there", "en"))
print("preprocess('Use the AI and API on RPi...'):", _preprocess_tts("Use the AI and API on RPi..."))

from voice import audio_manager  # noqa: E402

print("audio_ready before init:", audio_manager.audio_ready())
