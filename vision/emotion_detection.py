"""
DeepFace-based facial emotion detection for HariShiva V2.

DeepFace is an optional, heavy dependency (TensorFlow). If it can't be
imported (e.g. protobuf conflicts on a Pi), emotion detection is silently
disabled - `vision.face_detector` checks `DEEPFACE_AVAILABLE` before using it.
"""

import threading

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
    print("[Emotion] DeepFace loaded - emotion detection active.")
except Exception as exc:
    DeepFace = None
    DEEPFACE_AVAILABLE = False
    print(f"[Emotion] DeepFace unavailable ({exc.__class__.__name__}) - emotion detection disabled.")

EMOTION_EMOJI = {
    "happy": "\U0001F60A", "sad": "\U0001F622", "angry": "\U0001F620",
    "surprise": "\U0001F632", "fear": "\U0001F628", "disgust": "\U0001F922",
    "neutral": "\U0001F610",
}

# face_idx -> {"emotion": "happy", "score": 0.92}
emotion_cache: dict[int, dict] = {}
emotion_lock = threading.Lock()


def analyze_emotion_async(face_bgr, face_idx: int) -> None:
    """Run DeepFace emotion analysis on a face crop. Meant to be called in a
    background thread - updates `emotion_cache` when done."""
    if not DEEPFACE_AVAILABLE:
        return
    try:
        result = DeepFace.analyze(face_bgr, actions=["emotion"], enforce_detection=False, silent=True)
        if isinstance(result, list):
            result = result[0]
        dominant = result.get("dominant_emotion", "neutral")
        score = result["emotion"].get(dominant, 0) / 100.0
        with emotion_lock:
            emotion_cache[face_idx] = {"emotion": dominant, "score": round(score, 2)}
    except Exception:
        pass


def get_emotion(face_idx: int) -> dict:
    with emotion_lock:
        return dict(emotion_cache.get(face_idx, {}))
