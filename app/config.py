"""
Central configuration for HariShiva V2.

All values are loaded from environment variables (see .env at the project
root). Nothing here should be hardcoded secrets - the original project kept
the Groq API key inline in source, V2 keeps it out of the codebase.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = parent of the "app" package
BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")


def _bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

USERS_DIR = DATA_DIR / "users"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
MEMORIES_DIR = DATA_DIR / "memories"
BACKUPS_DIR = DATA_DIR / "backups"

FACE_MODELS_DIR = MODELS_DIR / "face_models"
EMOTION_MODELS_DIR = MODELS_DIR / "emotion_models"
CUSTOM_MODELS_DIR = MODELS_DIR / "custom_models"

DATABASE_PATH = DATA_DIR / "harishiva.db"
FACE_ENCODINGS_FILE = EMBEDDINGS_DIR / "face_encodings.pkl"

# ── External services ───────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Delhi")

# ── Audio ────────────────────────────────────────────────────────────────
# Bluetooth speaker sink name, e.g. "bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink"
AUDIO_SINK = os.getenv("AUDIO_SINK", "")
XDG_RUNTIME_DIR = os.getenv("XDG_RUNTIME_DIR", "/run/user/1000")
PULSE_RUNTIME_PATH = os.getenv("PULSE_RUNTIME_PATH", "/run/user/1000/pulse")

# ── Assistant behaviour ──────────────────────────────────────────────────
WAKE_WORD = os.getenv("WAKE_WORD", "hari")
DEFAULT_LANG = os.getenv("DEFAULT_LANG", "en")
MAX_CONV_HISTORY = int(os.getenv("MAX_CONV_HISTORY", "200"))
MAX_FACTS = int(os.getenv("MAX_FACTS", "100"))
BEHAVIOR_INTERVAL_SECONDS = int(os.getenv("BEHAVIOR_INTERVAL_SECONDS", "1800"))

# ── Camera / vision ──────────────────────────────────────────────────────
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
ENABLE_VISION = _bool(os.getenv("ENABLE_VISION"), default=True)
ENABLE_VOICE = _bool(os.getenv("ENABLE_VOICE"), default=True)

# Show a live "HariShiva Vision" preview window (cv2.imshow) - requires a
# desktop/display (DISPLAY env var). Set to false to run fully headless.
SHOW_CAMERA_PREVIEW = _bool(os.getenv("SHOW_CAMERA_PREVIEW"), default=True)

# Lower = stricter face match (face_recognition encoding distance threshold)
RECOGNITION_TOLERANCE = float(os.getenv("RECOGNITION_TOLERANCE", "0.42"))
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")

# ── Logging ──────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def ensure_directories() -> None:
    """Create every directory this config references if it doesn't exist."""
    for directory in (
        DATA_DIR,
        LOGS_DIR,
        MODELS_DIR,
        USERS_DIR,
        EMBEDDINGS_DIR,
        MEMORIES_DIR,
        BACKUPS_DIR,
        FACE_MODELS_DIR,
        EMOTION_MODELS_DIR,
        CUSTOM_MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
