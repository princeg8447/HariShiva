# HariShiva V2

A clean, modular rebuild of the HariShiva voice/vision assistant. This is a
**separate project** from the original `HariShivaVision_v2.py` - the original
file is left untouched as a reference and is never imported or modified by V2.

## Project structure

```
HariShiva_V2/
├── app/                # entry point, config, startup/bootstrap
│   ├── config.py
│   ├── startup.py
│   └── main.py
├── ai/                 # Groq LLM integration
│   ├── groq_client.py
│   ├── llm_manager.py
│   └── response_engine.py
├── conversation/       # dialogue loop, command dispatcher, behaviour analysis
│   ├── offline_conversation.py
│   ├── prompt_builder.py
│   └── emotion_questions.py
├── memory/             # adaptive learning + per-person memory
│   ├── memory_manager.py
│   ├── memory_extractor.py
│   └── context_retriever.py
├── vision/             # camera loop, face recognition, emotion detection
│   ├── face_detector.py
│   ├── face_recognition.py
│   ├── emotion_detection.py
│   ├── enrollment.py
│   └── scene_state.py
├── voice/              # speech-to-text, text-to-speech, audio device handling
│   ├── speech_to_text.py
│   ├── text_to_speech.py
│   └── audio_manager.py
├── database/           # SQLite schema + repositories
│   ├── database.py
│   └── models.py
├── dashboard/          # standalone health-check CLI
│   └── diagnostics.py
├── tests/              # smoke tests for each layer
├── data/               # SQLite DB, face encodings, enrolled user photos (created at runtime)
├── models/             # downloaded model weights (created at runtime)
├── logs/               # system.log / errors.log (created at runtime)
├── requirements.txt
└── .env                # local secrets/config, not committed
```

`data/`, `models/` and `logs/` are created automatically by
`app.config.ensure_directories()` on first run - they start empty and are
populated as the assistant runs (fresh database, no data carried over from
the original project).

## Setup

1. Create and activate a virtualenv, then install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Copy `.env` and fill in the values that matter for your setup:

   | Variable | Purpose |
   | --- | --- |
   | `GROQ_API_KEY` | Required for LLM responses (https://console.groq.com) |
   | `WEATHER_API_KEY` | Optional - OpenWeatherMap key for "what's the weather" |
   | `DEFAULT_CITY` | City used when the user doesn't name one (default `Delhi`) |
   | `AUDIO_SINK` | Bluetooth/output PulseAudio sink name, e.g. `bluez_sink.AA_BB_CC_DD_EE_FF.a2dp_sink` |
   | `WAKE_WORD` | Wake word (default `hari`) |
   | `DEFAULT_LANG` | `en` or `hi` |
   | `CAMERA_INDEX` | OpenCV camera index (default `0`) |
   | `ENABLE_VISION` / `ENABLE_VOICE` | Set to `false` to disable a subsystem |
   | `RECOGNITION_TOLERANCE` | Face match strictness (lower = stricter, default `0.42`) |
   | `YOLO_MODEL_PATH` | Path/name of the YOLO weights (default `yolov8n.pt`) |

## Running

```bash
cd HariShiva_V2
PYTHONPATH=. python app/main.py
```

This will:
- run startup (logging, audio sink, database init),
- start the vision thread (camera + face recognition + object/emotion detection), if `ENABLE_VISION=true`,
- start the main voice conversation loop, if `ENABLE_VOICE=true`.

### Health check

To check what's working without starting the full assistant:

```bash
PYTHONPATH=. python dashboard/diagnostics.py
```

This prints config, database table counts, the adaptive-learning summary,
audio readiness, and which vision libraries/models are available.

## Tests

Each layer has a smoke test under `tests/`:

```bash
PYTHONPATH=. python tests/test_memory_smoke.py
PYTHONPATH=. python tests/test_ai_smoke.py
PYTHONPATH=. python tests/test_voice_smoke.py
PYTHONPATH=. python tests/test_conversation_smoke.py
PYTHONPATH=. python tests/test_vision_smoke.py
```

These exercise real modules against a local SQLite database (no mocks) and
report import/availability of optional libraries (face_recognition,
mediapipe, deepface, ultralytics) without requiring a camera.

## What changed vs. the original

- **Storage**: SQLite (`database/`) replaces the original's JSON memory
  files. All tables are created fresh by `database.database.init_db()` -
  no data is migrated from the old project.
- **Headless vision**: `vision/face_detector.py` is a port of the original's
  `camera_detect_loop` with all `cv2.imshow`/`xdotool` GUI/window-management
  code removed, since the Pi runs this as a background service with no
  display attached. All detection logic (YOLO object detection, face
  recognition, DeepFace emotion, MediaPipe finger counting, greetings) is
  preserved.
- **Shared vision state**: `vision/scene_state.py` is a small thread-safe
  module holding the current frame, detected objects/scene description and
  the currently-visible person, used to avoid circular imports between the
  vision and conversation layers.
- **Config & secrets**: all API keys and tunables live in `.env` /
  `app/config.py` instead of being hardcoded in source.
- **Modular layers**: AI, voice, memory, conversation and vision are each
  isolated packages with lazy imports between them, so any layer can be
  developed/tested independently.

The original `HariShivaVision_v2.py` is unchanged and remains the reference
implementation this project was refactored from.
# HariShiva
