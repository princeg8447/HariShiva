"""
Command-line health check for HariShiva V2.

Run with:

    PYTHONPATH=. python dashboard/diagnostics.py

Prints config, database, memory, audio and vision status so you can quickly
tell what's working on a given Pi without starting the full assistant.
"""

from __future__ import annotations

from app import config


def _check(label: str, ok: bool, detail: str = "") -> str:
    mark = "OK " if ok else "!! "
    line = f"[{mark}] {label}"
    if detail:
        line += f" - {detail}"
    return line


def check_config() -> list[str]:
    lines = ["== Config =="]
    lines.append(_check("GROQ_API_KEY set", bool(config.GROQ_API_KEY)))
    lines.append(_check("WEATHER_API_KEY set", bool(config.WEATHER_API_KEY)))
    lines.append(f"     Database path: {config.DATABASE_PATH}")
    lines.append(f"     Wake word: {config.WAKE_WORD}")
    lines.append(f"     Default language: {config.DEFAULT_LANG}")
    lines.append(f"     Vision enabled: {config.ENABLE_VISION}")
    lines.append(f"     Voice enabled: {config.ENABLE_VOICE}")
    return lines


def check_database() -> list[str]:
    lines = ["== Database =="]
    try:
        from database.database import get_connection, init_db

        init_db()
        with get_connection() as conn:
            tables = [
                "persons", "person_facts", "conversations", "learned_facts",
                "corrections", "frequent_topics", "profile",
                "behavior_reports",
            ]
            for table in tables:
                count = conn.execute(f"SELECT count(*) AS c FROM {table}").fetchone()["c"]
                lines.append(f"     {table}: {count} rows")
        lines.append(_check("Database reachable", True))
    except Exception as e:
        lines.append(_check("Database reachable", False, str(e)))
    return lines


def check_memory() -> list[str]:
    lines = ["== Memory =="]
    try:
        from memory.memory_manager import get_learner

        learner = get_learner()
        lines.append(_check("AdaptiveLearning loaded", True))
        lines.append(f"     {learner.get_learning_summary()}")
    except Exception as e:
        lines.append(_check("AdaptiveLearning loaded", False, str(e)))
    return lines


def check_audio() -> list[str]:
    lines = ["== Audio =="]
    try:
        from voice import audio_manager

        ready = audio_manager.init_audio()
        lines.append(_check("pygame mixer ready", ready))
    except Exception as e:
        lines.append(_check("pygame mixer ready", False, str(e)))
    return lines


def check_vision() -> list[str]:
    lines = ["== Vision =="]

    try:
        import cv2  # noqa: F401
        lines.append(_check("opencv-python", True))
    except Exception as e:
        lines.append(_check("opencv-python", False, str(e)))

    try:
        import face_recognition  # noqa: F401
        lines.append(_check("face_recognition", True))
    except Exception as e:
        lines.append(_check("face_recognition", False, str(e)))

    try:
        from vision.emotion_detection import DEEPFACE_AVAILABLE
        lines.append(_check("DeepFace (emotion detection)", DEEPFACE_AVAILABLE))
    except Exception as e:
        lines.append(_check("DeepFace (emotion detection)", False, str(e)))

    try:
        import mediapipe  # noqa: F401
        lines.append(_check("MediaPipe (finger counting)", True))
    except Exception as e:
        lines.append(_check("MediaPipe (finger counting)", False, str(e)))

    try:
        from ultralytics import YOLO  # noqa: F401
        lines.append(_check("ultralytics (YOLO)", True))
    except Exception as e:
        lines.append(_check("ultralytics (YOLO)", False, str(e)))

    try:
        from vision.face_recognition import list_known_people, load_face_model

        load_face_model()
        people = list_known_people()
        lines.append(f"     Known people: {len(people)} ({', '.join(people) or 'none'})")
    except Exception as e:
        lines.append(_check("Face model", False, str(e)))

    return lines


def run_diagnostics() -> str:
    sections = [
        check_config(),
        check_database(),
        check_memory(),
        check_audio(),
        check_vision(),
    ]
    return "\n".join(line for section in sections for line in section)


if __name__ == "__main__":
    print(run_diagnostics())
