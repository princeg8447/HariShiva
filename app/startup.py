"""
Startup / bootstrap routines for HariShiva V2.

Anything that needs to happen exactly once, before the assistant's main
loop starts, lives here: directory creation, logging setup, audio sink
selection and database initialisation.
"""

import logging
import os
import subprocess
from pathlib import Path

from app import config


def setup_logging() -> logging.Logger:
    """Configure root logging to write to logs/system.log and logs/errors.log."""
    config.ensure_directories()

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(config.LOG_LEVEL)
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    system_handler = logging.FileHandler(config.LOGS_DIR / "system.log", encoding="utf-8")
    system_handler.setFormatter(formatter)
    root_logger.addHandler(system_handler)

    error_handler = logging.FileHandler(config.LOGS_DIR / "errors.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    return root_logger


def set_audio_sink(logger: logging.Logger) -> None:
    """Point PulseAudio at the configured Bluetooth/output sink, if set."""
    if not config.AUDIO_SINK:
        logger.info("No AUDIO_SINK configured, skipping audio sink setup.")
        return

    env = os.environ.copy()
    env["XDG_RUNTIME_DIR"] = config.XDG_RUNTIME_DIR
    env["PULSE_RUNTIME_PATH"] = config.PULSE_RUNTIME_PATH

    try:
        subprocess.run(
            ["pactl", "set-default-sink", config.AUDIO_SINK],
            env=env,
            check=True,
            capture_output=True,
            timeout=10,
        )
        logger.info("Audio sink set to %s", config.AUDIO_SINK)
    except Exception as exc:  # noqa: BLE001 - log and continue, audio is not critical
        logger.warning("Could not set audio sink %s: %s", config.AUDIO_SINK, exc)


def init_database(logger: logging.Logger) -> None:
    """Initialise the V2 database (separate from the original project's data)."""
    try:
        from database.database import init_db

        init_db()
        logger.info("Database initialised at %s", config.DATABASE_PATH)
    except ModuleNotFoundError:
        logger.warning("database.database.init_db not implemented yet, skipping.")
    except Exception:
        logger.exception("Database initialisation failed.")


def health_check(logger: logging.Logger) -> bool:
    """Lightweight system health check: camera, mic, database, AI key.

    Logs a status line per subsystem. Camera/mic problems are warnings (the
    vision/voice layers degrade gracefully); a broken database is fatal.
    """
    all_ok = True

    try:
        from database.database import get_connection

        with get_connection() as conn:
            conn.execute("SELECT 1")
        logger.info("[Health] Database: OK")
    except Exception as exc:  # noqa: BLE001
        all_ok = False
        logger.error("[Health] Database: FAIL (%s)", exc)

    if config.GROQ_API_KEY:
        logger.info("[Health] AI (Groq key): OK")
    else:
        logger.warning("[Health] AI (Groq key): missing - offline replies only.")

    if config.ENABLE_VISION:
        cam = Path(f"/dev/video{config.CAMERA_INDEX}")
        if cam.exists():
            logger.info("[Health] Camera: OK (%s)", cam)
        else:
            logger.warning("[Health] Camera: %s not found - vision may not start.", cam)

    if config.ENABLE_VOICE:
        try:
            cards = Path("/proc/asound/cards").read_text(encoding="utf-8").strip()
            if cards and "no soundcards" not in cards:
                logger.info("[Health] Microphone/sound cards: OK")
            else:
                logger.warning("[Health] Microphone: no sound cards detected.")
        except OSError:
            logger.warning("[Health] Microphone: could not read sound card list.")

    logger.info("[Health] %s", "All systems OK." if all_ok else "Some checks FAILED - see above.")
    return all_ok


def run_startup() -> logging.Logger:
    """Run all startup steps and return the configured logger."""
    logger = setup_logging()
    logger.info("HariShiva V2 starting up...")
    set_audio_sink(logger)
    init_database(logger)
    health_check(logger)
    return logger
