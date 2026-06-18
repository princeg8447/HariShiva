"""
HariShiva V2 - entry point.

This orchestrates startup, the vision (camera/face) thread and the main
conversation loop. Individual subsystems (vision, conversation, ai, voice,
memory) are implemented in their own packages and are imported lazily so
that this file keeps working as those packages are filled in one by one.
"""

import sys
import time

from app import config, startup


def _start_vision_thread(logger):
    if not config.ENABLE_VISION:
        logger.info("Vision disabled (ENABLE_VISION=false in .env).")
        return None

    try:
        import threading

        from vision.face_detector import camera_detect_loop
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning("Vision module not implemented yet (%s), skipping.", exc)
        return None

    thread = threading.Thread(target=camera_detect_loop, daemon=True, name="vision")
    thread.start()
    logger.info("Vision thread started.")
    return thread


def _run_conversation_loop(logger):
    try:
        from conversation.offline_conversation import run_conversation_loop
    except (ImportError, ModuleNotFoundError) as exc:
        logger.warning("Conversation module not implemented yet (%s).", exc)
        logger.info(
            "HariShiva V2 has no conversation loop yet. "
            "Implement conversation/offline_conversation.py to enable it."
        )
        while True:
            time.sleep(60)

    run_conversation_loop()


def main():
    logger = startup.run_startup()
    logger.info("Starting HariShiva V2...")

    _start_vision_thread(logger)

    try:
        _run_conversation_loop(logger)
    except KeyboardInterrupt:
        logger.info("Shutdown requested, exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
