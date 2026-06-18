"""
Shared, thread-safe state describing what the camera currently sees.

`vision.face_detector` writes this every frame; `conversation.offline_conversation`
and `vision.face_recognition` read it. Keeping it in its own module avoids a
circular import between those two packages.
"""

import threading

_lock = threading.Lock()
_frame = None
_scene_objects: list[str] = []
_scene_desc: str = ""
_visible_person: str | None = None


def set_frame(frame) -> None:
    global _frame
    with _lock:
        _frame = frame


def get_frame():
    """Return a copy of the latest camera frame, or None if not available."""
    with _lock:
        return None if _frame is None else _frame.copy()


def set_scene_objects(objects) -> None:
    global _scene_objects
    with _lock:
        _scene_objects = list(objects)


def get_scene_objects() -> list[str]:
    with _lock:
        return list(_scene_objects)


def set_scene_desc(desc: str) -> None:
    global _scene_desc
    with _lock:
        _scene_desc = desc


def get_scene_desc() -> str:
    with _lock:
        return _scene_desc


def set_visible_person(name: str | None) -> None:
    global _visible_person
    with _lock:
        _visible_person = name


def get_visible_person() -> str | None:
    with _lock:
        return _visible_person
