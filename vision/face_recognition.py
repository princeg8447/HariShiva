"""
Face encoding storage, training and recognition for HariShiva V2.

Ported from the original's `load_face_model` / `train_face_model` /
`recognize_face_now`. Enrollment images live under
`data/users/<name>/*.jpg` (one folder per person); encodings are cached as a
pickle at `config.FACE_ENCODINGS_FILE`.
"""

from __future__ import annotations

import os
import pickle

import cv2
import face_recognition as fr
import numpy as np

from app import config
from vision import scene_state

known_encodings: list = []
known_names: list[str] = []
model_trained = False


def load_face_model() -> None:
    """Load cached encodings from disk, if present."""
    global known_encodings, known_names, model_trained
    if config.FACE_ENCODINGS_FILE.exists():
        with open(config.FACE_ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        known_encodings = data.get("encodings", [])
        known_names = data.get("names", [])
        model_trained = bool(known_encodings)
        print(f"[FaceRecognition] Loaded {len(set(known_names))} known people.")
    else:
        print("[FaceRecognition] No saved face model - enroll someone first.")


def train_face_model() -> None:
    """Rebuild encodings from every image under data/users/<name>/."""
    global known_encodings, known_names, model_trained

    encodings: list = []
    names: list[str] = []

    if config.USERS_DIR.exists():
        for person_name in os.listdir(config.USERS_DIR):
            person_dir = config.USERS_DIR / person_name
            if not person_dir.is_dir():
                continue
            person_encs = []
            for img_file in os.listdir(person_dir):
                img_bgr = cv2.imread(str(person_dir / img_file))
                if img_bgr is None:
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                encs = fr.face_encodings(img_rgb)
                if encs:
                    person_encs.append(encs[0])
            if person_encs:
                encodings.append(np.mean(person_encs, axis=0))
                names.append(person_name)

    known_encodings, known_names = encodings, names
    model_trained = bool(encodings)

    if encodings:
        config.FACE_ENCODINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(config.FACE_ENCODINGS_FILE, "wb") as f:
            pickle.dump({"encodings": encodings, "names": names}, f)
        print(f"[FaceRecognition] Model trained: {len(encodings)} people enrolled.")
    else:
        print("[FaceRecognition] No face data found to train.")


def list_known_people() -> list[str]:
    return list(dict.fromkeys(known_names))


def recognize_face_now() -> tuple[str | None, int]:
    """Grab the latest camera frame and do fresh face recognition - no cache."""
    frame = scene_state.get_frame()
    if frame is None:
        return None, 0

    try:
        small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = fr.face_locations(rgb, model="hog")
        if not locs or not known_encodings:
            return None, 0
        encs = fr.face_encodings(rgb, locs)
        if not encs:
            return None, 0
        dists = fr.face_distance(known_encodings, encs[0])
        best = int(np.argmin(dists))
        if dists[best] <= config.RECOGNITION_TOLERANCE:
            return known_names[best], int((1 - dists[best]) * 100)
    except Exception as e:
        print(f"[FaceRecognition] recognize_face_now error: {e}")

    return None, 0
