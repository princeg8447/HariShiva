"""
Live-camera face enrollment for HariShiva V2.

The voice/conversation layer puts a name onto `enrollment_queue`;
`vision.face_detector.camera_detect_loop` picks it up (it owns the camera)
and calls `enroll_face`. This is a headless port of the original's
`enroll_face` - it captures face crops to `data/users/<name>/` and retrains
the face model, but skips the cv2 preview window (the Pi normally runs
without a display attached).
"""

from __future__ import annotations

import queue
import shutil
import threading
import time

import cv2
import face_recognition as fr

from app import config
from vision import face_recognition as face_recog
from voice.text_to_speech import speak

enrollment_queue: "queue.Queue[str]" = queue.Queue()

_TARGET_PHOTOS = 20
_MIN_PHOTOS = 10
_TIMEOUT_SECONDS = 60


def enroll_face(name: str, cap) -> int:
    """Capture up to `_TARGET_PHOTOS` face crops for `name` from `cap` (an
    open cv2.VideoCapture, owned by the camera thread). Returns the number
    of images captured."""
    person_dir = config.USERS_DIR / name
    person_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Enrollment] Enrolling: {name}")

    count = 0
    frame_num = 0
    start_time = time.time()
    locs: list = []

    while count < _TARGET_PHOTOS:
        if time.time() - start_time > _TIMEOUT_SECONDS:
            print(f"[Enrollment] Timeout - {count} photos captured")
            break

        ret, frame = cap.read()
        if not ret:
            cv2.waitKey(30)
            continue

        frame_num += 1
        frame = cv2.flip(frame, 1)

        if frame_num % 2 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = fr.face_locations(rgb, model="hog")

        for (top, right, bottom, left) in locs:
            face_roi = frame[top:bottom, left:right]
            if face_roi.size > 0 and frame_num % 3 == 0 and count < _TARGET_PHOTOS:
                face_roi_resized = cv2.resize(face_roi, (150, 150))
                cv2.imwrite(str(person_dir / f"{name}_{count:03d}.jpg"), face_roi_resized)
                count += 1
                print(f"[Enrollment]   captured {count}/{_TARGET_PHOTOS}")

    if count >= _MIN_PHOTOS:
        print(f"[Enrollment] {count} images captured for {name}. Building encodings...")
        threading.Thread(target=speak, args=(f"Almost done {name}, building your face model.", "en"), daemon=True).start()
        face_recog.train_face_model()
        threading.Thread(
            target=speak,
            args=(f"Done! I have learned your face {name}. I will recognize you from now on.", "en"),
            daemon=True,
        ).start()
    else:
        print(f"[Enrollment] Only {count} photos - need at least {_MIN_PHOTOS}")
        if count > 0:
            face_recog.train_face_model()
        threading.Thread(
            target=speak,
            args=("I could not capture enough samples. Please try again in better lighting.", "en"),
            daemon=True,
        ).start()

    return count


def delete_person(name: str) -> bool:
    person_dir = config.USERS_DIR / name
    if person_dir.exists():
        shutil.rmtree(person_dir)
        face_recog.train_face_model()
        return True
    return False
