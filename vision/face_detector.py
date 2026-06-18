"""
Camera loop for HariShiva V2: YOLO object detection, face recognition,
emotion detection and finger counting, all feeding into
`vision.scene_state` so the conversation layer can describe "what I see"
and know who's currently in front of the camera.

This is a port of the original's `camera_detect_loop`. Detection logic
(YOLO every N frames on a worker thread, face recognition every N frames,
DeepFace emotion every N frames, MediaPipe finger counting) is preserved.

A live preview window (cv2.imshow, same as the original "HariShiva Vision"
window) is shown when `config.SHOW_CAMERA_PREVIEW` is true (default) and a
display is available - set SHOW_CAMERA_PREVIEW=false in .env to run headless.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from app import config
from conversation.emotion_questions import maybe_emotion_feedback
from conversation.offline_conversation import set_active_person
from database.models import LearnedFactRepository
from memory.memory_manager import get_person_memory
from vision import emotion_detection, scene_state
from vision import face_recognition as face_recog
from vision.enrollment import enroll_face, enrollment_queue
from voice.text_to_speech import speak

try:
    import face_recognition as fr
    _FACE_RECOGNITION_AVAILABLE = True
except Exception as _fr_err:
    fr = None
    _FACE_RECOGNITION_AVAILABLE = False
    print(f"[Vision] face_recognition unavailable ({_fr_err.__class__.__name__}) - face features disabled.")

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except Exception as _mp_err:
    mp = None
    _MEDIAPIPE_AVAILABLE = False
    print(f"[Vision] MediaPipe unavailable ({_mp_err.__class__.__name__}) - finger counting disabled.")

try:
    from ultralytics import YOLO
    _yolo_model = YOLO(config.YOLO_MODEL_PATH)
    try:
        import torch
        torch.set_num_threads(2)
    except Exception:
        pass
    _YOLO_AVAILABLE = True
    print("[Vision] YOLO model loaded - object detection active.")
except Exception as _yolo_err:
    _yolo_model = None
    _YOLO_AVAILABLE = False
    print(f"[Vision] YOLO unavailable ({_yolo_err.__class__.__name__}) - object detection disabled.")


YOLO_EVERY = 6      # run YOLO every 6 frames - RPi needs breathing room
FACE_EVERY = 8      # run face recognition every 8 frames
EMOTION_EVERY = 15  # run DeepFace every 15 frames

PREVIEW_WINDOW = "HariShiva Vision"

_yolo_lock = threading.Lock()
_yolo_boxes: list = []
_yolo_running = False
_sticky_obj_ttl: dict[str, float] = {}
_yolo_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")

_greeted_names: set[str] = set()


def get_scene_objects() -> list[str]:
    """Public accessor used by conversation.offline_conversation."""
    return scene_state.get_scene_objects()


def _open_camera(retries: int = 5, delay: float = 0.8):
    """Open the camera, retrying a few times while the driver settles."""
    for attempt in range(retries):
        for idx in (config.CAMERA_INDEX, 1 - config.CAMERA_INDEX):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    print(f"[Vision] Camera opened at index {idx} (attempt {attempt + 1})")
                    return cap
                cap.release()
        print(f"[Vision] Camera not ready, retry {attempt + 1}/{retries}...")
        time.sleep(delay)
    print("[Vision] No camera found!")
    return None


def _yolo_worker(small_frame) -> None:
    """Runs YOLO in the background; updates `_yolo_boxes` + sticky TTLs."""
    global _yolo_running, _yolo_boxes
    try:
        results = _yolo_model(small_frame, imgsz=320, conf=0.40, verbose=False)[0]
        boxes = []
        now = time.time()
        sx = 640 / small_frame.shape[1]
        sy = 480 / small_frame.shape[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            label = _yolo_model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            boxes.append((x1, y1, x2, y2, label, conf))
            _sticky_obj_ttl[label] = now + 4.0
        with _yolo_lock:
            _yolo_boxes = boxes
    except Exception as e:
        print(f"[Vision] YOLO worker error: {e}")
    finally:
        _yolo_running = False


def _update_scene_description(objects: set[str]) -> None:
    faces = [o.replace("person:", "") for o in objects if o.startswith("person:")]
    emotions = [o.replace("emotion:", "") for o in objects if o.startswith("emotion:")]
    others = [o for o in objects if not o.startswith("person:") and not o.startswith("emotion:")]

    parts = []
    if faces:
        parts.append("People in camera: " + ", ".join(faces))
    if others:
        parts.append("Objects: " + ", ".join(others[:5]))
    if emotions:
        parts.append("Emotion: " + ", ".join(emotions[:2]))
    scene_state.set_scene_desc(". ".join(parts))


def camera_detect_loop() -> None:
    """Main vision thread entry point (started by app.main)."""
    global _yolo_running, _yolo_boxes

    face_recog.load_face_model()

    cap = _open_camera()
    if cap is None:
        print("[Vision] Camera thread exiting - no camera available.")
        return

    hands = None
    mp_draw = None
    if _MEDIAPIPE_AVAILABLE:
        try:
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5)
            mp_draw = mp.solutions.drawing_utils
        except Exception as e:
            print(f"[Vision] MediaPipe Hands init failed ({e}) - finger counting disabled.")

    show_preview = config.SHOW_CAMERA_PREVIEW
    if show_preview:
        try:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WINDOW, 800, 450)
            cv2.moveWindow(PREVIEW_WINDOW, 50, 50)
        except Exception as e:
            print(f"[Vision] Could not open preview window ({e}) - running headless.")
            show_preview = False

    print("[Vision] Starting YOLO + face recognition + emotion detection.")

    frame_count = 0
    fail_count = 0
    sticky_faces: set[str] = set()
    unknown_seen_since = 0.0
    last_unknown_prompt = 0.0

    while True:
        # ── Enrollment requests from the conversation layer ──────────────
        if not enrollment_queue.empty():
            enroll_name = enrollment_queue.get()
            threading.Thread(
                target=speak,
                args=(f"Okay {enroll_name}, please look into my eyes. I will capture 20 photos.", "en"),
                daemon=True,
            ).start()
            time.sleep(2)
            print(f"[Vision] Starting enrollment for: {enroll_name}")
            enroll_face(enroll_name, cap)
            print("[Vision] Enrollment done - resuming main loop")
            continue

        ret, frame = cap.read()
        if not ret:
            fail_count += 1
            if fail_count % 30 == 0:
                print(f"[Vision] Camera read failed {fail_count}x - retrying...")
            if fail_count > 100:
                print("[Vision] Reopening camera...")
                cap.release()
                time.sleep(1)
                cap = _open_camera()
                if cap is None:
                    return
                fail_count = 0
            cv2.waitKey(1)
            continue
        fail_count = 0

        frame_count += 1
        frame = cv2.flip(frame, 1)
        scene_state.set_frame(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        objects: set[str] = set(sticky_faces)

        # ── YOLO object detection (background, cached) ───────────────────
        if _YOLO_AVAILABLE and frame_count % YOLO_EVERY == 0 and not _yolo_running:
            _yolo_running = True
            small = cv2.resize(frame, (320, 240))
            _yolo_executor.submit(_yolo_worker, small)

        with _yolo_lock:
            cached_boxes = list(_yolo_boxes)
        for (x1, y1, x2, y2, label, conf) in cached_boxes:
            objects.add(label)
            if show_preview:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

        # ── DeepFace emotion detection (background) ──────────────────────
        if _FACE_RECOGNITION_AVAILABLE and frame_count % EMOTION_EVERY == 0:
            small_locs = fr.face_locations(cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5))
            for idx, (top, right, bottom, left) in enumerate(small_locs):
                t2, r2, b2, l2 = top * 2, right * 2, bottom * 2, left * 2
                face_roi = frame[t2:b2, l2:r2]
                if face_roi.size > 0:
                    threading.Thread(
                        target=emotion_detection.analyze_emotion_async,
                        args=(face_roi.copy(), idx),
                        daemon=True,
                    ).start()

        # ── Face recognition ──────────────────────────────────────────────
        current_run_faces: set[str] = set()
        if _FACE_RECOGNITION_AVAILABLE and frame_count % FACE_EVERY == 0:
            small_rgb = cv2.resize(rgb, (0, 0), fx=0.5, fy=0.5)
            face_locations = fr.face_locations(small_rgb)
            face_encs = (
                fr.face_encodings(small_rgb, face_locations)
                if face_recog.model_trained and face_recog.known_encodings
                else []
            )

            for i, (top, right, bottom, left) in enumerate(face_locations):
                top, right, bottom, left = top * 2, right * 2, bottom * 2, left * 2
                x, y, w, h = left, top, right - left, bottom - top

                name = "Unknown"
                confidence_pct = 0
                if face_recog.model_trained and face_recog.known_encodings and i < len(face_encs):
                    dists = fr.face_distance(face_recog.known_encodings, face_encs[i])
                    best = int(np.argmin(dists))
                    if dists[best] <= config.RECOGNITION_TOLERANCE:
                        name = face_recog.known_names[best]
                        confidence_pct = int((1 - dists[best]) * 100)

                emo = emotion_detection.get_emotion(i)
                emotion_label = emo.get("emotion", "")
                emotion_score = emo.get("score", 0)
                if emotion_label:
                    objects.add(f"emotion:{emotion_label}")

                if show_preview:
                    if name != "Unknown":
                        box_color, text_color, status_txt = (255, 100, 0), (255, 255, 255), name
                    else:
                        box_color, text_color, status_txt = (0, 0, 200), (255, 255, 255), "Unknown"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                    cv2.rectangle(frame, (x, y - 30), (x + w, y), box_color, cv2.FILLED)
                    cv2.putText(frame, status_txt, (x + 5, y - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

                    if emotion_label:
                        emo_text = f"{emotion_label} {int(emotion_score * 100)}%"
                        cv2.putText(frame, emo_text, (x + 5, y + h + 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

                    if name != "Unknown":
                        cv2.putText(frame, f"{confidence_pct}%", (x + 5, y + h + 18),
                                     cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 1)

                if name != "Unknown":
                    objects.add(f"person:{name}")
                    current_run_faces.add(f"person:{name}")
                    scene_state.set_visible_person(name)
                    unknown_seen_since = 0.0

                    pm = get_person_memory(name)
                    set_active_person(name)

                    if emotion_label and emotion_score > 0.6:
                        pm.add_mood(emotion_label)
                        msg = maybe_emotion_feedback(emotion_label, "en")
                        if msg:
                            threading.Thread(target=speak, args=(msg, "en"), daemon=True).start()

                    if name not in _greeted_names and len(name) > 2 and name.lower() not in ("unk", "unknown"):
                        _greeted_names.add(name)
                        returning = pm.visit_count > 0
                        pm.on_seen()
                        greeting = (
                            f"Hello {name}, welcome back! How can I help you?"
                            if returning
                            else f"Nice to meet you, {name}! How can I help you?"
                        )
                        threading.Thread(target=speak, args=(greeting, "en"), daemon=True).start()
                        print(f"[Vision] Greeted: {name}")
                else:
                    objects.add("unknown person")
                    scene_state.set_visible_person(None)

                    # Flowchart "Unknown User" branch: after an unrecognised
                    # face persists ~5s, invite them (once per 2 min) to enroll
                    # via the existing voice command.
                    now_u = time.time()
                    if not current_run_faces:
                        if unknown_seen_since == 0.0:
                            unknown_seen_since = now_u
                        elif (now_u - unknown_seen_since >= 5.0
                              and now_u - last_unknown_prompt >= 120.0):
                            last_unknown_prompt = now_u
                            threading.Thread(
                                target=speak,
                                args=(
                                    "Hello! I don't recognize you yet. Say Hari, "
                                    "and then tell me your name, so I can "
                                    "remember your face.",
                                    "en",
                                ),
                                daemon=True,
                            ).start()
                            print("[Vision] Asked unknown visitor to register.")

            sticky_faces = current_run_faces

        # ── Finger counting (MediaPipe, every 3rd frame) ──────────────────
        if hands is not None and frame_count % 3 == 0:
            try:
                hand_results = hands.process(rgb)
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        if show_preview and mp_draw is not None:
                            mp_draw.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                        tips = [8, 12, 16, 20]
                        count = sum(
                            1 for t in tips
                            if hand_landmarks.landmark[t].y < hand_landmarks.landmark[t - 2].y
                        )
                        if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
                            count += 1
                        objects.add(f"{count} fingers")
                        if show_preview:
                            cv2.putText(frame, f"Fingers: {count}", (10, 70),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            except Exception:
                pass

        # ── Sticky object TTL (objects stay "visible" for a few seconds) ──
        now_t = time.time()
        for lbl, exp in list(_sticky_obj_ttl.items()):
            if now_t < exp:
                objects.add(lbl)
            else:
                del _sticky_obj_ttl[lbl]

        # Remove raw "person"/"unknown person" noise once we have a named face
        if any(o.startswith("person:") for o in objects):
            objects.discard("person")
            objects.discard("unknown person")

        if objects:
            scene_state.set_scene_objects(list(objects))
            _update_scene_description(objects)

        # ── Preview window ─────────────────────────────────────────────────
        if show_preview:
            try:
                known_count = len(face_recog.known_names)
                facts_count = len(LearnedFactRepository.all_with_dates())
                cv2.putText(frame, f"Known: {known_count}  |  Facts: {facts_count}",
                            (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)

                cv2.imshow(PREVIEW_WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[Vision] Preview closed by user ('q') - vision thread exiting.")
                    break
            except Exception as e:
                print(f"[Vision] Preview error ({e}) - switching to headless.")
                show_preview = False

    cap.release()
    if show_preview:
        cv2.destroyAllWindows()
