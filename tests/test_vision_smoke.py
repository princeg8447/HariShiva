"""Smoke test for the vision layer - imports + pure-state checks only.

Does not open the camera (headless agent has no /dev/video0 guarantee).
"""

from vision import scene_state

print("scene objects (empty):", scene_state.get_scene_objects())
print("frame (none):", scene_state.get_frame())
print("visible person (none):", scene_state.get_visible_person())

scene_state.set_scene_objects(["person:Rahul", "bottle", "emotion:happy"])
scene_state.set_visible_person("Rahul")
scene_state.set_scene_desc("People in camera: Rahul. Objects: bottle. Emotion: happy")
print("scene objects (set):", scene_state.get_scene_objects())
print("visible person (set):", scene_state.get_visible_person())
print("scene desc (set):", scene_state.get_scene_desc())

from vision import face_recognition as face_recog  # noqa: E402

print("\n-- face_recognition --")
face_recog.load_face_model()
print("known people:", face_recog.list_known_people())
print("recognize_face_now (no frame):", face_recog.recognize_face_now())

from vision import emotion_detection  # noqa: E402

print("\n-- emotion_detection --")
print("DEEPFACE_AVAILABLE:", emotion_detection.DEEPFACE_AVAILABLE)
print("get_emotion(0) (empty):", emotion_detection.get_emotion(0))

from vision import enrollment  # noqa: E402

print("\n-- enrollment --")
print("enrollment_queue empty:", enrollment.enrollment_queue.empty())

from vision.face_detector import get_scene_objects  # noqa: E402

print("\n-- face_detector --")
print("get_scene_objects():", get_scene_objects())
