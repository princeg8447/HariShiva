"""Smoke test for the memory layer - run manually with the venv python."""

from database.database import init_db
from memory.memory_manager import get_learner, get_person_memory

init_db()

learner = get_learner()
learner.new_session()
learner.record_conversation("my name is Hari and i live in Pune", "Nice to meet you!", lang="en")
print("learning context:")
print(learner.build_learning_context())

pm = get_person_memory("TestUser")
pm.on_seen()
pm.add_fact("likes chess")
pm.add_preference("cricket", like=True)
pm.record_conversation("hi", "hello TestUser", lang="en")
print()
print("person context:")
print(pm.build_context())
print()
print("greeting:", pm.get_greeting())
print("summary:", pm.get_summary())
