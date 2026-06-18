"""Smoke test for the conversation layer - dispatcher + helpers, no audio/LLM."""

from conversation.offline_conversation import (
    _extract_city_from_text,
    get_current_datetime,
    get_weather,
    handle_custom_commands,
)
from conversation.emotion_questions import (
    behavior_summary_spoken,
    generate_daily_behavior_analysis,
    maybe_emotion_feedback,
)

print("now:", get_current_datetime())

print("extract_city('weather in mumbai'):", _extract_city_from_text("weather in mumbai"))
print("extract_city('delhi ka mausam'):", _extract_city_from_text("delhi ka mausam"))
print("get_weather() (no API key):", get_weather())

print("\n-- handle_custom_commands --")
print("'what time is it' ->", handle_custom_commands("what time is it", "en"))
print("'aaj kya tarikh hai' ->", handle_custom_commands("aaj kya tarikh hai", "hi"))
print("'your name' ->", handle_custom_commands("what is your name", "en"))
print("'my name is Rahul' ->", handle_custom_commands("my name is Rahul", "en"))
print("'what's my name' ->", handle_custom_commands("what's my name", "en"))
print("'tell me about Rahul' ->", handle_custom_commands("tell me about Rahul", "en"))
print("'i like cricket' ->", handle_custom_commands("i like cricket", "en"))
print("'sahi hai' ->", handle_custom_commands("sahi hai", "en"))
print("'galat hai' ->", handle_custom_commands("galat hai", "en"))
print("'actually the capital is Paris' ->", handle_custom_commands("actually the capital is Paris", "en"))
print("'yaad rakh ki mera birthday June mein hai' ->", handle_custom_commands("yaad rakh ki mera birthday june mein hai", "hi"))
print("'kya yaad hai' ->", handle_custom_commands("kya yaad hai", "hi"))
print("'how many people' (no vision) ->", handle_custom_commands("how many people are there", "en"))
print("'what can you see' (no vision) ->", handle_custom_commands("what can you see", "en"))
print("'unrelated question' ->", handle_custom_commands("tell me a joke about robots", "en"))

print("\n-- emotion / behavior --")
print("maybe_emotion_feedback('sad'):", maybe_emotion_feedback("sad", "en"))
print("maybe_emotion_feedback('sad') again (cooldown):", maybe_emotion_feedback("sad", "en"))
report = generate_daily_behavior_analysis()
print("behavior report:", report)
print("behavior summary:", behavior_summary_spoken(report, "en"))
