"""Smoke test for dashboard.diagnostics - runs every check section and makes
sure the report comes back as a non-empty string covering each section.
"""

from dashboard.diagnostics import (
    check_audio,
    check_config,
    check_database,
    check_memory,
    check_vision,
    run_diagnostics,
)

print("\n-- check_config --")
for line in check_config():
    print(line)

print("\n-- check_database --")
for line in check_database():
    print(line)

print("\n-- check_memory --")
for line in check_memory():
    print(line)

print("\n-- check_audio --")
for line in check_audio():
    print(line)

print("\n-- check_vision --")
for line in check_vision():
    print(line)

print("\n-- run_diagnostics --")
report = run_diagnostics()
assert "== Config ==" in report
assert "== Database ==" in report
assert "== Memory ==" in report
assert "== Audio ==" in report
assert "== Vision ==" in report
print(report)
print("\nOK: diagnostics report generated successfully.")
