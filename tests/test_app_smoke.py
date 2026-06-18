"""Smoke test for app.config and app.startup - does not run the main loop
(that blocks forever waiting for audio/camera), just the bootstrap pieces.
"""

from app import config, startup

print("BASE_DIR:", config.BASE_DIR)
print("DATABASE_PATH:", config.DATABASE_PATH)

config.ensure_directories()
for d in (config.DATA_DIR, config.LOGS_DIR, config.MODELS_DIR, config.USERS_DIR,
          config.EMBEDDINGS_DIR, config.MEMORIES_DIR, config.BACKUPS_DIR):
    assert d.exists(), f"{d} was not created"
print("All config directories exist.")

logger = startup.run_startup()
logger.info("test_app_smoke: startup completed")
print("OK: startup.run_startup() completed without error.")
