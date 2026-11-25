"""
core/settings.py
----------------
Central, app-wide configuration for the WHOLE platform (backend, database,
simulator, frontend, logging).

Why this file exists
====================
In a small script you can hard-code things like a database URL. In a
production app you want ONE place that reads configuration from the
environment (or an .env file) so the same code can run on your laptop
(SQLite) and inside Docker (PostgreSQL) without any code changes.

We use `pydantic-settings`, which:
  * reads values from environment variables / a `.env` file,
  * validates their types,
  * gives every value a sensible default.

Anything ML-specific (feature names, model thresholds) lives in `ml/config.py`
instead, so the machine-learning code stays self-contained and unchanged.
"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ─── Project paths ────────────────────────────────────────────────────────────
# BASE_DIR is the project root: the folder that contains core/, ml/, backend/ …
# `__file__` -> core/settings.py, so `.parent.parent` climbs up to the root.
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """All runtime configuration as a single, typed object."""

    # Tells pydantic to load variables from a local ".env" file if present.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── General ───────────────────────────────────────────────────────────────
    APP_NAME: str = "CNC Predictive Maintenance Platform"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"  # "development" | "production"

    # ── Database ──────────────────────────────────────────────────────────────
    # Default = a local SQLite file so the project runs with ZERO setup.
    # docker-compose overrides this with a PostgreSQL URL via the DATABASE_URL
    # environment variable, e.g.:
    #   postgresql+psycopg2://cnc:cnc@db:5432/cnc
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'data' / 'cnc.db'}"

    # ── API (FastAPI backend) ─────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    # The URL the frontend and simulator use to reach the backend.
    # Inside Docker this becomes "http://backend:8000".
    API_BASE_URL: str = "http://localhost:8000"

    # ── Simulator ─────────────────────────────────────────────────────────────
    SIM_INTERVAL_SECONDS: float = 2.0   # seconds between generated readings
    SIM_MACHINES: int = 4               # how many machines to simulate by default

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"             # DEBUG | INFO | WARNING | ERROR

    # ── Derived paths (computed, not read from env) ───────────────────────────
    @property
    def DATA_DIR(self) -> Path:
        return BASE_DIR / "data"

    @property
    def MODEL_DIR(self) -> Path:
        return BASE_DIR / "models"

    @property
    def LOG_DIR(self) -> Path:
        return BASE_DIR / "logs"

    @property
    def is_sqlite(self) -> bool:
        """Handy flag — SQLite needs a special connect arg (see connection.py)."""
        return self.DATABASE_URL.startswith("sqlite")


# A single shared instance imported everywhere: `from core.settings import settings`
settings = Settings()

# Make sure the important folders exist on first import (cheap, idempotent).
for _d in (settings.DATA_DIR, settings.MODEL_DIR, settings.LOG_DIR):
    os.makedirs(_d, exist_ok=True)
