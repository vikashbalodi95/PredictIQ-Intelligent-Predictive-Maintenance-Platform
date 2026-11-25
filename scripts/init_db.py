"""
scripts/init_db.py
------------------
One-shot setup script:
  1. create all database tables, and
  2. seed a handful of demo machines so the dashboard isn't empty.

Run it once after cloning:
    python -m scripts.init_db

It is safe to run multiple times — table creation is "if not exists" and the
seeding skips machines that already exist.
"""

from core.settings import settings
from database.connection import SessionLocal, init_db
from database import crud
from utils.logger import get_logger

log = get_logger("scripts.init_db")

# A small fleet covering all three AI4I machine types (L/M/H).
DEFAULT_MACHINES = [
    {"machine_id": "CNC-001", "name": "Lathe A",   "type": "L", "location": "Bay 1"},
    {"machine_id": "CNC-002", "name": "Mill B",    "type": "M", "location": "Bay 1"},
    {"machine_id": "CNC-003", "name": "Router C",  "type": "H", "location": "Bay 2"},
    {"machine_id": "CNC-004", "name": "Grinder D", "type": "L", "location": "Bay 2"},
]


def main() -> None:
    log.info("Creating database tables …")
    init_db()

    db = SessionLocal()
    try:
        # Seed only as many machines as SIM_MACHINES requests (default 4).
        for spec in DEFAULT_MACHINES[: settings.SIM_MACHINES]:
            m = crud.create_machine(db, **spec)
            log.info("Seeded machine %s (%s)", m.machine_id, m.name)
    finally:
        db.close()

    log.info("Database initialised. ✅")


if __name__ == "__main__":
    main()
