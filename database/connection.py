"""
database/connection.py
----------------------
Sets up the SQLAlchemy "engine" (the actual connection to the database) and a
"session" factory (the object you use to read/write rows).

This file is database-agnostic: the SAME code talks to SQLite on your laptop
and PostgreSQL inside Docker. The only difference is the DATABASE_URL in
core/settings.py, which is read from the environment.

Key SQLAlchemy concepts (good to know for interviews):
  * Engine   – manages the pool of low-level DB connections.
  * Session  – a "workspace" for your objects; you add/query, then commit.
  * Base     – the parent class every ORM model inherits from. SQLAlchemy
               uses it to know which Python classes map to which tables.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from core.settings import settings

# SQLite (a single file) needs this flag because, unlike a real server,
# its connections cannot be shared across threads by default. FastAPI uses
# multiple threads, so we relax that check. PostgreSQL ignores this arg.
connect_args = {"check_same_thread": False} if settings.is_sqlite else {}

# The engine is created ONCE and reused for the whole app's lifetime.
engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,   # quietly drops dead connections instead of erroring
    echo=False,           # set True to print every SQL statement (debugging)
)

# SessionLocal() gives you a brand-new session each time it is called.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Every ORM model (Machine, SensorReading, …) will subclass this Base.
Base = declarative_base()


def init_db() -> None:
    """
    Create all tables that don't exist yet.

    Importing the models module first is essential: SQLAlchemy only knows
    about tables whose classes have been imported (which registers them on
    `Base.metadata`).
    """
    from database import models  # noqa: F401  (import side-effect registers tables)
    Base.metadata.create_all(bind=engine)


def get_db():
    """
    FastAPI dependency that yields a database session and ALWAYS closes it.

    Usage in a route:
        def my_route(db: Session = Depends(get_db)):
            ...

    The try/finally guarantees the connection returns to the pool even if the
    request raises an error — preventing connection leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
