"""
database/models.py
------------------
The ORM models — Python classes that map to database tables.

Each class = one table. Each attribute = one column. SQLAlchemy translates
your Python objects into rows and back, so you rarely write raw SQL.

The schema (four tables):

    machines            one row per physical CNC machine
      │
      ├──< sensor_readings   the raw sensor snapshots streamed in over time
      │       │
      │       └──< predictions   one ML result per reading (failure prob, RUL…)
      │
      └──< alerts           notifications raised when something looks wrong

The "<" symbols mean one-to-many: one machine has many readings, etc.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String
)
from sqlalchemy.orm import relationship

from database.connection import Base


def _utcnow() -> datetime:
    """Timezone-aware UTC timestamp (stored consistently regardless of server TZ)."""
    return datetime.now(timezone.utc)


class Machine(Base):
    """A single CNC machine we monitor."""
    __tablename__ = "machines"

    id          = Column(Integer, primary_key=True, index=True)
    machine_id  = Column(String, unique=True, index=True, nullable=False)  # e.g. "CNC-001"
    name        = Column(String, nullable=False)                           # e.g. "Lathe A"
    type        = Column(String, default="L")     # AI4I machine type: L / M / H
    location    = Column(String, default="Shop Floor")
    status      = Column(String, default="Healthy")  # latest known status
    created_at  = Column(DateTime, default=_utcnow)

    # relationship() lets you write `machine.readings` instead of a manual query.
    # cascade="all, delete-orphan" => deleting a machine cleans up its children.
    readings    = relationship("SensorReading", back_populates="machine",
                               cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="machine",
                               cascade="all, delete-orphan")
    alerts      = relationship("Alert", back_populates="machine",
                               cascade="all, delete-orphan")


class SensorReading(Base):
    """One snapshot of the six sensor values at a point in time."""
    __tablename__ = "sensor_readings"

    id            = Column(Integer, primary_key=True, index=True)
    machine_id    = Column(Integer, ForeignKey("machines.id"), index=True, nullable=False)
    timestamp     = Column(DateTime, default=_utcnow, index=True)

    # The six model inputs (column names are friendly; the ML code maps them
    # back to the AI4I feature names like "Air temperature [K]").
    type              = Column(String, default="L")
    air_temp          = Column(Float)   # Air temperature [K]
    process_temp      = Column(Float)   # Process temperature [K]
    rotational_speed  = Column(Float)   # Rotational speed [rpm]
    torque            = Column(Float)   # Torque [Nm]
    tool_wear         = Column(Float)   # Tool wear [min]

    machine    = relationship("Machine", back_populates="readings")
    prediction = relationship("Prediction", back_populates="reading",
                              uselist=False, cascade="all, delete-orphan")


class Prediction(Base):
    """The ML output computed from one SensorReading."""
    __tablename__ = "predictions"

    id                  = Column(Integer, primary_key=True, index=True)
    machine_id          = Column(Integer, ForeignKey("machines.id"), index=True, nullable=False)
    reading_id          = Column(Integer, ForeignKey("sensor_readings.id"), index=True, nullable=False)
    timestamp           = Column(DateTime, default=_utcnow, index=True)

    failure_prediction  = Column(Integer)  # 0 = no failure, 1 = failure predicted
    failure_probability = Column(Float)    # 0..1
    health_score        = Column(Integer)  # 0..100
    status              = Column(String)   # Healthy / Warning / Critical
    rul_minutes         = Column(Integer)  # remaining useful life estimate
    rul_band            = Column(String)   # OK / Plan soon / Urgent

    machine = relationship("Machine", back_populates="predictions")
    reading = relationship("SensorReading", back_populates="prediction")


class Alert(Base):
    """A notification raised by the alerting rules (utils/alerting.py)."""
    __tablename__ = "alerts"

    id           = Column(Integer, primary_key=True, index=True)
    machine_id   = Column(Integer, ForeignKey("machines.id"), index=True, nullable=False)
    timestamp    = Column(DateTime, default=_utcnow, index=True)

    severity     = Column(String)   # Critical / Warning / Info
    title        = Column(String)
    message      = Column(String)
    acknowledged = Column(Boolean, default=False)  # has an operator seen it?

    machine = relationship("Machine", back_populates="alerts")
