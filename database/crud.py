"""
database/crud.py
----------------
CRUD = Create, Read, Update, Delete.

Every function here takes a `db` session and performs ONE focused database
operation. Keeping all DB access in this single module (instead of scattering
queries across the API routes) is a common, clean pattern: the routes stay
readable, and if the schema changes you only edit one file.

There is also a small helper, `reading_to_features`, that converts a stored
SensorReading row into the exact dict shape the ML model expects.
"""

from datetime import datetime, timezone

from sqlalchemy import desc
from sqlalchemy.orm import Session

from database import models


# ─── Machines ─────────────────────────────────────────────────────────────────

def create_machine(db: Session, machine_id: str, name: str,
                    type: str = "L", location: str = "Shop Floor") -> models.Machine:
    """Insert a new machine (or return the existing one if machine_id is taken)."""
    existing = get_machine_by_code(db, machine_id)
    if existing:
        return existing
    machine = models.Machine(machine_id=machine_id, name=name,
                             type=type, location=location)
    db.add(machine)
    db.commit()
    db.refresh(machine)   # reload so machine.id (auto-generated) is populated
    return machine


def get_machine(db: Session, pk: int) -> models.Machine | None:
    """Fetch a machine by its numeric primary key."""
    return db.query(models.Machine).filter(models.Machine.id == pk).first()


def get_machine_by_code(db: Session, machine_id: str) -> models.Machine | None:
    """Fetch a machine by its human code, e.g. 'CNC-001'."""
    return db.query(models.Machine).filter(
        models.Machine.machine_id == machine_id
    ).first()


def list_machines(db: Session) -> list[models.Machine]:
    """Return all machines, newest first."""
    return db.query(models.Machine).order_by(models.Machine.id).all()


def update_machine_status(db: Session, machine: models.Machine, status: str) -> None:
    """Keep the machine's 'latest status' in sync with its newest prediction."""
    machine.status = status
    db.commit()


# ─── Sensor readings ──────────────────────────────────────────────────────────

def create_reading(db: Session, machine: models.Machine,
                    reading_data: dict) -> models.SensorReading:
    """
    Store one sensor snapshot.

    `reading_data` uses the friendly DB keys (air_temp, torque, …); see
    backend/schemas.py for the request shape.
    """
    reading = models.SensorReading(
        machine_id=machine.id,
        timestamp=reading_data.get("timestamp", datetime.now(timezone.utc)),
        type=reading_data.get("type", machine.type),
        air_temp=reading_data["air_temp"],
        process_temp=reading_data["process_temp"],
        rotational_speed=reading_data["rotational_speed"],
        torque=reading_data["torque"],
        tool_wear=reading_data["tool_wear"],
    )
    db.add(reading)
    db.commit()
    db.refresh(reading)
    return reading


def get_readings(db: Session, machine_pk: int, limit: int = 100) -> list[models.SensorReading]:
    """Most recent readings for a machine (newest first)."""
    return (
        db.query(models.SensorReading)
        .filter(models.SensorReading.machine_id == machine_pk)
        .order_by(desc(models.SensorReading.timestamp))
        .limit(limit)
        .all()
    )


# ─── Predictions ──────────────────────────────────────────────────────────────

def create_prediction(db: Session, machine: models.Machine,
                       reading: models.SensorReading, result: dict) -> models.Prediction:
    """
    Persist the ML result for a reading.

    `result` is the dict returned by CNCPredictor.predict().
    """
    prediction = models.Prediction(
        machine_id=machine.id,
        reading_id=reading.id,
        failure_prediction=result["prediction"],
        failure_probability=result["probability"],
        health_score=result["health_score"],
        status=result["status"],
        rul_minutes=result["rul"]["rul_minutes"],
        rul_band=result["rul"]["band"],
    )
    db.add(prediction)
    db.commit()
    db.refresh(prediction)
    return prediction


def get_predictions(db: Session, machine_pk: int, limit: int = 100) -> list[models.Prediction]:
    """Most recent predictions for a machine (newest first)."""
    return (
        db.query(models.Prediction)
        .filter(models.Prediction.machine_id == machine_pk)
        .order_by(desc(models.Prediction.timestamp))
        .limit(limit)
        .all()
    )


def get_latest_prediction(db: Session, machine_pk: int) -> models.Prediction | None:
    """The single newest prediction (used for the live dashboard tiles)."""
    return (
        db.query(models.Prediction)
        .filter(models.Prediction.machine_id == machine_pk)
        .order_by(desc(models.Prediction.timestamp))
        .first()
    )


# ─── Alerts ───────────────────────────────────────────────────────────────────

def create_alert(db: Session, machine: models.Machine, alert_data: dict) -> models.Alert:
    """Insert one alert row."""
    alert = models.Alert(
        machine_id=machine.id,
        severity=alert_data["severity"],
        title=alert_data["title"],
        message=alert_data["message"],
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    return alert


def get_alerts(db: Session, machine_pk: int | None = None,
               limit: int = 100, only_unacknowledged: bool = False) -> list[models.Alert]:
    """
    Fetch alerts, optionally filtered by machine and/or acknowledgement state.
    """
    q = db.query(models.Alert)
    if machine_pk is not None:
        q = q.filter(models.Alert.machine_id == machine_pk)
    if only_unacknowledged:
        q = q.filter(models.Alert.acknowledged.is_(False))
    return q.order_by(desc(models.Alert.timestamp)).limit(limit).all()


def acknowledge_alert(db: Session, alert_pk: int) -> models.Alert | None:
    """Mark an alert as seen so it stops showing up as 'new'."""
    alert = db.query(models.Alert).filter(models.Alert.id == alert_pk).first()
    if alert:
        alert.acknowledged = True
        db.commit()
        db.refresh(alert)
    return alert


# ─── Helper: DB row → ML feature dict ─────────────────────────────────────────

def reading_to_features(reading: models.SensorReading) -> dict:
    """
    Convert a SensorReading row into the dict the ML model expects.

    The database uses friendly column names; the AI4I model was trained on the
    original bracketed names, so we translate here in ONE place.
    """
    return {
        "Type":                    reading.type,
        "Air temperature [K]":     reading.air_temp,
        "Process temperature [K]": reading.process_temp,
        "Rotational speed [rpm]":  reading.rotational_speed,
        "Torque [Nm]":             reading.torque,
        "Tool wear [min]":         reading.tool_wear,
    }
