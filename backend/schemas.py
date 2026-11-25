"""
backend/schemas.py
------------------
Pydantic "schemas" describe the SHAPE of data going in and out of the API.

Two jobs:
  1. Validation  — FastAPI uses these to reject malformed requests with a clear
     422 error before your code ever runs.
  2. Serialization — they define exactly which fields appear in JSON responses
     (so you never accidentally leak internal columns).

Naming convention used here:
  * `...Create`   – what a client must SEND to create something.
  * `...Out`      – what the API SENDS BACK.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


# ─── Machines ─────────────────────────────────────────────────────────────────

class MachineCreate(BaseModel):
    """Body for POST /machines."""
    machine_id: str = Field(..., examples=["CNC-001"])
    name: str = Field(..., examples=["Lathe A"])
    type: str = Field("L", examples=["L", "M", "H"])
    location: str = Field("Shop Floor")


class MachineOut(BaseModel):
    # from_attributes=True lets Pydantic read values straight off an ORM object.
    model_config = ConfigDict(from_attributes=True)
    id: int
    machine_id: str
    name: str
    type: str
    location: str
    status: str
    created_at: datetime


# ─── Sensor readings ──────────────────────────────────────────────────────────

class ReadingCreate(BaseModel):
    """
    Body for POST /machines/{id}/readings — one sensor snapshot.
    `type` is optional; if omitted we use the machine's own type.
    """
    type: str | None = Field(None, examples=["L"])
    air_temp: float = Field(..., examples=[300.0])
    process_temp: float = Field(..., examples=[310.5])
    rotational_speed: float = Field(..., examples=[1500])
    torque: float = Field(..., examples=[42.0])
    tool_wear: float = Field(..., examples=[120])


class ReadingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    machine_id: int
    timestamp: datetime
    type: str
    air_temp: float
    process_temp: float
    rotational_speed: float
    torque: float
    tool_wear: float


# ─── Predictions ──────────────────────────────────────────────────────────────

class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    machine_id: int
    reading_id: int
    timestamp: datetime
    failure_prediction: int
    failure_probability: float
    health_score: int
    status: str
    rul_minutes: int
    rul_band: str


class IngestResult(BaseModel):
    """
    Response for POST /machines/{id}/readings: the stored reading, the ML
    result, and any alerts that were raised — everything in one round trip.
    """
    reading: ReadingOut
    prediction: PredictionOut
    alerts_created: int


class PredictRequest(BaseModel):
    """Body for the stateless POST /predict (no database write)."""
    type: str = "L"
    air_temp: float
    process_temp: float
    rotational_speed: float
    torque: float
    tool_wear: float


class PredictResponse(BaseModel):
    """Stateless prediction response (mirrors CNCPredictor.predict)."""
    prediction: int
    probability: float
    health_score: int
    status: str
    rul_minutes: int
    rul_band: str
    rul_explanation: str


# ─── Alerts ───────────────────────────────────────────────────────────────────

class AlertOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    machine_id: int
    timestamp: datetime
    severity: str
    title: str
    message: str
    acknowledged: bool
