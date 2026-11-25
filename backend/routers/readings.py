"""
backend/routers/readings.py
---------------------------
The most important router: ingesting live sensor data.

    POST   /machines/{pk}/readings    ingest one reading (the full pipeline)
    GET    /machines/{pk}/readings    recent readings
    GET    /machines/{pk}/latest      newest reading + prediction (live tiles)

The ingest endpoint runs the whole loop in one place:
    store reading -> ML predict -> store prediction -> raise alerts -> update status
This is exactly what the simulator hammers a few times per second.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import schemas
from backend.deps import get_db, get_predictor
from database import crud
from ml.predictor import CNCPredictor
from utils.alerting import evaluate_alerts
from utils.logger import get_logger

log = get_logger("backend.readings")
router = APIRouter(prefix="/machines", tags=["readings"])


def ingest_reading(db: Session, predictor: CNCPredictor,
                   machine, payload: schemas.ReadingCreate) -> dict:
    """
    Core ingestion pipeline (kept as a plain function so it's easy to read/test).

    Steps:
      1. Save the raw reading.
      2. Convert it to ML features and run the model.
      3. Save the prediction.
      4. Evaluate alert rules and save any alerts.
      5. Update the machine's 'latest status'.
    """
    # 1) Store the raw sensor snapshot.
    reading_data = payload.model_dump()
    if not reading_data.get("type"):
        reading_data["type"] = machine.type
    reading = crud.create_reading(db, machine, reading_data)

    # 2) Run the ML model on the stored reading.
    features = crud.reading_to_features(reading)
    result = predictor.predict(features)

    # 3) Persist the prediction.
    prediction = crud.create_prediction(db, machine, reading, result)

    # 4) Evaluate + store alerts.
    alert_dicts = evaluate_alerts(features, result)
    for a in alert_dicts:
        crud.create_alert(db, machine, a)
    if alert_dicts:
        log.warning("%s: %d alert(s) raised (status=%s)",
                    machine.machine_id, len(alert_dicts), result["status"])

    # 5) Keep the machine's headline status current.
    crud.update_machine_status(db, machine, result["status"])

    return {"reading": reading, "prediction": prediction,
            "alerts_created": len(alert_dicts)}


@router.post("/{pk}/readings", response_model=schemas.IngestResult)
def post_reading(pk: int, payload: schemas.ReadingCreate,
                 db: Session = Depends(get_db),
                 predictor: CNCPredictor = Depends(get_predictor)):
    """Ingest one sensor reading and return the reading + prediction + alert count."""
    machine = crud.get_machine(db, pk)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return ingest_reading(db, predictor, machine, payload)


@router.get("/{pk}/readings", response_model=list[schemas.ReadingOut])
def get_readings(pk: int, limit: int = 100, db: Session = Depends(get_db)):
    """Return the most recent sensor readings for a machine (newest first)."""
    machine = crud.get_machine(db, pk)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return crud.get_readings(db, pk, limit=limit)


@router.get("/{pk}/latest")
def get_latest(pk: int, db: Session = Depends(get_db)):
    """
    Return the newest reading + prediction together. Handy for the dashboard's
    live status tiles, which need both in a single call.
    """
    machine = crud.get_machine(db, pk)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    readings = crud.get_readings(db, pk, limit=1)
    prediction = crud.get_latest_prediction(db, pk)

    return {
        "machine": schemas.MachineOut.model_validate(machine).model_dump(mode="json"),
        "reading": (schemas.ReadingOut.model_validate(readings[0]).model_dump(mode="json")
                    if readings else None),
        "prediction": (schemas.PredictionOut.model_validate(prediction).model_dump(mode="json")
                       if prediction else None),
    }
