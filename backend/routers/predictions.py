"""
backend/routers/predictions.py
------------------------------
    POST   /predict                  stateless prediction (no DB write)
    GET    /machines/{pk}/predictions stored prediction history

The stateless /predict is great for the dashboard's manual "what-if" form and
for quick testing in /docs — you send six numbers and get the model's verdict
back without touching the database.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import schemas
from backend.deps import get_db, get_predictor
from database import crud
from ml.predictor import CNCPredictor

router = APIRouter(tags=["predictions"])


@router.post("/predict", response_model=schemas.PredictResponse)
def predict(payload: schemas.PredictRequest,
            predictor: CNCPredictor = Depends(get_predictor)):
    """Score a single set of sensor values without storing anything."""
    features = {
        "Type": payload.type,
        "Air temperature [K]": payload.air_temp,
        "Process temperature [K]": payload.process_temp,
        "Rotational speed [rpm]": payload.rotational_speed,
        "Torque [Nm]": payload.torque,
        "Tool wear [min]": payload.tool_wear,
    }
    r = predictor.predict(features)
    return schemas.PredictResponse(
        prediction=r["prediction"],
        probability=r["probability"],
        health_score=r["health_score"],
        status=r["status"],
        rul_minutes=r["rul"]["rul_minutes"],
        rul_band=r["rul"]["band"],
        rul_explanation=r["rul"]["explanation"],
    )


@router.get("/machines/{pk}/predictions", response_model=list[schemas.PredictionOut])
def prediction_history(pk: int, limit: int = 100, db: Session = Depends(get_db)):
    """Return stored predictions for a machine (newest first) — powers trend charts."""
    machine = crud.get_machine(db, pk)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return crud.get_predictions(db, pk, limit=limit)
