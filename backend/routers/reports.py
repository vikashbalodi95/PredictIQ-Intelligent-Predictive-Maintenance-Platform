"""
backend/routers/reports.py
--------------------------
    GET   /machines/{pk}/report      download a PDF maintenance report

This route gathers the machine's latest state from the database, asks the
recommender for advice, builds the PDF (utils/pdf_report.py), and streams it
back with the right headers so the browser downloads a .pdf file.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session

from backend.deps import get_db
from database import crud
from ml.recommender import get_recommendations
from utils.logger import get_logger
from utils.pdf_report import build_maintenance_report

log = get_logger("backend.reports")
router = APIRouter(prefix="/machines", tags=["reports"])


@router.get("/{pk}/report")
def download_report(pk: int, db: Session = Depends(get_db)):
    """Build and return a PDF maintenance report for the machine."""
    machine = crud.get_machine(db, pk)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")

    readings = crud.get_readings(db, pk, limit=1)
    prediction = crud.get_latest_prediction(db, pk)
    if not readings or not prediction:
        raise HTTPException(
            status_code=400,
            detail="No data yet for this machine — start the simulator first.",
        )
    reading = readings[0]

    # Build dicts for the (DB-agnostic) PDF builder.
    machine_d = {
        "machine_id": machine.machine_id, "name": machine.name,
        "type": machine.type, "location": machine.location,
    }
    reading_d = {
        "type": reading.type, "air_temp": reading.air_temp,
        "process_temp": reading.process_temp,
        "rotational_speed": reading.rotational_speed,
        "torque": reading.torque, "tool_wear": reading.tool_wear,
    }
    prediction_d = {
        "failure_prediction": prediction.failure_prediction,
        "failure_probability": prediction.failure_probability,
        "health_score": prediction.health_score,
        "status": prediction.status,
        "rul_minutes": prediction.rul_minutes,
        "rul_band": prediction.rul_band,
    }

    # Recommendations come from the existing rule-based engine.
    features = crud.reading_to_features(reading)
    recommendations = get_recommendations(features, prediction.failure_probability)

    # Recent alerts as plain dicts.
    alerts = [
        {"severity": a.severity, "title": a.title, "message": a.message}
        for a in crud.get_alerts(db, machine_pk=pk, limit=8)
    ]

    pdf_bytes = build_maintenance_report(
        machine_d, reading_d, prediction_d, alerts, recommendations
    )
    log.info("Generated PDF report for %s (%d bytes)", machine.machine_id, len(pdf_bytes))

    filename = f"maintenance_report_{machine.machine_id}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
