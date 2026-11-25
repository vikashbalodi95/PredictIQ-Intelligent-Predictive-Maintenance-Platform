"""
backend/routers/alerts.py
-------------------------
    GET   /alerts                    all alerts (optionally only unacknowledged)
    GET   /machines/{pk}/alerts      alerts for one machine
    POST  /alerts/{alert_id}/ack     mark an alert as acknowledged
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend import schemas
from backend.deps import get_db
from database import crud

router = APIRouter(tags=["alerts"])


@router.get("/alerts", response_model=list[schemas.AlertOut])
def list_alerts(limit: int = 100, unacknowledged_only: bool = False,
                db: Session = Depends(get_db)):
    """Global alert feed across all machines."""
    return crud.get_alerts(db, machine_pk=None, limit=limit,
                           only_unacknowledged=unacknowledged_only)


@router.get("/machines/{pk}/alerts", response_model=list[schemas.AlertOut])
def machine_alerts(pk: int, limit: int = 100, db: Session = Depends(get_db)):
    """Alerts for a single machine."""
    machine = crud.get_machine(db, pk)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return crud.get_alerts(db, machine_pk=pk, limit=limit)


@router.post("/alerts/{alert_id}/ack", response_model=schemas.AlertOut)
def acknowledge(alert_id: int, db: Session = Depends(get_db)):
    """Mark an alert as seen so the dashboard stops flagging it as new."""
    alert = crud.acknowledge_alert(db, alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return alert
