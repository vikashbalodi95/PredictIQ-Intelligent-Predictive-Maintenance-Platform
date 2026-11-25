"""
backend/routers/machines.py
---------------------------
Endpoints for managing CNC machines.

    POST   /machines            create a machine
    GET    /machines            list all machines
    GET    /machines/{pk}       get one machine

A "router" is just a group of related endpoints. main.py plugs each router
into the app. Splitting routes by resource keeps files small and focused.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend import schemas
from backend.deps import get_db
from database import crud
from utils.logger import get_logger

log = get_logger("backend.machines")

# prefix => every path here starts with /machines; tags => groups them in /docs.
router = APIRouter(prefix="/machines", tags=["machines"])


@router.post("", response_model=schemas.MachineOut, status_code=status.HTTP_201_CREATED)
def create_machine(payload: schemas.MachineCreate, db: Session = Depends(get_db)):
    """Register a new machine. Idempotent: re-posting the same machine_id is OK."""
    machine = crud.create_machine(
        db, machine_id=payload.machine_id, name=payload.name,
        type=payload.type, location=payload.location,
    )
    log.info("Machine ready: %s (%s)", machine.machine_id, machine.name)
    return machine


@router.get("", response_model=list[schemas.MachineOut])
def list_machines(db: Session = Depends(get_db)):
    """Return every machine — used to populate the dashboard selector."""
    return crud.list_machines(db)


@router.get("/{pk}", response_model=schemas.MachineOut)
def get_machine(pk: int, db: Session = Depends(get_db)):
    """Fetch one machine by its numeric id."""
    machine = crud.get_machine(db, pk)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine not found")
    return machine
