from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import Plan

router = APIRouter()

@router.get("/plans")
def list_plans(db: Session = Depends(get_db)):
    plans = db.query(Plan).order_by(Plan.precio.asc()).all()
    return [{"id_plan": p.id_plan, "nombre": p.nombre, "precio": float(p.precio), "periodo": p.periodo, "duracion_meses": p.duracion_meses} for p in plans]