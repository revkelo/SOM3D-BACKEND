from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import Plan, Suscripcion, Pago
from ..auth import get_current_user
from ..schemas import PlanIn, PlanUpdateIn, PlanOut

router = APIRouter()

@router.get("/plans", response_model=list[PlanOut])
def list_plans(db: Session = Depends(get_db)):
    return db.query(Plan).order_by(Plan.precio.asc()).all()


def _ensure_admin(user):
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")


def _validate_periodo_duracion(periodo: str, duracion: int):
    expected = {"MENSUAL": 1, "TRIMESTRAL": 3, "ANUAL": 12}.get(periodo)
    if expected is None or duracion != expected:
        raise HTTPException(status_code=400, detail="periodo y duracion_meses no son coherentes")


@router.get("/plans/{plan_id}", response_model=PlanOut)
def get_plan(plan_id: int, db: Session = Depends(get_db)):
    plan = db.query(Plan).filter(Plan.id_plan == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan no encontrado")
    return plan


@router.post("/plans", response_model=PlanOut, status_code=201)
def create_plan(payload: PlanIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    # unicidad por nombre
    if db.query(Plan).filter(Plan.nombre == payload.nombre).first():
        raise HTTPException(status_code=409, detail="Nombre de plan ya existe")
    _validate_periodo_duracion(payload.periodo, payload.duracion_meses)
    plan = Plan(
        nombre=payload.nombre,
        precio=payload.precio,
        periodo=payload.periodo,
        duracion_meses=payload.duracion_meses,
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


@router.patch("/plans/{plan_id}", response_model=PlanOut)
def update_plan(plan_id: int, payload: PlanUpdateIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    plan = db.query(Plan).filter(Plan.id_plan == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan no encontrado")
    # unicidad de nombre si cambia
    if payload.nombre is not None and payload.nombre != plan.nombre:
        if db.query(Plan).filter(Plan.nombre == payload.nombre).first():
            raise HTTPException(status_code=409, detail="Nombre de plan ya existe")
    # validar periodo/duracion coherentes con mezcla de actuales y nuevos
    new_periodo = payload.periodo if payload.periodo is not None else plan.periodo
    new_duracion = payload.duracion_meses if payload.duracion_meses is not None else plan.duracion_meses
    _validate_periodo_duracion(new_periodo, new_duracion)

    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(plan, k, v)
    db.commit()
    db.refresh(plan)
    return plan


@router.delete("/plans/{plan_id}", status_code=204)
def delete_plan(plan_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    plan = db.query(Plan).filter(Plan.id_plan == plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan no encontrado")

    # Eliminar pagos y suscripciones de este plan para cumplir FKs
    sus_ids = [row[0] for row in db.query(Suscripcion.id_suscripcion).filter(Suscripcion.id_plan == plan_id).all()]
    if sus_ids:
        db.query(Pago).filter(Pago.id_suscripcion.in_(sus_ids)).delete(synchronize_session=False)
        db.query(Suscripcion).filter(Suscripcion.id_suscripcion.in_(sus_ids)).delete(synchronize_session=False)

    db.query(Plan).filter(Plan.id_plan == plan_id).delete(synchronize_session=False)
    db.commit()
    return
