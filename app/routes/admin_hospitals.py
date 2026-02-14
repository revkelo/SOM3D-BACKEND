from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..db import get_db
from ..core.security import get_current_user
from ..models import Hospital, Medico, Suscripcion, Pago, Plan
from ..schemas import HospitalIn, HospitalOut, HospitalUpdateIn


router = APIRouter()


def _ensure_admin(user):
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")


def _gen_unique_code(db: Session, length: int = 8) -> str:
    import secrets, string
    alphabet = string.ascii_uppercase + string.digits
    for _ in range(50):
        code = "".join(secrets.choice(alphabet) for _ in range(length))
        if not db.query(Hospital).filter(Hospital.codigo == code).first():
            return code
    raise HTTPException(status_code=500, detail="No fue posible generar un código único")


@router.get("/admin/hospitals/generate-code")
def admin_generate_hospital_code(db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    return {"codigo": _gen_unique_code(db)}


@router.get("/admin/hospitals", response_model=list[HospitalOut])
def admin_list_hospitals(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    estado: str | None = Query(None, description="ACTIVO/INACTIVO"),
    q: str | None = Query(None, description="Buscar por nombre/ciudad/código"),
):
    _ensure_admin(user)
    qry = db.query(Hospital)
    if estado in ("ACTIVO", "INACTIVO"):
        qry = qry.filter(Hospital.estado == estado)
    if q:
        like = f"%{q}%"
        qry = qry.filter(
            (Hospital.nombre.ilike(like))
            | (Hospital.ciudad.ilike(like))
            | (Hospital.codigo.ilike(like))
        )
    return qry.order_by(Hospital.nombre.asc()).all()


@router.post("/admin/hospitals", response_model=HospitalOut, status_code=201)
def admin_create_hospital(payload: HospitalIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    codigo = (payload.codigo or "").strip() if getattr(payload, "codigo", None) else None
    if not codigo:
        codigo = _gen_unique_code(db)
    else:
        if db.query(Hospital).filter(Hospital.codigo == codigo).first():
            raise HTTPException(status_code=409, detail="Código ya existe")

    hosp = Hospital(
        nombre=payload.nombre,
        direccion=payload.direccion,
        ciudad=payload.ciudad,
        telefono=payload.telefono,
        correo=payload.correo,
        codigo=codigo,
        estado="ACTIVO",
    )
    db.add(hosp)
    db.commit()
    db.refresh(hosp)

    # Si enviaron plan_id, crear Suscripcion en PAUSADA para el hospital
    if getattr(payload, "plan_id", None):
        plan = db.query(Plan).filter(Plan.id_plan == int(payload.plan_id)).first()
        if plan:
            sus = Suscripcion(
                id_medico=None,
                id_hospital=hosp.id_hospital,
                id_plan=plan.id_plan,
                estado="PAUSADA",
            )
            db.add(sus)
            db.commit()

    return hosp


@router.get("/admin/hospitals/{hospital_id}", response_model=HospitalOut)
def admin_get_hospital(hospital_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    hosp = db.query(Hospital).filter(Hospital.id_hospital == hospital_id).first()
    if not hosp:
        raise HTTPException(status_code=404, detail="Hospital no encontrado")
    return hosp


@router.patch("/admin/hospitals/{hospital_id}", response_model=HospitalOut)
def admin_update_hospital(
    hospital_id: int,
    payload: HospitalUpdateIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    _ensure_admin(user)
    hosp = db.query(Hospital).filter(Hospital.id_hospital == hospital_id).first()
    if not hosp:
        raise HTTPException(status_code=404, detail="Hospital no encontrado")

    if payload.codigo is not None and payload.codigo != hosp.codigo:
        dup = db.query(Hospital).filter(Hospital.codigo == payload.codigo).first()
        if dup:
            raise HTTPException(status_code=409, detail="Código ya existe")

    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(hosp, k, v)
    db.commit()
    db.refresh(hosp)
    return hosp


@router.delete("/admin/hospitals/{hospital_id}", status_code=204)
def admin_delete_hospital(hospital_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)

    hosp = db.query(Hospital).filter(Hospital.id_hospital == hospital_id).first()
    if not hosp:
        raise HTTPException(status_code=404, detail="Hospital no encontrado")

    # Recolectar suscripciones asociadas al hospital
    sus_ids = [row[0] for row in db.query(Suscripcion.id_suscripcion).filter(Suscripcion.id_hospital == hospital_id).all()]

    if sus_ids:
        # Borrar pagos de esas suscripciones
        db.query(Pago).filter(Pago.id_suscripcion.in_(sus_ids)).delete(synchronize_session=False)
        # Borrar suscripciones del hospital
        db.query(Suscripcion).filter(Suscripcion.id_suscripcion.in_(sus_ids)).delete(synchronize_session=False)

    # Desasociar médicos del hospital (id_hospital puede ser NULL)
    db.query(Medico).filter(Medico.id_hospital == hospital_id).update({Medico.id_hospital: None}, synchronize_session=False)

    # Borrar el hospital
    db.query(Hospital).filter(Hospital.id_hospital == hospital_id).delete(synchronize_session=False)

    db.commit()
    return
