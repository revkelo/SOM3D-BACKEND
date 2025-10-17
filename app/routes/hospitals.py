from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import Hospital, Medico, Suscripcion, Pago
from ..schemas import HospitalIn, HospitalOut, HospitalUpdateIn
from fastapi import Body

router = APIRouter()


def _ensure_admin(user):
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")


def _ensure_doctor(user):
    if getattr(user, "rol", None) != "MEDICO":
        raise HTTPException(status_code=403, detail="Requiere rol MEDICO")


@router.get("/hospitals", response_model=list[HospitalOut])
def list_hospitals(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    include_all: bool = Query(False, description="Si true y ADMIN, incluye INACTIVO"),
):
    q = db.query(Hospital)
    if not include_all or getattr(user, "rol", None) != "ADMINISTRADOR":
        q = q.filter(Hospital.estado == "ACTIVO")
    return q.order_by(Hospital.nombre.asc()).all()


@router.get("/hospitals/{hospital_id}", response_model=HospitalOut)
def get_hospital(hospital_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    hosp = db.query(Hospital).filter(Hospital.id_hospital == hospital_id).first()
    if not hosp:
        raise HTTPException(status_code=404, detail="Hospital no encontrado")
    if hosp.estado != "ACTIVO" and getattr(user, "rol", None) != "ADMINISTRADOR":
        raise HTTPException(status_code=404, detail="Hospital no encontrado")
    return hosp


@router.get("/hospitals/by-code/{codigo}", response_model=HospitalOut)
def get_hospital_by_code(codigo: str, db: Session = Depends(get_db)):
    hosp = db.query(Hospital).filter(Hospital.codigo == codigo, Hospital.estado == "ACTIVO").first()
    if not hosp:
        raise HTTPException(status_code=404, detail="Hospital no encontrado o inactivo")
    return hosp


@router.post("/hospitals", response_model=HospitalOut, status_code=201)
def create_hospital(payload: HospitalIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)

    exists = db.query(Hospital).filter(Hospital.codigo == payload.codigo).first()
    if exists:
        raise HTTPException(status_code=409, detail="Código ya existe")

    hosp = Hospital(
        nombre=payload.nombre,
        direccion=payload.direccion,
        ciudad=payload.ciudad,
        telefono=payload.telefono,
        correo=payload.correo,
        codigo=payload.codigo,
        estado="ACTIVO",
    )
    db.add(hosp)
    db.commit()
    db.refresh(hosp)
    return hosp


@router.patch("/hospitals/{hospital_id}", response_model=HospitalOut)
def update_hospital(
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


@router.delete("/hospitals/{hospital_id}", status_code=204)
def delete_hospital(hospital_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
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


@router.post("/hospitals/link-by-code", response_model=HospitalOut)
def link_by_code(
    payload: dict = Body(...),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """Vincula al usuario (MEDICO) con un hospital por su código y activa la cuenta.
    payload: { "codigo": "ABC123" }
    """
    _ensure_doctor(user)
    codigo = (payload or {}).get("codigo")
    if not codigo:
        raise HTTPException(status_code=400, detail="Falta 'codigo'")

    hosp = db.query(Hospital).filter(Hospital.codigo == codigo, Hospital.estado == "ACTIVO").first()
    if not hosp:
        raise HTTPException(status_code=404, detail="Código inválido o hospital inactivo")

    # Asegurar registro Medico del usuario
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not med:
        med = Medico(id_usuario=user.id_usuario)
        db.add(med)
        db.flush()

    # Vincular y activar cuenta
    med.id_hospital = hosp.id_hospital
    try:
        user.activo = True
    except Exception:
        pass
    db.commit()
    return hosp
