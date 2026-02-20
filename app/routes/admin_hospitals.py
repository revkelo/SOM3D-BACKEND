from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import Hospital, Medico, Suscripcion, Pago, Plan, HospitalCode
from ..schemas import HospitalIn, HospitalOut, HospitalUpdateIn


router = APIRouter()


def _ensure_admin(user):
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")


def _gen_unique_code(db: Session, length: int = 8) -> str:
    import secrets
    import string

    alphabet = string.ascii_uppercase + string.digits
    for _ in range(50):
        code = "".join(secrets.choice(alphabet) for _ in range(length))
        in_hospital = db.query(Hospital).filter(Hospital.codigo == code).first()
        in_codes = db.query(HospitalCode).filter(HospitalCode.codigo == code).first()
        if not in_hospital and not in_codes:
            return code
    raise HTTPException(status_code=500, detail="No fue posible generar un codigo unico")


def _issue_hospital_code(
    db: Session,
    hospital_id: int,
    actor_user_id: int | None,
    explicit_code: str | None = None,
    expires_days: int = 30,
    revoke_previous: bool = True,
) -> HospitalCode:
    if revoke_previous:
        now = datetime.utcnow()
        (
            db.query(HospitalCode)
            .filter(
                HospitalCode.id_hospital == hospital_id,
                HospitalCode.used_at.is_(None),
                HospitalCode.revoked_at.is_(None),
            )
            .update({HospitalCode.revoked_at: now}, synchronize_session=False)
        )

    code = (explicit_code or "").strip() or _gen_unique_code(db)
    if db.query(HospitalCode).filter(HospitalCode.codigo == code).first():
        raise HTTPException(status_code=409, detail="Codigo ya existe")

    expires_at = datetime.utcnow() + timedelta(days=max(1, int(expires_days)))
    hc = HospitalCode(
        id_hospital=hospital_id,
        codigo=code,
        creado_por_id_usuario=actor_user_id,
        expires_at=expires_at,
    )
    db.add(hc)

    hosp = db.query(Hospital).filter(Hospital.id_hospital == hospital_id).first()
    if hosp:
        hosp.codigo = code
    db.flush()
    return hc


@router.get("/admin/hospitals/generate-code")
def admin_generate_hospital_code(db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    return {"codigo": _gen_unique_code(db)}


@router.get("/admin/hospitals", response_model=list[HospitalOut])
def admin_list_hospitals(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    estado: str | None = Query(None, description="ACTIVO/INACTIVO"),
    q: str | None = Query(None, description="Buscar por nombre/ciudad/codigo"),
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
    codigo = (payload.codigo or "").strip() if getattr(payload, "codigo", None) else _gen_unique_code(db)
    if db.query(Hospital).filter(Hospital.codigo == codigo).first():
        raise HTTPException(status_code=409, detail="Codigo ya existe")

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
    db.flush()
    _issue_hospital_code(
        db=db,
        hospital_id=hosp.id_hospital,
        actor_user_id=getattr(user, "id_usuario", None),
        explicit_code=codigo,
        expires_days=30,
        revoke_previous=True,
    )
    db.commit()
    db.refresh(hosp)

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
        dup = db.query(HospitalCode).filter(HospitalCode.codigo == payload.codigo).first()
        if dup:
            raise HTTPException(status_code=409, detail="Codigo ya existe")

    data = payload.model_dump(exclude_unset=True)
    nuevo_codigo = data.pop("codigo", None)
    for k, v in data.items():
        setattr(hosp, k, v)

    if nuevo_codigo is not None:
        _issue_hospital_code(
            db=db,
            hospital_id=hosp.id_hospital,
            actor_user_id=getattr(user, "id_usuario", None),
            explicit_code=nuevo_codigo,
            expires_days=30,
            revoke_previous=True,
        )

    db.commit()
    db.refresh(hosp)
    return hosp


@router.post("/admin/hospitals/{hospital_id}/codes/reissue")
def admin_reissue_hospital_code(
    hospital_id: int,
    expires_days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    _ensure_admin(user)
    hosp = db.query(Hospital).filter(Hospital.id_hospital == hospital_id).first()
    if not hosp:
        raise HTTPException(status_code=404, detail="Hospital no encontrado")

    hc = _issue_hospital_code(
        db=db,
        hospital_id=hospital_id,
        actor_user_id=getattr(user, "id_usuario", None),
        explicit_code=None,
        expires_days=expires_days,
        revoke_previous=True,
    )
    db.commit()
    return {
        "id_hospital": hospital_id,
        "codigo": hc.codigo,
        "expires_at": hc.expires_at.isoformat() if hc.expires_at else None,
    }


@router.post("/admin/hospitals/{hospital_id}/codes/revoke-active")
def admin_revoke_active_hospital_codes(
    hospital_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    _ensure_admin(user)
    hosp = db.query(Hospital).filter(Hospital.id_hospital == hospital_id).first()
    if not hosp:
        raise HTTPException(status_code=404, detail="Hospital no encontrado")

    now = datetime.utcnow()
    revoked = (
        db.query(HospitalCode)
        .filter(
            HospitalCode.id_hospital == hospital_id,
            HospitalCode.used_at.is_(None),
            HospitalCode.revoked_at.is_(None),
        )
        .update({HospitalCode.revoked_at: now}, synchronize_session=False)
    )
    db.commit()
    return {"ok": True, "revoked": int(revoked or 0)}


@router.delete("/admin/hospitals/{hospital_id}", status_code=204)
def admin_delete_hospital(hospital_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)

    hosp = db.query(Hospital).filter(Hospital.id_hospital == hospital_id).first()
    if not hosp:
        raise HTTPException(status_code=404, detail="Hospital no encontrado")

    sus_ids = [row[0] for row in db.query(Suscripcion.id_suscripcion).filter(Suscripcion.id_hospital == hospital_id).all()]

    if sus_ids:
        db.query(Pago).filter(Pago.id_suscripcion.in_(sus_ids)).delete(synchronize_session=False)
        db.query(Suscripcion).filter(Suscripcion.id_suscripcion.in_(sus_ids)).delete(synchronize_session=False)

    now = datetime.utcnow()
    (
        db.query(HospitalCode)
        .filter(HospitalCode.id_hospital == hospital_id, HospitalCode.revoked_at.is_(None))
        .update({HospitalCode.revoked_at: now}, synchronize_session=False)
    )

    db.query(Medico).filter(Medico.id_hospital == hospital_id).update({Medico.id_hospital: None}, synchronize_session=False)

    db.query(Hospital).filter(Hospital.id_hospital == hospital_id).delete(synchronize_session=False)

    db.commit()
    return
