from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..db import get_db
from datetime import datetime, timedelta
from ..core.security import get_current_user
from ..models import Hospital, Medico, Paciente, Estudio, Suscripcion, Pago, Usuario, Plan


router = APIRouter()


def _ensure_admin(user):
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")


@router.get("/admin/metrics")
def admin_metrics(db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)

    total_hosp = db.query(func.count(Hospital.id_hospital)).scalar() or 0
    active_hosp = db.query(func.count(Hospital.id_hospital)).filter(Hospital.estado == "ACTIVO").scalar() or 0

    total_doc = db.query(func.count(Medico.id_medico)).scalar() or 0
    active_doc = db.query(func.count(Medico.id_medico)).filter(Medico.estado == "ACTIVO").scalar() or 0

    total_pac = db.query(func.count(Paciente.id_paciente)).scalar() or 0
    total_est = db.query(func.count(Estudio.id_estudio)).scalar() or 0

    subs_total = db.query(func.count(Suscripcion.id_suscripcion)).scalar() or 0
    subs_activas = db.query(func.count(Suscripcion.id_suscripcion)).filter(Suscripcion.estado == "ACTIVA").scalar() or 0

    pagos_count = db.query(func.count(Pago.id_pago)).scalar() or 0
    pagos_monto = db.query(func.coalesce(func.sum(Pago.monto), 0)).scalar() or 0
    try:
        pagos_monto = float(pagos_monto)
    except Exception:
        pagos_monto = 0.0


    # Doctores por hospital
    by_hosp = (
        db.query(Hospital.id_hospital, Hospital.nombre, func.count(Medico.id_medico))
        .outerjoin(Medico, Medico.id_hospital == Hospital.id_hospital)
        .group_by(Hospital.id_hospital, Hospital.nombre)
        .order_by(Hospital.nombre.asc())
        .all()
    )
    doctors_by_hospital = [
        {"id_hospital": row[0], "nombre": row[1], "doctores": int(row[2] or 0)} for row in by_hosp
    ]

    return {
        "hospitals": {"total": int(total_hosp), "activos": int(active_hosp)},
        "doctors": {"total": int(total_doc), "activos": int(active_doc)},
        "patients": int(total_pac),
        "studies": int(total_est),
        "subscriptions": {"total": int(subs_total), "activas": int(subs_activas)},
        "payments": {"total": int(pagos_count), "monto": pagos_monto},
        "doctors_by_hospital": doctors_by_hospital,
    }


@router.get("/admin/overview/recent-payments")
def recent_payments(limit: int = 10, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    # Join Pago -> Suscripcion and derive owner (medico->usuario or hospital)
    from sqlalchemy import desc
    rows = (
        db.query(
            Pago.id_pago,
            Pago.fecha_pago,
            Pago.monto,
            Pago.referencia_epayco,
            Suscripcion.id_suscripcion,
            Suscripcion.id_medico,
            Suscripcion.id_hospital,
        )
        .join(Suscripcion, Suscripcion.id_suscripcion == Pago.id_suscripcion)
        .order_by(desc(Pago.fecha_pago))
        .limit(limit)
        .all()
    )
    # Fetch owners in bulk
    medico_ids = [r[5] for r in rows if r[5] is not None]
    hospital_ids = [r[6] for r in rows if r[6] is not None]
    usuarios_by_medico = {}
    if medico_ids:
        pairs = (
            db.query(Medico.id_medico, Medico.id_usuario)
            .filter(Medico.id_medico.in_(medico_ids))
            .all()
        )
        uid_map = {mid: uid for (mid, uid) in pairs}
        if uid_map:
            users = db.query(Usuario).filter(Usuario.id_usuario.in_(uid_map.values())).all()
            usuarios_by_id = {u.id_usuario: u for u in users}
            for mid, uid in uid_map.items():
                u = usuarios_by_id.get(uid)
                if u:
                    usuarios_by_medico[mid] = {"id_usuario": u.id_usuario, "nombre": u.nombre, "apellido": u.apellido, "correo": u.correo}
    hospitales_by_id = {}
    if hospital_ids:
        hs = db.query(Hospital).filter(Hospital.id_hospital.in_(hospital_ids)).all()
        hospitales_by_id = {h.id_hospital: {"id_hospital": h.id_hospital, "nombre": h.nombre} for h in hs}

    out = []
    for r in rows:
        owner = None
        if r[5] is not None:
            owner = {"type": "MEDICO", **usuarios_by_medico.get(r[5], {})}
        elif r[6] is not None:
            owner = {"type": "HOSPITAL", **hospitales_by_id.get(r[6], {})}
        out.append({
            "id_pago": r[0],
            "fecha_pago": str(r[1]) if r[1] else None,
            "monto": float(r[2]) if r[2] is not None else 0.0,
            "referencia": r[3],
            "id_suscripcion": r[4],
            "owner": owner,
        })
    return out


@router.get("/admin/overview/active-users-subs")
def active_users_subs(db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    # Usuarios activos (MEDICO) con su suscripción ACTIVA más reciente y expiración
    from sqlalchemy import desc
    med_users = (
        db.query(Medico, Usuario)
        .join(Usuario, Usuario.id_usuario == Medico.id_usuario)
        .filter(Usuario.activo == True)
        .all()
    )
    out = []
    for med, usr in med_users:
        sus = (
            db.query(Suscripcion)
            .filter(Suscripcion.id_medico == med.id_medico, Suscripcion.estado == "ACTIVA")
            .order_by(desc(Suscripcion.creado_en))
            .first()
        )
        if not sus:
            continue
        plan = db.query(Plan).filter(Plan.id_plan == sus.id_plan).first()
        hosp = db.query(Hospital).filter(Hospital.id_hospital == med.id_hospital).first() if med.id_hospital else None
        out.append({
            "id_usuario": usr.id_usuario,
            "nombre": usr.nombre,
            "apellido": usr.apellido,
            "correo": usr.correo,
            "hospital": hosp.nombre if hosp else None,
            "suscripcion_id": sus.id_suscripcion,
            "plan": plan.nombre if plan else None,
            "periodo": plan.periodo if plan else None,
            "fecha_inicio": str(sus.fecha_inicio) if sus.fecha_inicio else None,
            "fecha_expiracion": str(sus.fecha_expiracion) if sus.fecha_expiracion else None,
        })
    return out


@router.get("/admin/overview/hospitals-billing")
def hospitals_billing(db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    # Por hospital: suscripciones (medicos vinculados o id_hospital directo), pagos: count y monto
    from sqlalchemy import func, desc
    hospitals = db.query(Hospital).order_by(Hospital.nombre.asc()).all()
    results = []
    for h in hospitals:
        # Suscripciones por medicos del hospital o directas al hospital
        sus_q = db.query(Suscripcion.id_suscripcion).join(Medico, isouter=True)
        sus_q = sus_q.filter(
            (Suscripcion.id_hospital == h.id_hospital)
            | ((Suscripcion.id_medico == Medico.id_medico) & (Medico.id_hospital == h.id_hospital))
        )
        sus_ids = [row[0] for row in sus_q.all()]
        sus_count = len(sus_ids)
        pagos_count = 0
        pagos_sum = 0.0
        last_pago = None
        if sus_ids:
            pagos_count = int(db.query(func.count(Pago.id_pago)).filter(Pago.id_suscripcion.in_(sus_ids)).scalar() or 0)
            pagos_sum_val = db.query(func.coalesce(func.sum(Pago.monto), 0)).filter(Pago.id_suscripcion.in_(sus_ids)).scalar() or 0
            try:
                pagos_sum = float(pagos_sum_val)
            except Exception:
                pagos_sum = 0.0
            last = (
                db.query(Pago.fecha_pago)
                .filter(Pago.id_suscripcion.in_(sus_ids))
                .order_by(desc(Pago.fecha_pago))
                .first()
            )
            if last:
                last_pago = str(last[0]) if last[0] else None
        results.append({
            "id_hospital": h.id_hospital,
            "nombre": h.nombre,
            "suscripciones": sus_count,
            "pagos": pagos_count,
            "monto": pagos_sum,
            "ultimo_pago": last_pago,
        })
    return results
