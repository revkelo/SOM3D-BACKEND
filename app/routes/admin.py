from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..core.security import get_current_user
from ..db import get_db
from ..models import ClinicalAudit, Estudio, Hospital, Medico, Paciente, Pago, Plan, Suscripcion, Usuario


router = APIRouter()


def _ensure_admin(user):
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")


@router.get("/admin/metrics")
def admin_metrics(db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)

    now = datetime.utcnow()
    in_30_days = now + timedelta(days=30)
    month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    next_month_start = (
        month_start.replace(year=month_start.year + 1, month=1)
        if month_start.month == 12
        else month_start.replace(month=month_start.month + 1)
    )
    last_30_days = now - timedelta(days=30)

    total_hospitals = int(db.query(func.count(Hospital.id_hospital)).scalar() or 0)
    active_hospitals = int(
        db.query(func.count(Hospital.id_hospital))
        .filter(Hospital.estado == "ACTIVO")
        .scalar()
        or 0
    )
    total_doctors = int(db.query(func.count(Medico.id_medico)).scalar() or 0)
    active_doctors = int(
        db.query(func.count(Medico.id_medico))
        .filter(Medico.estado == "ACTIVO")
        .scalar()
        or 0
    )
    total_patients = int(db.query(func.count(Paciente.id_paciente)).scalar() or 0)
    active_patients = int(
        db.query(func.count(Paciente.id_paciente))
        .filter(Paciente.estado == "ACTIVO")
        .scalar()
        or 0
    )
    total_studies = int(db.query(func.count(Estudio.id_estudio)).scalar() or 0)
    studies_last_30_days = int(
        db.query(func.count(Estudio.id_estudio))
        .filter(Estudio.fecha_estudio >= last_30_days)
        .scalar()
        or 0
    )

    total_admins = int(
        db.query(func.count(Usuario.id_usuario))
        .filter(Usuario.rol == "ADMINISTRADOR")
        .scalar()
        or 0
    )

    total_subscriptions = int(db.query(func.count(Suscripcion.id_suscripcion)).scalar() or 0)
    active_subscriptions = int(
        db.query(func.count(Suscripcion.id_suscripcion))
        .filter(Suscripcion.estado == "ACTIVA")
        .scalar()
        or 0
    )
    expiring_subscriptions_30d = int(
        db.query(func.count(Suscripcion.id_suscripcion))
        .filter(
            Suscripcion.estado == "ACTIVA",
            Suscripcion.fecha_expiracion.is_not(None),
            Suscripcion.fecha_expiracion >= now,
            Suscripcion.fecha_expiracion <= in_30_days,
        )
        .scalar()
        or 0
    )

    payments_count_month = int(
        db.query(func.count(Pago.id_pago))
        .filter(Pago.fecha_pago >= month_start, Pago.fecha_pago < next_month_start)
        .scalar()
        or 0
    )
    revenue_month_raw = (
        db.query(func.coalesce(func.sum(Pago.monto), 0))
        .filter(Pago.fecha_pago >= month_start, Pago.fecha_pago < next_month_start)
        .scalar()
        or 0
    )
    revenue_month = float(revenue_month_raw)

    hospitals_without_doctors = int(
        db.query(Hospital.id_hospital)
        .outerjoin(Medico, Medico.id_hospital == Hospital.id_hospital)
        .group_by(Hospital.id_hospital)
        .having(func.count(Medico.id_medico) == 0)
        .count()
    )

    doctors_without_patients = int(
        db.query(Medico.id_medico)
        .outerjoin(Paciente, Paciente.id_medico == Medico.id_medico)
        .group_by(Medico.id_medico)
        .having(func.count(Paciente.id_paciente) == 0)
        .count()
    )

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

    avg_patients_per_doctor = float(total_patients / total_doctors) if total_doctors > 0 else 0.0

    return {
        "summary": {
            "hospitals_total": total_hospitals,
            "doctors_total": total_doctors,
            "patients_total": total_patients,
            "studies_total": total_studies,
            "admins_total": total_admins,
        },
        "clinical": {
            "active_patients": active_patients,
            "studies_last_30_days": studies_last_30_days,
            "avg_patients_per_doctor": round(avg_patients_per_doctor, 2),
            "doctors_without_patients": doctors_without_patients,
        },
        "finance": {
            "payments_month_count": payments_count_month,
            "revenue_month": revenue_month,
        },
        # Compatibilidad con frontend anterior
        "hospitals": {
            "total": total_hospitals,
            "activos": active_hospitals,
            "active": active_hospitals,
            "without_doctors": hospitals_without_doctors,
        },
        "doctors": {
            "total": total_doctors,
            "activos": active_doctors,
        },
        "patients": total_patients,
        "studies": total_studies,
        "subscriptions_legacy": {
            "total": total_subscriptions,
            "activas": active_subscriptions,
        },
        "subscriptions": {
            "total": total_subscriptions,
            "active": active_subscriptions,
            "activas": active_subscriptions,
            "expiring_30_days": expiring_subscriptions_30d,
        },
        "payments": {
            "total": payments_count_month,
            "monto": revenue_month,
        },
        "doctors_by_hospital": doctors_by_hospital,
    }


@router.get("/admin/overview/recent-payments")
def recent_payments(limit: int = 10, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
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
        .order_by(Pago.fecha_pago.desc())
        .limit(limit)
        .all()
    )

    medico_ids = [r[5] for r in rows if r[5] is not None]
    hospital_ids = [r[6] for r in rows if r[6] is not None]

    usuarios_by_medico = {}
    if medico_ids:
        pairs = db.query(Medico.id_medico, Medico.id_usuario).filter(Medico.id_medico.in_(medico_ids)).all()
        uid_map = {mid: uid for (mid, uid) in pairs}
        if uid_map:
            users = db.query(Usuario).filter(Usuario.id_usuario.in_(uid_map.values())).all()
            usuarios_by_id = {u.id_usuario: u for u in users}
            for mid, uid in uid_map.items():
                u = usuarios_by_id.get(uid)
                if u:
                    usuarios_by_medico[mid] = {
                        "id_usuario": u.id_usuario,
                        "nombre": u.nombre,
                        "apellido": u.apellido,
                        "correo": u.correo,
                    }

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
        out.append(
            {
                "id_pago": r[0],
                "fecha_pago": str(r[1]) if r[1] else None,
                "monto": float(r[2]) if r[2] is not None else 0.0,
                "referencia": r[3],
                "id_suscripcion": r[4],
                "owner": owner,
            }
        )
    return out


@router.get("/admin/overview/active-users-subs")
def active_users_subs(db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
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
            .order_by(Suscripcion.creado_en.desc())
            .first()
        )
        if not sus:
            continue
        plan = db.query(Plan).filter(Plan.id_plan == sus.id_plan).first()
        hosp = (
            db.query(Hospital).filter(Hospital.id_hospital == med.id_hospital).first()
            if med.id_hospital
            else None
        )
        out.append(
            {
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
            }
        )
    return out


@router.get("/admin/overview/hospitals-billing")
def hospitals_billing(db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    hospitals = db.query(Hospital).order_by(Hospital.nombre.asc()).all()
    results = []
    for h in hospitals:
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
            pagos_count = int(
                db.query(func.count(Pago.id_pago)).filter(Pago.id_suscripcion.in_(sus_ids)).scalar() or 0
            )
            pagos_sum_val = (
                db.query(func.coalesce(func.sum(Pago.monto), 0))
                .filter(Pago.id_suscripcion.in_(sus_ids))
                .scalar()
                or 0
            )
            pagos_sum = float(pagos_sum_val)
            last = (
                db.query(Pago.fecha_pago)
                .filter(Pago.id_suscripcion.in_(sus_ids))
                .order_by(Pago.fecha_pago.desc())
                .first()
            )
            if last:
                last_pago = str(last[0]) if last[0] else None
        results.append(
            {
                "id_hospital": h.id_hospital,
                "nombre": h.nombre,
                "suscripciones": sus_count,
                "pagos": pagos_count,
                "monto": pagos_sum,
                "ultimo_pago": last_pago,
            }
        )
    return results


@router.get("/admin/audit/clinical")
def clinical_audit(
    limit: int = 100,
    entity_type: str | None = None,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    _ensure_admin(user)
    q = db.query(ClinicalAudit)
    if entity_type:
        q = q.filter(ClinicalAudit.entity_type == entity_type.upper())
    rows = q.order_by(ClinicalAudit.created_at.desc()).limit(max(1, min(limit, 500))).all()
    out = []
    for r in rows:
        out.append(
            {
                "id_audit": r.id_audit,
                "entity_type": r.entity_type,
                "entity_id": r.entity_id,
                "action": r.action,
                "actor_id_usuario": r.actor_id_usuario,
                "before_json": r.before_json,
                "after_json": r.after_json,
                "meta_json": r.meta_json,
                "created_at": str(r.created_at) if r.created_at else None,
            }
        )
    return out
