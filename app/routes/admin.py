from datetime import datetime, timedelta
import os
import subprocess
import sys

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


@router.get("/admin/gpu-check")
def admin_gpu_check(user=Depends(get_current_user)):
    _ensure_admin(user)
    out = {
        "gpu_ready": False,
        "gpu_detected": False,
        "backend": "cpu",
        "post_restart_check": {
            "title": "Reiniciar backend y verificar",
            "command": 'python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"',
            "output": "",
            "ok": False,
            "error": None,
        },
        "python": sys.version.split(" ")[0],
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": {"available": False, "error": None, "devices": []},
        "torch": {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
            "devices": [],
            "devices_info": [],
            "arch_list": [],
            "arch_flags": "",
            "matmul": {"ok": False, "sample": None, "error": None},
            "error": None,
        },
        "cupy": {
            "installed": False,
            "cuda_available": False,
            "device_count": 0,
            "error": None,
        },
    }

    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=6,
        )
        if smi.returncode == 0:
            rows = [ln.strip() for ln in (smi.stdout or "").splitlines() if ln.strip()]
            devices = []
            for row in rows:
                parts = [p.strip() for p in row.split(",")]
                if len(parts) >= 3:
                    devices.append(
                        {
                            "name": parts[0],
                            "memory_mb": int(float(parts[1])) if parts[1] else 0,
                            "driver_version": parts[2],
                        }
                    )
                else:
                    devices.append({"raw": row})
            out["nvidia_smi"]["available"] = bool(devices)
            out["nvidia_smi"]["devices"] = devices
        else:
            out["nvidia_smi"]["error"] = (smi.stderr or smi.stdout or "").strip() or f"return_code={smi.returncode}"
    except Exception as ex:
        out["nvidia_smi"]["error"] = str(ex)

    try:
        import torch  # type: ignore

        out["torch"]["installed"] = True
        out["torch"]["version"] = getattr(torch, "__version__", None)
        out["torch"]["cuda_version"] = getattr(torch.version, "cuda", None)
        cuda_ok = bool(torch.cuda.is_available())
        out["torch"]["cuda_available"] = cuda_ok

        if hasattr(torch.cuda, "get_arch_list"):
            try:
                out["torch"]["arch_list"] = torch.cuda.get_arch_list()
            except Exception:
                out["torch"]["arch_list"] = []

        try:
            out["torch"]["arch_flags"] = str(torch._C._cuda_getArchFlags() or "")
        except Exception:
            out["torch"]["arch_flags"] = ""

        if cuda_ok:
            cnt = int(torch.cuda.device_count())
            out["torch"]["device_count"] = cnt
            out["torch"]["devices"] = [torch.cuda.get_device_name(i) for i in range(cnt)]
            devices_info = []
            for i in range(cnt):
                try:
                    props = torch.cuda.get_device_properties(i)
                    major = int(getattr(props, "major", 0))
                    minor = int(getattr(props, "minor", 0))
                    total_mem = int(getattr(props, "total_memory", 0) or 0)
                    devices_info.append(
                        {
                            "index": i,
                            "name": getattr(props, "name", None),
                            "capability": f"sm_{major}{minor} (compute {major}.{minor})",
                            "total_memory_mb": int(total_mem / (1024 * 1024)) if total_mem else 0,
                        }
                    )
                except Exception as ex:
                    devices_info.append({"index": i, "error": str(ex)})
            out["torch"]["devices_info"] = devices_info

            try:
                a = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
                b = torch.randn((2048, 2048), device="cuda", dtype=torch.float16)
                c = a @ b
                torch.cuda.synchronize()
                out["torch"]["matmul"]["ok"] = True
                out["torch"]["matmul"]["sample"] = float(c[0, 0])
            except Exception as ex:
                out["torch"]["matmul"]["ok"] = False
                out["torch"]["matmul"]["error"] = str(ex)
    except Exception as ex:
        out["torch"]["error"] = str(ex)

    try:
        import cupy as cp  # type: ignore

        out["cupy"]["installed"] = True
        cnt = int(cp.cuda.runtime.getDeviceCount())
        out["cupy"]["device_count"] = cnt
        out["cupy"]["cuda_available"] = cnt > 0
    except Exception as ex:
        out["cupy"]["error"] = str(ex)

    out["gpu_detected"] = bool(out["nvidia_smi"]["devices"]) or bool(out["torch"]["cuda_available"]) or bool(out["cupy"]["cuda_available"])
    out["gpu_ready"] = bool(out["torch"]["cuda_available"]) or bool(out["cupy"]["cuda_available"])
    out["backend"] = "cuda" if out["gpu_ready"] else "cpu"

    try:
        chk = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )
        cmd_out = (chk.stdout or "").strip()
        cmd_err = (chk.stderr or "").strip()
        out["post_restart_check"]["output"] = cmd_out
        out["post_restart_check"]["ok"] = chk.returncode == 0
        out["post_restart_check"]["error"] = cmd_err or (None if chk.returncode == 0 else f"return_code={chk.returncode}")
    except Exception as ex:
        out["post_restart_check"]["ok"] = False
        out["post_restart_check"]["error"] = str(ex)
    return out


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
