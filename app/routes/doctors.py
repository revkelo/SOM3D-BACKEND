from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..db import get_db
from ..core.security import get_current_user, hash_password
from ..models import Usuario, Medico, Hospital, Suscripcion, Pago, Paciente, Estudio, VisorEstado, JobSTL, JobConv, Mensaje
from ..schemas import DoctorIn, DoctorUpdateIn, DoctorOut


router = APIRouter()


def _ensure_admin(user):
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")


def _doctor_to_out(med: Medico) -> DoctorOut:
    u = med.usuario
    return DoctorOut(
        id_medico=med.id_medico,
        id_usuario=med.id_usuario,
        nombre=u.nombre,
        apellido=u.apellido,
        correo=u.correo,
        telefono=u.telefono,
        direccion=u.direccion,
        ciudad=u.ciudad,
        id_hospital=med.id_hospital,
        referenciado=bool(med.referenciado),
        estado=med.estado,
        activo=bool(u.activo),
    )


@router.get("/admin/doctors", response_model=list[DoctorOut])
def list_doctors(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    hospital_id: int | None = Query(None),
    estado: str | None = Query(None, description="ACTIVO/INACTIVO"),
    q: str | None = Query(None, description="Buscar por nombre/apellido/correo"),
):
    _ensure_admin(user)
    qry = db.query(Medico).join(Usuario, Medico.id_usuario == Usuario.id_usuario)
    if hospital_id:
        qry = qry.filter(Medico.id_hospital == hospital_id)
    if estado in ("ACTIVO", "INACTIVO"):
        qry = qry.filter(Medico.estado == estado)
    if q:
        like = f"%{q}%"
        qry = qry.filter(
            (Usuario.nombre.ilike(like))
            | (Usuario.apellido.ilike(like))
            | (Usuario.correo.ilike(like))
        )
    medicos = qry.order_by(Usuario.nombre.asc(), Usuario.apellido.asc()).all()
    return [_doctor_to_out(m) for m in medicos]


@router.get("/admin/doctors/count")
def count_doctors(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    hospital_id: int | None = Query(None),
):
    _ensure_admin(user)
    qry = db.query(func.count(Medico.id_medico))
    if hospital_id:
        qry = qry.filter(Medico.id_hospital == hospital_id)
    total = qry.scalar() or 0
    activos = db.query(func.count(Medico.id_medico)).filter(Medico.estado == "ACTIVO")
    if hospital_id:
        activos = activos.filter(Medico.id_hospital == hospital_id)
    inactivos = db.query(func.count(Medico.id_medico)).filter(Medico.estado == "INACTIVO")
    if hospital_id:
        inactivos = inactivos.filter(Medico.id_hospital == hospital_id)
    return {
        "total": int(total),
        "activos": int(activos.scalar() or 0),
        "inactivos": int(inactivos.scalar() or 0),
    }


@router.post("/admin/doctors", response_model=DoctorOut, status_code=201)
def create_doctor(payload: DoctorIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)

    if db.query(Usuario).filter(Usuario.correo == payload.correo).first():
        raise HTTPException(status_code=409, detail="Correo ya registrado")

    if payload.id_hospital is not None:
        hosp = db.query(Hospital).filter(Hospital.id_hospital == payload.id_hospital).first()
        if not hosp:
            raise HTTPException(status_code=404, detail="Hospital no encontrado")

    u = Usuario(
        nombre=payload.nombre,
        apellido=payload.apellido,
        correo=str(payload.correo),
        contrasena=hash_password(payload.password),
        telefono=payload.telefono,
        direccion=payload.direccion,
        ciudad=payload.ciudad,
        rol="MEDICO",
        activo=bool(payload.activo),
    )
    db.add(u)
    db.flush()

    m = Medico(
        id_usuario=u.id_usuario,
        id_hospital=payload.id_hospital,
        referenciado=bool(payload.referenciado),
        estado="ACTIVO",
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    db.refresh(u)
    return _doctor_to_out(m)


@router.patch("/admin/doctors/{id_medico}", response_model=DoctorOut)
def update_doctor(
    id_medico: int,
    payload: DoctorUpdateIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    _ensure_admin(user)
    m = db.query(Medico).filter(Medico.id_medico == id_medico).first()
    if not m:
        raise HTTPException(status_code=404, detail="Medico no encontrado")
    u = db.query(Usuario).filter(Usuario.id_usuario == m.id_usuario).first()

    data = payload.model_dump(exclude_unset=True)

    if "correo" in data and data["correo"] and data["correo"] != u.correo:
        dup = db.query(Usuario).filter(Usuario.correo == data["correo"]).first()
        if dup:
            raise HTTPException(status_code=409, detail="Correo ya registrado")
        u.correo = data["correo"]

    if "password" in data and data["password"]:
        u.contrasena = hash_password(data["password"])

    # Usuario fields
    for field in ("nombre", "apellido", "telefono", "direccion", "ciudad", "activo"):
        if field in data and data[field] is not None:
            setattr(u, field, data[field])

    # Medico fields
    if "id_hospital" in data:
        if data["id_hospital"] is not None:
            hosp = db.query(Hospital).filter(Hospital.id_hospital == data["id_hospital"]).first()
            if not hosp:
                raise HTTPException(status_code=404, detail="Hospital no encontrado")
        m.id_hospital = data["id_hospital"]

    if "referenciado" in data and data["referenciado"] is not None:
        m.referenciado = bool(data["referenciado"])

    if "estado" in data and data["estado"] in ("ACTIVO", "INACTIVO"):
        m.estado = data["estado"]

    db.commit()
    db.refresh(m)
    db.refresh(u)
    return _doctor_to_out(m)


@router.delete("/admin/doctors/{id_medico}", status_code=204)
def delete_doctor(id_medico: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    m = db.query(Medico).filter(Medico.id_medico == id_medico).first()
    if not m:
        raise HTTPException(status_code=404, detail="Medico no encontrado")

    # Cascade delete dependencias del médico
    uid = m.id_usuario

    # Pacientes del médico
    p_ids = [row[0] for row in db.query(Paciente.id_paciente).filter(Paciente.id_medico == id_medico).all()]

    if p_ids:
        db.query(Mensaje).filter((Mensaje.id_medico == id_medico) | (Mensaje.id_paciente.in_(p_ids))).delete(synchronize_session=False)
        # VisorEstado primero (depende de Paciente y JobSTL)
        db.query(VisorEstado).filter((VisorEstado.id_medico == id_medico) | (VisorEstado.id_paciente.in_(p_ids))).delete(synchronize_session=False)
        # Estudio por paciente o por médico
        db.query(Estudio).filter((Estudio.id_medico == id_medico) | (Estudio.id_paciente.in_(p_ids))).delete(synchronize_session=False)
        # JobSTL por pacientes
        db.query(JobSTL).filter(JobSTL.id_paciente.in_(p_ids)).delete(synchronize_session=False)
        # Pacientes
        db.query(Paciente).filter(Paciente.id_paciente.in_(p_ids)).delete(synchronize_session=False)
    else:
        db.query(Mensaje).filter(Mensaje.id_medico == id_medico).delete(synchronize_session=False)
        # Aún así limpiar VisorEstado y Estudio por médico
        db.query(VisorEstado).filter(VisorEstado.id_medico == id_medico).delete(synchronize_session=False)
        db.query(Estudio).filter(Estudio.id_medico == id_medico).delete(synchronize_session=False)

    # Suscripciones y pagos del médico
    sus_ids = [row[0] for row in db.query(Suscripcion.id_suscripcion).filter(Suscripcion.id_medico == id_medico).all()]
    if sus_ids:
        db.query(Pago).filter(Pago.id_suscripcion.in_(sus_ids)).delete(synchronize_session=False)
        db.query(Suscripcion).filter(Suscripcion.id_suscripcion.in_(sus_ids)).delete(synchronize_session=False)

    # Jobs del usuario (si existen)
    job_ids = [row[0] for row in db.query(JobConv.job_id).filter(JobConv.id_usuario == uid).all()]
    if job_ids:
        db.query(JobSTL).filter(JobSTL.job_id.in_(job_ids)).delete(synchronize_session=False)
        db.query(JobConv).filter(JobConv.job_id.in_(job_ids)).delete(synchronize_session=False)

    # Borrar médico y su usuario
    db.query(Medico).filter(Medico.id_medico == id_medico).delete(synchronize_session=False)
    db.query(Usuario).filter(Usuario.id_usuario == uid).delete(synchronize_session=False)
    db.commit()
    return
