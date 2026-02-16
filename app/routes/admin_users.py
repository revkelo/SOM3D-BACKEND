from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func

from ..db import get_db
from ..core.security import get_current_user, hash_password
from ..models import Usuario, Medico, Paciente, Estudio, Suscripcion, Pago, VisorEstado, JobSTL, JobConv, Mensaje
from ..schemas import UserOut, AdminCreateIn


router = APIRouter()


def _ensure_admin(user):
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")


@router.get("/admin/users/inactive", response_model=list[UserOut])
def list_inactive_users(db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    users = db.query(Usuario).filter(Usuario.activo == False).order_by(Usuario.creado_en.desc()).all()
    return users


@router.get("/admin/users/admins", response_model=list[UserOut])
def list_admin_users(db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    return (
        db.query(Usuario)
        .filter(Usuario.rol == "ADMINISTRADOR")
        .order_by(Usuario.creado_en.desc())
        .all()
    )


@router.post("/admin/users/admins", response_model=UserOut, status_code=201)
def create_admin_user(payload: AdminCreateIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    if db.query(Usuario).filter(Usuario.correo == str(payload.correo)).first():
        raise HTTPException(status_code=409, detail="Correo ya registrado")

    u = Usuario(
        nombre=payload.nombre,
        apellido=payload.apellido,
        correo=str(payload.correo),
        contrasena=hash_password(payload.password),
        telefono=payload.telefono,
        direccion=payload.direccion,
        ciudad=payload.ciudad,
        rol="ADMINISTRADOR",
        activo=bool(payload.activo),
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@router.delete("/admin/users/{id_usuario}", status_code=204)
def delete_inactive_user(id_usuario: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    u = db.query(Usuario).filter(Usuario.id_usuario == id_usuario).first()
    if not u:
        raise HTTPException(status_code=404, detail="Usuario no encontrado")
    if u.activo:
        raise HTTPException(status_code=409, detail="No se puede borrar un usuario activo")
    if u.rol == "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="No se pueden borrar usuarios ADMINISTRADOR")

    # Si es médico, eliminar en cascada dependencias, suscripciones/pagos y jobs
    m = db.query(Medico).filter(Medico.id_usuario == u.id_usuario).first()
    if m:
        id_medico = m.id_medico
        p_ids = [row[0] for row in db.query(Paciente.id_paciente).filter(Paciente.id_medico == id_medico).all()]
        # Limpiar mensajes vinculados al medico/pacientes antes de borrar pacientes
        if p_ids:
            db.query(Mensaje).filter((Mensaje.id_medico == id_medico) | (Mensaje.id_paciente.in_(p_ids))).delete(synchronize_session=False)
        else:
            db.query(Mensaje).filter(Mensaje.id_medico == id_medico).delete(synchronize_session=False)
        if p_ids:
            db.query(VisorEstado).filter((VisorEstado.id_medico == id_medico) | (VisorEstado.id_paciente.in_(p_ids))).delete(synchronize_session=False)
            db.query(Estudio).filter((Estudio.id_medico == id_medico) | (Estudio.id_paciente.in_(p_ids))).delete(synchronize_session=False)
            db.query(JobSTL).filter(JobSTL.id_paciente.in_(p_ids)).delete(synchronize_session=False)
            db.query(Paciente).filter(Paciente.id_paciente.in_(p_ids)).delete(synchronize_session=False)
        else:
            db.query(VisorEstado).filter(VisorEstado.id_medico == id_medico).delete(synchronize_session=False)
            db.query(Estudio).filter(Estudio.id_medico == id_medico).delete(synchronize_session=False)

        sus_ids = [row[0] for row in db.query(Suscripcion.id_suscripcion).filter(Suscripcion.id_medico == id_medico).all()]
        if sus_ids:
            db.query(Pago).filter(Pago.id_suscripcion.in_(sus_ids)).delete(synchronize_session=False)
            db.query(Suscripcion).filter(Suscripcion.id_suscripcion.in_(sus_ids)).delete(synchronize_session=False)

        job_ids = [row[0] for row in db.query(JobConv.job_id).filter(JobConv.id_usuario == u.id_usuario).all()]
        if job_ids:
            db.query(JobSTL).filter(JobSTL.job_id.in_(job_ids)).delete(synchronize_session=False)
            db.query(JobConv).filter(JobConv.job_id.in_(job_ids)).delete(synchronize_session=False)

        db.query(Medico).filter(Medico.id_medico == id_medico).delete(synchronize_session=False)

    db.query(Usuario).filter(Usuario.id_usuario == id_usuario).delete(synchronize_session=False)
    db.commit()
    return
