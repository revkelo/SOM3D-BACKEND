from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import Paciente, Medico, Usuario
from ..schemas import PacienteIn, PacienteOut, PacienteUpdateIn

router = APIRouter()


def _ensure_doctor(user: Usuario):
    if getattr(user, "rol", None) != "MEDICO":
        raise HTTPException(status_code=403, detail="Requiere rol MEDICO")


@router.get("/patients", response_model=list[PacienteOut])
def list_patients(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    estado: str | None = Query(None, description="Filtrar por estado"),
):
    q = db.query(Paciente)
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        # Medico: limitar a sus pacientes
        med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
        if not med:
            # auto-crear perfil Medico si no existe, para facilitar onboarding
            med = Medico(id_usuario=user.id_usuario)
            db.add(med)
            db.commit()
        q = q.filter(Paciente.id_medico == med.id_medico)
    if estado:
        q = q.filter(Paciente.estado == estado)
    return q.order_by(Paciente.creado_en.desc()).all()


@router.post("/patients", response_model=PacienteOut, status_code=201)
def create_patient(payload: PacienteIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_doctor(user)
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not med:
        med = Medico(id_usuario=user.id_usuario)
        db.add(med)
        db.commit()

    if payload.doc_tipo and payload.doc_numero:
        dup = db.query(Paciente).filter(
            Paciente.doc_tipo == payload.doc_tipo,
            Paciente.doc_numero == payload.doc_numero,
        ).first()
        if dup:
            raise HTTPException(status_code=409, detail="Documento ya registrado")

    p = Paciente(
        id_medico=med.id_medico,
        doc_tipo=payload.doc_tipo,
        doc_numero=payload.doc_numero,
        nombres=payload.nombres,
        apellidos=payload.apellidos,
        fecha_nacimiento=payload.fecha_nacimiento,
        sexo=payload.sexo,
        telefono=payload.telefono,
        correo=payload.correo,
        direccion=payload.direccion,
        ciudad=payload.ciudad,
        estado="ACTIVO",
    )
    db.add(p)
    db.commit()
    db.refresh(p)
    return p


def _get_patient_owned(db: Session, user: Usuario, paciente_id: int) -> Paciente:
    p = db.query(Paciente).filter(Paciente.id_paciente == paciente_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    if getattr(user, "rol", None) == "ADMINISTRADOR":
        return p
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not med or med.id_medico != p.id_medico:
        raise HTTPException(status_code=403, detail="No autorizado")
    return p


@router.get("/patients/{paciente_id}", response_model=PacienteOut)
def get_patient(paciente_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    p = _get_patient_owned(db, user, paciente_id)
    return p


@router.patch("/patients/{paciente_id}", response_model=PacienteOut)
def update_patient(paciente_id: int, payload: PacienteUpdateIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    p = _get_patient_owned(db, user, paciente_id)

    data = payload.model_dump(exclude_unset=True)
    if ("doc_tipo" in data or "doc_numero" in data) and (data.get("doc_tipo") or p.doc_tipo) and (data.get("doc_numero") or p.doc_numero):
        new_tipo = data.get("doc_tipo", p.doc_tipo)
        new_num = data.get("doc_numero", p.doc_numero)
        dup = db.query(Paciente).filter(
            Paciente.doc_tipo == new_tipo,
            Paciente.doc_numero == new_num,
            Paciente.id_paciente != p.id_paciente,
        ).first()
        if dup:
            raise HTTPException(status_code=409, detail="Documento ya registrado")

    for k, v in data.items():
        setattr(p, k, v)
    db.commit()
    db.refresh(p)
    return p


@router.delete("/patients/{paciente_id}", status_code=204)
def delete_patient(paciente_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    p = _get_patient_owned(db, user, paciente_id)
    db.delete(p)
    db.commit()
    return
