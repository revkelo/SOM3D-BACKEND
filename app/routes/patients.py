import json
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import Paciente, Medico, Usuario, ClinicalAudit, ClinicalNote
from ..schemas import PacienteIn, PacienteOut, PacienteUpdateIn, ClinicalNoteIn, ClinicalNoteOut

router = APIRouter()


def _ensure_doctor(user: Usuario):
    if getattr(user, "rol", None) != "MEDICO":
        raise HTTPException(status_code=403, detail="Requiere rol MEDICO")


def _to_json_safe(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _paciente_to_dict(p: Paciente) -> dict:
    return {
        "id_paciente": p.id_paciente,
        "id_medico": p.id_medico,
        "doc_tipo": p.doc_tipo,
        "doc_numero": p.doc_numero,
        "nombres": p.nombres,
        "apellidos": p.apellidos,
        "fecha_nacimiento": _to_json_safe(p.fecha_nacimiento),
        "sexo": p.sexo,
        "telefono": p.telefono,
        "correo": p.correo,
        "direccion": p.direccion,
        "ciudad": p.ciudad,
        "estado": p.estado,
    }


def _audit(
    db: Session,
    *,
    entity_type: str,
    entity_id: int,
    action: str,
    actor_id_usuario: int | None,
    before_payload: dict | None,
    after_payload: dict | None,
    meta: dict | None = None,
):
    row = ClinicalAudit(
        entity_type=entity_type,
        entity_id=entity_id,
        action=action,
        actor_id_usuario=actor_id_usuario,
        before_json=(json.dumps(before_payload, ensure_ascii=False) if before_payload is not None else None),
        after_json=(json.dumps(after_payload, ensure_ascii=False) if after_payload is not None else None),
        meta_json=(json.dumps(meta or {}, ensure_ascii=False) if meta is not None else None),
    )
    db.add(row)


@router.get("/patients", response_model=list[PacienteOut])
def list_patients(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    estado: str | None = Query(None, description="Filtrar por estado"),
    doc_numero: str | None = Query(None, description="Buscar por cedula/documento"),
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
    if doc_numero:
        like = f"%{doc_numero}%"
        q = q.filter(Paciente.doc_numero.like(like))
    return q.order_by(Paciente.creado_en.desc()).all()


@router.post("/patients", response_model=PacienteOut, status_code=201)
def create_patient(payload: PacienteIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_doctor(user)
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not med:
        med = Medico(id_usuario=user.id_usuario)
        db.add(med)
        db.flush()

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
    db.flush()
    _audit(
        db,
        entity_type="PACIENTE",
        entity_id=p.id_paciente,
        action="CREATE",
        actor_id_usuario=getattr(user, "id_usuario", None),
        before_payload=None,
        after_payload=_paciente_to_dict(p),
    )
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
    before = _paciente_to_dict(p)

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

    after = _paciente_to_dict(p)
    _audit(
        db,
        entity_type="PACIENTE",
        entity_id=p.id_paciente,
        action="UPDATE",
        actor_id_usuario=getattr(user, "id_usuario", None),
        before_payload=before,
        after_payload=after,
        meta={"changed_fields": sorted(list(data.keys()))},
    )

    db.commit()
    db.refresh(p)
    return p


@router.delete("/patients/{paciente_id}", status_code=204)
def delete_patient(paciente_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    p = _get_patient_owned(db, user, paciente_id)
    before = _paciente_to_dict(p)
    _audit(
        db,
        entity_type="PACIENTE",
        entity_id=p.id_paciente,
        action="DELETE",
        actor_id_usuario=getattr(user, "id_usuario", None),
        before_payload=before,
        after_payload=None,
    )
    db.delete(p)
    db.commit()
    return


@router.get("/patients/{paciente_id}/notes", response_model=list[ClinicalNoteOut])
def list_patient_notes(
    paciente_id: int,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    segmento: str | None = Query(None, description="Filtrar por segmento"),
    limit: int = Query(50, ge=1, le=300),
):
    p = _get_patient_owned(db, user, paciente_id)
    q = db.query(ClinicalNote).filter(ClinicalNote.id_paciente == p.id_paciente)
    if segmento:
        q = q.filter(ClinicalNote.segmento == segmento.strip().upper())
    return q.order_by(ClinicalNote.created_at.desc()).limit(limit).all()


@router.post("/patients/{paciente_id}/notes", response_model=ClinicalNoteOut, status_code=201)
def create_patient_note(
    paciente_id: int,
    payload: ClinicalNoteIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    _ensure_doctor(user)
    p = _get_patient_owned(db, user, paciente_id)
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not med or med.id_medico != p.id_medico:
        raise HTTPException(status_code=403, detail="No autorizado para crear notas en este paciente")

    segmento = (payload.segmento or "GENERAL").strip().upper()[:60]
    note = ClinicalNote(
        id_paciente=p.id_paciente,
        id_medico=med.id_medico,
        segmento=segmento,
        texto=payload.texto.strip(),
        anchor_json=payload.anchor_json,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return note
