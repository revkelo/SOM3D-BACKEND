import json
from datetime import date, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import Estudio, Paciente, Medico, Usuario, ClinicalAudit
from ..schemas import EstudioIn, EstudioOut, EstudioUpdateIn

router = APIRouter()


def _owned_or_admin(db: Session, user: Usuario, paciente_id: int) -> Paciente:
    p = db.query(Paciente).filter(Paciente.id_paciente == paciente_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Paciente no encontrado")
    if getattr(user, "rol", None) == "ADMINISTRADOR":
        return p
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not med or med.id_medico != p.id_medico:
        raise HTTPException(status_code=403, detail="No autorizado")
    return p


def _to_json_safe(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _estudio_to_dict(e: Estudio) -> dict:
    return {
        "id_estudio": e.id_estudio,
        "id_paciente": e.id_paciente,
        "id_medico": e.id_medico,
        "modalidad": e.modalidad,
        "fecha_estudio": _to_json_safe(e.fecha_estudio),
        "descripcion": e.descripcion,
        "notas": e.notas,
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
    db.add(
        ClinicalAudit(
            entity_type=entity_type,
            entity_id=entity_id,
            action=action,
            actor_id_usuario=actor_id_usuario,
            before_json=(json.dumps(before_payload, ensure_ascii=False) if before_payload is not None else None),
            after_json=(json.dumps(after_payload, ensure_ascii=False) if after_payload is not None else None),
            meta_json=(json.dumps(meta or {}, ensure_ascii=False) if meta is not None else None),
        )
    )


@router.post("/studies", response_model=EstudioOut, status_code=201)
def create_study(payload: EstudioIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    p = _owned_or_admin(db, user, payload.id_paciente)
    id_medico = p.id_medico

    e = Estudio(
        id_paciente=p.id_paciente,
        id_medico=id_medico,
        modalidad=payload.modalidad,
        fecha_estudio=payload.fecha_estudio or None,
        descripcion=payload.descripcion,
        notas=payload.notas,
    )
    db.add(e)
    db.flush()
    _audit(
        db,
        entity_type="ESTUDIO",
        entity_id=e.id_estudio,
        action="CREATE",
        actor_id_usuario=getattr(user, "id_usuario", None),
        before_payload=None,
        after_payload=_estudio_to_dict(e),
    )
    try:
        db.commit()
    except Exception as ex:
        db.rollback()
        raise HTTPException(status_code=409, detail="Restriccion unica de estudio (paciente/fecha/modalidad)") from ex
    db.refresh(e)
    return e


@router.get("/studies", response_model=list[EstudioOut])
def list_studies(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    paciente_id: int | None = Query(None),
):
    q = db.query(Estudio)
    if paciente_id is not None:
        p = _owned_or_admin(db, user, paciente_id)
        q = q.filter(Estudio.id_paciente == p.id_paciente)
    else:
        if getattr(user, "rol", None) != "ADMINISTRADOR":
            med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
            if not med:
                return []
            q = q.filter(Estudio.id_medico == med.id_medico)
    return q.order_by(Estudio.creado_en.desc()).all()


@router.get("/studies/{estudio_id}", response_model=EstudioOut)
def get_study(estudio_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    e = db.query(Estudio).filter(Estudio.id_estudio == estudio_id).first()
    if not e:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
        if not med or med.id_medico != e.id_medico:
            raise HTTPException(status_code=403, detail="No autorizado")
    return e


@router.patch("/studies/{estudio_id}", response_model=EstudioOut)
def update_study(
    estudio_id: int,
    payload: EstudioUpdateIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    e = db.query(Estudio).filter(Estudio.id_estudio == estudio_id).first()
    if not e:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")
    _owned_or_admin(db, user, e.id_paciente)

    before = _estudio_to_dict(e)
    data = payload.model_dump(exclude_unset=True)
    for k, v in data.items():
        setattr(e, k, v)

    _audit(
        db,
        entity_type="ESTUDIO",
        entity_id=e.id_estudio,
        action="UPDATE",
        actor_id_usuario=getattr(user, "id_usuario", None),
        before_payload=before,
        after_payload=_estudio_to_dict(e),
        meta={"changed_fields": sorted(list(data.keys()))},
    )

    db.add(e)
    db.commit()
    db.refresh(e)
    return e


@router.delete("/studies/{estudio_id}", status_code=204)
def delete_study(estudio_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    e = db.query(Estudio).filter(Estudio.id_estudio == estudio_id).first()
    if not e:
        raise HTTPException(status_code=404, detail="Estudio no encontrado")
    _owned_or_admin(db, user, e.id_paciente)

    before = _estudio_to_dict(e)
    _audit(
        db,
        entity_type="ESTUDIO",
        entity_id=e.id_estudio,
        action="DELETE",
        actor_id_usuario=getattr(user, "id_usuario", None),
        before_payload=before,
        after_payload=None,
    )
    db.delete(e)
    db.commit()
    return
