from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import Estudio, Paciente, Medico, Usuario
from ..schemas import EstudioIn, EstudioOut

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
    )
    db.add(e)
    try:
        db.commit()
    except Exception as ex:
        db.rollback()
        raise HTTPException(status_code=409, detail="Restricción única de estudio (paciente/fecha/modalidad)") from ex
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
        if getattr(user, "rol", None) == "ADMINISTRADOR":
            pass
        else:
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

