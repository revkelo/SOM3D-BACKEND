from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import VisorEstado, Paciente, Medico, Usuario
from ..schemas import VisorEstadoIn, VisorEstadoOut

router = APIRouter()


def _ensure_owner(db: Session, user: Usuario, paciente_id: int) -> int:
    if getattr(user, "rol", None) == "ADMINISTRADOR":
        # Admin may save states but requires a Medico id context; deny for simplicity
        raise HTTPException(status_code=403, detail="Solo medicos pueden guardar estados")
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not med:
        raise HTTPException(status_code=400, detail="Usuario no tiene perfil de Medico")
    p = db.query(Paciente).filter(Paciente.id_paciente == paciente_id).first()
    if not p or p.id_medico != med.id_medico:
        raise HTTPException(status_code=403, detail="No autorizado")
    return med.id_medico


@router.post("/visor/states", response_model=VisorEstadoOut, status_code=201)
def create_state(payload: VisorEstadoIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    id_medico = _ensure_owner(db, user, payload.id_paciente)
    st = VisorEstado(
        id_medico=id_medico,
        id_paciente=payload.id_paciente,
        id_jobstl=payload.id_jobstl,
        titulo=payload.titulo,
        descripcion=payload.descripcion,
        ui_json=payload.ui_json,
        modelos_json=payload.modelos_json,
        notas_json=payload.notas_json,
        i18n_json=payload.i18n_json,
    )
    db.add(st)
    db.commit()
    db.refresh(st)
    return st


@router.get("/visor/states", response_model=list[VisorEstadoOut])
def list_states(db: Session = Depends(get_db), user=Depends(get_current_user), paciente_id: int | None = Query(None)):
    if getattr(user, "rol", None) == "ADMINISTRADOR":
        q = db.query(VisorEstado)
        if paciente_id is not None:
            q = q.filter(VisorEstado.id_paciente == paciente_id)
        return q.order_by(VisorEstado.creado_en.desc()).all()
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not med:
        return []
    q = db.query(VisorEstado).filter(VisorEstado.id_medico == med.id_medico)
    if paciente_id is not None:
        q = q.filter(VisorEstado.id_paciente == paciente_id)
    return q.order_by(VisorEstado.creado_en.desc()).all()


@router.get("/visor/states/{estado_id}", response_model=VisorEstadoOut)
def get_state(estado_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    st = db.query(VisorEstado).filter(VisorEstado.id_visor_estado == estado_id).first()
    if not st:
        raise HTTPException(status_code=404, detail="Estado no encontrado")
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
        if not med or st.id_medico != med.id_medico:
            raise HTTPException(status_code=403, detail="No autorizado")
    return st


@router.delete("/visor/states/{estado_id}", status_code=204)
def delete_state(estado_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    st = db.query(VisorEstado).filter(VisorEstado.id_visor_estado == estado_id).first()
    if not st:
        return
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
        if not med or st.id_medico != med.id_medico:
            raise HTTPException(status_code=403, detail="No autorizado")
    db.delete(st)
    db.commit()
    return

