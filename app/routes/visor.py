from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import VisorEstado, Paciente, Medico, Usuario, JobSTL, JobConv
from ..schemas import VisorEstadoIn, VisorEstadoOut

router = APIRouter()


def _ensure_owner(db: Session, user: Usuario, paciente_id: int | None, id_jobstl: int | None) -> tuple[int, int | None]:
    role = getattr(user, "rol", None)
    if role == "ADMINISTRADOR":
        if paciente_id:
            p_admin = db.query(Paciente).filter(Paciente.id_paciente == paciente_id).first()
            if not p_admin or not p_admin.id_medico:
                raise HTTPException(status_code=400, detail="Paciente invalido para guardar estado")
            return int(p_admin.id_medico), int(p_admin.id_paciente)
        if id_jobstl:
            js = db.query(JobSTL).filter(JobSTL.id_jobstl == id_jobstl).first()
            if js:
                if js.id_paciente:
                    p = db.query(Paciente).filter(Paciente.id_paciente == js.id_paciente).first()
                    if p and p.id_medico:
                        return int(p.id_medico), int(p.id_paciente)
                jc = db.query(JobConv).filter(JobConv.job_id == js.job_id).first()
                if jc:
                    med_owner = db.query(Medico).filter(Medico.id_usuario == jc.id_usuario).first()
                    if med_owner:
                        return int(med_owner.id_medico), None
        # Fallback para admin sin paciente: usar su perfil de medico si existe,
        # o el primer medico disponible en el sistema.
        med_self = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
        if med_self:
            return int(med_self.id_medico), None
        med_any = db.query(Medico).order_by(Medico.id_medico.asc()).first()
        if med_any:
            return int(med_any.id_medico), None
        raise HTTPException(status_code=400, detail="No hay medicos disponibles para asociar estado de visor")

    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not med:
        raise HTTPException(status_code=400, detail="Usuario no tiene perfil de Medico")
    if paciente_id:
        p = db.query(Paciente).filter(Paciente.id_paciente == paciente_id).first()
        if not p or p.id_medico != med.id_medico:
            raise HTTPException(status_code=403, detail="No autorizado")
        return int(med.id_medico), int(p.id_paciente)
    if id_jobstl:
        js = db.query(JobSTL).filter(JobSTL.id_jobstl == id_jobstl).first()
        if js and js.id_paciente:
            p = db.query(Paciente).filter(Paciente.id_paciente == js.id_paciente).first()
            if p and p.id_medico == med.id_medico:
                return int(med.id_medico), int(p.id_paciente)
    return int(med.id_medico), None


@router.post("/visor/states", response_model=VisorEstadoOut, status_code=201)
def create_state(payload: VisorEstadoIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    id_medico, resolved_patient = _ensure_owner(db, user, payload.id_paciente, payload.id_jobstl)
    st = VisorEstado(
        id_medico=id_medico,
        id_paciente=resolved_patient,
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
def list_states(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    paciente_id: int | None = Query(None),
    job_id: str | None = Query(None),
):
    if getattr(user, "rol", None) == "ADMINISTRADOR":
        q = db.query(VisorEstado)
        if paciente_id is not None:
            q = q.filter(VisorEstado.id_paciente == paciente_id)
        if job_id:
            q = q.join(JobSTL, VisorEstado.id_jobstl == JobSTL.id_jobstl).filter(JobSTL.job_id == job_id)
        return q.order_by(VisorEstado.creado_en.desc()).all()
    med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not med:
        return []
    q = db.query(VisorEstado).filter(VisorEstado.id_medico == med.id_medico)
    if paciente_id is not None:
        q = q.filter(VisorEstado.id_paciente == paciente_id)
    if job_id:
        q = q.join(JobSTL, VisorEstado.id_jobstl == JobSTL.id_jobstl).filter(JobSTL.job_id == job_id)
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


@router.patch("/visor/states/{estado_id}", response_model=VisorEstadoOut)
def update_state(estado_id: int, payload: VisorEstadoIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    st = db.query(VisorEstado).filter(VisorEstado.id_visor_estado == estado_id).first()
    if not st:
        raise HTTPException(status_code=404, detail="Estado no encontrado")

    if getattr(user, "rol", None) != "ADMINISTRADOR":
        med = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
        if not med or st.id_medico != med.id_medico:
            raise HTTPException(status_code=403, detail="No autorizado")

    id_medico, resolved_patient = _ensure_owner(db, user, payload.id_paciente, payload.id_jobstl)
    st.id_medico = id_medico
    st.id_paciente = resolved_patient
    st.id_jobstl = payload.id_jobstl
    st.titulo = payload.titulo
    st.descripcion = payload.descripcion
    st.ui_json = payload.ui_json
    st.modelos_json = payload.modelos_json
    st.notas_json = payload.notas_json
    st.i18n_json = payload.i18n_json
    db.commit()
    db.refresh(st)
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

