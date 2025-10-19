# app/routes/messages.py
from datetime import datetime
from typing import Optional, Literal, Any

from fastapi import (
    APIRouter, Depends, UploadFile, File, Form,
    Query, HTTPException, status
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
# ⬇️ Usa tu modelo real. Si tu tabla es MensajeSistema, alíasalo como Mensaje:
from ..models import Mensaje as Mensaje, Medico
from ..schemas import MensajeOut, MensajeList
from ..core.security import get_current_user

# ⬇️ El front llama /mensajes, así que dejamos ese prefix
router = APIRouter(prefix="/mensajes", tags=["Mensajes"])

# ---------------------------
# Helpers
# ---------------------------
def _get_user_id(current_user: Any) -> Optional[int]:
    """Obtiene id_usuario del 'current_user' (dict u objeto)."""
    if isinstance(current_user, dict):
        return current_user.get("id_usuario")
    return getattr(current_user, "id_usuario", None)

def _get_current_medico(db: Session, user_id: int) -> Medico:
    """Valida que el usuario autenticado sea un médico y lo devuelve."""
    medico = db.query(Medico).filter(Medico.id_usuario == user_id).first()
    if not medico:
        raise HTTPException(status_code=403, detail="Solo médicos pueden usar mensajería")
    return medico

# ---------------------------
# Crear mensaje (FormData)
# ---------------------------
@router.post("/", response_model=MensajeOut, status_code=status.HTTP_201_CREATED)
async def crear_mensaje(
    tipo: Literal["sugerencia", "error"] = Form(...),
    titulo: str = Form(...),
    descripcion: str = Form(...),
    severidad: Literal["baja", "media", "alta"] = Form(...),
    id_paciente: Optional[int] = Form(None),
    adjunto: Optional[UploadFile] = File(None),  # no se persiste por ahora
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    if not user_id:
        raise HTTPException(status_code=401, detail="No autenticado")
    medico = _get_current_medico(db, user_id)

    # Si más adelante quieres guardar archivo local, aquí iría la lógica.
    adjunto_url = None

    ahora = datetime.utcnow()
    m = Mensaje(
        id_medico=medico.id_medico,
        id_paciente=id_paciente,
        tipo=tipo,
        titulo=titulo.strip(),
        descripcion=descripcion.strip(),
        severidad=severidad,
        adjunto_url=adjunto_url,
        estado="nuevo",
        leido_medico=False,
        leido_admin=False,
        # Estos asignados sólo si existen en el modelo/tabla:
        creado_en=ahora if hasattr(Mensaje, "creado_en") else None,
        actualizado_en=ahora if hasattr(Mensaje, "actualizado_en") else None,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m

# ---------------------------
# Listar mensajes (paginado + filtros)
# ---------------------------
@router.get("/", response_model=MensajeList)
def listar_mensajes(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    tipo: Optional[str] = Query(None),
    estado: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    if not user_id:
        raise HTTPException(status_code=401, detail="No autenticado")
    medico = _get_current_medico(db, user_id)

    q = db.query(Mensaje).filter(Mensaje.id_medico == medico.id_medico)
    # Ignora strings vacíos del front
    if tipo:
        q = q.filter(Mensaje.tipo == tipo)
    if estado:
        q = q.filter(Mensaje.estado == estado)

    total = q.count()
    items = (
        q.order_by(Mensaje.creado_en.desc() if hasattr(Mensaje, "creado_en") else Mensaje.id_mensaje.desc())
         .offset((page - 1) * page_size)
         .limit(page_size)
         .all()
    )
    return {"total": total, "items": items}

# ---------------------------
# Marcar como leído (JSON body)
# ---------------------------
class LeidoIn(BaseModel):
    leido_medico: bool

@router.put("/{id_mensaje}/leido")
def marcar_leido(
    id_mensaje: int,
    body: LeidoIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    if not user_id:
        raise HTTPException(status_code=401, detail="No autenticado")
    medico = _get_current_medico(db, user_id)

    m = (
        db.query(Mensaje)
          .filter(
              Mensaje.id_mensaje == id_mensaje,
              Mensaje.id_medico == medico.id_medico
          )
          .first()
    )
    if not m:
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")

    m.leido_medico = bool(body.leido_medico)
    if hasattr(Mensaje, "actualizado_en"):
        m.actualizado_en = datetime.utcnow()
    db.commit()
    return {"ok": True}
