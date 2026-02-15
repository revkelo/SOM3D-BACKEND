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
from ..models import Mensaje as Mensaje, Medico, Usuario
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

def _ensure_admin(current_user: Any) -> None:
    rol = getattr(current_user, "rol", None)
    if str(rol).upper() != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")

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


# ---------------------------
# Admin: listar mensajes
# ---------------------------
@router.get("/admin")
def listar_mensajes_admin(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    tipo: Optional[str] = Query(None),
    estado: Optional[str] = Query(None),
    leido_admin: Optional[bool] = Query(None),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _ensure_admin(current_user)

    q = db.query(Mensaje)
    if tipo:
        q = q.filter(Mensaje.tipo == tipo)
    if estado:
        q = q.filter(Mensaje.estado == estado)
    if leido_admin is not None:
        q = q.filter(Mensaje.leido_admin == bool(leido_admin))

    total = q.count()
    items = (
        q.order_by(Mensaje.creado_en.desc() if hasattr(Mensaje, "creado_en") else Mensaje.id_mensaje.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    medico_ids = list({m.id_medico for m in items if getattr(m, "id_medico", None) is not None})
    medicos = db.query(Medico).filter(Medico.id_medico.in_(medico_ids)).all() if medico_ids else []
    medicos_by_id = {m.id_medico: m for m in medicos}

    user_ids = list({m.id_usuario for m in medicos if getattr(m, "id_usuario", None) is not None})
    users = db.query(Usuario).filter(Usuario.id_usuario.in_(user_ids)).all() if user_ids else []
    users_by_id = {u.id_usuario: u for u in users}

    out_items = []
    for m in items:
        medico = medicos_by_id.get(m.id_medico)
        usuario = users_by_id.get(medico.id_usuario) if medico else None
        out_items.append({
            "id_mensaje": m.id_mensaje,
            "id_medico": m.id_medico,
            "id_paciente": m.id_paciente,
            "tipo": m.tipo,
            "titulo": m.titulo,
            "descripcion": m.descripcion,
            "severidad": m.severidad,
            "adjunto_url": m.adjunto_url,
            "estado": m.estado,
            "respuesta_admin": m.respuesta_admin,
            "leido_admin": m.leido_admin,
            "leido_medico": m.leido_medico,
            "creado_en": m.creado_en,
            "actualizado_en": m.actualizado_en,
            "medico_nombre": getattr(usuario, "nombre", None),
            "medico_apellido": getattr(usuario, "apellido", None),
            "medico_correo": getattr(usuario, "correo", None),
        })

    return {"total": total, "items": out_items}


class MensajeAdminUpdateIn(BaseModel):
    estado: Optional[str] = None
    respuesta_admin: Optional[str] = None
    leido_admin: Optional[bool] = None


# ---------------------------
# Admin: responder/actualizar
# ---------------------------
@router.patch("/admin/{id_mensaje}")
def actualizar_mensaje_admin(
    id_mensaje: int,
    body: MensajeAdminUpdateIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _ensure_admin(current_user)

    m = db.query(Mensaje).filter(Mensaje.id_mensaje == id_mensaje).first()
    if not m:
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")

    if body.estado is not None:
        estado = body.estado.strip()
        if not estado:
            raise HTTPException(status_code=400, detail="Estado invalido")
        m.estado = estado
    if body.respuesta_admin is not None:
        txt = body.respuesta_admin.strip()
        m.respuesta_admin = txt if txt else None
    if body.leido_admin is not None:
        m.leido_admin = bool(body.leido_admin)

    if hasattr(Mensaje, "actualizado_en"):
        m.actualizado_en = datetime.utcnow()

    db.commit()
    db.refresh(m)
    return {"ok": True, "item": m}
