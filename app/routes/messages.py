# app/routes/messages.py
import json
import re
from datetime import datetime
from typing import Optional, Literal, Any

from fastapi import (
    APIRouter,
    Depends,
    Form,
    Query,
    HTTPException,
    status,
)
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import (
    Mensaje as Mensaje,
    Medico,
    Usuario,
    MensajeGestion,
    MensajeEvento,
)
from ..schemas import MensajeOut, MensajeList
from ..core.security import get_current_user

router = APIRouter(prefix="/mensajes", tags=["Mensajes"])

_RE_MSG_TITLE = re.compile("^[A-Za-z\\u00C0-\\u00FF0-9()_.,:;!?\"' -]{5,200}$")


def _norm_spaces(value: str) -> str:
    return re.sub(r"\s{2,}", " ", str(value or "").strip())


def _clean_text(value: str, max_len: int) -> str:
    return _norm_spaces(value).replace("<", "").replace(">", "")[:max_len]


# ---------------------------
# Helpers
# ---------------------------
def _get_user_id(current_user: Any) -> Optional[int]:
    if isinstance(current_user, dict):
        return current_user.get("id_usuario")
    return getattr(current_user, "id_usuario", None)


def _get_current_medico(db: Session, user_id: int) -> Medico:
    medico = db.query(Medico).filter(Medico.id_usuario == user_id).first()
    if not medico:
        raise HTTPException(status_code=403, detail="Solo medicos pueden usar mensajeria")
    return medico


def _ensure_admin(current_user: Any):
    rol = (current_user.get("rol") if isinstance(current_user, dict) else getattr(current_user, "rol", None)) or ""
    if str(rol).upper() != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")


def _append_event(
    db: Session,
    *,
    id_mensaje: int,
    actor_id_usuario: int | None,
    accion: str,
    detalle: dict | None = None,
):
    db.add(
        MensajeEvento(
            id_mensaje=id_mensaje,
            id_actor_usuario=actor_id_usuario,
            accion=accion,
            detalle_json=(json.dumps(detalle or {}, ensure_ascii=False) if detalle is not None else None),
        )
    )


def _parse_json(raw: str | None):
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


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
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    user_id = _get_user_id(current_user)
    if not user_id:
        raise HTTPException(status_code=401, detail="No autenticado")
    medico = _get_current_medico(db, user_id)

    titulo_clean = _clean_text(titulo, 200)
    descripcion_clean = _clean_text(descripcion, 2000)
    if not _RE_MSG_TITLE.fullmatch(titulo_clean):
        raise HTTPException(status_code=422, detail="Titulo invalido")
    if len(descripcion_clean) < 10:
        raise HTTPException(status_code=422, detail="Descripcion invalida")

    ahora = datetime.utcnow()
    m = Mensaje(
        id_medico=medico.id_medico,
        id_paciente=id_paciente,
        tipo=tipo,
        titulo=titulo_clean,
        descripcion=descripcion_clean,
        severidad=severidad,
        adjunto_url=None,
        estado="nuevo",
        leido_medico=False,
        leido_admin=False,
        creado_en=ahora if hasattr(Mensaje, "creado_en") else None,
        actualizado_en=ahora if hasattr(Mensaje, "actualizado_en") else None,
    )
    db.add(m)
    db.flush()
    _append_event(
        db,
        id_mensaje=m.id_mensaje,
        actor_id_usuario=user_id,
        accion="CREADO",
        detalle={"tipo": tipo, "severidad": severidad},
    )
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
# Marcar como leido (JSON body)
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
            Mensaje.id_medico == medico.id_medico,
        )
        .first()
    )
    if not m:
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")

    m.leido_medico = bool(body.leido_medico)
    if hasattr(Mensaje, "actualizado_en"):
        m.actualizado_en = datetime.utcnow()
    _append_event(
        db,
        id_mensaje=id_mensaje,
        actor_id_usuario=user_id,
        accion="LEIDO_MEDICO",
        detalle={"leido_medico": bool(body.leido_medico)},
    )
    db.commit()
    return {"ok": True}


class AdminMensajeUpdateIn(BaseModel):
    estado: Optional[Literal["nuevo", "analisis", "en_curso", "resuelto"]] = None
    respuesta_admin: Optional[str] = None
    leido_admin: Optional[bool] = None
    asignado_admin_id_usuario: Optional[int] = None


@router.get("/admin/inbox")
def admin_listar_mensajes(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    tipo: Optional[str] = Query(None),
    estado: Optional[str] = Query(None),
    severidad: Optional[str] = Query(None),
    q: Optional[str] = Query(None, description="Buscar por titulo/descripcion/correo"),
    only_unread: bool = Query(False),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _ensure_admin(current_user)

    qry = (
        db.query(Mensaje, Medico, Usuario, MensajeGestion)
        .join(Medico, Medico.id_medico == Mensaje.id_medico)
        .join(Usuario, Usuario.id_usuario == Medico.id_usuario)
        .outerjoin(MensajeGestion, MensajeGestion.id_mensaje == Mensaje.id_mensaje)
    )
    if tipo:
        qry = qry.filter(Mensaje.tipo == tipo)
    if estado:
        qry = qry.filter(Mensaje.estado == estado)
    if severidad:
        qry = qry.filter(Mensaje.severidad == severidad)
    if only_unread:
        qry = qry.filter(Mensaje.leido_admin == False)
    if q:
        like = f"%{q}%"
        qry = qry.filter(
            (Mensaje.titulo.ilike(like))
            | (Mensaje.descripcion.ilike(like))
            | (Usuario.correo.ilike(like))
        )

    total = qry.count()
    rows = (
        qry.order_by(Mensaje.actualizado_en.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    msg_ids = [msg.id_mensaje for (msg, _, _, _) in rows]
    events_by_msg = {k: [] for k in msg_ids}
    if msg_ids:
        ev_rows = (
            db.query(MensajeEvento, Usuario)
            .outerjoin(Usuario, Usuario.id_usuario == MensajeEvento.id_actor_usuario)
            .filter(MensajeEvento.id_mensaje.in_(msg_ids))
            .order_by(MensajeEvento.creado_en.desc())
            .all()
        )
        for ev, actor in ev_rows:
            if len(events_by_msg.get(ev.id_mensaje, [])) >= 3:
                continue
            events_by_msg.setdefault(ev.id_mensaje, []).append(
                {
                    "id_evento": ev.id_evento,
                    "accion": ev.accion,
                    "detalle": _parse_json(ev.detalle_json),
                    "creado_en": ev.creado_en,
                    "actor": (
                        {
                            "id_usuario": actor.id_usuario,
                            "nombre": actor.nombre,
                            "apellido": actor.apellido,
                            "correo": actor.correo,
                        }
                        if actor
                        else None
                    ),
                }
            )

    assignee_ids = [g.asignado_admin_id_usuario for (_, _, _, g) in rows if g and g.asignado_admin_id_usuario is not None]
    assignees = {}
    if assignee_ids:
        for u in db.query(Usuario).filter(Usuario.id_usuario.in_(assignee_ids)).all():
            assignees[u.id_usuario] = {
                "id_usuario": u.id_usuario,
                "nombre": u.nombre,
                "apellido": u.apellido,
                "correo": u.correo,
            }

    items = []
    for msg, med, usr, gestion in rows:
        assignee = None
        if gestion and gestion.asignado_admin_id_usuario is not None:
            assignee = assignees.get(gestion.asignado_admin_id_usuario)
            if assignee is None:
                assignee = {"id_usuario": gestion.asignado_admin_id_usuario}

        items.append(
            {
                "id_mensaje": msg.id_mensaje,
                "id_medico": msg.id_medico,
                "id_paciente": msg.id_paciente,
                "tipo": msg.tipo,
                "titulo": msg.titulo,
                "descripcion": msg.descripcion,
                "severidad": msg.severidad,
                "adjunto_url": msg.adjunto_url,
                "estado": msg.estado,
                "respuesta_admin": msg.respuesta_admin,
                "leido_admin": bool(msg.leido_admin),
                "leido_medico": bool(msg.leido_medico),
                "creado_en": msg.creado_en,
                "actualizado_en": msg.actualizado_en,
                "asignado_admin": assignee,
                "bitacora_preview": events_by_msg.get(msg.id_mensaje, []),
                "emisor": {
                    "id_medico": med.id_medico,
                    "id_usuario": usr.id_usuario,
                    "nombre": usr.nombre,
                    "apellido": usr.apellido,
                    "correo": usr.correo,
                },
            }
        )
    return {"total": total, "items": items}


@router.get("/admin/inbox/{id_mensaje}/events")
def admin_listar_eventos_mensaje(
    id_mensaje: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _ensure_admin(current_user)
    exists = db.query(Mensaje.id_mensaje).filter(Mensaje.id_mensaje == id_mensaje).first()
    if not exists:
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")

    rows = (
        db.query(MensajeEvento, Usuario)
        .outerjoin(Usuario, Usuario.id_usuario == MensajeEvento.id_actor_usuario)
        .filter(MensajeEvento.id_mensaje == id_mensaje)
        .order_by(MensajeEvento.creado_en.desc())
        .all()
    )
    out = []
    for ev, actor in rows:
        out.append(
            {
                "id_evento": ev.id_evento,
                "accion": ev.accion,
                "detalle": _parse_json(ev.detalle_json),
                "creado_en": ev.creado_en,
                "actor": (
                    {
                        "id_usuario": actor.id_usuario,
                        "nombre": actor.nombre,
                        "apellido": actor.apellido,
                        "correo": actor.correo,
                    }
                    if actor
                    else None
                ),
            }
        )
    return {"items": out}


@router.patch("/admin/inbox/{id_mensaje}")
def admin_actualizar_mensaje(
    id_mensaje: int,
    body: AdminMensajeUpdateIn,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    _ensure_admin(current_user)
    actor_id = _get_user_id(current_user)

    m = db.query(Mensaje).filter(Mensaje.id_mensaje == id_mensaje).first()
    if not m:
        raise HTTPException(status_code=404, detail="Mensaje no encontrado")

    gestion = db.query(MensajeGestion).filter(MensajeGestion.id_mensaje == id_mensaje).first()
    if not gestion:
        gestion = MensajeGestion(id_mensaje=id_mensaje)
        db.add(gestion)
        db.flush()

    data = body.model_dump(exclude_unset=True)
    respuesta_nueva = None
    changed = {}

    if "respuesta_admin" in data:
        respuesta_nueva = _clean_text((data.get("respuesta_admin") or ""), 4000)
        data["respuesta_admin"] = respuesta_nueva or None

    for field in ("estado", "respuesta_admin", "leido_admin"):
        if field in data:
            old = getattr(m, field)
            new = data[field]
            if old != new:
                setattr(m, field, new)
                changed[field] = {"old": old, "new": new}

    if "asignado_admin_id_usuario" in data:
        new_assignee = data.get("asignado_admin_id_usuario")
        if new_assignee is not None:
            admin_exists = (
                db.query(Usuario)
                .filter(Usuario.id_usuario == new_assignee, Usuario.rol == "ADMINISTRADOR")
                .first()
            )
            if not admin_exists:
                raise HTTPException(status_code=404, detail="Administrador asignado no encontrado")
        old_assignee = gestion.asignado_admin_id_usuario
        if old_assignee != new_assignee:
            gestion.asignado_admin_id_usuario = new_assignee
            changed["asignado_admin_id_usuario"] = {"old": old_assignee, "new": new_assignee}

    # Si el admin respondio, el medico debe ver ese mensaje como no leido.
    if respuesta_nueva:
        m.leido_medico = False
        changed["leido_medico"] = {"old": True, "new": False}

    if hasattr(Mensaje, "actualizado_en"):
        m.actualizado_en = datetime.utcnow()

    if changed:
        _append_event(
            db,
            id_mensaje=id_mensaje,
            actor_id_usuario=actor_id,
            accion="ACTUALIZADO_ADMIN",
            detalle={"changes": changed},
        )

    db.commit()
    return {"ok": True}

