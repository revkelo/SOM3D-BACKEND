from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import Hospital, HospitalCode, Medico, Plan, Suscripcion, Pago
from ..schemas import HospitalLinkByCodeIn, HospitalStartSubscriptionIn
from ..core.config import EPAYCO_PUBLIC_KEY, EPAYCO_TEST, BASE_URL


router = APIRouter()


def _build_onpage_html(amount: str, name: str, description: str, invoice: str, extra1: str):
    response_url = f"{BASE_URL}/epayco/response?ngrok-skip-browser-warning=1"
    confirmation_url = f"{BASE_URL}/epayco/confirmation"
    return f"""
<script src=\"https://s3-us-west-2.amazonaws.com/epayco/v1.0/checkoutEpayco.js\"
  class=\"epayco-button\"
  data-epayco-key=\"{EPAYCO_PUBLIC_KEY}\"
  data-epayco-amount=\"{amount}\"
  data-epayco-name=\"{name}\"
  data-epayco-description=\"{description}\"
  data-epayco-currency=\"cop\"
  data-epayco-test=\"{'true' if EPAYCO_TEST else 'false'}\"
  data-epayco-response=\"{response_url}\"
  data-epayco-confirmation=\"{confirmation_url}\"
  data-epayco-invoice=\"{invoice}\"
  data-epayco-extra1=\"{extra1}\">\n</script>""".strip()


def _resolve_hospital_by_code(db: Session, codigo: str, require_code_usable: bool = False):
    now = datetime.utcnow()
    hc = (
        db.query(HospitalCode)
        .filter(HospitalCode.codigo == codigo)
        .order_by(HospitalCode.created_at.desc())
        .first()
    )

    if hc:
        h = db.query(Hospital).filter(Hospital.id_hospital == hc.id_hospital).first()
        if not h or h.estado != "ACTIVO":
            raise HTTPException(status_code=404, detail="Hospital no encontrado o inactivo")
        if require_code_usable:
            if hc.revoked_at is not None:
                raise HTTPException(status_code=409, detail="Codigo revocado")
            if hc.used_at is not None:
                raise HTTPException(status_code=409, detail="Codigo ya usado")
            if hc.expires_at and hc.expires_at <= now:
                raise HTTPException(status_code=409, detail="Codigo expirado")
        return h, hc

    # Compatibilidad con datos legacy: Hospital.codigo sin registro en HospitalCode
    h = db.query(Hospital).filter(Hospital.codigo == codigo).first()
    if not h or h.estado != "ACTIVO":
        raise HTTPException(status_code=404, detail="Hospital no encontrado o inactivo")

    if require_code_usable:
        # El primer uso de codigo legacy se registra y queda consumido.
        hc = HospitalCode(
            id_hospital=h.id_hospital,
            codigo=codigo,
            expires_at=now,
            used_at=now,
        )
        db.add(hc)
        db.flush()
    return h, hc


@router.get("/hospitals/by-code/{codigo}")
def hospital_by_code(codigo: str, db: Session = Depends(get_db)):
    h, hc = _resolve_hospital_by_code(db, codigo, require_code_usable=False)
    # Determinar si tiene una suscripcion ACTIVA directa
    has_active = (
        db.query(Suscripcion)
        .filter(
            Suscripcion.id_hospital == h.id_hospital,
            Suscripcion.estado == "ACTIVA",
        )
        .first()
        is not None
    )
    code_info = None
    if hc:
        code_info = {
            "codigo": hc.codigo,
            "expires_at": hc.expires_at.isoformat() if hc.expires_at else None,
            "used_at": hc.used_at.isoformat() if hc.used_at else None,
            "revoked_at": hc.revoked_at.isoformat() if hc.revoked_at else None,
        }
    return {
        "id_hospital": h.id_hospital,
        "nombre": h.nombre,
        "ciudad": h.ciudad,
        "estado": h.estado,
        "has_active_subscription": has_active,
        "code": code_info,
    }


@router.get("/hospitals/status/{codigo}")
def hospital_status(codigo: str, db: Session = Depends(get_db)):
    """Devuelve estado de suscripcion del hospital por codigo: plan, expiracion y ultimo pago."""
    h, _ = _resolve_hospital_by_code(db, codigo, require_code_usable=False)

    # Preferir suscripcion ACTIVA, si no, la ultima PAUSADA
    sus_act = (
        db.query(Suscripcion)
        .filter(
            Suscripcion.id_hospital == h.id_hospital,
            Suscripcion.estado == "ACTIVA",
        )
        .order_by(Suscripcion.creado_en.desc())
        .first()
    )

    sus_pau = (
        db.query(Suscripcion)
        .filter(
            Suscripcion.id_hospital == h.id_hospital,
            Suscripcion.estado == "PAUSADA",
        )
        .order_by(Suscripcion.creado_en.desc())
        .first()
    )

    sus = sus_act or sus_pau
    if not sus:
        return {
            "hospital": {"id_hospital": h.id_hospital, "nombre": h.nombre, "ciudad": h.ciudad},
            "has_subscription": False,
            "estado": None,
            "plan": None,
            "fecha_expiracion": None,
            "last_payment": None,
            "can_pay": True,
        }

    plan = db.query(Plan).filter(Plan.id_plan == sus.id_plan).first()
    last_pago = (
        db.query(Pago)
        .filter(Pago.id_suscripcion == sus.id_suscripcion)
        .order_by(Pago.fecha_pago.desc())
        .first()
    )

    now = datetime.utcnow()
    exp = getattr(sus, "fecha_expiracion", None)
    exp_dt = exp if isinstance(exp, datetime) else None
    is_active = sus.estado == "ACTIVA"
    is_expired = bool(exp_dt and exp_dt <= now)
    can_pay = (not is_active) or is_expired

    return {
        "hospital": {"id_hospital": h.id_hospital, "nombre": h.nombre, "ciudad": h.ciudad},
        "has_subscription": True,
        "suscripcion_id": sus.id_suscripcion,
        "estado": sus.estado,
        "plan": {
            "id_plan": plan.id_plan if plan else None,
            "nombre": plan.nombre if plan else None,
            "precio": float(plan.precio) if plan else None,
            "periodo": plan.periodo if plan else None,
            "duracion_meses": plan.duracion_meses if plan else None,
        },
        "fecha_expiracion": sus.fecha_expiracion.isoformat() if sus.fecha_expiracion else None,
        "last_payment": (
            {
                "fecha_pago": last_pago.fecha_pago.isoformat() if last_pago and last_pago.fecha_pago else None,
                "monto": float(last_pago.monto) if last_pago else None,
                "referencia": last_pago.referencia_epayco if last_pago else None,
            }
            if last_pago
            else None
        ),
        "can_pay": can_pay,
    }


@router.get("/hospitals/mine-status")
def my_hospital_status(db: Session = Depends(get_db), user=Depends(get_current_user)):
    """Estado de suscripcion del hospital vinculado al medico autenticado."""
    if getattr(user, "rol", None) != "MEDICO":
        return {"has_hospital": False}

    m = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not m or not m.id_hospital:
        return {"has_hospital": False}

    h = db.query(Hospital).filter(Hospital.id_hospital == m.id_hospital).first()
    if not h:
        return {"has_hospital": False}

    sus_act = (
        db.query(Suscripcion)
        .filter(
            Suscripcion.id_hospital == h.id_hospital,
            Suscripcion.estado == "ACTIVA",
        )
        .first()
    )

    return {
        "has_hospital": True,
        "active": bool(sus_act),
        "hospital": {"id_hospital": h.id_hospital, "nombre": h.nombre},
    }


@router.post("/hospitals/link-by-code")
def link_hospital_by_code(payload: HospitalLinkByCodeIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    # Solo medicos pueden vincularse
    if getattr(user, "rol", None) != "MEDICO":
        raise HTTPException(status_code=403, detail="Solo medicos pueden vincularse a un hospital")

    h, hc = _resolve_hospital_by_code(db, payload.codigo, require_code_usable=True)

    m = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not m:
        m = Medico(id_usuario=user.id_usuario, id_hospital=h.id_hospital, referenciado=False, estado="ACTIVO")
        db.add(m)
        db.flush()
    else:
        m.id_hospital = h.id_hospital

    # Consumir codigo para asegurar uso unico.
    if hc and (hc.used_at is None or hc.usado_por_id_medico is None):
        if hc.used_at is None:
            hc.used_at = datetime.utcnow()
        hc.usado_por_id_medico = m.id_medico

    try:
        user.activo = True
    except Exception:
        pass

    has_active = (
        db.query(Suscripcion)
        .filter(Suscripcion.id_hospital == h.id_hospital, Suscripcion.estado == "ACTIVA")
        .first()
        is not None
    )
    db.commit()
    return {
        "ok": True,
        "id_hospital": h.id_hospital,
        "nombre": h.nombre,
        "has_active_subscription": has_active,
    }


@router.post("/hospitals/start-subscription")
def hospital_start_subscription(payload: HospitalStartSubscriptionIn, db: Session = Depends(get_db)):
    h, _ = _resolve_hospital_by_code(db, payload.codigo, require_code_usable=False)

    plan = None
    sus = None

    if payload.plan_id:
        plan = db.query(Plan).filter(Plan.id_plan == payload.plan_id).first()
        if not plan:
            raise HTTPException(status_code=404, detail="Plan no encontrado")
        # Reutilizar suscripcion PAUSADA del mismo hospital y plan si existe
        sus = (
            db.query(Suscripcion)
            .filter(
                Suscripcion.id_hospital == h.id_hospital,
                Suscripcion.id_plan == plan.id_plan,
                Suscripcion.estado == "PAUSADA",
            )
            .order_by(Suscripcion.creado_en.desc())
            .first()
        )
        if not sus:
            sus = Suscripcion(
                id_medico=None,
                id_hospital=h.id_hospital,
                id_plan=plan.id_plan,
                estado="PAUSADA",
            )
            db.add(sus)
            db.commit()
            db.refresh(sus)
    else:
        # Sin plan_id: tomar la ultima suscripcion PAUSADA del hospital
        sus = (
            db.query(Suscripcion)
            .filter(
                Suscripcion.id_hospital == h.id_hospital,
                Suscripcion.estado == "PAUSADA",
            )
            .order_by(Suscripcion.creado_en.desc())
            .first()
        )
        if not sus:
            active = (
                db.query(Suscripcion)
                .filter(
                    Suscripcion.id_hospital == h.id_hospital,
                    Suscripcion.estado == "ACTIVA",
                )
                .first()
            )
            if active:
                raise HTTPException(status_code=409, detail="El hospital ya tiene una suscripcion ACTIVA")
            raise HTTPException(status_code=404, detail="El hospital no tiene una suscripcion pendiente (PAUSADA)")
        plan = db.query(Plan).filter(Plan.id_plan == sus.id_plan).first()

    # ePayco requiere invoice unico por intento
    import secrets

    invoice = f"3DVinciStudio-H{h.id_hospital}-S{sus.id_suscripcion}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(2)}"
    amount = f"{float(plan.precio):.2f}"
    name = f"Plan {plan.nombre} (Hospital)"
    description = f"Suscripcion {plan.periodo} ({plan.duracion_meses} meses)"

    checkout = {
        "key": EPAYCO_PUBLIC_KEY,
        "amount": amount,
        "name": name,
        "description": description,
        "currency": "cop",
        "test": EPAYCO_TEST,
        "response": f"{BASE_URL}/epayco/response",
        "confirmation": f"{BASE_URL}/epayco/confirmation",
        "invoice": invoice,
        "extra1": str(sus.id_suscripcion),
    }
    html = _build_onpage_html(amount, name, description, invoice, str(sus.id_suscripcion))
    return {
        "suscripcion_id": sus.id_suscripcion,
        "checkout": checkout,
        "onpage_html": html,
        "hospital": {"id_hospital": h.id_hospital, "nombre": h.nombre},
    }
