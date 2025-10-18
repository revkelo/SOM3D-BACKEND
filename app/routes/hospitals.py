from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import Hospital, Medico, Plan, Suscripcion, Pago
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


@router.get("/hospitals/by-code/{codigo}")
def hospital_by_code(codigo: str, db: Session = Depends(get_db)):
    h = db.query(Hospital).filter(Hospital.codigo == codigo).first()
    if not h or h.estado != "ACTIVO":
        raise HTTPException(status_code=404, detail="Hospital no encontrado o inactivo")
    # Determinar si tiene una suscripcion ACTIVA directa
    has_active = db.query(Suscripcion).filter(
        Suscripcion.id_hospital == h.id_hospital,
        Suscripcion.estado == "ACTIVA",
    ).first() is not None
    return {
        "id_hospital": h.id_hospital,
        "nombre": h.nombre,
        "ciudad": h.ciudad,
        "estado": h.estado,
        "has_active_subscription": has_active,
    }


@router.get("/hospitals/status/{codigo}")
def hospital_status(codigo: str, db: Session = Depends(get_db)):
    """Devuelve estado de suscripción del hospital por código: plan, expiración y último pago.
    - can_pay: True si no tiene ACTIVA o si está expirada.
    """
    from datetime import datetime, timezone

    h = db.query(Hospital).filter(Hospital.codigo == codigo).first()
    if not h or h.estado != "ACTIVO":
        raise HTTPException(status_code=404, detail="Hospital no encontrado o inactivo")

    # Preferir suscripción ACTIVA, si no, la última PAUSADA
    sus_act = db.query(Suscripcion).filter(
        Suscripcion.id_hospital == h.id_hospital,
        Suscripcion.estado == "ACTIVA",
    ).order_by(Suscripcion.creado_en.desc()).first()

    sus_pau = db.query(Suscripcion).filter(
        Suscripcion.id_hospital == h.id_hospital,
        Suscripcion.estado == "PAUSADA",
    ).order_by(Suscripcion.creado_en.desc()).first()

    sus = sus_act or sus_pau
    if not sus:
        return {
            "hospital": {"id_hospital": h.id_hospital, "nombre": h.nombre, "ciudad": h.ciudad},
            "has_subscription": False,
            "estado": None,
            "plan": None,
            "fecha_expiracion": None,
            "last_payment": None,
            "can_pay": True,  # No tiene suscripción: permitir crear/pagar
        }

    plan = db.query(Plan).filter(Plan.id_plan == sus.id_plan).first()
    last_pago = db.query(Pago).filter(Pago.id_suscripcion == sus.id_suscripcion).order_by(Pago.fecha_pago.desc()).first()

    # Determinar si se puede pagar ahora: si no está ACTIVA o si está expirada
    now = datetime.utcnow()
    exp = getattr(sus, "fecha_expiracion", None)
    exp_dt = exp if isinstance(exp, datetime) else None
    is_active = (sus.estado == "ACTIVA")
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
    """Estado de suscripción del hospital vinculado al médico autenticado.
    Devuelve { has_hospital, active, hospital }.
    """
    if getattr(user, "rol", None) != "MEDICO":
        return {"has_hospital": False}

    m = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not m or not m.id_hospital:
        return {"has_hospital": False}

    h = db.query(Hospital).filter(Hospital.id_hospital == m.id_hospital).first()
    if not h:
        return {"has_hospital": False}

    # ¿Tiene ACTIVA?
    sus_act = db.query(Suscripcion).filter(
        Suscripcion.id_hospital == h.id_hospital,
        Suscripcion.estado == "ACTIVA",
    ).first()

    return {
        "has_hospital": True,
        "active": bool(sus_act),
        "hospital": {"id_hospital": h.id_hospital, "nombre": h.nombre}
    }


@router.post("/hospitals/link-by-code")
def link_hospital_by_code(payload: HospitalLinkByCodeIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    # Solo medicos pueden vincularse
    if getattr(user, "rol", None) != "MEDICO":
        raise HTTPException(status_code=403, detail="Solo medicos pueden vincularse a un hospital")

    h = db.query(Hospital).filter(Hospital.codigo == payload.codigo).first()
    if not h or h.estado != "ACTIVO":
        raise HTTPException(status_code=404, detail="Codigo de hospital invalido o inactivo")

    m = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not m:
        m = Medico(id_usuario=user.id_usuario, id_hospital=h.id_hospital, referenciado=False, estado="ACTIVO")
        db.add(m)
    else:
        m.id_hospital = h.id_hospital

    # A solicitud: el medico no debe pagar si se vincula a un hospital.
    # Para simplificar el flujo del frontend (que redirige si has_active_subscription es True),
    # devolvemos has_active_subscription=True siempre que el codigo sea valido y la vinculacion sea exitosa.
    has_active = True
    try:
        # Opcional: activar el usuario inmediatamente tras la vinculacion.
        user.activo = True
    except Exception:
        pass
    db.commit()
    return {"ok": True, "id_hospital": h.id_hospital, "nombre": h.nombre, "has_active_subscription": has_active}


@router.post("/hospitals/start-subscription")
def hospital_start_subscription(payload: HospitalStartSubscriptionIn, db: Session = Depends(get_db)):
    h = db.query(Hospital).filter(Hospital.codigo == payload.codigo).first()
    if not h or h.estado != "ACTIVO":
        raise HTTPException(status_code=404, detail="Hospital no encontrado o inactivo")

    plan = None
    sus = None

    if payload.plan_id:
        plan = db.query(Plan).filter(Plan.id_plan == payload.plan_id).first()
        if not plan:
            raise HTTPException(status_code=404, detail="Plan no encontrado")
        # Reutilizar suscripcion PAUSADA del mismo hospital y plan si existe
        sus = db.query(Suscripcion).filter(
            Suscripcion.id_hospital == h.id_hospital,
            Suscripcion.id_plan == plan.id_plan,
            Suscripcion.estado == "PAUSADA",
        ).order_by(Suscripcion.creado_en.desc()).first()
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
        # Sin plan_id: tomar la última suscripción PAUSADA del hospital
        sus = db.query(Suscripcion).filter(
            Suscripcion.id_hospital == h.id_hospital,
            Suscripcion.estado == "PAUSADA",
        ).order_by(Suscripcion.creado_en.desc()).first()
        if not sus:
            # Si ya está activo, informar
            active = db.query(Suscripcion).filter(
                Suscripcion.id_hospital == h.id_hospital,
                Suscripcion.estado == "ACTIVA",
            ).first()
            if active:
                raise HTTPException(status_code=409, detail="El hospital ya tiene una suscripcion ACTIVA")
            raise HTTPException(status_code=404, detail="El hospital no tiene una suscripcion pendiente (PAUSADA)")
        plan = db.query(Plan).filter(Plan.id_plan == sus.id_plan).first()

    # ePayco requiere invoice unico por intento
    from datetime import datetime
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
    return {"suscripcion_id": sus.id_suscripcion, "checkout": checkout, "onpage_html": html, "hospital": {"id_hospital": h.id_hospital, "nombre": h.nombre}}
