from datetime import datetime
import os
from dateutil.relativedelta import relativedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import Plan, Suscripcion, Medico, Pago, Hospital
from ..schemas import (
    StartSubscriptionIn,
    CheckoutOut,
    SubscriptionOut,
    SubscriptionUpdateIn,
    SubscriptionAdminCreateIn,
    SubscriptionAdminUpdateIn,
    PaymentOut,
)
from ..core.config import EPAYCO_PUBLIC_KEY, EPAYCO_TEST, BASE_URL

router = APIRouter()


def _public_base_url() -> str:
    raw = (os.getenv("NGROK_URL") or BASE_URL or "").strip().rstrip("/")
    if raw and not raw.lower().startswith(("http://", "https://")):
        raw = f"https://{raw}"
    return raw


def _unique_invoice(prefix: str, sus_id: int) -> str:
    from datetime import datetime
    import secrets
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    rnd = secrets.token_hex(2)
    return f"{prefix}-S{sus_id}-{ts}-{rnd}"

def _build_onpage_html(amount: str, name: str, description: str, invoice: str, extra1: str):
    base_url = _public_base_url()
    response_url = f"{base_url}/epayco/response?ngrok-skip-browser-warning=1"
    confirmation_url = f"{base_url}/epayco/confirmation"
    return f"""
<script src="https://s3-us-west-2.amazonaws.com/epayco/v1.0/checkoutEpayco.js"
  class="epayco-button"
  data-epayco-key="{EPAYCO_PUBLIC_KEY}"
  data-epayco-amount="{amount}"
  data-epayco-name="{name}"
  data-epayco-description="{description}"
  data-epayco-currency="cop"
  data-epayco-test="{'true' if EPAYCO_TEST else 'false'}"
  data-epayco-response="{response_url}"
  data-epayco-confirmation="{confirmation_url}"
  data-epayco-invoice="{invoice}"
  data-epayco-extra1="{extra1}">
</script>""".strip()


@router.post("/subscriptions/start", response_model=CheckoutOut)
def start_subscription(payload: StartSubscriptionIn, db: Session = Depends(get_db), user=Depends(get_current_user)):
    plan = db.query(Plan).filter(Plan.id_plan == payload.plan_id).first()
    if not plan:
        raise HTTPException(status_code=404, detail="Plan no encontrado")

    medico = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not medico and getattr(user, "rol", None) == "MEDICO":
        medico = Medico(id_usuario=user.id_usuario)
        db.add(medico)
        db.commit()
        db.refresh(medico)

    medico_id = medico.id_medico if medico else None
    if medico_id is None:
        raise HTTPException(status_code=400, detail="No se pudo resolver el medico titular.")

    active_q = db.query(Suscripcion).filter(
        Suscripcion.estado == "ACTIVA",
        Suscripcion.id_medico == medico_id,
    )
    if active_q.first():
        raise HTTPException(status_code=409, detail="Ya existe una suscripcion ACTIVA para este titular. Debe cancelarla o esperar a su expiracion.")

    paused_q = db.query(Suscripcion).filter(
        Suscripcion.estado == "PAUSADA",
        Suscripcion.id_plan == plan.id_plan,
        Suscripcion.id_medico == medico_id,
    )
    sus = paused_q.order_by(Suscripcion.creado_en.desc()).first()

    if not sus:
        sus = Suscripcion(
            id_medico=medico_id,
            id_hospital=None,
            id_plan=plan.id_plan,
            estado="PAUSADA",                                                       
        )
        db.add(sus)
        db.commit()
        db.refresh(sus)

    invoice = _unique_invoice("3DVinciStudio", sus.id_suscripcion)
    amount = f"{float(plan.precio):.2f}"
    name = f"Plan {plan.nombre}"
    description = f"Suscripcion {plan.periodo} ({plan.duracion_meses} meses)"

    base_url = _public_base_url()
    checkout = {
        "key": EPAYCO_PUBLIC_KEY,
        "amount": amount,
        "name": name,
        "description": description,
        "currency": "cop",
        "test": EPAYCO_TEST,
        "response": f"{base_url}/epayco/response",
        "confirmation": f"{base_url}/epayco/confirmation",
        "invoice": invoice,
        "extra1": str(sus.id_suscripcion),
    }

    html = _build_onpage_html(amount, name, description, invoice, str(sus.id_suscripcion))
    return {"suscripcion_id": sus.id_suscripcion, "checkout": checkout, "onpage_html": html}


@router.get("/subscriptions/mine")
def my_subscription(db: Session = Depends(get_db), user=Depends(get_current_user)):
    medico = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    medico_id = medico.id_medico if medico else None
    if medico_id is None:
        return {"has": False, "active": False}

    base_q = db.query(Suscripcion).filter(Suscripcion.id_medico == medico_id)

    act = base_q.filter(Suscripcion.estado == "ACTIVA").order_by(Suscripcion.creado_en.desc()).first()
    pau = base_q.filter(Suscripcion.estado == "PAUSADA").order_by(Suscripcion.creado_en.desc()).first()
    sus = act or pau

    if not sus:
        return {"has": False, "active": False}

    plan = db.query(Plan).filter(Plan.id_plan == sus.id_plan).first()
    plan_out = {
        "id_plan": plan.id_plan,
        "nombre": plan.nombre,
        "precio": float(plan.precio),
        "periodo": plan.periodo,
        "duracion_meses": plan.duracion_meses,
    } if plan else None

    return {
        "has": True,
        "active": sus.estado == "ACTIVA",
        "estado": sus.estado,
        "suscripcion_id": sus.id_suscripcion,
        "plan": plan_out,
        "can_resume": sus.estado == "PAUSADA",
        "can_start": sus.estado != "ACTIVA",
    }


@router.get("/subscriptions/mine/status")
def my_subscription_status(db: Session = Depends(get_db), user=Depends(get_current_user)):
    """Detalle de la suscripción del médico autenticado: plan, expiración y último pago.
    Devuelve can_pay=True si no está ACTIVA o está expirada.
    """
    from datetime import datetime

    medico = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    medico_id = medico.id_medico if medico else None
    if medico_id is None:
        return {"has": False}

    base_q = db.query(Suscripcion).filter(Suscripcion.id_medico == medico_id)
    act = base_q.filter(Suscripcion.estado == "ACTIVA").order_by(Suscripcion.creado_en.desc()).first()
    pau = base_q.filter(Suscripcion.estado == "PAUSADA").order_by(Suscripcion.creado_en.desc()).first()
    sus = act or pau
    if not sus:
        return {"has": False}

    plan = db.query(Plan).filter(Plan.id_plan == sus.id_plan).first()
    last_pago = db.query(Pago).filter(Pago.id_suscripcion == sus.id_suscripcion).order_by(Pago.fecha_pago.desc()).first()

    now = datetime.utcnow()
    exp = getattr(sus, "fecha_expiracion", None)
    is_active = sus.estado == "ACTIVA"
    is_expired = bool(exp and exp <= now)
    can_pay = (not is_active) or is_expired

    return {
        "has": True,
        "estado": sus.estado,
        "suscripcion_id": sus.id_suscripcion,
        "plan": {
            "id_plan": plan.id_plan if plan else None,
            "nombre": plan.nombre if plan else None,
            "precio": float(plan.precio) if plan else None,
            "periodo": plan.periodo if plan else None,
            "duracion_meses": plan.duracion_meses if plan else None,
        } if plan else None,
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



def _ensure_admin(user):
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")


def _validate_subscription_owner(id_medico: int | None, id_hospital: int | None):
    if (id_medico is None and id_hospital is None) or (id_medico is not None and id_hospital is not None):
        raise HTTPException(status_code=400, detail="Debe existir exactamente un pagador: id_medico o id_hospital")


def _validate_subscription_refs(db: Session, id_medico: int | None, id_hospital: int | None, id_plan: int):
    if id_medico is not None and not db.query(Medico).filter(Medico.id_medico == id_medico).first():
        raise HTTPException(status_code=404, detail="Medico no encontrado")
    if id_hospital is not None and not db.query(Hospital).filter(Hospital.id_hospital == id_hospital).first():
        raise HTTPException(status_code=404, detail="Hospital no encontrado")
    if not db.query(Plan).filter(Plan.id_plan == id_plan).first():
        raise HTTPException(status_code=404, detail="Plan no encontrado")


@router.get("/subscriptions", response_model=list[SubscriptionOut])
def list_subscriptions(
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
    id_medico: int | None = Query(None),
    estado: str | None = Query(None, pattern="^(ACTIVA|PAUSADA)$"),
    plan_id: int | None = Query(None),
):
    _ensure_admin(user)

    q = db.query(Suscripcion)
    if id_medico is not None:
        q = q.filter(Suscripcion.id_medico == id_medico)
    if estado is not None:
        q = q.filter(Suscripcion.estado == estado)
    if plan_id is not None:
        q = q.filter(Suscripcion.id_plan == plan_id)
    return q.order_by(Suscripcion.creado_en.desc()).all()


@router.post("/subscriptions", response_model=SubscriptionOut, status_code=201)
def create_subscription_admin(
    payload: SubscriptionAdminCreateIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    _ensure_admin(user)
    _validate_subscription_owner(payload.id_medico, payload.id_hospital)
    _validate_subscription_refs(db, payload.id_medico, payload.id_hospital, payload.id_plan)

    sus = Suscripcion(
        id_medico=payload.id_medico,
        id_hospital=payload.id_hospital,
        id_plan=payload.id_plan,
        estado=payload.estado,
        fecha_inicio=payload.fecha_inicio or datetime.utcnow(),
        fecha_expiracion=payload.fecha_expiracion,
    )
    db.add(sus)
    db.commit()
    db.refresh(sus)
    return sus


@router.get("/subscriptions/{suscripcion_id}", response_model=SubscriptionOut)
def get_subscription(suscripcion_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    sus = db.query(Suscripcion).filter(Suscripcion.id_suscripcion == suscripcion_id).first()
    if not sus:
        raise HTTPException(status_code=404, detail="Suscripcion no encontrada")
    return sus


@router.patch("/subscriptions/{suscripcion_id}", response_model=SubscriptionOut)
def update_subscription(
    suscripcion_id: int,
    payload: SubscriptionAdminUpdateIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    _ensure_admin(user)
    sus = db.query(Suscripcion).filter(Suscripcion.id_suscripcion == suscripcion_id).first()
    if not sus:
        raise HTTPException(status_code=404, detail="Suscripcion no encontrada")

    data = payload.model_dump(exclude_unset=True)
    new_id_medico = data["id_medico"] if "id_medico" in data else sus.id_medico
    new_id_hospital = data["id_hospital"] if "id_hospital" in data else sus.id_hospital
    new_id_plan = data["id_plan"] if "id_plan" in data else sus.id_plan
    new_estado = data["estado"] if "estado" in data else sus.estado

    _validate_subscription_owner(new_id_medico, new_id_hospital)
    _validate_subscription_refs(db, new_id_medico, new_id_hospital, new_id_plan)

    if new_estado == "ACTIVA":
        conflict_q = db.query(Suscripcion).filter(
            Suscripcion.estado == "ACTIVA",
            Suscripcion.id_suscripcion != sus.id_suscripcion,
        )
        if new_id_medico is not None:
            conflict_q = conflict_q.filter(Suscripcion.id_medico == new_id_medico)
        if new_id_hospital is not None:
            conflict_q = conflict_q.filter(Suscripcion.id_hospital == new_id_hospital)
        if conflict_q.first():
            raise HTTPException(status_code=409, detail="Ya existe otra suscripcion ACTIVA para este titular")

    for k, v in data.items():
        setattr(sus, k, v)
    db.commit()
    db.refresh(sus)
    return sus


@router.delete("/subscriptions/{suscripcion_id}", status_code=204)
def delete_subscription(suscripcion_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    sus = db.query(Suscripcion).filter(Suscripcion.id_suscripcion == suscripcion_id).first()
    if not sus:
        raise HTTPException(status_code=404, detail="Suscripcion no encontrada")

    db.query(Pago).filter(Pago.id_suscripcion == suscripcion_id).delete(synchronize_session=False)
    db.query(Suscripcion).filter(Suscripcion.id_suscripcion == suscripcion_id).delete(synchronize_session=False)
    db.commit()
    return


@router.get("/subscriptions/{suscripcion_id}/payments", response_model=list[PaymentOut])
def list_subscription_payments(suscripcion_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    sus = db.query(Suscripcion).filter(Suscripcion.id_suscripcion == suscripcion_id).first()
    if not sus:
        raise HTTPException(status_code=404, detail="Suscripcion no encontrada")
    pagos = db.query(Pago).filter(Pago.id_suscripcion == suscripcion_id).order_by(Pago.fecha_pago.desc()).all()
    return pagos

