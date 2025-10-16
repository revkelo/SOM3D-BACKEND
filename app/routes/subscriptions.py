from datetime import datetime
from dateutil.relativedelta import relativedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ..db import get_db
from ..core.security import get_current_user
from ..models import Plan, Suscripcion, Medico, Pago
from ..schemas import StartSubscriptionIn, CheckoutOut, SubscriptionOut, SubscriptionUpdateIn, PaymentOut
from ..core.config import EPAYCO_PUBLIC_KEY, EPAYCO_TEST, BASE_URL

router = APIRouter()


def _build_onpage_html(amount: str, name: str, description: str, invoice: str, extra1: str):
    response_url = f"{BASE_URL}/epayco/response?ngrok-skip-browser-warning=1"
    confirmation_url = f"{BASE_URL}/epayco/confirmation"
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

    # Resolver titular (médico del usuario autenticado)
    medico = db.query(Medico).filter(Medico.id_usuario == user.id_usuario).first()
    if not medico and getattr(user, "rol", None) == "MEDICO":
        medico = Medico(id_usuario=user.id_usuario)
        db.add(medico)
        db.commit()
        db.refresh(medico)

    medico_id = medico.id_medico if medico else None
    if medico_id is None:
        raise HTTPException(status_code=400, detail="No se pudo resolver el medico titular.")

    # Evitar múltiples suscripciones ACTIVA para el mismo titular (médico)
    active_q = db.query(Suscripcion).filter(
        Suscripcion.estado == "ACTIVA",
        Suscripcion.id_medico == medico_id,
    )
    if active_q.first():
        raise HTTPException(status_code=409, detail="Ya existe una suscripcion ACTIVA para este titular. Debe cancelarla o esperar a su expiracion.")

    # Reutilizar una PAUSADA del mismo titular y plan si existe
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
            estado="PAUSADA",  # Se activará en /epayco/confirmation (pago aprobado)
        )
        db.add(sus)
        db.commit()
        db.refresh(sus)

    invoice = f"3DVinciStudio-{sus.id_suscripcion}"
    amount = f"{float(plan.precio):.2f}"
    name = f"Plan {plan.nombre}"
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
    return {"suscripcion_id": sus.id_suscripcion, "checkout": checkout, "onpage_html": html}


@router.get("/subscriptions/mine")
def my_subscription(db: Session = Depends(get_db), user=Depends(get_current_user)):
    # Resolver titular (médico del usuario)
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


# --------------------
# Admin endpoints
# --------------------

def _ensure_admin(user):
    if getattr(user, "rol", None) != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")


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
    payload: SubscriptionUpdateIn,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    _ensure_admin(user)
    sus = db.query(Suscripcion).filter(Suscripcion.id_suscripcion == suscripcion_id).first()
    if not sus:
        raise HTTPException(status_code=404, detail="Suscripcion no encontrada")

    # Si se va a activar, verificar que no exista otra ACTIVA para el mismo médico
    if payload.estado == "ACTIVA":
        conflict_q = db.query(Suscripcion).filter(
            Suscripcion.estado == "ACTIVA",
            Suscripcion.id_suscripcion != sus.id_suscripcion,
        )
        if sus.id_medico is not None:
            conflict_q = conflict_q.filter(Suscripcion.id_medico == sus.id_medico)
        if conflict_q.first():
            raise HTTPException(status_code=409, detail="Ya existe otra suscripcion ACTIVA para este titular")

    sus.estado = payload.estado
    db.commit()
    db.refresh(sus)
    return sus


@router.delete("/subscriptions/{suscripcion_id}", status_code=204)
def delete_subscription(suscripcion_id: int, db: Session = Depends(get_db), user=Depends(get_current_user)):
    _ensure_admin(user)
    sus = db.query(Suscripcion).filter(Suscripcion.id_suscripcion == suscripcion_id).first()
    if not sus:
        raise HTTPException(status_code=404, detail="Suscripcion no encontrada")

    # Borrar pagos primero, luego la suscripción
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

