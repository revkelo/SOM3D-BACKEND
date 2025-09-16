from datetime import datetime
from dateutil.relativedelta import relativedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..auth import get_current_user
from ..models import Plan, Suscripcion, Medico
from ..schemas import StartSubscriptionIn, CheckoutOut
from ..config import EPAYCO_PUBLIC_KEY, EPAYCO_TEST, BASE_URL

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
    hospital_id = payload.hospital_id

    # Para rol MÉDICO, ignorar hospital_id; para ADMIN exigir hospital_id
    if getattr(user, "rol", None) == "MEDICO":
        hospital_id = None
    if getattr(user, "rol", None) == "ADMINISTRADOR" and hospital_id is None:
        raise HTTPException(status_code=400, detail="Para ADMINISTRADOR debe enviar hospital_id.")

    # Regla: exactamente uno debe estar presente (XOR)
    if (medico_id is None and hospital_id is None) or (medico_id is not None and hospital_id is not None):
        raise HTTPException(status_code=400, detail="Debe especificar exactamente un pagador: id_medico XOR id_hospital.")

    # Evitar múltiples suscripciones ACTIVA para el mismo titular
    active_q = db.query(Suscripcion).filter(Suscripcion.estado == "ACTIVA")
    if medico_id is not None:
        active_q = active_q.filter(Suscripcion.id_medico == medico_id)
    else:
        active_q = active_q.filter(Suscripcion.id_hospital == hospital_id)
    if active_q.first():
        raise HTTPException(status_code=409, detail="Ya existe una suscripción ACTIVA para este titular. Debe cancelarla o esperar a su expiración.")

    # Reutilizar una PAUSADA del mismo titular y plan si existe
    paused_q = db.query(Suscripcion).filter(
        Suscripcion.estado == "PAUSADA",
        Suscripcion.id_plan == plan.id_plan,
    )
    if medico_id is not None:
        paused_q = paused_q.filter(Suscripcion.id_medico == medico_id)
    else:
        paused_q = paused_q.filter(Suscripcion.id_hospital == hospital_id)
    sus = paused_q.order_by(Suscripcion.creado_en.desc()).first()

    if not sus:
        sus = Suscripcion(
            id_medico=medico_id,
            id_hospital=hospital_id,
            id_plan=plan.id_plan,
            estado="PAUSADA",  # Se activará en /epayco/confirmation (pago aprobado)
        )
        db.add(sus)
        db.commit()
        db.refresh(sus)

    invoice = f"SOM3D-{sus.id_suscripcion}"
    amount = f"{float(plan.precio):.2f}"
    name = f"Plan {plan.nombre}"
    description = f"Suscripción {plan.periodo} ({plan.duracion_meses} meses)"

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
    hospital_id = None

    base_q = db.query(Suscripcion)
    if medico_id is not None:
        base_q = base_q.filter(Suscripcion.id_medico == medico_id)
    else:
        base_q = base_q.filter(Suscripcion.id_hospital == hospital_id)

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

