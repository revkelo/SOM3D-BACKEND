import hashlib, json
from datetime import datetime
import html
from typing import Dict, Any, Optional
import hmac

import httpx
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..db import get_db
from ..models import Suscripcion, Pago, Plan, Medico, Hospital, Usuario, PaymentWebhookEvent
from ..core.config import P_CUST_ID_CLIENTE, P_KEY
from ..services.mailer import send_email
from ..services.mail_templates import template_payment_confirm

router = APIRouter(prefix="/epayco", tags=["epayco"])

STATUS_MAP = {"1": "APROBADO", "2": "RECHAZADO", "3": "PENDIENTE", "4": "FALLIDO"}


def _esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _safe_bool(getattr_value: Any, default: bool = False) -> bool:
    try:
        return bool(getattr_value)
    except Exception:
        return default


def _upsert_webhook_event(
    db: Session,
    *,
    ref_payco: str,
    transaction_id: Optional[str],
    estado: Optional[str],
    firma_valida: bool,
    payload: dict,
) -> PaymentWebhookEvent:
    ev = db.query(PaymentWebhookEvent).filter(PaymentWebhookEvent.ref_payco == ref_payco).first()
    if not ev:
        ev = PaymentWebhookEvent(ref_payco=ref_payco)
        db.add(ev)

    ev.transaction_id = transaction_id
    ev.estado = estado
    ev.firma_valida = bool(firma_valida)
    ev.payload_json = json.dumps(payload, ensure_ascii=False)
    ev.attempts = int(getattr(ev, "attempts", 0) or 0) + 1
    ev.last_error = None

    if not hasattr(ev, "processed"):
        pass

    db.flush()
    return ev


def _send_payment_confirmation_email(db: Session, sus: Suscripcion, plan: Optional[Plan], ref: str, amount: str, when: datetime) -> None:
    try:
        if sus.id_medico is not None:
            med = db.query(Medico).filter(Medico.id_medico == sus.id_medico).first()
            if not med:
                return
            usr = db.query(Usuario).filter(Usuario.id_usuario == med.id_usuario).first()
            if not usr or not usr.correo:
                return
            body_html = template_payment_confirm(
                nombre=(usr.nombre or "Usuario"),
                plan_nombre=(plan.nombre if plan else f"Plan {sus.id_plan}"),
                monto=str(amount),
                referencia=str(ref),
                fecha_iso=when.isoformat(timespec="seconds"),
            )
            send_email(str(usr.correo), "Pago confirmado - 3DVinci Health", body_html)
            return

        if sus.id_hospital is not None:
            hosp = db.query(Hospital).filter(Hospital.id_hospital == sus.id_hospital).first()
            if not hosp or not hosp.correo:
                return
            body_html = template_payment_confirm(
                nombre=(hosp.nombre or "Hospital"),
                plan_nombre=(plan.nombre if plan else f"Plan {sus.id_plan}"),
                monto=str(amount),
                referencia=str(ref),
                fecha_iso=when.isoformat(timespec="seconds"),
            )
            send_email(str(hosp.correo), "Pago confirmado - 3DVinci Health", body_html)
    except Exception:
        pass


def _page(title: str, body_html: str) -> HTMLResponse:
    safe_title = _esc(title)
    html_doc = f"""
    <!doctype html><html lang="es"><head>
    <meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>{safe_title}</title>
    <style>
      body{{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;padding:24px;max-width:960px;margin:auto}}
      .card{{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0}}
      table{{border-collapse:collapse;width:100%}}
      td{{border-bottom:1px solid #eee;padding:6px 8px;vertical-align:top}}
      pre{{background:#f6f8fa;padding:12px;border-radius:8px;overflow:auto}}
      .ok{{color:#0b7d62;font-weight:600}} .warn{{color:#b58900;font-weight:600}} .err{{color:#b00020;font-weight:600}}
      small{{opacity:.7}}
    </style></head><body>
    <h2>{safe_title}</h2>{body_html}</body></html>"""
    return HTMLResponse(html_doc)


@router.get("/response", response_class=HTMLResponse)
async def epayco_response(request: Request):
    params = dict(request.query_params)
    ref_payco = params.get("ref_payco")
    detail = "<p class='warn'>No recibí <code>ref_payco</code> en la URL.</p>"
    valid_json = {}

    if ref_payco:
        url = f"https://secure.epayco.co/validation/v1/reference/{ref_payco}"
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url)
            valid_json = r.json()

        if valid_json.get("success"):
            d = valid_json.get("data", {})
            cod = str(d.get("x_cod_response", ""))
            estado = STATUS_MAP.get(cod, "DESCONOCIDO")
            detail = f"""
            <p class="ok">Validación OK — estado: <b>{_esc(estado)}</b></p>
            <table>
              <tr><td>ref_payco</td><td><code>{_esc(ref_payco)}</code></td></tr>
              <tr><td>x_id_invoice</td><td>{_esc(d.get('x_id_invoice'))}</td></tr>
              <tr><td>x_response</td><td>{_esc(d.get('x_response'))}</td></tr>
              <tr><td>x_response_reason_text</td><td>{_esc(d.get('x_response_reason_text'))}</td></tr>
              <tr><td>x_transaction_id</td><td>{_esc(d.get('x_transaction_id'))}</td></tr>
              <tr><td>x_amount x_currency</td><td>{_esc(d.get('x_amount'))} {_esc(d.get('x_currency_code'))}</td></tr>
              <tr><td>x_transaction_date</td><td>{_esc(d.get('x_transaction_date'))}</td></tr>
            </table>
            """
        else:
            detail = f"<p class='err'>La validación respondió error para <code>{_esc(ref_payco)}</code>.</p>"

    body = f"""
    <div class="card">
      <h3>Respuesta de ePayco (cliente)</h3>
      <p>QueryString recibido:</p>
      <pre>{_esc(json.dumps(params, indent=2, ensure_ascii=False))}</pre>
      <hr/>
      <h4>Validación por <code>ref_payco</code></h4>
      {detail}
      <h4>Raw de validación</h4>
      <pre>{_esc(json.dumps(valid_json, indent=2, ensure_ascii=False))}</pre>
    </div>
    """
    return _page("Respuesta de Pago", body)


@router.get("/validate")
async def epayco_validate(ref_payco: str):
    url = f"https://secure.epayco.co/validation/v1/reference/{ref_payco}"
    valid_json: Dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url)
            valid_json = r.json()
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=502)

    if not valid_json.get("success"):
        return {"success": False, "ref_payco": ref_payco, "raw": valid_json}

    d = valid_json.get("data", {})
    cod = str(d.get("x_cod_response", ""))
    estado = STATUS_MAP.get(cod, "DESCONOCIDO")
    return {
        "success": True,
        "ref_payco": ref_payco,
        "estado": estado,
        "data": {
            "x_id_invoice": d.get("x_id_invoice"),
            "x_response": d.get("x_response"),
            "x_response_reason_text": d.get("x_response_reason_text"),
            "x_transaction_id": d.get("x_transaction_id"),
            "x_amount": d.get("x_amount"),
            "x_currency_code": d.get("x_currency_code"),
            "x_transaction_date": d.get("x_transaction_date"),
        },
        "raw": valid_json,
    }


async def _read_epayco_payload(request: Request, method: str) -> Dict[str, Any]:
    """
    ePayco suele mandar x_* como FORM (application/x-www-form-urlencoded),
    pero algunas integraciones / proxies lo envían como JSON.
    """
    if method == "GET":
        return dict(request.query_params)

    form = await request.form()
    if form:
        return dict(form)

    js = await request.json() if request.headers.get("content-type", "").startswith("application/json") else None
    if isinstance(js, dict):
        return js

    js2 = await request.json() if request.body else None
    return js2 if isinstance(js2, dict) else {}


def _verify_signature(payload: Dict[str, Any]) -> tuple[bool, str]:
    x_ref_payco      = str(payload.get("x_ref_payco", "") or "")
    x_transaction_id = str(payload.get("x_transaction_id", "") or "")
    x_amount         = str(payload.get("x_amount", "") or "")
    x_currency_code  = str(payload.get("x_currency_code", "") or "")
    x_signature      = str(payload.get("x_signature", "") or "")

    sign_raw = f"{P_CUST_ID_CLIENTE}^{P_KEY}^{x_ref_payco}^{x_transaction_id}^{x_amount}^{x_currency_code}"
    expected = hashlib.sha256(sign_raw.encode("utf-8")).hexdigest()

    return hmac.compare_digest(expected, x_signature), expected


async def _epayco_confirmation_impl(method: str, request: Request, db: Session) -> JSONResponse:
    payload = await _read_epayco_payload(request, method)

    x_ref_payco      = str(payload.get("x_ref_payco", "") or "")
    x_transaction_id = str(payload.get("x_transaction_id", "") or "")
    x_amount_raw     = str(payload.get("x_amount", "") or "")
    x_currency_code  = str(payload.get("x_currency_code", "") or "")
    x_cod_response   = str(payload.get("x_cod_response", "") or "")
    x_extra1         = payload.get("x_extra1")                      
    x_id_invoice     = payload.get("x_id_invoice")                            

    firma_ok, expected_sig = _verify_signature(payload)
    estado = STATUS_MAP.get(x_cod_response, "DESCONOCIDO")

    webhook_event = None
    try:
        if x_ref_payco:
            webhook_event = _upsert_webhook_event(
                db,
                ref_payco=x_ref_payco,
                transaction_id=x_transaction_id or None,
                estado=estado,
                firma_valida=firma_ok,
                payload=payload,
            )

            if webhook_event and hasattr(webhook_event, "processed") and _safe_bool(getattr(webhook_event, "processed", False)):
                db.commit()
                return JSONResponse(
                    {
                        "ok": True,
                        "duplicado": True,
                        "firma_valida": firma_ok,
                        "estado": estado,
                        "ref": x_ref_payco,
                        "suscripcion": x_extra1,
                        "invoice": x_id_invoice,
                    },
                    status_code=200,
                )
    except SQLAlchemyError as e:
        db.rollback()
        return JSONResponse({"ok": True, "warning": "db_log_failed", "detail": str(e)}, status_code=200)

    if firma_ok and estado == "APROBADO" and x_extra1:
        try:
            sus_id = int(x_extra1)
        except Exception:
            if webhook_event and hasattr(webhook_event, "last_error"):
                webhook_event.last_error = "x_extra1 no es int"
                db.commit()
            return JSONResponse({"ok": True, "warning": "bad_extra1", "ref": x_ref_payco}, status_code=200)

        try:
            sus = db.query(Suscripcion).filter(Suscripcion.id_suscripcion == sus_id).first()
            if not sus:
                if webhook_event and hasattr(webhook_event, "last_error"):
                    webhook_event.last_error = "suscripcion no encontrada"
                db.commit()
                return JSONResponse({"ok": True, "warning": "sus_not_found", "ref": x_ref_payco, "suscripcion": sus_id}, status_code=200)

            existing_pago = db.query(Pago).filter(Pago.referencia_epayco == x_ref_payco).first()
            if existing_pago:
                if webhook_event and hasattr(webhook_event, "processed"):
                    webhook_event.processed = True
                db.commit()
                return JSONResponse({"ok": True, "duplicado": True, "ref": x_ref_payco, "suscripcion": sus_id}, status_code=200)

            try:
                monto_num = float(x_amount_raw) if x_amount_raw else 0.0
            except Exception:
                monto_num = 0.0

            pago = Pago(
                id_suscripcion=sus.id_suscripcion,
                referencia_epayco=x_ref_payco,
                monto=monto_num,
            )
            db.add(pago)

            now = datetime.utcnow()
            db.flush()

            db.execute(
                text("CALL sp_activate_subscription(:sus_id, :now_dt)"),
                {"sus_id": int(sus.id_suscripcion), "now_dt": now},
            )
            db.refresh(sus)

            plan = db.query(Plan).filter(Plan.id_plan == sus.id_plan).first()

            if webhook_event and hasattr(webhook_event, "processed"):
                webhook_event.processed = True

            db.commit()

            _send_payment_confirmation_email(db=db, sus=sus, plan=plan, ref=x_ref_payco, amount=x_amount_raw or "0", when=now)

        except SQLAlchemyError as e:
            db.rollback()
            if webhook_event and hasattr(webhook_event, "last_error"):
                try:
                    webhook_event.last_error = str(e)
                    db.commit()
                except Exception:
                    db.rollback()
            return JSONResponse({"ok": True, "warning": "db_failed", "detail": str(e), "ref": x_ref_payco}, status_code=200)

    else:
        try:
            db.commit()
        except SQLAlchemyError:
            db.rollback()

    return JSONResponse(
        {
            "ok": True,
            "firma_valida": firma_ok,
            "estado": estado,
            "ref": x_ref_payco,
            "suscripcion": x_extra1,
            "invoice": x_id_invoice,
        },
        status_code=200,
    )


@router.get("/confirmation", name="epayco_confirmation_get", operation_id="epayco_confirmation_get")
async def epayco_confirmation_get(request: Request, db: Session = Depends(get_db)):
    return await _epayco_confirmation_impl("GET", request, db)


@router.post("/confirmation", name="epayco_confirmation_post", operation_id="epayco_confirmation_post")
async def epayco_confirmation_post(request: Request, db: Session = Depends(get_db)):
    return await _epayco_confirmation_impl("POST", request, db)
