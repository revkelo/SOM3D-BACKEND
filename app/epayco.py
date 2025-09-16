import os, hashlib, json
from datetime import datetime
from typing import Dict, Any

import httpx
from fastapi import APIRouter, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.orm import Session

from .db import get_db
from .models import Suscripcion, Pago, Plan, Medico
from .config import EPAYCO_PUBLIC_KEY, EPAYCO_TEST, BASE_URL, P_CUST_ID_CLIENTE, P_KEY

router = APIRouter(prefix="/epayco", tags=["epayco"])

STATUS_MAP = {"1": "APROBADO", "2": "RECHAZADO", "3": "PENDIENTE", "4": "FALLIDO"}
PROCESADOS = set()

def _page(title: str, body_html: str) -> HTMLResponse:
    html = f"""
    <!doctype html><html lang="es"><head>
    <meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>{title}</title>
    <style>
      body{{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;padding:24px;max-width:960px;margin:auto}}
      .card{{border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin:12px 0}}
      table{{border-collapse:collapse;width:100%}}
      td{{border-bottom:1px solid #eee;padding:6px 8px;vertical-align:top}}
      pre{{background:#f6f8fa;padding:12px;border-radius:8px;overflow:auto}}
      .ok{{color:#0b7d62;font-weight:600}} .warn{{color:#b58900;font-weight:600}} .err{{color:#b00020;font-weight:600}}
      small{{opacity:.7}}
    </style></head><body>
    <h2>{title}</h2>{body_html}</body></html>"""
    return HTMLResponse(html)

@router.get("/response", response_class=HTMLResponse)
async def epayco_response(request: Request):
    params = dict(request.query_params)
    ref_payco = params.get("ref_payco")
    detail = "<p class='warn'>No recibÃ­ <code>ref_payco</code> en la URL.</p>"
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
            <p class="ok">ValidaciÃ³n OK â€” estado: <b>{estado}</b></p>
            <table>
              <tr><td>ref_payco</td><td><code>{ref_payco}</code></td></tr>
              <tr><td>x_id_invoice</td><td>{d.get('x_id_invoice')}</td></tr>
              <tr><td>x_response</td><td>{d.get('x_response')}</td></tr>
              <tr><td>x_response_reason_text</td><td>{d.get('x_response_reason_text')}</td></tr>
              <tr><td>x_transaction_id</td><td>{d.get('x_transaction_id')}</td></tr>
              <tr><td>x_amount x_currency</td><td>{d.get('x_amount')} {d.get('x_currency_code')}</td></tr>
              <tr><td>x_transaction_date</td><td>{d.get('x_transaction_date')}</td></tr>
            </table>
            """
        else:
            detail = f"<p class='err'>La validaciÃ³n respondiÃ³ error para <code>{ref_payco}</code>.</p>"

    body = f"""
    <div class="card">
      <h3>Respuesta de ePayco (cliente)</h3>
      <p>QueryString recibido:</p>
      <pre>{json.dumps(params, indent=2, ensure_ascii=False)}</pre>
      <hr/>
      <h4>ValidaciÃ³n por <code>ref_payco</code></h4>
      {detail}
      <h4>Raw de validaciÃ³n</h4>
      <pre>{json.dumps(valid_json, indent=2, ensure_ascii=False)}</pre>
    </div>
    """
    return _page("Respuesta de Pago", body)

@router.get("/validate")
async def epayco_validate(ref_payco: str):
    """Devuelve en JSON el resultado de validar un ref_payco en ePayco."""
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

@router.api_route("/confirmation", methods=["GET", "POST"])
async def epayco_confirmation(request: Request, db: Session = Depends(get_db)):
    payload = dict(await request.form()) if request.method == "POST" else dict(request.query_params)

    x_ref_payco      = payload.get("x_ref_payco", "")
    x_transaction_id = payload.get("x_transaction_id", "")
    x_amount         = payload.get("x_amount", "")
    x_currency_code  = payload.get("x_currency_code", "")
    x_signature      = payload.get("x_signature", "")
    x_cod_response   = str(payload.get("x_cod_response", ""))
    x_extra1         = payload.get("x_extra1")  # suscripcion_id que enviamos desde OnPage
    x_id_invoice     = payload.get("x_id_invoice")  # "SOM3D-<suscripcion_id>"

    sign_raw = f"{P_CUST_ID_CLIENTE}^{P_KEY}^{x_ref_payco}^{x_transaction_id}^{x_amount}^{x_currency_code}"
    expected = hashlib.sha256(sign_raw.encode("utf-8")).hexdigest()
    firma_ok = (expected == x_signature)
    estado = STATUS_MAP.get(x_cod_response, "DESCONOCIDO")

    if x_ref_payco and x_ref_payco in PROCESADOS:
        return JSONResponse({"ok": True, "duplicado": True, "firma_valida": firma_ok, "estado": estado, "ref": x_ref_payco})

    # Procesar solo si la firma es vÃ¡lida
    if firma_ok and estado == "APROBADO" and x_extra1:
        # Buscar suscripciÃ³n
        sus = db.query(Suscripcion).filter(Suscripcion.id_suscripcion == int(x_extra1)).first()
        if sus:
            # Registrar pago
            pago = Pago(
                id_suscripcion=sus.id_suscripcion,
                referencia_epayco=x_ref_payco,
                monto=float(x_amount or 0)
            )
            db.add(pago)
            # Verificar conflicto de suscripción ACTIVA antes de activar
            conflict_q = db.query(Suscripcion).filter(
                Suscripcion.estado == "ACTIVA",
                Suscripcion.id_suscripcion != sus.id_suscripcion,
            )
            if sus.id_medico is not None:
                conflict_q = conflict_q.filter(Suscripcion.id_medico == sus.id_medico)
            elif sus.id_hospital is not None:
                conflict_q = conflict_q.filter(Suscripcion.id_hospital == sus.id_hospital)
            conflict = conflict_q.first()
            if conflict:
                # Guardar solo el pago y reportar conflicto sin activar
                db.commit()
                PROCESADOS.add(x_ref_payco)
                return JSONResponse({
                    "ok": True,
                    "firma_valida": True,
                    "estado": estado,
                    "ref": x_ref_payco,
                    "suscripcion": x_extra1,
                    "activada": False,
                    "motivo": "Ya existe otra Suscripción ACTIVA para este titular",
                })

            # Activar y extender suscripciÃ³n segÃºn plan
            plan = db.query(Plan).filter(Plan.id_plan == sus.id_plan).first()
            now = datetime.utcnow()
            sus.estado = "ACTIVA"
            sus.fecha_inicio = now
            from dateutil.relativedelta import relativedelta
            sus.fecha_expiracion = now + relativedelta(months=+int(getattr(plan, "duracion_meses", 1)))
            # Activar usuario tras pago aprobado
            try:
                if sus.id_medico is not None:
                    med = db.query(Medico).filter(Medico.id_medico == sus.id_medico).first()
                    if med and getattr(med, 'usuario', None):
                        med.usuario.activo = True
            except Exception:
                pass
            db.commit()
            PROCESADOS.add(x_ref_payco)

    return JSONResponse({"ok": True, "firma_valida": firma_ok, "estado": estado, "ref": x_ref_payco, "suscripcion": x_extra1, "invoice": x_id_invoice})
