from html import escape as _esc
from datetime import datetime

# ======== SHELL ========

def _shell_html(
    title: str,
    content: str,
    brand: str = "3DVinci Health",
) -> str:
    """
    Layout de email alineado al look de la app:
    - BG #0b0f17 (igual a tu página)
    - Card #111827 con borde #1f2937 y radio grande
    - Headings cian, textos grises
    - Todo centrado y con buen respiro
    - Sin menú
    """
    year = datetime.now().year
    title_e = _esc(title)

    # Paleta + estilos pensados para clientes de correo (inline en lo crítico)
    css = """
    body{margin:0;padding:0;background:#0b0f17;color:#cbd5e1;}
    a{color:#10d4ed;text-decoration:none;}
    .wrap{max-width:860px;margin:0 auto;padding:24px 16px;
          font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;}
    .header{padding:10px 12px 6px;text-align:center;}
    .brand{font-weight:900;font-size:26px;color:#10d4ed;letter-spacing:.2px;}
    .by{font-size:12px;color:#9aa3b2;margin-top:4px;}
    .card{background:#111827;color:#e5e7eb;border-radius:18px;padding:28px 28px;margin:22px 6px;
          border:1px solid #1f2937;}
    h1{margin:0 0 12px;font-size:22px;color:#10d4ed;text-align:center;}
    h2{margin:0 0 8px;font-size:18px;color:#e5e7eb;text-align:center;}
    p{margin:8px 0;line-height:1.6;color:#cbd5e1;}
    .muted{color:#9aa3b2;font-size:12px;}
    .footer{margin-top:14px;text-align:center;color:#9aa3b2;font-size:12px;padding:18px 0 26px;}
    .contacts p{margin:2px 0;}
    .code-wrap{margin:18px auto 8px;max-width:360px;background:#000;border:1px solid #2a3a4f;
               border-radius:10px;padding:14px 20px;text-align:center;}
    /* Estilo del código solicitado */
    .code{font-size:22px; letter-spacing:4px; font-weight:700; color:#e5e7eb;}
    .lead{max-width:640px;margin:0 auto;text-align:center;color:#cbd5e1}
    .btn{display:inline-block;padding:12px 18px;font-weight:800;background:#10d4ed;color:#0b1321;
         border-radius:10px;text-decoration:none;}
    .bold{font-weight:800;}
    /* Apariencia de cita sin disparar el colapsado de Gmail */
    .quote{border-left:3px solid #2a3a4f;padding-left:12px;margin:12px 0;}
    """

    # Soporte <bold> y neutralización de patrones que Gmail colapsa
    content_html = (
        content
        .replace("<bold>", "<b>").replace("</bold>", "</b>")          # alias <bold>
        .replace("<blockquote", "<div class='quote'")                 # evita blockquote
        .replace("</blockquote>", "</div>")
        .replace("gmail_quote", "x-gmail-quote")                      # evita clase especial de Gmail
        .replace("\n-- ", "\n— ")                                     # evita separador de firma
    )

    return f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <meta http-equiv="x-ua-compatible" content="ie=edge"/>
  <title>{title_e}</title>
  <style>{css}</style>
</head>
<body style="margin:0;padding:0;background:#0b0f17;color:#cbd5e1;">
  <div class="wrap">
    <div class="header">
      <div class="brand">{_esc(brand)}</div>
      <div class="by">por 3DVinci Studios</div>
    </div>

    <div class="card">
      {content_html}
    </div>

    <div class="footer">
      <div style="margin:8px 0;font-weight:700;color:#cbd5e1">{_esc(brand)}</div>
      <p class="muted">Una nueva experiencia al servicio de la salud</p>
      <div class="contacts" style="margin-top:10px">
        <p>Celular: <a href="tel:+573134184657">+57 313 418 4657</a> · <a href="tel:+573058732339">+57 305 873 2339</a></p>
        <p>Dirección: Calle 74a # 20C-75 / Estudio 74, Bogotá</p>
        <p>Correo: <a href="mailto:info@3dvincistudios.com">info@3dvincistudios.com</a></p>
      </div>
      <div style="margin-top:10px">Facebook · Instagram · YouTube · Behance</div>
      <div style="margin-top:6px" class="muted">© {year} 3DVinci Studios. Todos los derechos reservados.</div>
    </div>
  </div>
</body>
</html>"""

# ======== HELPERS ========

def _btn_primary(href: str, label: str = "Abrir") -> str:
    """Botón con estilos inline para compatibilidad alta (Outlook, etc.)."""
    return (
        f"<a href=\"{_esc(href, quote=True)}\" target=\"_blank\" "
        "style=\"display:inline-block;padding:12px 18px;font-weight:800;"
        "background:#10d4ed;color:#0b1321;border-radius:10px;text-decoration:none;"
        "mso-padding-alt:0;\"><span style=\"mso-text-raise:12pt;\">"
        f"{_esc(label)}</span></a>"
    )

# ======== TEMPLATES ========

def template_reset_link(nombre: str, link: str, expire_min: int) -> str:
    body = (
        "<h1>Restablecer contraseña</h1>"
        "<p class='lead'>Hola " + _esc(nombre) +
        ", para restablecer tu contraseña usa el botón siguiente.</p>"
        f"<p style='margin:16px 0;text-align:center'>{_btn_primary(link, 'Restablecer contraseña')}</p>"
        "<p class='muted' style='text-align:center'>El enlace expira en "
        + _esc(str(expire_min)) + " minutos.</p>"
    )
    return _shell_html("Restablecer contraseña — 3DVinci Health", body)

def template_reset_code(nombre: str, code: str, expire_min: int) -> str:
    body = (
        "<h1>Restablecer contraseña</h1>"
        "<p class='lead'>Hola " + _esc(nombre) +
        ", usa este código para restablecer tu contraseña:</p>"
        "<div class='code-wrap'><div class='code'>" + _esc(str(code)) + "</div></div>"
        "<p class='muted' style='text-align:center'>El código expira en "
        + _esc(str(expire_min)) + " minutos.</p>"
    )
    return _shell_html("Código de restablecimiento — 3DVinci Health", body)

def template_verify_code(nombre: str, code: str, expire_min: int) -> str:
    body = (
        "<h1>Verifica tu correo</h1>"
        "<p class='lead'>Hola " + _esc(nombre) +
        ", usa el siguiente código para confirmar tu registro:</p>"
        "<div class='code-wrap'><div class='code'>" + _esc(str(code)) + "</div></div>"
        "<p class='muted' style='text-align:center'>El código expira en "
        + _esc(str(expire_min)) + " minutos.</p>"
    )
    return _shell_html("Verifica tu correo — 3DVinci Health", body)


def template_payment_confirm(
    nombre: str,
    plan_nombre: str,
    monto: str,
    referencia: str,
    fecha_iso: str | None = None,
) -> str:
    fecha_txt = _esc(fecha_iso or datetime.now().isoformat(timespec="seconds"))
    body = (
        "<h1>Pago confirmado</h1>"
        "<p class='lead'>Hola " + _esc(nombre) + ", tu pago fue aprobado correctamente.</p>"
        "<div class='code-wrap' style='max-width:520px;text-align:left'>"
        "<p><span class='bold'>Plan:</span> " + _esc(plan_nombre) + "</p>"
        "<p><span class='bold'>Monto:</span> " + _esc(monto) + "</p>"
        "<p><span class='bold'>Referencia:</span> " + _esc(referencia) + "</p>"
        "<p><span class='bold'>Fecha:</span> " + fecha_txt + "</p>"
        "</div>"
        "<p class='muted' style='text-align:center'>Tu suscripción quedó activada.</p>"
    )
    return _shell_html("Pago confirmado — 3DVinci Health", body)
