import smtplib
from typing import Optional, Dict, Any
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.header import Header
from email.policy import SMTP as SMTP_POLICY 

from ..core.config import (
    SMTP_HOST,
    SMTP_PORT,
    SMTP_USER,
    SMTP_PASS,
    SMTP_FROM,
    SMTP_USE_TLS,
    SMTP_USE_SSL,
)

def _build_message(frm: str, to: str, subject: str, html: str, text: Optional[str]) -> MIMEMultipart:
    msg = MIMEMultipart("alternative")
    msg["From"] = frm              
    msg["To"] = to                 
    msg["Subject"] = str(Header(subject or "(sin asunto)", "utf-8"))  

    if text is None:
        text = "Correo en HTML adjunto."
    msg.attach(MIMEText(text, "plain", "utf-8"))
    if html:
        msg.attach(MIMEText(html, "html", "utf-8"))
    return msg


def send_email(to: str, subject: str, html: str, text: Optional[str] = None) -> Dict[str, Any]:
    if not SMTP_HOST:
        return {"ok": False, "error": "SMTP_HOST no configurado"}

    msg = _build_message(SMTP_FROM, to, subject, html, text)

    try:
        if SMTP_USE_SSL:
            server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT)
        else:
            server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)

        server.ehlo()
        if SMTP_USE_TLS and not SMTP_USE_SSL:
            server.starttls()
            server.ehlo()

        if SMTP_USER:
            server.login(SMTP_USER, SMTP_PASS)

        mail_options = []
        if "smtputf8" in getattr(server, "esmtp_features", {}):
            mail_options.append("SMTPUTF8")

        raw = msg.as_bytes(policy=SMTP_POLICY)
        server.sendmail(SMTP_FROM, [to], raw, mail_options=mail_options)
        server.quit()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}

