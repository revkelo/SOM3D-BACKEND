from fastapi import APIRouter, Depends, HTTPException, Query, Response, Request
import os
from pydantic import EmailStr
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Usuario
from ..schemas import (
    RegisterIn,
    LoginIn,
    TokenOut,
    UserOut,
    UserUpdateIn,
    ForgotPasswordIn,
    ResetPasswordIn,
    ConfirmCodeIn,
    ResetPasswordCodeIn,
)
from ..services.mailer import send_email
from ..services.email_templates import (
    template_reset_link,
    template_reset_code,
    template_verify_code,
)
from ..core.config import BASE_URL, FRONTEND_BASE_URL, VERIFY_EMAIL_EXPIRE_MIN, RESET_PASS_EXPIRE_MIN
from ..core.config import (
    JWT_SECRET,
    JWT_ALG,
    REFRESH_TOKEN_EXPIRE_MINUTES,
    REFRESH_COOKIE_NAME,
    REFRESH_COOKIE_SAMESITE,
    REFRESH_COOKIE_SECURE,
)
from ..core.tokens import (
    make_pre_register_token,
    parse_pre_register_token,
    make_reset_token,
    parse_reset_token,
    token_fp_matches,
    make_reset_code_token,
    parse_reset_code_token,
    _fp_password,
)
from ..core.tokens import make_refresh_token, parse_refresh_token
from ..core.security import hash_password, verify_password, create_access_token, get_current_user
from fastapi.responses import HTMLResponse


router = APIRouter()


@router.post("/login", response_model=TokenOut)
def login(payload: LoginIn, response: Response, db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(Usuario.correo == payload.correo).first()
    if not user or not verify_password(payload.password, user.contrasena):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    # Permitimos login aunque esté inactivo para completar el pago

    access = create_access_token({"sub": str(user.id_usuario), "rol": user.rol, "email": user.correo})
    refresh = make_refresh_token(user.id_usuario, user.contrasena, REFRESH_TOKEN_EXPIRE_MINUTES)
    # Eliminar posibles cookies antiguas con path más específico para evitar colisiones
    try:
        response.delete_cookie(REFRESH_COOKIE_NAME, path="/auth")
    except Exception:
        pass
    # usar helper que fija Path=/ para que el navegador la envíe en todas las rutas
    try:
        _set_refresh_cookie(response, refresh)
    except NameError:
        # fallback (por si el helper cambia de nombre)
        response.set_cookie(
            key=REFRESH_COOKIE_NAME,
            value=refresh,
            httponly=True,
            secure=bool(REFRESH_COOKIE_SECURE),
            samesite=(REFRESH_COOKIE_SAMESITE or "lax").lower(),
            max_age=REFRESH_TOKEN_EXPIRE_MINUTES * 60,
            path="/",
        )
    return {"access_token": access, "token_type": "bearer"}


@router.get("/me", response_model=UserOut)
def whoami(current=Depends(get_current_user)):
    return current


@router.patch("/me", response_model=UserOut)
def update_me(payload: UserUpdateIn, db: Session = Depends(get_db), current=Depends(get_current_user)):
    data = payload.model_dump(exclude_unset=True)

    if "correo" in data and data["correo"] and data["correo"] != current.correo:
        exists = db.query(Usuario).filter(
            Usuario.correo == data["correo"],
            Usuario.id_usuario != current.id_usuario,
        ).first()
        if exists:
            raise HTTPException(status_code=409, detail="Correo ya registrado")
        current.correo = str(data["correo"])

    if "password" in data and data["password"]:
        current.contrasena = hash_password(data["password"])

    for field in ("nombre", "apellido", "telefono", "direccion", "ciudad"):
        if field in data and data[field] is not None:
            setattr(current, field, data[field])

    db.commit()
    db.refresh(current)
    return current


@router.get("/email-exists")
def email_exists(correo: EmailStr, db: Session = Depends(get_db)):
    """Verifica si un correo ya esta registrado.
    Devuelve { exists: true/false } sin revelar mas datos.
    """
    exists = db.query(Usuario).filter(Usuario.correo == str(correo)).first() is not None
    return {"exists": exists}


# -----------------------
# Password reset
# -----------------------

@router.post("/forgot-password")
def forgot_password(payload: ForgotPasswordIn, db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(Usuario.correo == payload.correo).first()
    if user:
        try:
            token = make_reset_token(user.id_usuario, user.contrasena, RESET_PASS_EXPIRE_MIN)
            link = f"{BASE_URL}/auth/reset-password/form?token={token}"
            if FRONTEND_BASE_URL:
                link = f"{FRONTEND_BASE_URL}/reset_password.html?token={token}"
            html = template_reset_link(user.nombre or "Usuario", link, RESET_PASS_EXPIRE_MIN)
            send_email(user.correo, "Restablecer contrasena - 3DVinci Health", html)
        except Exception:
            pass
    # Siempre OK para no filtrar existencia
    return {"ok": True}


@router.post("/reset-password")
def reset_password(payload: ResetPasswordIn, db: Session = Depends(get_db)):
    data = parse_reset_token(payload.token)
    if not data:
        raise HTTPException(status_code=400, detail="Token invalido o expirado")
    user = db.query(Usuario).filter(Usuario.id_usuario == int(data.get("sub", 0))).first()
    if not user or not token_fp_matches(data, user.contrasena):
        raise HTTPException(status_code=400, detail="Token invalido o expirado")
    user.contrasena = hash_password(payload.new_password)
    db.commit()
    return {"ok": True}


# --- Alternativa: código de 6 dígitos para restablecer ---
@router.post("/forgot-password/code")
def forgot_password_code(payload: ForgotPasswordIn, db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(Usuario.correo == payload.correo).first()
    if user:
        try:
            # Generar código 6 dígitos + token con fingerprint
            import secrets
            code = str(secrets.randbelow(1_000_000)).zfill(6)
            data = {
                "sub": user.id_usuario,
                "fp": _fp_password(user.contrasena),
                "code": code,
            }
            token = make_reset_code_token(data, RESET_PASS_EXPIRE_MIN)
            # Enviar correo con el código
            html = template_reset_code(user.nombre or "Usuario", code, RESET_PASS_EXPIRE_MIN)
            send_email(user.correo, "Codigo de restablecimiento - 3DVinci Health", html)
            return {"ok": True, "token": token, "expires_in": RESET_PASS_EXPIRE_MIN}
        except Exception:
            pass
    # Siempre OK para no filtrar existencia
    return {"ok": True}


@router.post("/reset-password/code")
def reset_password_code(payload: ResetPasswordCodeIn, db: Session = Depends(get_db)):
    data = parse_reset_code_token(payload.token)
    if not data:
        raise HTTPException(status_code=400, detail="Token invalido o expirado")
    # Validar código
    code_expected = (data.get("code") or "").strip()
    if (payload.code or "").strip() != code_expected:
        raise HTTPException(status_code=400, detail="Codigo invalido o expirado")
    # Validar fingerprint y actualizar contraseña
    user = db.query(Usuario).filter(Usuario.id_usuario == int(data.get("sub", 0))).first()
    if not user:
        raise HTTPException(status_code=400, detail="Token invalido o expirado")
    # fp guardado en token al generar
    saved_fp = data.get("fp")
    from ..core.tokens import token_fp_matches as _tfm
    if not _tfm({"fp": saved_fp}, user.contrasena):
        raise HTTPException(status_code=400, detail="Token invalido o expirado")
    user.contrasena = hash_password(payload.new_password)
    db.commit()
    return {"ok": True}


@router.post("/pre-register")
def pre_register(payload: RegisterIn, db: Session = Depends(get_db)):
    exists = db.query(Usuario).filter(Usuario.correo == payload.correo).first()
    if exists:
        raise HTTPException(status_code=409, detail="Correo ya registrado")

    hashed = hash_password(payload.password)
    # Generate 6-digit verification code
    import secrets
    code = str(secrets.randbelow(1_000_000)).zfill(6)

    data = {
        "nombre": payload.nombre,
        "apellido": payload.apellido,
        "correo": str(payload.correo),
        "hashed": hashed,
        "telefono": payload.telefono,
        "direccion": payload.direccion,
        "ciudad": payload.ciudad,
        "rol": payload.rol,
        "code": code,
    }
    token = make_pre_register_token(data, VERIFY_EMAIL_EXPIRE_MIN)
    # Send email with verification code (no link)
    html = template_verify_code(payload.nombre or "Usuario", code, VERIFY_EMAIL_EXPIRE_MIN)
    res = send_email(str(payload.correo), "Verifica tu correo - 3DVinci Health", html)
    if not res.get("ok"):
        raise HTTPException(status_code=502, detail=f"Fallo enviando correo: {res}")
    # Return token to be used with code on frontend
    return {"ok": True, "token": token, "expires_in": VERIFY_EMAIL_EXPIRE_MIN}


@router.post("/register/confirm-code")
def confirm_register_code(payload: ConfirmCodeIn, db: Session = Depends(get_db)):
    data = parse_pre_register_token(payload.token)
    if not data:
        raise HTTPException(status_code=400, detail="Token invalido o expirado")
    # Check 6-digit code
    if (payload.code or "").strip() != (data.get("code") or "").strip():
        raise HTTPException(status_code=400, detail="Codigo invalido o expirado")
    # Avoid duplicates
    if db.query(Usuario).filter(Usuario.correo == data.get("correo")).first():
        raise HTTPException(status_code=409, detail="Correo ya registrado")
    # Create user
    user = Usuario(
        nombre=data.get("nombre"),
        apellido=data.get("apellido"),
        correo=data.get("correo"),
        contrasena=data.get("hashed"),
        telefono=data.get("telefono"),
        direccion=data.get("direccion"),
        ciudad=data.get("ciudad"),
        rol=data.get("rol") or "MEDICO",
        activo=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"ok": True, "id_usuario": user.id_usuario}


@router.get("/reset-password/form")
def reset_password_form(token: str = Query(...)):
    template = """<!doctype html><html lang='es'><head>
    <meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/>
    <title>Restablecer contrasena</title>
    <style>
      body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,sans-serif;padding:24px;max-width:720px;margin:auto;background:#0b0f17;color:#e5e7eb}
      .card{background:#111827;border:1px solid #374151;border-radius:12px;padding:16px;margin:12px 0}
      input{width:100%;padding:10px 12px;background:#0b0f17;border:1px solid #374151;border-radius:8px;color:#e5e7eb}
      button{padding:10px 16px;border-radius:10px;background:#22d3ee;color:#111827;font-weight:700;border:0;cursor:pointer}
      small{opacity:.8}
    </style></head><body>
      <h2>Restablecer contrasena</h2>
      <div class='card'>
        <p>Ingresa y confirma tu nueva contrasena.</p>
        <div style='display:grid;gap:10px;margin-top:10px'>
          <input id='p1' type='password' placeholder='Nueva contrasena (min 6)' />
          <input id='p2' type='password' placeholder='Repite contrasena' />
          <button id='go'>Cambiar contrasena</button>
          <div id='msg'></div>
        </div>
      </div>
      <script>
        const TOKEN = '__TOKEN__';
        const msg = document.getElementById('msg');
        function show(t, ok){ msg.textContent = t; msg.style.color = ok ? '#22d3ee' : '#f87171'; }
        async function reset(){
          const p1 = document.getElementById('p1').value || '';
          const p2 = document.getElementById('p2').value || '';
          if (p1.length < 6) return show('Contrasena muy corta (min 6).', false);
          if (p1 !== p2) return show('Las contrasenas no coinciden.', false);
          try {
            const res = await fetch('/auth/reset-password', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ token: TOKEN, new_password: p1 }) });
            if (!res.ok) throw new Error(await res.text());
            show('Contrasena actualizada. Ya puedes ingresar.', true);
          } catch (e) { show('No fue posible actualizar. Token invalido o expirado.', false); }
        }
        document.getElementById('go').addEventListener('click', reset);
      </script>
    </body></html>"""
    html = template.replace('__TOKEN__', token)
    return HTMLResponse(html)


# -----------------------
# Refresh & Logout (cookies HttpOnly)
# -----------------------

def _set_refresh_cookie(response: Response, token: str):
    same = (REFRESH_COOKIE_SAMESITE or "lax").lower()
    if same not in ("lax", "strict", "none"):
        same = "lax"
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=bool(REFRESH_COOKIE_SECURE),
        samesite=same,
        max_age=REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        # Usar path raiz para que el navegador la envíe en cualquier ruta
        path="/",
    )


def _clear_refresh_cookie(response: Response):
    response.delete_cookie(REFRESH_COOKIE_NAME, path="/")


@router.post("/refresh", response_model=TokenOut)
def refresh_token(request: Request, response: Response, db: Session = Depends(get_db)):
    cookie = request.cookies.get(REFRESH_COOKIE_NAME)
    if not cookie:
        raise HTTPException(status_code=401, detail="No refresh token")
    payload = parse_refresh_token(cookie)
    if not payload:
        raise HTTPException(status_code=401, detail="Refresh invalido o expirado")
    user_id = int(payload.get("sub") or 0)
    user = db.query(Usuario).filter(Usuario.id_usuario == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    if not token_fp_matches(payload, user.contrasena):
        _clear_refresh_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh invalido")

    access = create_access_token({"sub": str(user.id_usuario), "rol": user.rol, "email": user.correo})
    new_refresh = make_refresh_token(user.id_usuario, user.contrasena, REFRESH_TOKEN_EXPIRE_MINUTES)
    _set_refresh_cookie(response, new_refresh)
    return {"access_token": access, "token_type": "bearer"}


@router.post("/logout")
def logout(response: Response):
    _clear_refresh_cookie(response)
    return {"ok": True}
