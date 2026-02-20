from fastapi import APIRouter, Depends, HTTPException, Query, Response, Request
import secrets
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from pydantic import EmailStr
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Usuario, AuthLoginAttempt, AuthRefreshSession
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
from ..services.mail_templates import (
    template_reset_link,
    template_reset_code,
    template_verify_code,
)
from ..core.config import BASE_URL, FRONTEND_BASE_URL, VERIFY_EMAIL_EXPIRE_MIN, RESET_PASS_EXPIRE_MIN
from ..core.config import (
    REFRESH_TOKEN_EXPIRE_MINUTES,
    REFRESH_COOKIE_NAME,
    REFRESH_COOKIE_SAMESITE,
    REFRESH_COOKIE_SECURE,
    CSRF_COOKIE_NAME,
    JWT_SECRET,
    TRUST_PROXY_HEADERS,
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

LOGIN_MAX_FAILED_ATTEMPTS = 5
LOGIN_WINDOW_MINUTES = 15
LOGIN_LOCK_MINUTES = 15
RESET_CODE_MAX_FAILED_ATTEMPTS = 8
RESET_CODE_WINDOW_MINUTES = 15
ACTION_WINDOW_MINUTES = 15
ACTION_MAX_ATTEMPTS = 6


def _client_ip(request: Request) -> str | None:
    xff = request.headers.get("x-forwarded-for")
    if TRUST_PROXY_HEADERS and xff:
        return xff.split(",")[0].strip() or None
    if request.client and request.client.host:
        return request.client.host
    return None


def _utcnow():
    return datetime.now(timezone.utc)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _token_sha(token: str) -> str:
    return hashlib.sha256((token or "").encode("utf-8")).hexdigest()


def _csrf_value() -> str:
    return secrets.token_urlsafe(32)


def _register_login_attempt(
    db: Session,
    *,
    correo: str,
    request: Request,
    success: bool,
    reason: str | None = None,
):
    try:
        db.add(
            AuthLoginAttempt(
                correo=str(correo).strip().lower(),
                ip_address=_client_ip(request),
                user_agent=request.headers.get("user-agent"),
                success=bool(success),
                reason=reason,
            )
        )
        db.commit()
    except Exception:
        db.rollback()


def _reset_code_digest(code: str) -> str:
    raw = str(code or "").strip()
    return hmac.new(
        JWT_SECRET.encode("utf-8"),
        raw.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _action_rate_limited(
    db: Session,
    *,
    request: Request,
    reason: str,
    correo: str | None = None,
    max_attempts: int = ACTION_MAX_ATTEMPTS,
    window_minutes: int = ACTION_WINDOW_MINUTES,
) -> tuple[bool, int]:
    now = _utcnow()
    window_start = now - timedelta(minutes=window_minutes)
    ip = _client_ip(request)
    correo_norm = str(correo or "").strip().lower() or None

    filters = [AuthLoginAttempt.reason == reason, AuthLoginAttempt.created_at >= window_start]
    if ip and correo_norm:
        filters.append((AuthLoginAttempt.ip_address == ip) | (AuthLoginAttempt.correo == correo_norm))
    elif ip:
        filters.append(AuthLoginAttempt.ip_address == ip)
    elif correo_norm:
        filters.append(AuthLoginAttempt.correo == correo_norm)
    else:
        return False, 0

    count = int(
        db.query(func.count(AuthLoginAttempt.id_attempt))
        .filter(*filters)
        .scalar()
        or 0
    )
    if count < max_attempts:
        return False, 0

    last_attempt = (
        db.query(func.max(AuthLoginAttempt.created_at))
        .filter(*filters)
        .scalar()
    )
    if not last_attempt:
        return False, 0
    if last_attempt.tzinfo is None:
        last_attempt = last_attempt.replace(tzinfo=timezone.utc)
    unlock_at = last_attempt + timedelta(minutes=window_minutes)
    if unlock_at <= now:
        return False, 0
    return True, int((unlock_at - now).total_seconds())


def _login_lock_status(db: Session, correo: str, request: Request) -> tuple[bool, int]:
    now = _utcnow()
    window_start = now - timedelta(minutes=LOGIN_WINDOW_MINUTES)
    lock_window_start = now - timedelta(minutes=LOGIN_LOCK_MINUTES)
    correo_norm = str(correo).strip().lower()
    ip = _client_ip(request)

    failed_email = int(
        db.query(func.count(AuthLoginAttempt.id_attempt))
        .filter(
            AuthLoginAttempt.correo == correo_norm,
            AuthLoginAttempt.success == False,
            AuthLoginAttempt.created_at >= window_start,
        )
        .scalar()
        or 0
    )
    failed_ip = 0
    if ip:
        failed_ip = int(
            db.query(func.count(AuthLoginAttempt.id_attempt))
            .filter(
                AuthLoginAttempt.ip_address == ip,
                AuthLoginAttempt.success == False,
                AuthLoginAttempt.created_at >= window_start,
            )
            .scalar()
            or 0
        )

    if failed_email < LOGIN_MAX_FAILED_ATTEMPTS and failed_ip < LOGIN_MAX_FAILED_ATTEMPTS:
        return False, 0

    last_failed_query = db.query(func.max(AuthLoginAttempt.created_at)).filter(
        AuthLoginAttempt.success == False,
        AuthLoginAttempt.created_at >= lock_window_start,
        (AuthLoginAttempt.correo == correo_norm) if not ip else (
            (AuthLoginAttempt.correo == correo_norm) | (AuthLoginAttempt.ip_address == ip)
        ),
    )
    last_failed = last_failed_query.scalar()
    if not last_failed:
        return False, 0
    if last_failed.tzinfo is None:
        last_failed = last_failed.replace(tzinfo=timezone.utc)

    unlock_at = last_failed + timedelta(minutes=LOGIN_LOCK_MINUTES)
    if unlock_at <= now:
        return False, 0
    return True, int((unlock_at - now).total_seconds())


def _reset_code_locked(db: Session, request: Request) -> tuple[bool, int]:
    now = _utcnow()
    window_start = now - timedelta(minutes=RESET_CODE_WINDOW_MINUTES)
    ip = _client_ip(request)
    if not ip:
        return False, 0
    failed = int(
        db.query(func.count(AuthLoginAttempt.id_attempt))
        .filter(
            AuthLoginAttempt.ip_address == ip,
            AuthLoginAttempt.success == False,
            AuthLoginAttempt.reason == "RESET_CODE_INVALID",
            AuthLoginAttempt.created_at >= window_start,
        )
        .scalar()
        or 0
    )
    if failed < RESET_CODE_MAX_FAILED_ATTEMPTS:
        return False, 0
    last_failed = (
        db.query(func.max(AuthLoginAttempt.created_at))
        .filter(
            AuthLoginAttempt.ip_address == ip,
            AuthLoginAttempt.success == False,
            AuthLoginAttempt.reason == "RESET_CODE_INVALID",
            AuthLoginAttempt.created_at >= window_start,
        )
        .scalar()
    )
    if not last_failed:
        return False, 0
    if last_failed.tzinfo is None:
        last_failed = last_failed.replace(tzinfo=timezone.utc)
    unlock_at = last_failed + timedelta(minutes=RESET_CODE_WINDOW_MINUTES)
    if unlock_at <= now:
        return False, 0
    return True, int((unlock_at - now).total_seconds())


def _set_csrf_cookie(response: Response, value: str):
    same = (REFRESH_COOKIE_SAMESITE or "lax").lower()
    if same not in ("lax", "strict", "none"):
        same = "lax"
    response.set_cookie(
        key=CSRF_COOKIE_NAME,
        value=value,
        httponly=False,
        secure=bool(REFRESH_COOKIE_SECURE),
        samesite=same,
        max_age=REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )


def _clear_csrf_cookie(response: Response):
    response.delete_cookie(CSRF_COOKIE_NAME, path="/")


def _validate_csrf(request: Request):
    cookie = request.cookies.get(CSRF_COOKIE_NAME) or ""
    header = request.headers.get("x-csrf-token") or ""
    if not cookie or not header or not hmac.compare_digest(cookie, header):
        raise HTTPException(status_code=403, detail="CSRF token invalido")


def _create_refresh_session(
    db: Session,
    *,
    user: Usuario,
    request: Request,
    jti: str,
) -> AuthRefreshSession:
    now = _utcnow()
    expires_at = now + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
    row = AuthRefreshSession(
        id_usuario=user.id_usuario,
        jti=jti,
        fp=_fp_password(user.contrasena),
        ip_address=_client_ip(request),
        user_agent=request.headers.get("user-agent"),
        expires_at=expires_at,
    )
    db.add(row)
    return row


def _revoke_user_refresh_sessions(db: Session, user_id: int):
    now = _utcnow()
    db.query(AuthRefreshSession).filter(
        AuthRefreshSession.id_usuario == user_id,
        AuthRefreshSession.revoked_at.is_(None),
    ).update({AuthRefreshSession.revoked_at: now}, synchronize_session=False)


@router.post("/login", response_model=TokenOut)
def login(payload: LoginIn, response: Response, request: Request, db: Session = Depends(get_db)):
    locked, retry_after_sec = _login_lock_status(db, payload.correo, request)
    if locked:
        raise HTTPException(
            status_code=429,
            detail=f"Demasiados intentos fallidos. Intenta nuevamente en {max(retry_after_sec, 1)} segundos.",
        )

    user = db.query(Usuario).filter(Usuario.correo == payload.correo).first()
    if not user or not verify_password(payload.password, user.contrasena):
        _register_login_attempt(
            db,
            correo=payload.correo,
            request=request,
            success=False,
            reason="INVALID_CREDENTIALS",
        )
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    _register_login_attempt(
        db,
        correo=payload.correo,
        request=request,
        success=True,
        reason="LOGIN_OK",
    )

    access = create_access_token({"sub": str(user.id_usuario), "rol": user.rol, "email": user.correo})
    refresh_jti = secrets.token_urlsafe(32)
    refresh = make_refresh_token(
        user.id_usuario,
        user.contrasena,
        REFRESH_TOKEN_EXPIRE_MINUTES,
        jti=refresh_jti,
    )
    _create_refresh_session(db, user=user, request=request, jti=refresh_jti)
    try:
        response.delete_cookie(REFRESH_COOKIE_NAME, path="/auth")
    except Exception:
        pass
    try:
        _set_refresh_cookie(response, refresh)
    except NameError:
        response.set_cookie(
            key=REFRESH_COOKIE_NAME,
            value=refresh,
            httponly=True,
            secure=bool(REFRESH_COOKIE_SECURE),
            samesite=(REFRESH_COOKIE_SAMESITE or "lax").lower(),
            max_age=REFRESH_TOKEN_EXPIRE_MINUTES * 60,
            path="/",
        )
    csrf = _csrf_value()
    _set_csrf_cookie(response, csrf)
    db.commit()
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
        _revoke_user_refresh_sessions(db, current.id_usuario)

    for field in ("nombre", "apellido", "telefono", "direccion", "ciudad"):
        if field in data and data[field] is not None:
            setattr(current, field, data[field])

    db.commit()
    db.refresh(current)
    return current


@router.get("/email-exists")
def email_exists(correo: EmailStr, db: Session = Depends(get_db)):
    """Respuesta neutral para evitar enumeracion de cuentas."""
    _ = correo
    _ = db
    return {"exists": False}



@router.post("/forgot-password")
def forgot_password(payload: ForgotPasswordIn, request: Request, db: Session = Depends(get_db)):
    limited, retry_after = _action_rate_limited(
        db,
        request=request,
        reason="FORGOT_PASSWORD_REQUEST",
        correo=str(payload.correo),
    )
    if limited:
        raise HTTPException(
            status_code=429,
            detail=f"Demasiadas solicitudes. Intenta nuevamente en {max(retry_after, 1)} segundos.",
        )
    _register_login_attempt(
        db,
        correo=str(payload.correo),
        request=request,
        success=True,
        reason="FORGOT_PASSWORD_REQUEST",
    )
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
    _revoke_user_refresh_sessions(db, user.id_usuario)
    db.commit()
    return {"ok": True}


@router.post("/forgot-password/code")
def forgot_password_code(payload: ForgotPasswordIn, request: Request, db: Session = Depends(get_db)):
    limited, retry_after = _action_rate_limited(
        db,
        request=request,
        reason="FORGOT_PASSWORD_CODE_REQUEST",
        correo=str(payload.correo),
    )
    if limited:
        raise HTTPException(
            status_code=429,
            detail=f"Demasiadas solicitudes. Intenta nuevamente en {max(retry_after, 1)} segundos.",
        )
    _register_login_attempt(
        db,
        correo=str(payload.correo),
        request=request,
        success=True,
        reason="FORGOT_PASSWORD_CODE_REQUEST",
    )
    user = db.query(Usuario).filter(Usuario.correo == payload.correo).first()
    fake_sub = 0
    fake_fp = _fp_password(secrets.token_urlsafe(24))
    fake_code = str(secrets.randbelow(1_000_000)).zfill(6)
    flow_data = {"sub": fake_sub, "fp": fake_fp, "ch": _reset_code_digest(fake_code)}
    if user:
        try:
            code = str(secrets.randbelow(1_000_000)).zfill(6)
            flow_data = {
                "sub": user.id_usuario,
                "fp": _fp_password(user.contrasena),
                "ch": _reset_code_digest(code),
            }
            html = template_reset_code(user.nombre or "Usuario", code, RESET_PASS_EXPIRE_MIN)
            send_email(user.correo, "Codigo de restablecimiento - 3DVinci Health", html)
        except Exception:
            pass
    token = make_reset_code_token(flow_data, RESET_PASS_EXPIRE_MIN)
    return {"ok": True, "token": token, "expires_in": RESET_PASS_EXPIRE_MIN}


@router.post("/reset-password/code")
def reset_password_code(payload: ResetPasswordCodeIn, request: Request, db: Session = Depends(get_db)):
    locked, retry_after = _reset_code_locked(db, request)
    if locked:
        raise HTTPException(
            status_code=429,
            detail=f"Demasiados intentos de cÃ³digo. Intenta nuevamente en {max(retry_after, 1)} segundos.",
        )
    data = parse_reset_code_token(payload.token)
    if not data:
        _register_login_attempt(
            db,
            correo="__reset__",
            request=request,
            success=False,
            reason="RESET_CODE_INVALID",
        )
        raise HTTPException(status_code=400, detail="Token invalido o expirado")
    provided_code = (payload.code or "").strip()
    expected_digest = str(data.get("ch") or "").strip()
    if expected_digest:
        provided_digest = _reset_code_digest(provided_code)
        code_ok = hmac.compare_digest(expected_digest, provided_digest)
    else:
        code_ok = provided_code == (str(data.get("code") or "").strip())
    if not code_ok:
        _register_login_attempt(
            db,
            correo="__reset__",
            request=request,
            success=False,
            reason="RESET_CODE_INVALID",
        )
        raise HTTPException(status_code=400, detail="Codigo invalido o expirado")
    user = db.query(Usuario).filter(Usuario.id_usuario == int(data.get("sub", 0))).first()
    if not user:
        _register_login_attempt(
            db,
            correo="__reset__",
            request=request,
            success=False,
            reason="RESET_CODE_INVALID",
        )
        raise HTTPException(status_code=400, detail="Token invalido o expirado")
    saved_fp = data.get("fp")
    from ..core.tokens import token_fp_matches as _tfm
    if not _tfm({"fp": saved_fp}, user.contrasena):
        _register_login_attempt(
            db,
            correo="__reset__",
            request=request,
            success=False,
            reason="RESET_CODE_INVALID",
        )
        raise HTTPException(status_code=400, detail="Token invalido o expirado")
    user.contrasena = hash_password(payload.new_password)
    _revoke_user_refresh_sessions(db, user.id_usuario)
    db.commit()
    return {"ok": True}


@router.post("/pre-register")
def pre_register(payload: RegisterIn, request: Request, db: Session = Depends(get_db)):
    limited, retry_after = _action_rate_limited(
        db,
        request=request,
        reason="PRE_REGISTER_REQUEST",
        correo=str(payload.correo),
    )
    if limited:
        raise HTTPException(
            status_code=429,
            detail=f"Demasiadas solicitudes. Intenta nuevamente en {max(retry_after, 1)} segundos.",
        )
    _register_login_attempt(
        db,
        correo=str(payload.correo),
        request=request,
        success=True,
        reason="PRE_REGISTER_REQUEST",
    )
    exists = db.query(Usuario).filter(Usuario.correo == payload.correo).first()
    if exists:
        raise HTTPException(status_code=409, detail="Correo ya registrado")

    hashed = hash_password(payload.password)
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
        "rol": "MEDICO",
        "ch": _reset_code_digest(code),
    }
    token = make_pre_register_token(data, VERIFY_EMAIL_EXPIRE_MIN)
    html = template_verify_code(payload.nombre or "Usuario", code, VERIFY_EMAIL_EXPIRE_MIN)
    res = send_email(str(payload.correo), "Verifica tu correo - 3DVinci Health", html)
    if not res.get("ok"):
        raise HTTPException(status_code=502, detail=f"Fallo enviando correo: {res}")
    return {"ok": True, "token": token, "expires_in": VERIFY_EMAIL_EXPIRE_MIN}


@router.post("/register/confirm-code")
def confirm_register_code(payload: ConfirmCodeIn, db: Session = Depends(get_db)):
    data = parse_pre_register_token(payload.token)
    if not data:
        raise HTTPException(status_code=400, detail="Token invalido o expirado")
    provided_code = (payload.code or "").strip()
    expected_digest = str(data.get("ch") or "").strip()
    if expected_digest:
        code_ok = hmac.compare_digest(expected_digest, _reset_code_digest(provided_code))
    else:
        code_ok = provided_code == (str(data.get("code") or "").strip())
    if not code_ok:
        raise HTTPException(status_code=400, detail="Codigo invalido o expirado")
    if db.query(Usuario).filter(Usuario.correo == data.get("correo")).first():
        raise HTTPException(status_code=409, detail="Correo ya registrado")
    user = Usuario(
        nombre=data.get("nombre"),
        apellido=data.get("apellido"),
        correo=data.get("correo"),
        contrasena=data.get("hashed"),
        telefono=data.get("telefono"),
        direccion=data.get("direccion"),
        ciudad=data.get("ciudad"),
        rol="MEDICO",
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
          <input id='p1' type='password' placeholder='Nueva contrasena (min 8)' />
          <input id='p2' type='password' placeholder='Repite contrasena' />
          <button id='go'>Cambiar contrasena</button>
          <div id='msg'></div>
        </div>
      </div>
      <script>
        const TOKEN = __TOKEN_JSON__;
        const msg = document.getElementById('msg');
        function show(t, ok){ msg.textContent = t; msg.style.color = ok ? '#22d3ee' : '#f87171'; }
        async function reset(){
          const p1 = document.getElementById('p1').value || '';
          const p2 = document.getElementById('p2').value || '';
          if (p1.length < 8) return show('Contrasena muy corta (min 8).', false);
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
    html = template.replace('__TOKEN_JSON__', json.dumps(token))
    return HTMLResponse(html)



def _set_refresh_cookie(response: Response, token: str):
    same = (REFRESH_COOKIE_SAMESITE or "lax").lower()
    if same not in ("lax", "strict", "none"):
        same = "lax"
    secure = bool(REFRESH_COOKIE_SECURE)
    if same == "none" and not secure:
        secure = True
    response.set_cookie(
        key=REFRESH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=secure,
        samesite=same,
        max_age=REFRESH_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )


def _clear_refresh_cookie(response: Response):
    response.delete_cookie(REFRESH_COOKIE_NAME, path="/")


@router.get("/csrf")
def issue_csrf(response: Response):
    value = _csrf_value()
    _set_csrf_cookie(response, value)
    return {"ok": True, "csrf": value}


@router.post("/refresh", response_model=TokenOut)
def refresh_token(request: Request, response: Response, db: Session = Depends(get_db)):
    _validate_csrf(request)
    cookie = request.cookies.get(REFRESH_COOKIE_NAME)
    if not cookie:
        raise HTTPException(status_code=401, detail="No refresh token")
    payload = parse_refresh_token(cookie)
    if not payload:
        _clear_refresh_cookie(response)
        _clear_csrf_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh invalido o expirado")
    user_id = int(payload.get("sub") or 0)
    jti = str(payload.get("jti") or "")
    if not user_id or not jti:
        _clear_refresh_cookie(response)
        _clear_csrf_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh invalido")

    session = db.query(AuthRefreshSession).filter(AuthRefreshSession.jti == jti).first()
    if not session or session.id_usuario != user_id:
        _clear_refresh_cookie(response)
        _clear_csrf_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh invalido")

    now = _utcnow()
    if session.revoked_at is not None:
        _revoke_user_refresh_sessions(db, user_id)
        db.commit()
        _clear_refresh_cookie(response)
        _clear_csrf_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh revocado")

    expires_at = _as_utc(session.expires_at)
    if not expires_at or expires_at <= now:
        session.revoked_at = now
        db.commit()
        _clear_refresh_cookie(response)
        _clear_csrf_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh expirado")

    user = db.query(Usuario).filter(Usuario.id_usuario == user_id).first()
    if not user:
        session.revoked_at = now
        db.commit()
        _clear_refresh_cookie(response)
        _clear_csrf_cookie(response)
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    if not token_fp_matches(payload, user.contrasena):
        session.revoked_at = now
        db.commit()
        _clear_refresh_cookie(response)
        _clear_csrf_cookie(response)
        raise HTTPException(status_code=401, detail="Refresh invalido")

    new_jti = secrets.token_urlsafe(32)
    access = create_access_token({"sub": str(user.id_usuario), "rol": user.rol, "email": user.correo})
    new_refresh = make_refresh_token(
        user.id_usuario,
        user.contrasena,
        REFRESH_TOKEN_EXPIRE_MINUTES,
        jti=new_jti,
    )
    session.revoked_at = now
    session.replaced_by_jti = new_jti
    session.last_used_at = now
    _create_refresh_session(db, user=user, request=request, jti=new_jti)
    _set_refresh_cookie(response, new_refresh)
    _set_csrf_cookie(response, _csrf_value())
    db.commit()
    return {"access_token": access, "token_type": "bearer"}


@router.post("/logout")
def logout(request: Request, response: Response, db: Session = Depends(get_db)):
    try:
        _validate_csrf(request)
    except HTTPException:
        pass
    cookie = request.cookies.get(REFRESH_COOKIE_NAME)
    payload = parse_refresh_token(cookie) if cookie else None
    if payload:
        jti = str(payload.get("jti") or "")
        if jti:
            row = db.query(AuthRefreshSession).filter(AuthRefreshSession.jti == jti).first()
            if row and row.revoked_at is None:
                row.revoked_at = _utcnow()
                db.commit()
    _clear_refresh_cookie(response)
    _clear_csrf_cookie(response)
    return {"ok": True}





