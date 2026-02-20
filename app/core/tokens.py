from datetime import datetime, timedelta, timezone
import hashlib
import jwt

from ..core.config import JWT_SECRET, JWT_ALG, REFRESH_TOKEN_EXPIRE_MINUTES


def _now():
    return datetime.now(timezone.utc)


def _fp_password(hashed_password: str) -> str:
    return hashlib.sha256((hashed_password or "").encode("utf-8")).hexdigest()[:16]


def make_pre_register_token(data: dict, expires_minutes: int) -> str:
    payload = {
        "k": "pre_register",
        "d": data,
        "iat": int(_now().timestamp()),
        "exp": int((_now() + timedelta(minutes=expires_minutes)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def parse_pre_register_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        if payload.get("k") != "pre_register":
            return None
        return payload.get("d") or {}
    except Exception:
        return None


def make_reset_token(user_id: int, hashed_password: str, expires_minutes: int) -> str:
    fp = _fp_password(hashed_password)
    payload = {
        "k": "reset_password",
        "sub": str(user_id),
        "fp": fp,
        "iat": int(_now().timestamp()),
        "exp": int((_now() + timedelta(minutes=expires_minutes)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def parse_reset_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        if payload.get("k") != "reset_password":
            return None
        return payload
    except Exception:
        return None


def token_fp_matches(token_payload: dict, current_hashed_password: str) -> bool:
    try:
        return token_payload.get("fp") == _fp_password(current_hashed_password)
    except Exception:
        return False


def make_reset_code_token(data: dict, expires_minutes: int) -> str:
    payload = {
        "k": "reset_code",
        "d": data,
        "iat": int(_now().timestamp()),
        "exp": int((_now() + timedelta(minutes=expires_minutes)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def parse_reset_code_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        if payload.get("k") != "reset_code":
            return None
        return payload.get("d") or {}
    except Exception:
        return None



def make_refresh_token(
    user_id: int,
    hashed_password: str,
    expires_minutes: int | None = None,
    jti: str | None = None,
) -> str:
    exp_min = expires_minutes if expires_minutes is not None else REFRESH_TOKEN_EXPIRE_MINUTES
    payload = {
        "k": "refresh",
        "sub": str(user_id),
        "jti": str(jti or ""),
        "fp": _fp_password(hashed_password),
        "iat": int(_now().timestamp()),
        "exp": int((_now() + timedelta(minutes=exp_min)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def parse_refresh_token(token: str) -> dict | None:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        if payload.get("k") != "refresh":
            return None
        return payload
    except Exception:
        return None

