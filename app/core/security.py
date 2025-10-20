from datetime import datetime, timedelta, timezone
import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from ..core.config import JWT_SECRET, JWT_ALG, JWT_EXPIRE_MINUTES
from ..db import get_db
from ..models import Usuario

security = HTTPBearer()
security_optional = HTTPBearer(auto_error=False)

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def create_access_token(data: dict, expires_minutes: int = JWT_EXPIRE_MINUTES) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALG)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expirado")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token invalido")

def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Falta token Bearer")
    payload = decode_token(creds.credentials)
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="Token sin 'sub'")
    user = db.query(Usuario).filter(Usuario.id_usuario == int(uid)).first()
    if not user:
        raise HTTPException(status_code=401, detail="Usuario no encontrado")
    return user


def get_current_user_optional(
    creds: HTTPAuthorizationCredentials | None = Depends(security_optional),
    db: Session = Depends(get_db),
):
    if not creds or creds.scheme.lower() != "bearer":
        return None
    try:
        payload = decode_token(creds.credentials)
        uid = payload.get("sub")
        if not uid:
            return None
        user = db.query(Usuario).filter(Usuario.id_usuario == int(uid)).first()
        return user
    except HTTPException:
        return None


def require_admin(user = Depends(get_current_user)):
    rol = getattr(user, "rol", None)
    if str(rol).upper() != "ADMINISTRADOR":
        raise HTTPException(status_code=403, detail="Requiere rol ADMINISTRADOR")
    return user
