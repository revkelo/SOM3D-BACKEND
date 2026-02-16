import os
from dotenv import load_dotenv
from urllib.parse import urlparse

# En local, priorizar el .env para evitar desalineaciones con variables del sistema
load_dotenv(override=True)

def mysql_url() -> str:
    url = os.getenv("MYSQL_URL")
    if url:
        return url
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    user = os.getenv("DB_USER", "root")
    pwd  = os.getenv("DB_PASS", "")
    db   = os.getenv("DB_NAME", "casaos")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}"

JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
if not JWT_SECRET:
    raise RuntimeError("JWT_SECRET is required and must be set in environment")
if JWT_SECRET in {"LeonardoDaVinci", "changeme", "change-me"}:
    raise RuntimeError("JWT_SECRET is insecure. Configure a strong random value.")

JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "15"))

# Refresh tokens (cookies)
REFRESH_TOKEN_EXPIRE_MINUTES = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", "43200"))  # 30 days
REFRESH_COOKIE_NAME = os.getenv("REFRESH_COOKIE_NAME", "refresh_token")
# SameSite values: 'lax' | 'strict' | 'none'
REFRESH_COOKIE_SAMESITE = os.getenv("REFRESH_COOKIE_SAMESITE", "lax").lower()
REFRESH_COOKIE_SECURE = os.getenv("REFRESH_COOKIE_SECURE", "true").lower() == "true"
CSRF_COOKIE_NAME = os.getenv("CSRF_COOKIE_NAME", "csrf_token")
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"

# CORS: Frontend origins (comma-separated)
FRONTEND_ORIGINS = [
    o.strip() for o in (os.getenv("FRONTEND_ORIGINS", "").split(",")) if o.strip()
]

# ePayco
EPAYCO_PUBLIC_KEY = os.getenv("EPAYCO_PUBLIC_KEY", "")
EPAYCO_TEST = os.getenv("EPAYCO_TEST", "true").lower() == "true"
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
P_CUST_ID_CLIENTE = os.getenv("P_CUST_ID_CLIENTE", "")
P_KEY = os.getenv("P_KEY", "")

# SMTP / Email
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "no-reply@example.com")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"

# Frontend base (para links en emails)
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "")

# Expiraciones de tokens
VERIFY_EMAIL_EXPIRE_MIN = int(os.getenv("VERIFY_EMAIL_EXPIRE_MIN", "120"))
RESET_PASS_EXPIRE_MIN = int(os.getenv("RESET_PASS_EXPIRE_MIN", "60"))


def _is_local_base_url(url: str) -> bool:
    try:
        parsed = urlparse(str(url or "").strip())
        host = (parsed.hostname or "").lower()
        return host in {"localhost", "127.0.0.1"}
    except Exception:
        return False


if REFRESH_COOKIE_SAMESITE == "none" and not REFRESH_COOKIE_SECURE:
    raise RuntimeError("REFRESH_COOKIE_SAMESITE=none requiere REFRESH_COOKIE_SECURE=true")

# En ambientes no locales no permitimos cookies de sesion inseguras por error de configuracion.
if not REFRESH_COOKIE_SECURE and not _is_local_base_url(BASE_URL):
    raise RuntimeError(
        "REFRESH_COOKIE_SECURE=false solo se permite con BASE_URL local. "
        "En despliegue usa REFRESH_COOKIE_SECURE=true con HTTPS."
    )

