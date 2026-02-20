from fastapi import FastAPI
import os

from dotenv import load_dotenv, find_dotenv
from starlette.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .routes.auth import router as auth_router
from .routes.plans import router as plans_router
from .routes.subscriptions import router as subs_router
from .routes.doctors import router as doctors_router
from .routes.admin import router as admin_router
from .routes.admin_hospitals import router as admin_hospitals_router
from .routes.admin_users import router as admin_users_router
from .api.epayco import router as epayco_router
from .routes.som3d import router as som3d_router
from .routes.patients import router as patients_router
from .routes.studies import router as studies_router
from .routes.visor import router as visor_router
from .routes.messages import router as mensajes_router
from .routes.hospitals import router as hospitals_router

try:
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=True)
except Exception:
    pass

from .core.config import FRONTEND_ORIGINS
from .core.health import ensure_services_ready, run_health_checks
from .db import ensure_runtime_tables

app = FastAPI(title="SOM3D Backend", version="1.0.0")

_cors_origins = FRONTEND_ORIGINS if FRONTEND_ORIGINS else [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(plans_router, tags=["plans"])
app.include_router(subs_router, tags=["subscriptions"])
app.include_router(epayco_router)               
app.include_router(som3d_router)               
app.include_router(patients_router, tags=["patients"])             
app.include_router(studies_router, tags=["studies"])             
app.include_router(visor_router, tags=["visor"])               
app.include_router(hospitals_router, tags=["hospitals"])             
app.include_router(doctors_router, tags=["doctors"])                   
app.include_router(admin_router, tags=["admin"])                       
app.include_router(admin_hospitals_router, tags=["admin-hospitals"])                    
app.include_router(admin_users_router, tags=["admin-users"])                    
app.include_router(mensajes_router)

static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    path = request.url.path or ""
    is_docs_ui = path.startswith("/docs") or path.startswith("/redoc")
    if is_docs_ui:
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https://fastapi.tiangolo.com; "
            "font-src 'self' data: https://cdn.jsdelivr.net; "
            "object-src 'none'; frame-ancestors 'none'; base-uri 'self'"
        )
    else:
        csp = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; object-src 'none'; "
            "frame-ancestors 'none'; base-uri 'self'"
        )
    response.headers.setdefault("Content-Security-Policy", csp)
    if request.url.scheme == "https":
        response.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
    return response

@app.on_event("startup")
async def startup_healthcheck():
    ensure_runtime_tables()
    ensure_services_ready()

@app.get("/health", tags=["health"])
def health():
    services = run_health_checks()
    overall = "ok" if all(s.get("status") == "ok" for s in services.values()) else "error"
    return {"status": overall, "services": services}
