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

# Cargar variables de entorno desde .env si existe
try:
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        # En desarrollo priorizamos el .env sobre variables del entorno del sistema
        load_dotenv(_dotenv_path, override=True)
except Exception:
    pass

from .core.config import FRONTEND_ORIGINS
from .core.health import ensure_services_ready, run_health_checks

app = FastAPI(title="SOM3D Backend", version="1.0.0")

_cors_origins = FRONTEND_ORIGINS if FRONTEND_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth_router, prefix="/auth", tags=["auth"])
app.include_router(plans_router, tags=["plans"])
app.include_router(subs_router, tags=["subscriptions"])
app.include_router(epayco_router)  # /epayco/...
app.include_router(som3d_router)   # /som3d/...
app.include_router(patients_router)  # /patients
app.include_router(studies_router)   # /studies
app.include_router(visor_router)     # /visor
app.include_router(hospitals_router) # /hospitals
app.include_router(doctors_router)   # /admin/doctors
app.include_router(admin_router)     # /admin/metrics
app.include_router(admin_hospitals_router)  # /admin/hospitals
app.include_router(admin_users_router)      # /admin/users
app.include_router(mensajes_router)

# Static files (admin dashboard)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.on_event("startup")
async def startup_healthcheck():
    # Ensure critical dependencies are available before serving traffic
    ensure_services_ready()

@app.get("/health")
def health():
    services = run_health_checks()
    overall = "ok" if all(s.get("status") == "ok" for s in services.values()) else "error"
    return {"status": overall, "services": services}
