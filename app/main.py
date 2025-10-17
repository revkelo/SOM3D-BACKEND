from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv

from .routes.auth import router as auth_router
from .routes.plans import router as plans_router
from .routes.subscriptions import router as subs_router
from .api.epayco import router as epayco_router
from .routes.som3d import router as som3d_router
from .routes.patients import router as patients_router
from .routes.studies import router as studies_router
from .routes.visor import router as visor_router

# Cargar variables de entorno desde .env si existe
try:
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
except Exception:
    pass

app = FastAPI(title="SOM3D Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.get("/health")
def health():
    return {"status": "ok"}
