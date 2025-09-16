from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.auth import router as auth_router
from .routes.plans import router as plans_router
from .routes.subscriptions import router as subs_router
from .epayco import router as epayco_router

app = FastAPI(title="SOM3D Backend + Auth + Planes + ePayco", version="1.0.0")

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

@app.get("/health")
def health():
    return {"status": "ok"}