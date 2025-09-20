from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.convert import router as convert_router

app = FastAPI(title="SOM3D — DICOM → STL")

# CORS mínimo para el front local (ajusta dominios para producción)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

# Pipeline
app.include_router(convert_router)
