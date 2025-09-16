from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Usuario
from ..schemas import RegisterIn, LoginIn, TokenOut, UserOut
from ..auth import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter()

@router.post("/register", response_model=UserOut, status_code=201)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    exists = db.query(Usuario).filter(Usuario.correo == payload.correo).first()
    if exists:
        raise HTTPException(status_code=409, detail="Correo ya registrado")

    user = Usuario(
        nombre=payload.nombre,
        apellido=payload.apellido,
        correo=payload.correo,
        contrasena=hash_password(payload.password),
        telefono=payload.telefono,
        direccion=payload.direccion,
        ciudad=payload.ciudad,
        rol=payload.rol,
        # Se activará tras pago aprobado en /epayco/confirmation
        activo=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.post("/login", response_model=TokenOut)
def login(payload: LoginIn, db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(Usuario.correo == payload.correo).first()
    if not user or not verify_password(payload.password, user.contrasena):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    # Permitimos login aunque esté inactivo para completar el pago

    token = create_access_token({"sub": str(user.id_usuario), "rol": user.rol, "email": user.correo})
    return {"access_token": token, "token_type": "bearer"}

@router.get("/me", response_model=UserOut)
def whoami(current=Depends(get_current_user)):
    return current
