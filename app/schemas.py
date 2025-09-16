from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal

class RegisterIn(BaseModel):
    nombre: str = Field(min_length=1, max_length=100)
    apellido: str = Field(min_length=1, max_length=100)
    correo: EmailStr
    password: str = Field(min_length=6, max_length=128)
    telefono: Optional[str] = None
    direccion: Optional[str] = None
    ciudad: Optional[str] = None
    rol: Literal["ADMINISTRADOR", "MEDICO"] = "MEDICO"

class LoginIn(BaseModel):
    correo: EmailStr
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id_usuario: int
    nombre: str
    apellido: str
    correo: EmailStr
    rol: str
    activo: bool

    class Config:
        from_attributes = True

class StartSubscriptionIn(BaseModel):
    plan_id: int
    hospital_id: Optional[int] = None

class CheckoutOut(BaseModel):
    suscripcion_id: int
    checkout: dict
    onpage_html: str