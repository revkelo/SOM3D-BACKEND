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


# --------------------
# Hospital
# --------------------
class HospitalIn(BaseModel):
    nombre: str = Field(min_length=1, max_length=150)
    direccion: Optional[str] = Field(default=None, max_length=255)
    ciudad: Optional[str] = Field(default=None, max_length=100)
    telefono: Optional[str] = Field(default=None, max_length=30)
    correo: Optional[EmailStr] = None
    codigo: str = Field(min_length=1, max_length=12)

class HospitalUpdateIn(BaseModel):
    nombre: Optional[str] = Field(default=None, max_length=150)
    direccion: Optional[str] = Field(default=None, max_length=255)
    ciudad: Optional[str] = Field(default=None, max_length=100)
    telefono: Optional[str] = Field(default=None, max_length=30)
    correo: Optional[EmailStr] = None
    codigo: Optional[str] = Field(default=None, max_length=12)
    estado: Optional[Literal["ACTIVO","INACTIVO"]] = None

class HospitalOut(BaseModel):
    id_hospital: int
    nombre: str
    direccion: Optional[str]
    ciudad: Optional[str]
    telefono: Optional[str]
    correo: Optional[EmailStr]
    codigo: str
    estado: Literal["ACTIVO","INACTIVO"]

    class Config:
        from_attributes = True


# --------------------
# Plan
# --------------------
class PlanIn(BaseModel):
    nombre: str = Field(min_length=1, max_length=80)
    precio: float = Field(gt=0)
    periodo: Literal["MENSUAL", "TRIMESTRAL", "ANUAL"]
    duracion_meses: int

class PlanUpdateIn(BaseModel):
    nombre: Optional[str] = Field(default=None, max_length=80)
    precio: Optional[float] = Field(default=None, gt=0)
    periodo: Optional[Literal["MENSUAL", "TRIMESTRAL", "ANUAL"]] = None
    duracion_meses: Optional[int] = None

class PlanOut(BaseModel):
    id_plan: int
    nombre: str
    precio: float
    periodo: Literal["MENSUAL", "TRIMESTRAL", "ANUAL"]
    duracion_meses: int

    class Config:
        from_attributes = True


# --------------------
# Suscripcion / Pago (admin)
# --------------------
class SubscriptionOut(BaseModel):
    id_suscripcion: int
    id_medico: Optional[int]
    id_hospital: Optional[int]
    id_plan: int
    fecha_inicio: Optional[str]
    fecha_expiracion: Optional[str]
    estado: Literal["ACTIVA", "PAUSADA"]

    class Config:
        from_attributes = True

class SubscriptionUpdateIn(BaseModel):
    estado: Literal["ACTIVA", "PAUSADA"]

class PaymentOut(BaseModel):
    id_pago: int
    id_suscripcion: int
    referencia_epayco: str
    monto: float
    fecha_pago: Optional[str]

    class Config:
        from_attributes = True


# --------------------
# Auth aux schemas
# --------------------
class VerifyEmailRequestIn(BaseModel):
    correo: EmailStr

class ForgotPasswordIn(BaseModel):
    correo: EmailStr

class ResetPasswordIn(BaseModel):
    token: str = Field(min_length=10, max_length=200)
    new_password: str = Field(min_length=6, max_length=128)


class ConfirmCodeIn(BaseModel):
    token: str = Field(min_length=10, max_length=4096)
    code: str = Field(min_length=6, max_length=6)
