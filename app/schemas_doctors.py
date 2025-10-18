from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal


class DoctorIn(BaseModel):
    nombre: str = Field(min_length=1, max_length=100)
    apellido: str = Field(min_length=1, max_length=100)
    correo: EmailStr
    password: str = Field(min_length=6, max_length=128)
    telefono: Optional[str] = Field(default=None, max_length=20)
    direccion: Optional[str] = Field(default=None, max_length=255)
    ciudad: Optional[str] = Field(default=None, max_length=100)
    id_hospital: Optional[int] = None
    referenciado: Optional[bool] = False
    activo: Optional[bool] = True


class DoctorUpdateIn(BaseModel):
    nombre: Optional[str] = Field(default=None, max_length=100)
    apellido: Optional[str] = Field(default=None, max_length=100)
    correo: Optional[EmailStr] = None
    password: Optional[str] = Field(default=None, min_length=6, max_length=128)
    telefono: Optional[str] = Field(default=None, max_length=20)
    direccion: Optional[str] = Field(default=None, max_length=255)
    ciudad: Optional[str] = Field(default=None, max_length=100)
    id_hospital: Optional[int] = None
    referenciado: Optional[bool] = None
    activo: Optional[bool] = None
    estado: Optional[Literal["ACTIVO","INACTIVO"]] = None


class DoctorOut(BaseModel):
    id_medico: int
    id_usuario: int
    nombre: str
    apellido: str
    correo: EmailStr
    telefono: Optional[str]
    direccion: Optional[str]
    ciudad: Optional[str]
    id_hospital: Optional[int]
    referenciado: bool
    estado: Literal["ACTIVO","INACTIVO"]
    activo: bool

    class Config:
        from_attributes = True

