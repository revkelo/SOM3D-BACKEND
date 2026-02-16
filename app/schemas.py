import re
from pydantic import BaseModel, EmailStr, Field, ConfigDict, field_validator
from typing import Optional, Literal, List
from datetime import date, datetime

_RE_NAME = re.compile("^[A-Za-z\\u00C0-\\u00FF' -]{2,100}$")
_RE_CITY = re.compile("^[A-Za-z\\u00C0-\\u00FF0-9' .-]{2,100}$")
_RE_PHONE = re.compile("^[0-9+]{7,15}$")
_RE_CODE = re.compile("^[A-Z0-9-]{1,20}$")
_RE_DOC = re.compile("^[A-Z0-9-]{4,24}$")
_RE_PLAN_NAME = re.compile("^[A-Za-z\\u00C0-\\u00FF0-9()_\\- .]{2,120}$")


def _blank_to_none(v):
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _norm_spaces(v: str) -> str:
    return re.sub(r"\s{2,}", " ", str(v or "").strip())


def _clean_text(v: str, max_len: int) -> str:
    return _norm_spaces(v).replace("<", "").replace(">", "")[:max_len]


def _clean_name(v: str, max_len: int = 100) -> str:
    return re.sub("[^A-Za-z\\u00C0-\\u00FF' -]", "", _norm_spaces(v))[:max_len]


def _clean_city(v: str, max_len: int = 100) -> str:
    return re.sub("[^A-Za-z\\u00C0-\\u00FF0-9' .-]", "", _norm_spaces(v))[:max_len]


def _clean_phone(v: str) -> str:
    s = re.sub(r"[^\d+]", "", str(v or ""))
    s = re.sub(r"(?!^)\+", "", s)
    return s[:15]


def _clean_code(v: str, max_len: int = 20) -> str:
    return re.sub(r"[^A-Z0-9-]", "", str(v or "").upper())[:max_len]


def _clean_doc(v: str, max_len: int = 24) -> str:
    return re.sub(r"[^A-Z0-9-]", "", str(v or "").upper())[:max_len]

class RegisterIn(BaseModel):
    nombre: str = Field(min_length=1, max_length=100)
    apellido: str = Field(min_length=1, max_length=100)
    correo: EmailStr
    password: str = Field(min_length=8, max_length=128)
    telefono: Optional[str] = None
    direccion: Optional[str] = None
    ciudad: Optional[str] = None
    # El registro público solo permite cuentas MEDICO.
    rol: Literal["MEDICO"] = "MEDICO"

    @field_validator("nombre", "apellido", mode="before")
    @classmethod
    def _validate_names(cls, v):
        s = _clean_name(v, 100)
        if not _RE_NAME.fullmatch(s):
            raise ValueError("nombre/apellido invalido")
        return s

    @field_validator("telefono", mode="before")
    @classmethod
    def _validate_phone(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_phone(raw)
        if not _RE_PHONE.fullmatch(s):
            raise ValueError("telefono invalido")
        return s

    @field_validator("direccion", mode="before")
    @classmethod
    def _validate_address(cls, v):
        raw = _blank_to_none(v)
        return _clean_text(raw, 255) if raw else None

    @field_validator("ciudad", mode="before")
    @classmethod
    def _validate_city(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_city(raw, 100)
        if not _RE_CITY.fullmatch(s):
            raise ValueError("ciudad invalida")
        return s

class LoginIn(BaseModel):
    correo: str = Field(min_length=3, max_length=100)
    password: str

    @field_validator("correo", mode="before")
    @classmethod
    def _validate_login_email(cls, v):
        s = _norm_spaces(str(v or "")).lower()
        # Login permite dominios internos/reservados (p. ej. *.local) para compatibilidad con seeds.
        if not re.fullmatch(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", s):
            raise ValueError("correo invalido")
        return s

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id_usuario: int
    nombre: str
    apellido: str
    correo: str
    telefono: Optional[str] = None
    direccion: Optional[str] = None
    ciudad: Optional[str] = None
    rol: str
    activo: bool

    class Config:
        from_attributes = True


class UserUpdateIn(BaseModel):
    nombre: Optional[str] = Field(default=None, min_length=1, max_length=100)
    apellido: Optional[str] = Field(default=None, min_length=1, max_length=100)
    correo: Optional[EmailStr] = None
    telefono: Optional[str] = Field(default=None, max_length=20)
    direccion: Optional[str] = Field(default=None, max_length=255)
    ciudad: Optional[str] = Field(default=None, max_length=100)
    password: Optional[str] = Field(default=None, min_length=8, max_length=128)

    @field_validator("nombre", "apellido", mode="before")
    @classmethod
    def _validate_names(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_name(raw, 100)
        if not _RE_NAME.fullmatch(s):
            raise ValueError("nombre/apellido invalido")
        return s

    @field_validator("telefono", mode="before")
    @classmethod
    def _validate_phone(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_phone(raw)
        if not _RE_PHONE.fullmatch(s):
            raise ValueError("telefono invalido")
        return s

    @field_validator("direccion", mode="before")
    @classmethod
    def _validate_address(cls, v):
        raw = _blank_to_none(v)
        return _clean_text(raw, 255) if raw else None

    @field_validator("ciudad", mode="before")
    @classmethod
    def _validate_city(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_city(raw, 100)
        if not _RE_CITY.fullmatch(s):
            raise ValueError("ciudad invalida")
        return s

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
    codigo: Optional[str] = Field(default=None, min_length=1, max_length=12)
    # Opcional: al crear el hospital, asociarlo a un plan creando una Suscripcion en PAUSADA
    plan_id: Optional[int] = None

    @field_validator("nombre", mode="before")
    @classmethod
    def _validate_name(cls, v):
        s = _clean_name(v, 120)
        if not _RE_NAME.fullmatch(s):
            raise ValueError("nombre invalido")
        return s

    @field_validator("direccion", mode="before")
    @classmethod
    def _validate_address(cls, v):
        raw = _blank_to_none(v)
        return _clean_text(raw, 255) if raw else None

    @field_validator("ciudad", mode="before")
    @classmethod
    def _validate_city(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_city(raw, 100)
        if not _RE_CITY.fullmatch(s):
            raise ValueError("ciudad invalida")
        return s

    @field_validator("telefono", mode="before")
    @classmethod
    def _validate_phone(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_phone(raw)
        if not _RE_PHONE.fullmatch(s):
            raise ValueError("telefono invalido")
        return s

    @field_validator("codigo", mode="before")
    @classmethod
    def _validate_code(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_code(raw, 12)
        if not _RE_CODE.fullmatch(s):
            raise ValueError("codigo invalido")
        return s

class HospitalUpdateIn(BaseModel):
    nombre: Optional[str] = Field(default=None, max_length=150)
    direccion: Optional[str] = Field(default=None, max_length=255)
    ciudad: Optional[str] = Field(default=None, max_length=100)
    telefono: Optional[str] = Field(default=None, max_length=30)
    correo: Optional[EmailStr] = None
    codigo: Optional[str] = Field(default=None, max_length=12)
    estado: Optional[Literal["ACTIVO","INACTIVO"]] = None

    @field_validator("nombre", mode="before")
    @classmethod
    def _validate_name(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_name(raw, 120)
        if not _RE_NAME.fullmatch(s):
            raise ValueError("nombre invalido")
        return s

    @field_validator("direccion", mode="before")
    @classmethod
    def _validate_address(cls, v):
        raw = _blank_to_none(v)
        return _clean_text(raw, 255) if raw else None

    @field_validator("ciudad", mode="before")
    @classmethod
    def _validate_city(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_city(raw, 100)
        if not _RE_CITY.fullmatch(s):
            raise ValueError("ciudad invalida")
        return s

    @field_validator("telefono", mode="before")
    @classmethod
    def _validate_phone(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_phone(raw)
        if not _RE_PHONE.fullmatch(s):
            raise ValueError("telefono invalido")
        return s

    @field_validator("codigo", mode="before")
    @classmethod
    def _validate_code(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_code(raw, 12)
        if not _RE_CODE.fullmatch(s):
            raise ValueError("codigo invalido")
        return s

class HospitalOut(BaseModel):
    id_hospital: int
    nombre: str
    direccion: Optional[str]
    ciudad: Optional[str]
    telefono: Optional[str]
    correo: Optional[str]
    codigo: str
    estado: Literal["ACTIVO","INACTIVO"]

    class Config:
        from_attributes = True


class HospitalLinkByCodeIn(BaseModel):
    codigo: str = Field(min_length=1, max_length=12)

    @field_validator("codigo", mode="before")
    @classmethod
    def _validate_code(cls, v):
        s = _clean_code(v, 12)
        if not _RE_CODE.fullmatch(s):
            raise ValueError("codigo invalido")
        return s


class HospitalStartSubscriptionIn(BaseModel):
    codigo: str = Field(min_length=1, max_length=12)
    plan_id: Optional[int] = None

    @field_validator("codigo", mode="before")
    @classmethod
    def _validate_code(cls, v):
        s = _clean_code(v, 12)
        if not _RE_CODE.fullmatch(s):
            raise ValueError("codigo invalido")
        return s


# --------------------
# Plan
# --------------------
class PlanIn(BaseModel):
    nombre: str = Field(min_length=1, max_length=80)
    precio: float = Field(gt=0)
    periodo: Literal["MENSUAL", "TRIMESTRAL", "ANUAL"]
    duracion_meses: int

    @field_validator("nombre", mode="before")
    @classmethod
    def _validate_name(cls, v):
        s = _clean_text(v, 80)
        if not _RE_PLAN_NAME.fullmatch(s):
            raise ValueError("nombre de plan invalido")
        return s

class PlanUpdateIn(BaseModel):
    nombre: Optional[str] = Field(default=None, max_length=80)
    precio: Optional[float] = Field(default=None, gt=0)
    periodo: Optional[Literal["MENSUAL", "TRIMESTRAL", "ANUAL"]] = None
    duracion_meses: Optional[int] = None

    @field_validator("nombre", mode="before")
    @classmethod
    def _validate_name(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_text(raw, 80)
        if not _RE_PLAN_NAME.fullmatch(s):
            raise ValueError("nombre de plan invalido")
        return s

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
    fecha_inicio: Optional[datetime]
    fecha_expiracion: Optional[datetime]
    estado: Literal["ACTIVA", "PAUSADA"]

    class Config:
        from_attributes = True

class SubscriptionUpdateIn(BaseModel):
    estado: Literal["ACTIVA", "PAUSADA"]


class SubscriptionAdminCreateIn(BaseModel):
    id_medico: Optional[int] = None
    id_hospital: Optional[int] = None
    id_plan: int
    estado: Literal["ACTIVA", "PAUSADA"] = "PAUSADA"
    fecha_inicio: Optional[datetime] = None
    fecha_expiracion: Optional[datetime] = None


class SubscriptionAdminUpdateIn(BaseModel):
    id_medico: Optional[int] = None
    id_hospital: Optional[int] = None
    id_plan: Optional[int] = None
    estado: Optional[Literal["ACTIVA", "PAUSADA"]] = None
    fecha_inicio: Optional[datetime] = None
    fecha_expiracion: Optional[datetime] = None

class PaymentOut(BaseModel):
    id_pago: int
    id_suscripcion: int
    referencia_epayco: str
    monto: float
    fecha_pago: Optional[datetime]

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
    new_password: str = Field(min_length=8, max_length=128)


class ConfirmCodeIn(BaseModel):
    token: str = Field(min_length=10, max_length=4096)
    code: str = Field(min_length=6, max_length=6)


class ResetPasswordCodeIn(BaseModel):
    token: str = Field(min_length=10, max_length=4096)
    code: str = Field(min_length=6, max_length=6)
    new_password: str = Field(min_length=8, max_length=128)


# --------------------
# Paciente
# --------------------
class PacienteIn(BaseModel):
    doc_tipo: Optional[str] = Field(default=None, max_length=20)
    doc_numero: Optional[str] = Field(default=None, max_length=40)
    nombres: str = Field(min_length=1, max_length=100)
    apellidos: str = Field(min_length=1, max_length=100)
    fecha_nacimiento: Optional[date] = None
    sexo: Optional[str] = Field(default=None, max_length=20)
    telefono: Optional[str] = Field(default=None, max_length=30)
    correo: Optional[EmailStr] = None
    direccion: Optional[str] = Field(default=None, max_length=200)
    ciudad: Optional[str] = Field(default=None, max_length=80)

    @field_validator("doc_tipo", mode="before")
    @classmethod
    def _validate_doc_tipo(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        return _clean_code(raw, 20)

    @field_validator("doc_numero", mode="before")
    @classmethod
    def _validate_doc_numero(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_doc(raw, 24)
        if not _RE_DOC.fullmatch(s):
            raise ValueError("documento invalido")
        return s

    @field_validator("nombres", "apellidos", mode="before")
    @classmethod
    def _validate_names(cls, v):
        s = _clean_name(v, 100)
        if not _RE_NAME.fullmatch(s):
            raise ValueError("nombre/apellido invalido")
        return s

    @field_validator("sexo", mode="before")
    @classmethod
    def _validate_sexo(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = str(raw).upper()
        if s not in {"M", "F", "O"}:
            raise ValueError("sexo invalido")
        return s

    @field_validator("telefono", mode="before")
    @classmethod
    def _validate_phone(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_phone(raw)
        if not _RE_PHONE.fullmatch(s):
            raise ValueError("telefono invalido")
        return s

    @field_validator("direccion", mode="before")
    @classmethod
    def _validate_address(cls, v):
        raw = _blank_to_none(v)
        return _clean_text(raw, 200) if raw else None

    @field_validator("ciudad", mode="before")
    @classmethod
    def _validate_city(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_city(raw, 80)
        if not _RE_CITY.fullmatch(s):
            raise ValueError("ciudad invalida")
        return s

class PacienteUpdateIn(BaseModel):
    doc_tipo: Optional[str] = Field(default=None, max_length=20)
    doc_numero: Optional[str] = Field(default=None, max_length=40)
    nombres: Optional[str] = Field(default=None, max_length=100)
    apellidos: Optional[str] = Field(default=None, max_length=100)
    fecha_nacimiento: Optional[date] = None
    sexo: Optional[str] = Field(default=None, max_length=20)
    telefono: Optional[str] = Field(default=None, max_length=30)
    correo: Optional[EmailStr] = None
    direccion: Optional[str] = Field(default=None, max_length=200)
    ciudad: Optional[str] = Field(default=None, max_length=80)
    estado: Optional[Literal["ACTIVO","INACTIVO"]] = None

    @field_validator("doc_tipo", mode="before")
    @classmethod
    def _validate_doc_tipo(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        return _clean_code(raw, 20)

    @field_validator("doc_numero", mode="before")
    @classmethod
    def _validate_doc_numero(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_doc(raw, 24)
        if not _RE_DOC.fullmatch(s):
            raise ValueError("documento invalido")
        return s

    @field_validator("nombres", "apellidos", mode="before")
    @classmethod
    def _validate_names(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_name(raw, 100)
        if not _RE_NAME.fullmatch(s):
            raise ValueError("nombre/apellido invalido")
        return s

    @field_validator("sexo", mode="before")
    @classmethod
    def _validate_sexo(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = str(raw).upper()
        if s not in {"M", "F", "O"}:
            raise ValueError("sexo invalido")
        return s

    @field_validator("telefono", mode="before")
    @classmethod
    def _validate_phone(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_phone(raw)
        if not _RE_PHONE.fullmatch(s):
            raise ValueError("telefono invalido")
        return s

    @field_validator("direccion", mode="before")
    @classmethod
    def _validate_address(cls, v):
        raw = _blank_to_none(v)
        return _clean_text(raw, 200) if raw else None

    @field_validator("ciudad", mode="before")
    @classmethod
    def _validate_city(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_city(raw, 80)
        if not _RE_CITY.fullmatch(s):
            raise ValueError("ciudad invalida")
        return s

class PacienteOut(BaseModel):
    id_paciente: int
    id_medico: int
    doc_tipo: Optional[str]
    doc_numero: Optional[str]
    nombres: str
    apellidos: str
    fecha_nacimiento: Optional[date]
    sexo: Optional[str]
    telefono: Optional[str]
    correo: Optional[str]
    direccion: Optional[str]
    ciudad: Optional[str]
    estado: Literal["ACTIVO","INACTIVO"]

    class Config:
        from_attributes = True


# --------------------
# Estudio
# --------------------
class EstudioIn(BaseModel):
    id_paciente: int
    modalidad: Optional[str] = Field(default=None, max_length=20)
    fecha_estudio: Optional[datetime] = None
    descripcion: Optional[str] = Field(default=None, max_length=200)
    notas: Optional[str] = None

class EstudioUpdateIn(BaseModel):
    modalidad: Optional[str] = Field(default=None, max_length=20)
    fecha_estudio: Optional[datetime] = None
    descripcion: Optional[str] = Field(default=None, max_length=200)
    notas: Optional[str] = None

class EstudioOut(BaseModel):
    id_estudio: int
    id_paciente: int
    id_medico: int
    modalidad: Optional[str]
    fecha_estudio: datetime
    descripcion: Optional[str]
    notas: Optional[str]

    class Config:
        from_attributes = True


class ClinicalNoteIn(BaseModel):
    texto: str = Field(min_length=1, max_length=10000)
    segmento: Optional[str] = Field(default="GENERAL", max_length=60)
    anchor_json: Optional[str] = None


class ClinicalNoteUpdateIn(BaseModel):
    texto: Optional[str] = Field(default=None, min_length=1, max_length=10000)
    segmento: Optional[str] = Field(default=None, max_length=60)
    anchor_json: Optional[str] = None


class ClinicalNoteOut(BaseModel):
    id_note: int
    id_paciente: int
    id_medico: int
    segmento: str
    texto: str
    anchor_json: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# --------------------
# Jobs (DB) + finalize
# --------------------
class JobConvOut(BaseModel):
    job_id: str
    id_usuario: int
    status: Literal["QUEUED","RUNNING","DONE","ERROR","CANCELED"]
    enable_ortopedia: bool
    enable_appendicular: bool
    enable_muscles: bool
    enable_skull: bool
    enable_teeth: bool
    enable_hip_implant: bool
    queue_name: Optional[str]
    started_at: Optional[str]
    finished_at: Optional[str]
    updated_at: Optional[str]

    class Config:
        from_attributes = True

class FinalizeJobIn(BaseModel):
    id_paciente: Optional[int] = None
    stl_size: Optional[int] = None
    num_stl_archivos: Optional[int] = None
    notas: Optional[str] = None

class JobSTLOut(BaseModel):
    id_jobstl: int
    job_id: str
    id_paciente: Optional[int] = None
    stl_size: Optional[int]
    num_stl_archivos: Optional[int]
    notas: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]

    class Config:
        from_attributes = True


# --------------------
# Doctor
# --------------------
class DoctorIn(BaseModel):
    nombre: str = Field(min_length=1, max_length=100)
    apellido: str = Field(min_length=1, max_length=100)
    correo: EmailStr
    password: str = Field(min_length=8, max_length=128)
    telefono: Optional[str] = Field(default=None, max_length=20)
    direccion: Optional[str] = Field(default=None, max_length=255)
    ciudad: Optional[str] = Field(default=None, max_length=100)
    id_hospital: Optional[int] = None
    referenciado: Optional[bool] = False
    activo: Optional[bool] = True

    @field_validator("nombre", "apellido", mode="before")
    @classmethod
    def _validate_names(cls, v):
        s = _clean_name(v, 100)
        if not _RE_NAME.fullmatch(s):
            raise ValueError("nombre/apellido invalido")
        return s

    @field_validator("telefono", mode="before")
    @classmethod
    def _validate_phone(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_phone(raw)
        if not _RE_PHONE.fullmatch(s):
            raise ValueError("telefono invalido")
        return s

    @field_validator("direccion", mode="before")
    @classmethod
    def _validate_address(cls, v):
        raw = _blank_to_none(v)
        return _clean_text(raw, 255) if raw else None

    @field_validator("ciudad", mode="before")
    @classmethod
    def _validate_city(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_city(raw, 100)
        if not _RE_CITY.fullmatch(s):
            raise ValueError("ciudad invalida")
        return s


class DoctorUpdateIn(BaseModel):
    nombre: Optional[str] = Field(default=None, max_length=100)
    apellido: Optional[str] = Field(default=None, max_length=100)
    correo: Optional[EmailStr] = None
    password: Optional[str] = Field(default=None, min_length=8, max_length=128)
    telefono: Optional[str] = Field(default=None, max_length=20)
    direccion: Optional[str] = Field(default=None, max_length=255)
    ciudad: Optional[str] = Field(default=None, max_length=100)
    id_hospital: Optional[int] = None
    referenciado: Optional[bool] = None
    activo: Optional[bool] = None
    estado: Optional[Literal["ACTIVO","INACTIVO"]] = None

    @field_validator("nombre", "apellido", mode="before")
    @classmethod
    def _validate_names(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_name(raw, 100)
        if not _RE_NAME.fullmatch(s):
            raise ValueError("nombre/apellido invalido")
        return s

    @field_validator("telefono", mode="before")
    @classmethod
    def _validate_phone(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_phone(raw)
        if not _RE_PHONE.fullmatch(s):
            raise ValueError("telefono invalido")
        return s

    @field_validator("direccion", mode="before")
    @classmethod
    def _validate_address(cls, v):
        raw = _blank_to_none(v)
        return _clean_text(raw, 255) if raw else None

    @field_validator("ciudad", mode="before")
    @classmethod
    def _validate_city(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_city(raw, 100)
        if not _RE_CITY.fullmatch(s):
            raise ValueError("ciudad invalida")
        return s


class DoctorOut(BaseModel):
    id_medico: int
    id_usuario: int
    nombre: str
    apellido: str
    correo: str
    telefono: Optional[str]
    direccion: Optional[str]
    ciudad: Optional[str]
    id_hospital: Optional[int]
    referenciado: bool
    estado: Literal["ACTIVO","INACTIVO"]
    activo: bool

    class Config:
        from_attributes = True


class AdminCreateIn(BaseModel):
    nombre: str = Field(min_length=1, max_length=100)
    apellido: str = Field(min_length=1, max_length=100)
    correo: EmailStr
    password: str = Field(min_length=8, max_length=128)
    telefono: Optional[str] = Field(default=None, max_length=20)
    direccion: Optional[str] = Field(default=None, max_length=255)
    ciudad: Optional[str] = Field(default=None, max_length=100)
    activo: Optional[bool] = True

    @field_validator("nombre", "apellido", mode="before")
    @classmethod
    def _validate_names(cls, v):
        s = _clean_name(v, 100)
        if not _RE_NAME.fullmatch(s):
            raise ValueError("nombre/apellido invalido")
        return s

    @field_validator("telefono", mode="before")
    @classmethod
    def _validate_phone(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_phone(raw)
        if not _RE_PHONE.fullmatch(s):
            raise ValueError("telefono invalido")
        return s

    @field_validator("direccion", mode="before")
    @classmethod
    def _validate_address(cls, v):
        raw = _blank_to_none(v)
        return _clean_text(raw, 255) if raw else None

    @field_validator("ciudad", mode="before")
    @classmethod
    def _validate_city(cls, v):
        raw = _blank_to_none(v)
        if raw is None:
            return None
        s = _clean_city(raw, 100)
        if not _RE_CITY.fullmatch(s):
            raise ValueError("ciudad invalida")
        return s


# --------------------
# Som3D: Pacientes con STL
# --------------------
class PatientJobSTLOut(BaseModel):
    id_jobstl: int
    job_id: str
    id_paciente: Optional[int] = None
    nombres: Optional[str] = None
    apellidos: Optional[str] = None
    doc_numero: Optional[str] = None
    notas: Optional[str] = None
    created_at: Optional[str] = None

class JobSTLNoteUpdateIn(BaseModel):
    notas: Optional[str] = None



# --------------------
# VisorEstado
# --------------------
class VisorEstadoIn(BaseModel):
    id_paciente: int
    id_jobstl: Optional[int] = None
    titulo: str = Field(min_length=1, max_length=200)
    descripcion: Optional[str] = Field(default=None, max_length=400)
    ui_json: str
    modelos_json: str
    notas_json: str
    i18n_json: str

    @field_validator("titulo", mode="before")
    @classmethod
    def _validate_title(cls, v):
        s = _clean_text(v, 200)
        if len(s) < 1:
            raise ValueError("titulo invalido")
        return s

    @field_validator("descripcion", mode="before")
    @classmethod
    def _validate_description(cls, v):
        raw = _blank_to_none(v)
        return _clean_text(raw, 400) if raw else None

class VisorEstadoOut(BaseModel):
    id_visor_estado: int
    id_medico: int
    id_paciente: int
    id_jobstl: Optional[int]
    titulo: str
    descripcion: Optional[str]
    ui_json: str
    modelos_json: str
    notas_json: str
    i18n_json: str

    class Config:
        from_attributes = True

# --------------------
# Mensajes
# --------------------
class MensajeBase(BaseModel):
    id_medico: int
    id_paciente: Optional[int] = None
    tipo: str
    titulo: str
    descripcion: str
    severidad: str
    adjunto_url: Optional[str] = None
    estado: str
    respuesta_admin: Optional[str] = None
    leido_admin: bool
    leido_medico: bool

class MensajeOut(MensajeBase):
    id_mensaje: int
    creado_en: Optional[datetime] = None
    actualizado_en: Optional[datetime] = None
    model_config = ConfigDict(from_attributes=True)  # <- correcto en v2

class MensajeList(BaseModel):
    total: int
    items: List[MensajeOut]
