from sqlalchemy import (
    Column, Integer, String, Enum, Boolean, TIMESTAMP, text, DateTime, Date, DECIMAL, ForeignKey, BigInteger, Text
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from .db import Base
from datetime import datetime
from sqlalchemy import func

class Hospital(Base):
    __tablename__ = "Hospital"
    id_hospital: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nombre: Mapped[str] = mapped_column(String(150), nullable=False)
    direccion: Mapped[str | None] = mapped_column(String(255))
    ciudad: Mapped[str | None] = mapped_column(String(100))
    telefono: Mapped[str | None] = mapped_column(String(30))
    correo: Mapped[str | None] = mapped_column(String(100))
    codigo: Mapped[str] = mapped_column(String(12), unique=True, nullable=False)
    estado: Mapped[str] = mapped_column(Enum("ACTIVO","INACTIVO", name="estado_hospital"), server_default="ACTIVO")
    creado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    actualizado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

class Usuario(Base):
    __tablename__ = "Usuario"
    id_usuario: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nombre: Mapped[str] = mapped_column(String(100), nullable=False)
    apellido: Mapped[str] = mapped_column(String(100), nullable=False)
    correo: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    contrasena: Mapped[str] = mapped_column(String(255), nullable=False)
    telefono: Mapped[str | None] = mapped_column(String(20))
    direccion: Mapped[str | None] = mapped_column(String(255))
    ciudad: Mapped[str | None] = mapped_column(String(100))
    rol: Mapped[str] = mapped_column(Enum("ADMINISTRADOR","MEDICO", name="rol_usuario"), nullable=False)
    activo: Mapped[bool] = mapped_column(Boolean, server_default=text("1"))
    creado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    actualizado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

    medico: Mapped["Medico"] = relationship(back_populates="usuario", uselist=False)

class Medico(Base):
    __tablename__ = "Medico"
    id_medico: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_usuario: Mapped[int] = mapped_column(ForeignKey("Usuario.id_usuario"), nullable=False, unique=True)
    id_hospital: Mapped[int | None] = mapped_column(ForeignKey("Hospital.id_hospital"), nullable=True)
    referenciado: Mapped[bool] = mapped_column(Boolean, server_default=text("0"), nullable=False)
    estado: Mapped[str] = mapped_column(Enum("ACTIVO","INACTIVO", name="estado_medico"), server_default="ACTIVO")
    creado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    actualizado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

    usuario: Mapped["Usuario"] = relationship(back_populates="medico")
    hospital: Mapped["Hospital"] = relationship()

class Plan(Base):
    __tablename__ = "Plan"
    id_plan: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nombre: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    precio: Mapped[DECIMAL] = mapped_column(DECIMAL(12,2), nullable=False)
    periodo: Mapped[str] = mapped_column(Enum("MENSUAL","TRIMESTRAL","ANUAL", name="periodo_plan"), nullable=False)
    duracion_meses: Mapped[int] = mapped_column(Integer, nullable=False)
    creado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    actualizado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

class Suscripcion(Base):
    __tablename__ = "Suscripcion"
    id_suscripcion: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_medico: Mapped[int | None] = mapped_column(ForeignKey("Medico.id_medico"))
    id_hospital: Mapped[int | None] = mapped_column(ForeignKey("Hospital.id_hospital"))
    id_plan: Mapped[int] = mapped_column(ForeignKey("Plan.id_plan"), nullable=False)
    fecha_inicio: Mapped[str] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    fecha_expiracion: Mapped[str | None] = mapped_column(DateTime, nullable=True)
    estado: Mapped[str] = mapped_column(Enum("ACTIVA","PAUSADA", name="estado_suscripcion"), server_default="ACTIVA", nullable=False)
    creado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    actualizado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

class Pago(Base):
    __tablename__ = "Pago"
    id_pago: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_suscripcion: Mapped[int] = mapped_column(ForeignKey("Suscripcion.id_suscripcion"), nullable=False)
    referencia_epayco: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    monto: Mapped[DECIMAL] = mapped_column(DECIMAL(12,2), nullable=False)
    fecha_pago: Mapped[str] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    creado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    actualizado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class Paciente(Base):
    __tablename__ = "Paciente"
    id_paciente: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_medico: Mapped[int] = mapped_column(ForeignKey("Medico.id_medico"), nullable=False)
    doc_tipo: Mapped[str | None] = mapped_column(String(20))
    doc_numero: Mapped[str | None] = mapped_column(String(40))
    nombres: Mapped[str] = mapped_column(String(100), nullable=False)
    apellidos: Mapped[str] = mapped_column(String(100), nullable=False)
    fecha_nacimiento: Mapped[str | None] = mapped_column(Date)
    sexo: Mapped[str | None] = mapped_column(String(20))
    telefono: Mapped[str | None] = mapped_column(String(30))
    correo: Mapped[str | None] = mapped_column(String(120))
    direccion: Mapped[str | None] = mapped_column(String(200))
    ciudad: Mapped[str | None] = mapped_column(String(80))
    estado: Mapped[str] = mapped_column(Enum("ACTIVO","INACTIVO", name="estado_paciente"), server_default="ACTIVO")
    creado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    actualizado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class Estudio(Base):
    __tablename__ = "Estudio"
    id_estudio: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_paciente: Mapped[int] = mapped_column(ForeignKey("Paciente.id_paciente"), nullable=False)
    id_medico: Mapped[int] = mapped_column(ForeignKey("Medico.id_medico"), nullable=False)
    modalidad: Mapped[str | None] = mapped_column(String(20))
    fecha_estudio: Mapped[datetime] = mapped_column(DateTime, server_default=text("CURRENT_TIMESTAMP"), nullable=False)
    descripcion: Mapped[str | None] = mapped_column(String(200))
    notas: Mapped[str | None] = mapped_column(Text)
    creado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    actualizado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class JobConv(Base):
    __tablename__ = "JobConv"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    id_usuario: Mapped[int] = mapped_column(ForeignKey("Usuario.id_usuario"), nullable=False)
    status: Mapped[str] = mapped_column(Enum("QUEUED","RUNNING","DONE","ERROR","CANCELED", name="status_jobconv"), server_default="QUEUED", nullable=False)
    enable_ortopedia: Mapped[bool] = mapped_column(Boolean, server_default=text("1"), nullable=False)
    enable_appendicular: Mapped[bool] = mapped_column(Boolean, server_default=text("0"), nullable=False)
    enable_muscles: Mapped[bool] = mapped_column(Boolean, server_default=text("0"), nullable=False)
    enable_skull: Mapped[bool] = mapped_column(Boolean, server_default=text("0"), nullable=False)
    enable_teeth: Mapped[bool] = mapped_column(Boolean, server_default=text("0"), nullable=False)
    enable_hip_implant: Mapped[bool] = mapped_column(Boolean, server_default=text("0"), nullable=False)
    extra_tasks_json: Mapped[str | None] = mapped_column(Text)
    queue_name: Mapped[str | None] = mapped_column(String(80))
    started_at: Mapped[str | None] = mapped_column(DateTime)
    finished_at: Mapped[str | None] = mapped_column(DateTime)
    updated_at: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class JobSTL(Base):
    __tablename__ = "JobSTL"
    id_jobstl: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("JobConv.job_id"), nullable=False)
    id_paciente: Mapped[int] = mapped_column(ForeignKey("Paciente.id_paciente"), nullable=False)
    stl_size: Mapped[int | None] = mapped_column(BigInteger)
    num_stl_archivos: Mapped[int | None] = mapped_column(Integer)
    notas: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updated_at: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))


class VisorEstado(Base):
    __tablename__ = "VisorEstado"
    id_visor_estado: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    id_medico: Mapped[int] = mapped_column(ForeignKey("Medico.id_medico"), nullable=False)
    id_paciente: Mapped[int] = mapped_column(ForeignKey("Paciente.id_paciente"), nullable=False)
    id_jobstl: Mapped[int | None] = mapped_column(ForeignKey("JobSTL.id_jobstl"))
    titulo: Mapped[str] = mapped_column(String(200), nullable=False)
    descripcion: Mapped[str | None] = mapped_column(String(400))
    ui_json: Mapped[str] = mapped_column(Text, nullable=False)
    modelos_json: Mapped[str] = mapped_column(Text, nullable=False)
    notas_json: Mapped[str] = mapped_column(Text, nullable=False)
    i18n_json: Mapped[str] = mapped_column(Text, nullable=False)
    creado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    actualizado_en: Mapped[str] = mapped_column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP"))

class Mensaje(Base):
    __tablename__ = "mensaje"

    id_mensaje: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # relaciones
    id_medico: Mapped[int] = mapped_column(Integer, ForeignKey("Medico.id_medico"), nullable=False)
    medico = relationship("Medico", backref="mensajes")

    # opcional: si quieres enlazar a paciente (el front lo manda como opcional)
    id_paciente: Mapped[int | None] = mapped_column(Integer, ForeignKey("Paciente.id_paciente"), nullable=True)
    paciente = relationship("Paciente", backref="mensajes", foreign_keys=[id_paciente])

    # datos del mensaje
    tipo: Mapped[str] = mapped_column(String(30), nullable=False)         # 'error' | 'sugerencia'
    titulo: Mapped[str] = mapped_column(String(200), nullable=False)
    descripcion: Mapped[str] = mapped_column(Text, nullable=False)
    severidad: Mapped[str] = mapped_column(String(20), nullable=False)     # 'baja' | 'media' | 'alta'
    adjunto_url: Mapped[str | None] = mapped_column(String(500), nullable=True)

    # workflow
    estado: Mapped[str] = mapped_column(String(30), nullable=False, default="nuevo")
    respuesta_admin: Mapped[str | None] = mapped_column(Text, nullable=True)
    leido_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    leido_medico: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # tiempos
    creado_en: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    actualizado_en: Mapped[DateTime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
