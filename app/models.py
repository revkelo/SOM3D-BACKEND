from sqlalchemy import (
    Column, Integer, String, Enum, Boolean, TIMESTAMP, text, DateTime, DECIMAL, ForeignKey
)
from sqlalchemy.orm import relationship, Mapped, mapped_column
from .db import Base

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