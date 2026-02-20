from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from ..core.config import mysql_url
import os
from datetime import datetime, date

from sqlalchemy import select, func

class Base(DeclarativeBase):
    pass

engine = create_engine(mysql_url(), pool_pre_ping=True, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_runtime_tables():
    # Crea todas las tablas si faltan (no migra/altera tablas existentes).
    # Importar modelos para registrar tablas en el metadata.
    from .. import models as _models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    if _truthy(os.getenv("DB_AUTO_SEED", "false")):
        _seed_minimal()


def _truthy(v: str | None) -> bool:
    return str(v or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _seed_minimal() -> None:
    # Seed idempotente: crea admin + planes base y (opcional) datos de prueba.
    from ..core.security import hash_password
    from ..models import Usuario, Plan, Medico, Paciente, Suscripcion, Pago

    admin_email = (os.getenv("SEED_ADMIN_EMAIL") or "admin@root.com").strip()
    admin_password = os.getenv("SEED_ADMIN_PASSWORD") or "Administrador.1"

    plan_seed = [
        ("Basico Mensual", 39900.00, "MENSUAL", 1),
        ("Profesional Mensual", 69900.00, "MENSUAL", 1),
        ("Profesional Trimestral", 179900.00, "TRIMESTRAL", 3),
        ("Empresarial Anual", 599900.00, "ANUAL", 12),
    ]

    db = SessionLocal()
    try:
        # Admin
        existing_admin = db.execute(
            select(Usuario).where(func.lower(Usuario.correo) == admin_email.lower())
        ).scalar_one_or_none()
        if not existing_admin:
            db.add(
                Usuario(
                    nombre="Admin",
                    apellido="Root",
                    correo=admin_email,
                    contrasena=hash_password(admin_password),
                    rol="ADMINISTRADOR",
                    activo=True,
                )
            )
            db.commit()

        # Planes base
        for nombre, precio, periodo, dur_meses in plan_seed:
            exists = db.execute(select(Plan).where(Plan.nombre == nombre)).scalar_one_or_none()
            if not exists:
                db.add(Plan(nombre=nombre, precio=precio, periodo=periodo, duracion_meses=dur_meses))
        db.commit()

        # Datos de prueba (medico + pacientes + suscripcion + pago)
        if _truthy(os.getenv("DB_AUTO_SEED_TEST", "false")):
            _seed_test_data(db)
            db.commit()
    finally:
        db.close()


def _seed_test_data(db) -> None:
    from dateutil.relativedelta import relativedelta

    from ..core.security import hash_password
    from ..models import Usuario, Plan, Medico, Paciente, Suscripcion, Pago

    medico_email = (os.getenv("SEED_TEST_MEDICO_EMAIL") or "medico.pruebas@som3d.local").strip()
    medico_password = os.getenv("SEED_TEST_MEDICO_PASSWORD") or "Medico.1"

    medico_user = db.execute(
        select(Usuario).where(func.lower(Usuario.correo) == medico_email.lower())
    ).scalar_one_or_none()
    if not medico_user:
        medico_user = Usuario(
            nombre="Medico",
            apellido="Pruebas",
            correo=medico_email,
            contrasena=hash_password(medico_password),
            telefono="3000000000",
            ciudad="Bogota",
            rol="MEDICO",
            activo=True,
        )
        db.add(medico_user)
        db.flush()

    medico = db.execute(select(Medico).where(Medico.id_usuario == medico_user.id_usuario)).scalar_one_or_none()
    if not medico:
        medico = Medico(id_usuario=medico_user.id_usuario, id_hospital=None, referenciado=False, estado="ACTIVO")
        db.add(medico)
        db.flush()

    # Pacientes (crea hasta 120 si faltan)
    existing_count = db.execute(select(func.count()).select_from(Paciente).where(Paciente.id_medico == medico.id_medico)).scalar_one()
    target = int(os.getenv("SEED_TEST_PACIENTES", "120"))
    to_create = max(0, target - int(existing_count or 0))
    if to_create:
        start_n = int(existing_count or 0) + 1
        today = date.today()
        for i in range(start_n, start_n + to_create):
            birth = date(today.year - (18 + (i % 50)), today.month, min(today.day, 28))
            db.add(
                Paciente(
                    id_medico=medico.id_medico,
                    doc_tipo="CC",
                    doc_numero=f"PRUEBA-{i:05d}",
                    nombres=f"Paciente {i:03d}",
                    apellidos="Carga",
                    fecha_nacimiento=birth,
                    sexo="M" if i % 2 == 0 else "F",
                    telefono=f"300{i:07d}",
                    correo=f"paciente{i:03d}@example.test",
                    direccion=f"Direccion {i}",
                    ciudad="Bogota",
                    estado="ACTIVO",
                )
            )

    # Suscripcion ACTIVA para el medico (si no hay)
    existing_active = db.execute(
        select(Suscripcion)
        .where(Suscripcion.id_medico == medico.id_medico, Suscripcion.estado == "ACTIVA")
        .order_by(Suscripcion.creado_en.desc())
    ).scalars().first()
    if not existing_active:
        plan = db.execute(select(Plan).where(Plan.nombre == "Profesional Mensual")).scalar_one_or_none()
        if plan:
            now = datetime.utcnow()
            sus = Suscripcion(
                id_medico=medico.id_medico,
                id_hospital=None,
                id_plan=plan.id_plan,
                fecha_inicio=now,
                fecha_expiracion=now + relativedelta(months=int(plan.duracion_meses)),
                estado="ACTIVA",
            )
            db.add(sus)
            db.flush()

            ref = f"SEED-PAGO-{sus.id_suscripcion}-{now.strftime('%Y%m%d%H%M%S')}"
            db.add(
                Pago(
                    id_suscripcion=sus.id_suscripcion,
                    referencia_epayco=ref,
                    monto=plan.precio,
                    fecha_pago=now,
                )
            )

