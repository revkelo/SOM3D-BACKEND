from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from ..core.config import mysql_url

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
    # Solo crea tablas auxiliares faltantes sin tocar el resto del esquema.
    from ..models import AuthLoginAttempt, AuthRefreshSession, PaymentWebhookEvent, ClinicalNote

    Base.metadata.create_all(
        bind=engine,
        tables=[
            AuthLoginAttempt.__table__,
            AuthRefreshSession.__table__,
            PaymentWebhookEvent.__table__,
            ClinicalNote.__table__,
        ],
    )

