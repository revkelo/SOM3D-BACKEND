import logging
import os
from typing import Any, Dict

from sqlalchemy import text

from app.db import engine
from app.services.s3_manager import S3Config, S3Manager

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse environment booleans in a tolerant way."""
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def check_database() -> Dict[str, Any]:
    """Verify the database connection by running a simple SELECT 1."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        logger.error("Database healthcheck failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def check_s3() -> Dict[str, Any]:
    """Verify S3/MinIO connectivity and bucket availability."""
    cfg = S3Config(
        endpoint=os.getenv("S3_ENDPOINT"),
        insecure=_env_bool("S3_INSECURE", False),
        bucket=os.getenv("S3_BUCKET", "som3d"),
        prefix=os.getenv("S3_PREFIX", "jobs/"),
        region=os.getenv("AWS_REGION", "us-east-1"),
        access_key=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    has_creds = bool(cfg.access_key and cfg.secret_key) or bool(
        os.getenv("AWS_PROFILE")
        or os.getenv("AWS_DEFAULT_PROFILE")
        or os.getenv("AWS_SHARED_CREDENTIALS_FILE")
    )
    if not has_creds:
        return {
            "status": "error",
            "error": "Missing S3 credentials (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS profile).",
        }
    if not cfg.bucket:
        return {"status": "error", "error": "Missing S3_BUCKET env var."}

    try:
        s3 = S3Manager(cfg)
        s3.ensure_bucket()
        return {
            "status": "ok",
            "bucket": cfg.bucket,
            "endpoint": cfg.endpoint,
        }
    except Exception as exc:
        logger.error("S3 healthcheck failed: %s", exc)
        return {"status": "error", "error": str(exc)}


def run_health_checks() -> Dict[str, Dict[str, Any]]:
    """Run all service checks and return a status map."""
    return {
        "database": check_database(),
        "storage": check_s3(),
    }


def ensure_services_ready() -> Dict[str, Dict[str, Any]]:
    """
    Run checks and raise an error if any dependency is unhealthy.
    FastAPI will abort startup if this raises.
    """
    results = run_health_checks()
    failed = [name for name, res in results.items() if res.get("status") != "ok"]
    if failed:
        raise RuntimeError(f"Unhealthy services on startup: {', '.join(failed)}")
    return results
