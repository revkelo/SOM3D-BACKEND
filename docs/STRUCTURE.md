# Estructura del proyecto

- `app/main.py`: aplicación FastAPI y montaje de routers.
- `app/routes/*`: endpoints de `auth`, `plans`, `subscriptions`, `hospitals`.
- `app/api/epayco.py`: router de ePayco (re-export en `app/epayco.py`).
- `app/models.py`: modelos SQLAlchemy.
- `app/schemas.py`: modelos Pydantic (request/response).
- `app/db/`: paquete con `Base`, `engine`, y `get_db`.
- `app/core/config.py`: variables de entorno y utilidades (re-export en `app/config.py`).
- `app/core/security.py`: utilidades de auth/JWT (re-export en `app/auth.py`).
- `app/core/tokens.py`: generación/parsing de tokens auxiliares (re-export en `app/tokens.py`).
- `app/services/mailer.py`: servicio de correo (re-export en `app/mailer.py`).

Notas
- Se mantienen archivos "shim" en la raíz de `app` para compatibilidad de imports existentes.
- Puedes migrar gradualmente los imports a los nuevos módulos si lo prefieres.
