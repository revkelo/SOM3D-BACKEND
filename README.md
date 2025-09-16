
SOM3D Backend (FastAPI + MySQL + ePayco)

Resumen
- API en FastAPI para autenticación, planes y suscripciones con ePayco.
- ORM con SQLAlchemy 2.x y conexión MySQL vía PyMySQL.
- JWT para auth; variables de entorno en `.env`.
- Dockerfile y docker-compose para desarrollo local.

Estructura
- `app/main.py`: aplicación FastAPI y rutas montadas.
- `app/routes/*`: endpoints de `auth`, `plans`, `subscriptions`, `epayco`.
- `app/models.py`: modelos SQLAlchemy.
- `app/schemas.py`: modelos Pydantic (request/response).
- `app/db.py`: engine y sesión.
- `app/config.py`: variables de entorno y utilidades.

Requisitos (sin Docker)
- Python 3.11+ recomendado.
- MySQL 8.x accesible y credenciales válidas.

Instalación local (solo backend)
1) Crear y editar `.env` a partir de `.env.example`.
2) Crear entorno virtual y deps del backend:
   - `python -m venv .venv`
   - `./.venv/Scripts/activate` (Windows) o `source .venv/bin/activate` (Linux/Mac)
   - `pip install -r requirements-backend.txt`
3) Ejecutar la API:
   - `uvicorn app.main:app --reload --port 8000`

Docker
- Build de imagen: `docker build -t som3d-backend .`
- Run simple: `docker run --env-file .env -p 8000:8000 som3d-backend`

Docker Compose (API + MySQL)
1) Copiar `.env.example` a `.env` y ajustar valores. Para usar el MySQL del compose, no cambies `DB_HOST` (se inyecta como `db`).
2) Levantar servicios: `docker compose up --build`
3) API en `http://localhost:8000`, MySQL en `localhost:3306` (usuario/clave según `.env`).

Variables de entorno clave
- DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME
- JWT_SECRET, JWT_ALG, JWT_EXPIRE_MINUTES
- EPAYCO_PUBLIC_KEY, EPAYCO_TEST, BASE_URL, P_CUST_ID_CLIENTE, P_KEY

Puntos de salud y rutas
- `GET /health` → `{ "status": "ok" }`
- `POST /auth/register`
- `POST /auth/login`
- `GET /auth/me` (Bearer)
- `GET /plans`
- `POST /subscriptions/start` (Bearer)
- `GET /subscriptions/mine` (Bearer)
- `GET /epayco/response` (HTML)
- `GET /epayco/validate?ref_payco=...`
- `GET|POST /epayco/confirmation`

Notas
- Este repo incluía `requirements.txt` con dependencias 3D no usadas por este API. Para el backend usa `requirements-backend.txt`.
- Si ya tienes una base MySQL externa, ajusta `DB_HOST`, `DB_PORT`, etc. y puedes omitir el servicio `db` en compose.
