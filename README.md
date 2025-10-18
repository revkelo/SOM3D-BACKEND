
SOM3D Backend (FastAPI + MySQL + ePayco)

Resumen
- API en FastAPI para autenticación, planes y suscripciones con ePayco.
- ORM con SQLAlchemy 2.x y conexión MySQL vía PyMySQL.
- JWT para auth; variables de entorno en `.env`.
- Dockerfile y docker-compose para desarrollo local.

Estructura
- `app/main.py`: aplicación FastAPI y rutas montadas.
- `app/routes/*`: endpoints de `auth`, `plans`, `subscriptions`, `epayco`, `patients`, `studies`, `visor`, `som3d`.
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
- `GET /plans/{id}`
- `POST /plans` (Bearer ADMIN)
- `PATCH /plans/{id}` (Bearer ADMIN)
- `DELETE /plans/{id}` (Bearer ADMIN; borra suscripciones y pagos del plan)
- `POST /subscriptions/start` (Bearer)
- `GET /subscriptions/mine` (Bearer)
- `GET /subscriptions` (Bearer ADMIN; filtros por id_medico/id_hospital/estado/plan_id)
- `GET /subscriptions/{id}` (Bearer ADMIN)
- `PATCH /subscriptions/{id}` (Bearer ADMIN; cambiar estado ACTIVA/PAUSADA con validación de conflicto)
- `DELETE /subscriptions/{id}` (Bearer ADMIN; borra pagos primero)
- `GET /subscriptions/{id}/payments` (Bearer ADMIN)
- `GET /hospitals` (Bearer; lista activos; admin puede incluir todos)
- `GET /hospitals/{id}` (Bearer)
- `GET /hospitals/by-code/{codigo}` (público)
- `POST /hospitals/link-by-code` (Bearer MEDICO; vincula por código y activa la cuenta)
- `POST /hospitals` (Bearer ADMIN)
- `PATCH /hospitals/{id}` (Bearer ADMIN)
- `DELETE /hospitals/{id}` (Bearer ADMIN; borra pagos y suscripciones asociadas y desasocia médicos)
- `GET /epayco/response` (HTML)
- `GET /epayco/validate?ref_payco=...`
- `GET|POST /epayco/confirmation`
- `GET /patients` | `POST /patients` | `GET/PATCH/DELETE /patients/{id}` (Bearer MEDICO/Admin)
- `GET /studies` | `POST /studies` | `GET /studies/{id}` (Bearer MEDICO/Admin)
- `POST /visor/states` | `GET /visor/states` | `GET/DELETE /visor/states/{id}` (Bearer MEDICO/Admin lectura)
- `POST /som3d/jobs` (requiere Bearer; sube ZIP DICOM; opcional `id_paciente`)
- `GET /som3d/jobs` | `GET /som3d/jobs/{id}` | `GET /som3d/jobs/{id}/progress` | `GET /som3d/jobs/{id}/stls` | `GET /som3d/jobs/{id}/result`
- `POST /som3d/jobs/{id}/finalize` (requiere Bearer; registra JobSTL con `id_paciente` y métricas)

Notas
- Este repo incluía `requirements.txt` con dependencias 3D no usadas por este API. Para el backend usa `requirements-backend.txt`.
- Si ya tienes una base MySQL externa, ajusta `DB_HOST`, `DB_PORT`, etc. y puedes omitir el servicio `db` en compose.

Datos y SQL
- Coloca el volcado completo en `data/casaos.sql` o impórtalo directo en tu MySQL. Consulta `data/README.md`.


para el ec2 el env
DB_USER=tu_usuario
DB_PASS=tu_password
DB_HOST=tu-endpoint-rds.amazonaws.com
DB_PORT=3306
DB_NAME=tu_base
JWT_SECRET=un-secreto-fuerte-y-largo
JWT_ALG=HS256
JWT_EXPIRE_MINUTES=60
EPAYCO_PUBLIC_KEY=tu_public_key
EPAYCO_TEST=false
BASE_URL=https://api.tu-dominio.com
P_CUST_ID_CLIENTE=tu_id_cliente
P_KEY=tu_p_key
USE_NGROK_SKIP=false
 
 
Admin (dashboard)
- `GET /admin/hospitals/generate-code` (Bearer ADMIN) — genera un código único disponible
- `GET /admin/hospitals` | `GET /admin/hospitals/{id}` | `POST /admin/hospitals` | `PATCH /admin/hospitals/{id}` | `DELETE /admin/hospitals/{id}` (Bearer ADMIN; en POST el código se genera automáticamente si no se envía)
- `GET /admin/doctors` | `PATCH /admin/doctors/{id}` | `DELETE /admin/doctors/{id}` (Bearer ADMIN)
- `GET /admin/doctors/count` (Bearer ADMIN) — admite `hospital_id` para filtrar
- `GET /admin/metrics` (Bearer ADMIN) — totales de hospitales, médicos, pacientes, estudios, suscripciones y pagos; incluye distribución de médicos por hospital
