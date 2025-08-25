from flask import Blueprint, request, Response
from db import cursor, db
import json
import re
import mysql.connector  # Para manejar errores específicos (ej. 1062)

hospital_bp = Blueprint("hospital", __name__)

# -------------------------------
# Helpers
# -------------------------------
def _json(data, status=200):
    return Response(
        json.dumps(data, ensure_ascii=False, sort_keys=False, indent=2),
        mimetype="application/json",
        status=status
    )

def _to_bool(v):
    # MariaDB/MySQL devuelve 0/1; también aceptamos True/False del payload
    if isinstance(v, bool):
        return v
    try:
        return bool(int(v))
    except Exception:
        return False

def _next_codigo():
    """
    Genera siguiente código con prefijo HOSP y 3 dígitos, según el máximo actual.
    Ej: HOSP001, HOSP002, ...
    """
    cursor.execute("""
        SELECT codigo
        FROM Hospital
        WHERE codigo REGEXP '^HOSP[0-9]+$'
        ORDER BY CAST(SUBSTRING(codigo, 5) AS UNSIGNED) DESC
        LIMIT 1
    """)
    row = cursor.fetchone()
    if not row or not row.get("codigo"):
        return "HOSP001"
    m = re.match(r"^HOSP(\d+)$", row["codigo"])
    if not m:
        return "HOSP001"
    n = int(m.group(1)) + 1
    return f"HOSP{n:03d}"


# -------------------------------
# Crear hospital
# -------------------------------
@hospital_bp.route('/', methods=['POST'])
def create_hospital():
    """
    Crear hospital
    ---
    tags:
      - Hospital
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - nombre
          properties:
            nombre: {type: string, example: "Clínica Central"}
            direccion: {type: string, example: "Calle 123 #45-67"}
            ciudad: {type: string, example: "Bogotá"}
            telefono: {type: string, example: "6011234567"}
            correo_contacto: {type: string, example: "contacto@clinica.com"}
            codigo: {type: string, example: "HOSP011", description: "Si no se envía, se autogenera"}
            activo: {type: boolean, example: true}
    responses:
      201:
        description: Hospital creado exitosamente
        schema:
          type: object
          properties:
            msg: {type: string, example: "Hospital creado con éxito"}
            id_hospital: {type: integer, example: 1}
            codigo: {type: string, example: "HOSP011"}
    """
    data = request.json or {}
    nombre = data.get('nombre')
    if not nombre:
        return _json({"error": "El campo 'nombre' es obligatorio"}, 400)

    codigo = data.get('codigo') or _next_codigo()
    activo = 1 if _to_bool(data.get('activo', True)) else 0

    try:
        query = """
            INSERT INTO Hospital (nombre, direccion, ciudad, telefono, correo_contacto, codigo, activo)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (
            nombre,
            data.get('direccion'),
            data.get('ciudad'),
            data.get('telefono'),
            data.get('correo_contacto'),
            codigo,
            activo
        ))
        db.commit()
        new_id = cursor.lastrowid
        return _json({"msg": "Hospital creado con éxito", "id_hospital": new_id, "codigo": codigo}, 201)

    except mysql.connector.Error as e:
        db.rollback()
        # 1062 = Duplicate entry
        if e.errno == 1062:
            return _json({"error": "Código duplicado", "detalle": str(e)}, 409)
        return _json({"error": "Error al crear hospital", "detalle": str(e)}, 500)
    except Exception as e:
        db.rollback()
        return _json({"error": "Error inesperado", "detalle": str(e)}, 500)


# -------------------------------
# Listar hospitales
# -------------------------------
@hospital_bp.route('/', methods=['GET'])
def get_hospitales():
    """
    Listar hospitales en el orden de la tabla
    ---
    tags:
      - Hospital
    produces:
      - application/json
    responses:
      200:
        description: OK
        schema:
          type: object
          properties:
            hospitales:
              type: array
              items:
                type: object
                properties:
                  id_hospital: {type: integer, example: 1}
                  nombre: {type: string, example: "Clínica Central"}
                  direccion: {type: string, example: "Calle 123 #45-67"}
                  ciudad: {type: string, example: "Bogotá"}
                  telefono: {type: string, example: "6011234567"}
                  correo_contacto: {type: string, example: "contacto@clinica.com"}
                  codigo: {type: string, example: "HOSP001"}
                  activo: {type: boolean, example: true}
    """
    cursor.execute("""
        SELECT id_hospital, nombre, direccion, ciudad, telefono, correo_contacto, codigo, activo
        FROM Hospital
        ORDER BY id_hospital ASC
    """)
    rows = cursor.fetchall() or []

    hospitales = []
    for row in rows:
        hospitales.append({
            "id_hospital": row["id_hospital"],
            "nombre": row["nombre"],
            "direccion": row["direccion"],
            "ciudad": row["ciudad"],
            "telefono": row["telefono"],
            "correo_contacto": row["correo_contacto"],
            "codigo": row.get("codigo"),
            "activo": bool(row.get("activo", 0)),
        })

    return _json({"hospitales": hospitales})


# -------------------------------
# Obtener hospital por ID
# -------------------------------
@hospital_bp.route('/<int:id_hospital>', methods=['GET'])
def get_hospital(id_hospital):
    """
    Obtener hospital por ID
    ---
    tags:
      - Hospital
    produces:
      - application/json
    parameters:
      - in: path
        name: id_hospital
        type: integer
        required: true
        description: ID del hospital
    responses:
      200:
        description: OK
        schema:
          type: object
          properties:
            hospital:
              type: object
              properties:
                id_hospital: {type: integer, example: 1}
                nombre: {type: string, example: "Clínica Central"}
                direccion: {type: string, example: "Calle 123 #45-67"}
                ciudad: {type: string, example: "Bogotá"}
                telefono: {type: string, example: "6011234567"}
                correo_contacto: {type: string, example: "contacto@clinica.com"}
                codigo: {type: string, example: "HOSP001"}
                activo: {type: boolean, example: true}
      404:
        description: No encontrado
        schema:
          type: object
          properties:
            error: {type: string, example: "Hospital no encontrado"}
    """
    cursor.execute("""
        SELECT id_hospital, nombre, direccion, ciudad, telefono, correo_contacto, codigo, activo
        FROM Hospital WHERE id_hospital=%s
    """, (id_hospital,))
    row = cursor.fetchone()

    if row:
        hospital = {
            "id_hospital": row["id_hospital"],
            "nombre": row["nombre"],
            "direccion": row["direccion"],
            "ciudad": row["ciudad"],
            "telefono": row["telefono"],
            "correo_contacto": row["correo_contacto"],
            "codigo": row.get("codigo"),
            "activo": bool(row.get("activo", 0)),
        }
        return _json({"hospital": hospital})

    return _json({"error": "Hospital no encontrado"}, 404)


# -------------------------------
# Actualizar hospital
# -------------------------------
@hospital_bp.route('/<int:id_hospital>', methods=['PUT'])
def update_hospital(id_hospital):
    """
    Actualizar hospital
    ---
    tags:
      - Hospital
    consumes:
      - application/json
    produces:
      - application/json
    parameters:
      - in: path
        name: id_hospital
        type: integer
        required: true
        description: ID del hospital
      - in: body
        name: body
        required: true
        schema:
          type: object
          properties:
            nombre: {type: string, example: "Clínica Actualizada"}
            direccion: {type: string, example: "Calle 99 #45-67"}
            ciudad: {type: string, example: "Bogotá"}
            telefono: {type: string, example: "6015555555"}
            correo_contacto: {type: string, example: "nuevo@clinica.com"}
            codigo: {type: string, example: "HOSP010"}
            activo: {type: boolean, example: true}
    responses:
      200:
        description: Actualizado
        schema:
          type: object
          properties:
            msg: {type: string, example: "Hospital actualizado"}
            hospital:
              type: object
              properties:
                id_hospital: {type: integer, example: 1}
                nombre: {type: string}
                direccion: {type: string}
                ciudad: {type: string}
                telefono: {type: string}
                correo_contacto: {type: string}
                codigo: {type: string}
                activo: {type: boolean}
      404:
        description: No encontrado
        schema:
          type: object
          properties:
            error: {type: string, example: "Hospital no encontrado"}
    """
    # 1. Existe
    cursor.execute("SELECT id_hospital FROM Hospital WHERE id_hospital=%s", (id_hospital,))
    if not cursor.fetchone():
        return _json({"error": "Hospital no encontrado"}, 404)

    data = request.json or {}

    try:
        query = """
            UPDATE Hospital 
            SET nombre=%s, direccion=%s, ciudad=%s, telefono=%s, correo_contacto=%s, codigo=%s, activo=%s
            WHERE id_hospital=%s
        """
        codigo = data.get('codigo')
        if not codigo:
            # Si no envían código, mantener el existente
            cursor.execute("SELECT codigo FROM Hospital WHERE id_hospital=%s", (id_hospital,))
            row = cursor.fetchone()
            codigo = row["codigo"] if row else None

        activo = data.get('activo')
        if activo is None:
            cursor.execute("SELECT activo FROM Hospital WHERE id_hospital=%s", (id_hospital,))
            row = cursor.fetchone()
            activo = row["activo"] if row else 1
        else:
            activo = 1 if _to_bool(activo) else 0

        cursor.execute(query, (
            data.get('nombre'),
            data.get('direccion'),
            data.get('ciudad'),
            data.get('telefono'),
            data.get('correo_contacto'),
            codigo,
            activo,
            id_hospital
        ))
        db.commit()

        # 3. Devolver hospital ya actualizado
        cursor.execute("""
            SELECT id_hospital, nombre, direccion, ciudad, telefono, correo_contacto, codigo, activo
            FROM Hospital WHERE id_hospital=%s
        """, (id_hospital,))
        updated = cursor.fetchone()

        hospital = {
            "id_hospital": updated["id_hospital"],
            "nombre": updated["nombre"],
            "direccion": updated["direccion"],
            "ciudad": updated["ciudad"],
            "telefono": updated["telefono"],
            "correo_contacto": updated["correo_contacto"],
            "codigo": updated.get("codigo"),
            "activo": bool(updated.get("activo", 0)),
        }
        return _json({"msg": "Hospital actualizado", "hospital": hospital})

    except mysql.connector.Error as e:
        db.rollback()
        if e.errno == 1062:
            return _json({"error": "Código duplicado", "detalle": str(e)}, 409)
        return _json({"error": "Error al actualizar hospital", "detalle": str(e)}, 500)
    except Exception as e:
        db.rollback()
        return _json({"error": "Error inesperado", "detalle": str(e)}, 500)


# -------------------------------
# Eliminar hospital
# -------------------------------
@hospital_bp.route('/<int:id_hospital>', methods=['DELETE'])
def delete_hospital(id_hospital):
    """
    Eliminar hospital
    ---
    tags:
      - Hospital
    produces:
      - application/json
    parameters:
      - in: path
        name: id_hospital
        type: integer
        required: true
        description: ID del hospital
    responses:
      200:
        description: Eliminado
        schema:
          type: object
          properties:
            msg: {type: string, example: "Hospital eliminado"}
            id_hospital: {type: integer, example: 1}
      404:
        description: No encontrado
        schema:
          type: object
          properties:
            error: {type: string, example: "Hospital no encontrado"}
    """
    cursor.execute("SELECT id_hospital FROM Hospital WHERE id_hospital=%s", (id_hospital,))
    if not cursor.fetchone():
        return _json({"error": "Hospital no encontrado"}, 404)

    cursor.execute("DELETE FROM Hospital WHERE id_hospital=%s", (id_hospital,))
    db.commit()

    return _json({"msg": "Hospital eliminado", "id_hospital": id_hospital})


# -------------------------------
# Buscar hospitales
# -------------------------------
@hospital_bp.route('/search', methods=['GET'])
def search_hospitals():
    """
    Buscar hospitales por nombre, ciudad o código
    ---
    tags:
      - Hospital
    produces:
      - application/json
    parameters:
      - in: query
        name: q
        type: string
        required: false
        description: Texto a buscar (nombre, ciudad o código)
      - in: query
        name: activo
        type: boolean
        required: false
        description: Filtrar por estado (true/false)
    responses:
      200:
        description: Lista de hospitales encontrados
        schema:
          type: object
          properties:
            hospitales:
              type: array
              items:
                type: object
                properties:
                  id_hospital: {type: integer, example: 1}
                  nombre: {type: string, example: "Clínica Central"}
                  direccion: {type: string, example: "Calle 123 #45-67"}
                  ciudad: {type: string, example: "Bogotá"}
                  telefono: {type: string, example: "6011234567"}
                  correo_contacto: {type: string, example: "contacto@clinica.com"}
                  codigo: {type: string, example: "HOSP001"}
                  activo: {type: boolean, example: true}
    """
    q = (request.args.get("q") or "").strip()
    activo_param = request.args.get("activo", "").strip().lower()

    params = []
    where = []
    if q:
        where.append("(nombre LIKE %s OR ciudad LIKE %s OR codigo LIKE %s)")
        like = f"%{q}%"
        params.extend([like, like, like])
    if activo_param in ("true", "false", "1", "0"):
        where.append("activo=%s")
        params.append(1 if activo_param in ("true", "1") else 0)

    sql = """
        SELECT id_hospital, nombre, direccion, ciudad, telefono, correo_contacto, codigo, activo
        FROM Hospital
    """
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id_hospital ASC"

    cursor.execute(sql, tuple(params))
    rows = cursor.fetchall() or []

    hospitales = []
    for row in rows:
        hospitales.append({
            "id_hospital": row["id_hospital"],
            "nombre": row["nombre"],
            "direccion": row["direccion"],
            "ciudad": row["ciudad"],
            "telefono": row["telefono"],
            "correo_contacto": row["correo_contacto"],
            "codigo": row.get("codigo"),
            "activo": bool(row.get("activo", 0)),
        })

    return _json({"hospitales": hospitales})
