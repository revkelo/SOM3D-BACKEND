from flask import Blueprint, request, Response
from db import cursor, db
import json  # 👈 Para serializar JSON manualmente

hospital_bp = Blueprint("hospital", __name__)

# -------------------------------
# Crear hospital
# -------------------------------
@hospital_bp.route('/', methods=['POST'])
def create_hospital():
    """
    Crear hospital
    ---
    tags: [Hospital]
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [nombre]
            properties:
              nombre: {type: string, example: "Clínica Central"}
              direccion: {type: string, example: "Calle 123 #45-67"}
              ciudad: {type: string, example: "Bogotá"}
              telefono: {type: string, example: "+57 3001234567"}
              correo_contacto: {type: string, example: "contacto@clinica.com"}
    responses:
      201:
        description: Creado
    """
    data = request.json
    query = """
        INSERT INTO Hospital (nombre, direccion, ciudad, telefono, correo_contacto)
        VALUES (%s, %s, %s, %s, %s)
    """
    cursor.execute(query, (
        data.get('nombre'),
        data.get('direccion'),
        data.get('ciudad'),
        data.get('telefono'),
        data.get('correo_contacto')
    ))
    db.commit()

    new_id = cursor.lastrowid
    return Response(
        json.dumps({"msg": "Hospital creado con éxito", "id_hospital": new_id},
                   ensure_ascii=False, sort_keys=False, indent=2),
        mimetype="application/json",
        status=201
    )


# -------------------------------
# Listar hospitales
# -------------------------------
@hospital_bp.route('/', methods=['GET'])
def get_hospitales():
    """
    Listar hospitales en el orden de la tabla
    ---
    tags: [Hospital]
    responses:
      200:
        description: OK
    """
    cursor.execute("""
        SELECT id_hospital, nombre, direccion, ciudad, telefono, correo_contacto
        FROM Hospital
        ORDER BY id_hospital ASC
    """)
    rows = cursor.fetchall()

    hospitales = []
    for row in rows:
        hospitales.append({
            "id_hospital": row["id_hospital"],
            "nombre": row["nombre"],
            "direccion": row["direccion"],
            "ciudad": row["ciudad"],
            "telefono": row["telefono"],
            "correo_contacto": row["correo_contacto"]
        })

    return Response(
        json.dumps({"hospitales": hospitales}, ensure_ascii=False, sort_keys=False, indent=2),
        mimetype="application/json"
    )


# -------------------------------
# Obtener hospital por ID
# -------------------------------
@hospital_bp.route('/<int:id_hospital>', methods=['GET'])
def get_hospital(id_hospital):
    """
    Obtener hospital por ID
    ---
    tags: [Hospital]
    parameters:
      - in: path
        name: id_hospital
        schema: {type: integer}
        required: true
    responses:
      200: {description: OK}
      404: {description: No encontrado}
    """
    cursor.execute("""
        SELECT id_hospital, nombre, direccion, ciudad, telefono, correo_contacto
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
            "correo_contacto": row["correo_contacto"]
        }
        return Response(
            json.dumps({"hospital": hospital}, ensure_ascii=False, sort_keys=False, indent=2),
            mimetype="application/json"
        )

    return Response(
        json.dumps({"error": "Hospital no encontrado"}, ensure_ascii=False, sort_keys=False, indent=2),
        mimetype="application/json",
        status=404
    )


@hospital_bp.route('/<int:id_hospital>', methods=['PUT'])
def update_hospital(id_hospital):
    """
    Actualizar hospital
    ---
    tags: [Hospital]
    parameters:
      - in: path
        name: id_hospital
        schema: {type: integer}
        required: true
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              nombre: {type: string, example: "Clínica Actualizada"}
              direccion: {type: string, example: "Calle 99 #45-67"}
              ciudad: {type: string, example: "Bogotá"}
              telefono: {type: string, example: "6015555555"}
              correo_contacto: {type: string, example: "nuevo@clinica.com"}
    responses:
      200: {description: Actualizado}
      404: {description: No encontrado}
    """
    # 1. Verificar si existe
    cursor.execute("SELECT id_hospital FROM Hospital WHERE id_hospital=%s", (id_hospital,))
    row = cursor.fetchone()
    if not row:
        return Response(
            json.dumps({"error": "Hospital no encontrado"}, ensure_ascii=False, sort_keys=False, indent=2),
            mimetype="application/json",
            status=404
        )

    # 2. Actualizar
    data = request.json
    query = """
        UPDATE Hospital 
        SET nombre=%s, direccion=%s, ciudad=%s, telefono=%s, correo_contacto=%s
        WHERE id_hospital=%s
    """
    cursor.execute(query, (
        data.get('nombre'),
        data.get('direccion'),
        data.get('ciudad'),
        data.get('telefono'),
        data.get('correo_contacto'),
        id_hospital
    ))
    db.commit()

    # 3. Devolver hospital ya actualizado
    cursor.execute("""
        SELECT id_hospital, nombre, direccion, ciudad, telefono, correo_contacto
        FROM Hospital WHERE id_hospital=%s
    """, (id_hospital,))
    updated = cursor.fetchone()

    hospital = {
        "id_hospital": updated["id_hospital"],
        "nombre": updated["nombre"],
        "direccion": updated["direccion"],
        "ciudad": updated["ciudad"],
        "telefono": updated["telefono"],
        "correo_contacto": updated["correo_contacto"]
    }

    return Response(
        json.dumps({"msg": "Hospital actualizado", "hospital": hospital},
                   ensure_ascii=False, sort_keys=False, indent=2),
        mimetype="application/json"
    )


# -------------------------------
# Eliminar hospital
# -------------------------------
@hospital_bp.route('/<int:id_hospital>', methods=['DELETE'])
def delete_hospital(id_hospital):
    """
    Eliminar hospital
    ---
    tags: [Hospital]
    parameters:
      - in: path
        name: id_hospital
        schema: {type: integer}
        required: true
    responses:
      200: {description: Eliminado}
      404: {description: No encontrado}
    """
    cursor.execute("SELECT id_hospital FROM Hospital WHERE id_hospital=%s", (id_hospital,))
    row = cursor.fetchone()

    if not row:
        return Response(
            json.dumps({"error": "Hospital no encontrado"}, ensure_ascii=False, sort_keys=False, indent=2),
            mimetype="application/json",
            status=404
        )

    cursor.execute("DELETE FROM Hospital WHERE id_hospital=%s", (id_hospital,))
    db.commit()

    return Response(
        json.dumps({"msg": "Hospital eliminado", "id_hospital": id_hospital},
                   ensure_ascii=False, sort_keys=False, indent=2),
        mimetype="application/json"
    )

# -------------------------------
# Buscar hospitales
# -------------------------------
@hospital_bp.route('/search', methods=['GET'])
def search_hospitals():
    """
    Buscar hospitales por nombre o ciudad
    ---
    tags: [Hospital]
    parameters:
      - in: query
        name: q
        schema: {type: string}
        description: Texto a buscar (nombre o ciudad)
    responses:
      200: {description: Lista de hospitales encontrados}
    """
    q = request.args.get("q", "").strip()

    if q:
        cursor.execute("""
            SELECT id_hospital, nombre, direccion, ciudad, telefono, correo_contacto
            FROM Hospital
            WHERE nombre LIKE %s OR ciudad LIKE %s
            ORDER BY id_hospital ASC
        """, (f"%{q}%", f"%{q}%"))
    else:
        cursor.execute("""
            SELECT id_hospital, nombre, direccion, ciudad, telefono, correo_contacto
            FROM Hospital
            ORDER BY id_hospital ASC
        """)

    rows = cursor.fetchall()
    hospitales = []
    for row in rows:
        hospitales.append({
            "id_hospital": row["id_hospital"],
            "nombre": row["nombre"],
            "direccion": row["direccion"],
            "ciudad": row["ciudad"],
            "telefono": row["telefono"],
            "correo_contacto": row["correo_contacto"]
        })

    return Response(
        json.dumps({"hospitales": hospitales}, ensure_ascii=False, sort_keys=False, indent=2),
        mimetype="application/json"
    )
