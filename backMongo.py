# app.py
from flask import Flask, request, jsonify, Response
from pymongo import MongoClient
from bson import ObjectId, errors as bson_errors
import gridfs
from flasgger import Swagger
from werkzeug.utils import secure_filename

app = Flask(__name__)
swagger = Swagger(app)

# ==== Configuración MongoDB ====
MONGO_URI = "mongodb://192.168.0.29:27017/"
DB_NAME = "SOM3D"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
pacientes_col = db["pacientes"]
fs = gridfs.GridFS(db)


# ==== Utilidades ====
def to_str_id(doc):
    """Convierte _id a str para respuestas JSON."""
    if not doc:
        return doc
    d = dict(doc)
    if "_id" in d and isinstance(d["_id"], ObjectId):
        d["_id"] = str(d["_id"])
    return d


# ==== Rutas ====
@app.route("/")
def health():
    """
    Estado de la API
    ---
    tags:
      - Sistema
    responses:
      200:
        description: La API está viva
    """
    return "API Flask + MongoDB GridFS + Swagger para SOM3D 🚀"


# ---------- Pacientes ----------
@app.route("/pacientes", methods=["POST"])
def crear_paciente():
    """
    Crear paciente
    ---
    tags:
      - Pacientes
    consumes:
      - application/json
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required: [nombre]
          properties:
            nombre:
              type: string
            edad:
              type: integer
            diagnostico:
              type: string
            notas:
              type: string
    responses:
      200:
        description: Paciente creado
    """
    data = request.get_json(force=True, silent=True) or {}
    if "nombre" not in data:
        return jsonify({"error": "El campo 'nombre' es requerido"}), 400

    result = pacientes_col.insert_one({
        "nombre": data.get("nombre"),
        "edad": data.get("edad"),
        "diagnostico": data.get("diagnostico"),
        "notas": data.get("notas")
    })
    return jsonify({"mensaje": "Paciente creado", "id": str(result.inserted_id)})


@app.route("/pacientes", methods=["GET"])
def listar_pacientes():
    """
    Listar pacientes
    ---
    tags:
      - Pacientes
    responses:
      200:
        description: Lista de pacientes
    """
    docs = [to_str_id(doc) for doc in pacientes_col.find({}).sort("_id", -1)]
    return jsonify(docs)


@app.route("/pacientes/<id>", methods=["GET"])
def obtener_paciente(id):
    """
    Obtener paciente por ID
    ---
    tags:
      - Pacientes
    parameters:
      - in: path
        name: id
        type: string
        required: true
    responses:
      200:
        description: Paciente
      404:
        description: No encontrado
    """
    try:
        doc = pacientes_col.find_one({"_id": ObjectId(id)})
    except bson_errors.InvalidId:
        return jsonify({"error": "ID inválido"}), 400

    if not doc:
        return jsonify({"error": "Paciente no encontrado"}), 404
    return jsonify(to_str_id(doc))


# ---------- Archivos STL (GridFS) ----------
@app.route("/stl", methods=["POST"])
def subir_stl():
    """
    Subir archivo STL (GridFS)
    ---
    tags:
      - Archivos STL
    consumes:
      - multipart/form-data
    parameters:
      - in: formData
        name: file
        type: file
        required: true
        description: Archivo .stl
      - in: formData
        name: paciente_id
        type: string
        required: false
        description: ObjectId del paciente relacionado
      - in: formData
        name: descripcion
        type: string
        required: false
    responses:
      200:
        description: Archivo almacenado en GridFS
    """
    if "file" not in request.files:
        return jsonify({"error": "Falta campo 'file'"}), 400

    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    filename = secure_filename(f.filename)
    paciente_id = request.form.get("paciente_id")
    descripcion = request.form.get("descripcion")

    metadata = {"tipo": "stl"}
    if descripcion:
        metadata["descripcion"] = descripcion
    if paciente_id:
        try:
            # valida que exista el paciente si enviaron el id
            pid = ObjectId(paciente_id)
            if not pacientes_col.find_one({"_id": pid}):
                return jsonify({"error": "paciente_id no existe"}), 400
            metadata["paciente_id"] = pid
        except bson_errors.InvalidId:
            return jsonify({"error": "paciente_id inválido"}), 400

    file_id = fs.put(
        f.stream,               # stream del archivo
        filename=filename,
        content_type="model/stl",
        metadata=metadata
    )
    return jsonify({"mensaje": "STL guardado", "file_id": str(file_id), "filename": filename})


@app.route("/stl/<file_id>", methods=["GET"])
def descargar_stl_por_id(file_id):
    """
    Descargar STL por file_id
    ---
    tags:
      - Archivos STL
    parameters:
      - in: path
        name: file_id
        type: string
        required: true
    responses:
      200:
        description: Devuelve el STL
      404:
        description: No encontrado
    """
    try:
        gridout = fs.get(ObjectId(file_id))
    except bson_errors.InvalidId:
        return jsonify({"error": "file_id inválido"}), 400
    except gridfs.NoFile:
        return jsonify({"error": "Archivo no encontrado"}), 404

    return Response(
        gridout.read(),
        mimetype=gridout.content_type or "application/sla",
        headers={"Content-Disposition": f'attachment; filename="{gridout.filename}"'}
    )


@app.route("/stl/by-name/<filename>", methods=["GET"])
def descargar_stl_por_nombre(filename):
    """
    Descargar STL por nombre (filename)
    ---
    tags:
      - Archivos STL
    parameters:
      - in: path
        name: filename
        type: string
        required: true
    responses:
      200:
        description: Devuelve el STL
      404:
        description: No encontrado
    """
    file = fs.find_one({"filename": filename})
    if not file:
        return jsonify({"error": "Archivo no encontrado"}), 404

    return Response(
        file.read(),
        mimetype=file.content_type or "application/sla",
        headers={"Content-Disposition": f'attachment; filename="{file.filename}"'}
    )


@app.route("/stl/by-paciente/<paciente_id>", methods=["GET"])
def listar_stl_por_paciente(paciente_id):
    """
    Listar STL por paciente_id
    ---
    tags:
      - Archivos STL
    parameters:
      - in: path
        name: paciente_id
        type: string
        required: true
    responses:
      200:
        description: Lista de archivos (solo metadatos)
    """
    try:
        pid = ObjectId(paciente_id)
    except bson_errors.InvalidId:
        return jsonify({"error": "paciente_id inválido"}), 400

    files = fs.find({"metadata.paciente_id": pid}).sort("uploadDate", -1)
    out = []
    for f in files:
        out.append({
            "file_id": str(f._id),
            "filename": f.filename,
            "length": f.length,
            "uploadDate": f.upload_date.isoformat(),
            "content_type": f.content_type,
            "metadata": {
                **({k: (str(v) if isinstance(v, ObjectId) else v) for k, v in (f.metadata or {}).items()})
            }
        })
    return jsonify(out)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
