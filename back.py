from flask import Flask, request, jsonify, send_file
from flasgger import Swagger
import mysql.connector
import io

app = Flask(__name__)
swagger = Swagger(app)

# Conexión DB
db = mysql.connector.connect(
    host="192.168.196.168",
    user="casaos",
    password="casaos",
    database="casaos"
)
cursor = db.cursor(dictionary=True)

# Endpoint raíz
@app.route('/')
def home():
    return jsonify({
        "msg": "API Flask corriendo con Swagger",
        "docs": "/apidocs",
        "endpoints": ["/upload", "/pacientes", "/imagen/<id>"]
    })

# Subir imagen como BLOB
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Subir una imagen de paciente
    ---
    consumes:
      - multipart/form-data
    parameters:
      - name: nombre
        in: formData
        type: string
        required: true
      - name: imagen
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Imagen guardada exitosamente en la base de datos
    """
    nombre = request.form['nombre']
    file = request.files['imagen']

    if file:
        imagen_bytes = file.read()
        cursor.execute("INSERT INTO pacientes (nombre, imagen) VALUES (%s, %s)", (nombre, imagen_bytes))
        db.commit()
        return jsonify({"msg": "Imagen subida con éxito"})
    return jsonify({"error": "No se recibió archivo"}), 400

# Listar pacientes
@app.route('/pacientes', methods=['GET'])
def get_pacientes():
    """
    Listar pacientes registrados
    ---
    responses:
      200:
        description: Lista de pacientes
        schema:
          type: array
          items:
            type: object
            properties:
              id:
                type: integer
              nombre:
                type: string
    """
    cursor.execute("SELECT id, nombre FROM pacientes")
    pacientes = cursor.fetchall()
    return jsonify(pacientes)

# Descargar imagen desde DB
@app.route('/imagen/<int:id>', methods=['GET'])
def get_imagen(id):
    """
    Obtener imagen de un paciente
    ---
    parameters:
      - name: id
        in: path
        type: integer
        required: true
    responses:
      200:
        description: Imagen en binario
        content:
          image/png: {}
    """
    cursor.execute("SELECT imagen FROM pacientes WHERE id=%s", (id,))
    result = cursor.fetchone()
    if result and result['imagen']:
        return send_file(io.BytesIO(result['imagen']), mimetype='image/png')
    return jsonify({"error": "Imagen no encontrada"}), 404

if __name__ == '__main__':
    app.run(debug=True)
