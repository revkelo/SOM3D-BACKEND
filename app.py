# app.py
from flask import Flask, jsonify
from flasgger import Swagger
from flask_cors import CORS

# ⬇️ Usa este import si hospital.py está en la raíz del proyecto
from routes.hospital import hospital_bp
# Si realmente lo tienes en routes/hospital.py, cambia a:
# from routes.hospital import hospital_bp

app = Flask(__name__)

# Flasgger usa Swagger 2.0 (no OpenAPI 3). Deja solo title y uiversion.
app.config['SWAGGER'] = {
    "title": "Hospital API",
    "uiversion": 3
}

# 🔓 CORS para tu front (Live Server, etc. en 5500)
CORS(
    app,
    resources={r"/*": {"origins": ["http://127.0.0.1:5500", "http://localhost:5500","http://localhost:5173"]}},
    supports_credentials=False,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

swagger = Swagger(app)

# Registra blueprint bajo /hospital
app.register_blueprint(hospital_bp, url_prefix="/hospital")

@app.route('/')
def home():
    return jsonify({
        "msg": "API Flask corriendo",
        "docs": "/apidocs",
        "endpoints": ["/hospital"]
    })

if __name__ == '__main__':
    app.run(debug=True)
