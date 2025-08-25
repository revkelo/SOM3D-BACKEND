# app.py
from flask import Flask, jsonify
from flasgger import Swagger
from flask_cors import CORS
import socket
import psutil  # pip install psutil  (para listar IPs ordenadas)
from routes.hospital import hospital_bp

app = Flask(__name__)

app.config['SWAGGER'] = {
    "title": "Hospital API",
    "uiversion": 3
}

# CORS: durante desarrollo, permite todo (ajusta en prod)
CORS(
    app,
    resources={r"/*": {"origins": "*"}},
    supports_credentials=False,
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"]
)

swagger = Swagger(app)
app.register_blueprint(hospital_bp, url_prefix="/hospital")

@app.route('/')
def home():
    return jsonify({
        "msg": "API Flask corriendo",
        "docs": "/apidocs",
        "endpoints": ["/hospital"]
    })

def print_urls(port: int):
    # Imprime estilo Vite: Local y Network
    try:
        hostname = socket.gethostname()
        try:
            local_name = socket.getfqdn()
        except:
            local_name = hostname
        print()
        print(f"➜  Local:   http://localhost:{port}/")
        print(f"➜  Local:   http://127.0.0.1:{port}/")
        # Recorre interfaces y muestra IPv4
        for ni in psutil.net_if_addrs().values():
            for addr in ni:
                if addr.family == socket.AF_INET:
                    ip = addr.address
                    print(f"➜  Network: http://{ip}:{port}/")
        print()
        print(f"Docs Swagger: http://localhost:{port}/apidocs")
    except Exception as e:
        print(f"[warn] No se pudieron listar las IPs: {e}")

if __name__ == '__main__':
    PORT = 5000
    print_urls(PORT)
    # Escucha en todas las interfaces
    app.run(host="0.0.0.0", port=PORT, debug=True)
