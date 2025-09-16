import os
from dotenv import load_dotenv

load_dotenv()

def mysql_url() -> str:
    url = os.getenv("MYSQL_URL")
    if url:
        return url
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    user = os.getenv("DB_USER", "root")
    pwd  = os.getenv("DB_PASS", "")
    db   = os.getenv("DB_NAME", "casaos")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}"

JWT_SECRET = os.getenv("JWT_SECRET", "LeonardoDaVinci")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))

# ePayco
EPAYCO_PUBLIC_KEY = os.getenv("EPAYCO_PUBLIC_KEY", "")
EPAYCO_TEST = os.getenv("EPAYCO_TEST", "true").lower() == "true"
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
P_CUST_ID_CLIENTE = os.getenv("P_CUST_ID_CLIENTE", "")
P_KEY = os.getenv("P_KEY", "")