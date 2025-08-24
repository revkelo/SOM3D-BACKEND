import mysql.connector
from config import DB_CONFIG

db = mysql.connector.connect(**DB_CONFIG)
cursor = db.cursor(dictionary=True)
