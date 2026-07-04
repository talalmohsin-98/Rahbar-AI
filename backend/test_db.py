import psycopg2
from dotenv import load_dotenv
import os

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()
cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
result = cur.fetchone()
print("pgvector enabled:", result is not None)
conn.close()