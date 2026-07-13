import os
import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
model = SentenceTransformer('BAAI/bge-large-en')

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # characters, not words — roughly 150-200 tokens
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", " ", ""]
)

def chunk_text(text):
    sections = text.split('\n## ')
    all_chunks = []
    for i, section in enumerate(sections):
        if i > 0:
            section = '## ' + section
        all_chunks.extend(splitter.split_text(section))
    return [c for c in all_chunks if c.strip()]

def ingest_file(service_id, filepath):
    conn = psycopg2.connect(os.getenv("DATABASE_URL"))
    cur = conn.cursor()

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    chunks = chunk_text(text)
    filename = os.path.basename(filepath)

    for chunk in chunks:
        embedding = model.encode(chunk).tolist()
        cur.execute(
            """INSERT INTO service_chunks (service_id, document_name, content, embedding)
               VALUES (%s, %s, %s, %s)""",
            (service_id, filename, chunk, embedding)
        )

    cur.execute(
        """INSERT INTO ingestion_log (service_id, document_count, chunk_count)
           VALUES (%s, %s, %s)""",
        (service_id, 1, len(chunks))
    )

    conn.commit()
    conn.close()
    print(f"{service_id}: {len(chunks)} chunks ingested from {filename}")

if __name__ == "__main__":
    ingest_file("nadra", "../data/NADRA.txt")
    ingest_file("fbr", "../data/FBR.txt")
    ingest_file("secp", "../data/SECP.txt")
    ingest_file("passport", "../data/Passport.txt")
    ingest_file("driving-license", "../data/DrivingLicense.txt")