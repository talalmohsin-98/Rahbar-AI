import os
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# 1. EMBEDDING MODEL
# ---------------------------------------------------------------------------
# "BAAI/bge-large-en" is a strong open-source embedding model.
# It converts any text into a list of 1024 numbers (a vector).
# Similar texts produce vectors that are close together in space.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en")

# We load the model once at module level.
# Why? Loading a model takes ~2-3 seconds. If we loaded it inside the function,
# every single query would pay that cost. Loading once = pay it only at startup.
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Returns the embedding model, loading it on first call (lazy loading).

    Lazy loading means: don't load until someone actually needs it.
    This keeps import time fast and lets tests mock it easily.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


# ---------------------------------------------------------------------------
# 2. DATABASE CONNECTION
# ---------------------------------------------------------------------------
def get_connection() -> psycopg2.extensions.connection:
    """
    Opens a connection to PostgreSQL (which has the pgvector extension).

    We read credentials from environment variables — NEVER hardcode passwords.
    In production this would use a connection pool (e.g. pgbouncer).
    For this project, a fresh connection per query is fine.
    """
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "rahbar"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )


# ---------------------------------------------------------------------------
# 3. EMBED A QUERY
# ---------------------------------------------------------------------------
def embed_query(text: str) -> list[float]:
    """
    Converts a text string into a vector (list of floats).

    Example:
        embed_query("What is NADRA?")
        → [0.021, -0.14, 0.83, ...]   # 1024 numbers

    The model was trained so that semantically similar texts
    produce vectors that are close together (high cosine similarity).
    """
    model = get_model()

    # encode() returns a numpy array; .tolist() converts to plain Python list.
    # pgvector expects a plain list when we pass it as a parameter.
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


# ---------------------------------------------------------------------------
# 4. DENSE RETRIEVAL
# ---------------------------------------------------------------------------
def dense_retrieve(query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """
    Given a query string, return the top_k most semantically similar chunks
    from the database, using pgvector cosine similarity search.

    Args:
        query:  The search query (could be original or rewritten)
        top_k:  How many results to return (default 20, reranker will trim later)

    Returns:
        List of dicts, each containing:
            - chunk_id:   unique ID of this text chunk
            - content:    the actual text of the chunk
            - source:     which document it came from
            - score:      cosine similarity (0.0 to 1.0, higher = more similar)
            - rank:       position in this result list (1 = best match)

    Why top_k=20?
        We retrieve more than we need (20) so the reranker has enough
        candidates to work with. The reranker will pick the best 5-10.
        Retrieving too few = might miss the right answer.
        Retrieving too many = slow reranker, noisy context.
        20 is the standard sweet spot in the literature.
    """
    # Step 1: Convert query text → vector
    query_vector = embed_query(query)

    # Step 2: Query pgvector using cosine distance operator (<=>)
    # The <=> operator is pgvector's cosine DISTANCE (not similarity).
    # Distance = 1 - similarity, so lower distance = more similar.
    # We convert back: similarity = 1 - distance, and ORDER BY distance ASC
    # to get the closest (most similar) chunks first.
    # NOTE: ingest.py inserts the filename into `document_name`, not a
    # `source` column — there is no `source` column populated during
    # ingestion (see ingest.py's INSERT statement). Selecting a `source`
    # column here would return NULL for every row (or error, if the column
    # doesn't exist at all), which is what caused the
    # "'NoneType' object has no attribute 'split'" crash in generator.py
    # and citation_verifier.py, both of which read chunk["source"].
    sql = """
        SELECT
            id                                    AS chunk_id,
            COALESCE(content, '')                 AS content,
            COALESCE(document_name, 'unknown')     AS source,
            1 - (embedding <=> %s::vector)  AS score
        FROM service_chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
    """

    conn = get_connection()
    try:
        # RealDictCursor makes each row a dict instead of a tuple.
        # So we get {"chunk_id": 5, "content": "...", ...} instead of (5, "...", ...)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # We pass query_vector TWICE because it appears twice in the SQL:
            # once for the score calculation, once for the ORDER BY.
            # pgvector needs the vector cast to ::vector type explicitly.
            cur.execute(sql, (query_vector, query_vector, top_k))
            rows = cur.fetchall()
    finally:
        # Always close the connection, even if an error occurs.
        # "finally" runs no matter what — success or exception.
        conn.close()

    # Step 3: Add rank (1-based position) to each result
    results = []
    for rank, row in enumerate(rows, start=1):
        results.append({
            "chunk_id": row["chunk_id"],
            "content":  row["content"],
            "source":   row["source"],
            "score":    float(row["score"]),   # numpy float → Python float
            "rank":     rank,
        })

    return results


# ---------------------------------------------------------------------------
# 5. QUICK SMOKE TEST  (run this file directly: python retrieval.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_query = "What documents are required for NADRA registration?"
    print(f"\nQuery: {test_query}\n")

    results = dense_retrieve(test_query, top_k=5)

    for r in results:
        print(f"  Rank {r['rank']} | Score: {r['score']:.4f} | Source: {r['source']}")
        print(f"  {r['content'][:120]}...")
        print()