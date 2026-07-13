import os
import json
import pickle
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor
from rank_bm25 import BM25Okapi

# ---------------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------------
# Where we save the BM25 index so we don't rebuild it on every query.
# Building BM25 requires reading ALL documents — expensive if done repeatedly.
BM25_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", "bm25_index.pkl"))


# ---------------------------------------------------------------------------
# 2. TOKENIZER
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """
    Converts a string into a list of lowercase word tokens.

    Example:
        tokenize("NADRA B-Form requirements!")
        → ["nadra", "b-form", "requirements"]

    We keep hyphens intact (don't split "b-form" into ["b", "form"])
    because in government documents, hyphenated terms are specific codes.

    Simple tokenization is intentional — BM25 doesn't need fancy NLP.
    Its power comes from term frequency math, not linguistic understanding.
    """
    import re
    # Split on whitespace and punctuation EXCEPT hyphens
    tokens = re.findall(r"[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)*", text.lower())
    return tokens


# ---------------------------------------------------------------------------
# 3. BUILD THE BM25 INDEX
# ---------------------------------------------------------------------------
def build_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Reads ALL document chunks from the database and builds a BM25 index.

    Returns:
        bm25:   The BM25Okapi index object (knows term frequencies for all docs)
        corpus: List of chunk dicts (so we can look up content by index position)

    This function is SLOW — it reads every chunk and tokenizes it.
    We call it once and save (pickle) the result to disk.
    On future runs, we load from disk instead of rebuilding.

    What is a BM25 index?
        It stores, for each unique word across all documents:
        - How many times it appears in each document (TF)
        - In how many documents it appears (IDF)
        BM25 uses these counts to score how relevant each document is
        to a query containing those words.
    """
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "rahbar"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Fetch every chunk — BM25 needs the full corpus to compute IDF
            # NOTE: ingest.py populates `document_name`, not a `source`
            # column — see retrieval.py for the full explanation. Alias it
            # here too so BM25 and dense retrieval return the same shape.
            cur.execute(
                "SELECT id AS chunk_id, COALESCE(content, '') AS content, "
                "COALESCE(document_name, 'unknown') AS source "
                "FROM service_chunks ORDER BY id"
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    # Convert rows to plain dicts (psycopg2 RealDictRow → regular dict)
    corpus = [dict(row) for row in rows]

    # Tokenize every document's content
    # tokenized_corpus = [["nadra", "registration", ...], ["passport", "form", ...], ...]
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]

    # Build the BM25 index from the tokenized corpus
    # BM25Okapi is the most common variant (Okapi BM25 = the standard formula)
    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, corpus


# ---------------------------------------------------------------------------
# 4. LOAD OR BUILD INDEX (with caching)
# ---------------------------------------------------------------------------
_bm25_index: BM25Okapi | None = None
_bm25_corpus: list[dict] | None = None


def get_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Returns the BM25 index and corpus, loading from disk cache if available.

    Cache strategy:
        1. Check memory (module-level variables) — fastest
        2. Check disk (pickle file) — fast
        3. Build from database — slow, only on first run

    This is the same lazy-loading pattern as retrieval.py,
    extended with a disk cache layer because BM25 index building
    is even more expensive than loading an embedding model.
    """
    global _bm25_index, _bm25_corpus

    # Already in memory — return immediately
    if _bm25_index is not None:
        return _bm25_index, _bm25_corpus

    # Try loading from disk cache
    if BM25_INDEX_PATH.exists():
        print(f"Loading BM25 index from {BM25_INDEX_PATH}...")
        with open(BM25_INDEX_PATH, "rb") as f:
            cached = pickle.load(f)
        _bm25_index = cached["bm25"]
        _bm25_corpus = cached["corpus"]
        return _bm25_index, _bm25_corpus

    # Not cached — build from scratch
    print("Building BM25 index from database (first run, will be cached)...")
    _bm25_index, _bm25_corpus = build_bm25_index()

    # Save to disk for next time
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": _bm25_index, "corpus": _bm25_corpus}, f)
    print(f"BM25 index saved to {BM25_INDEX_PATH}")

    return _bm25_index, _bm25_corpus


def invalidate_cache() -> None:
    """
    Deletes the disk cache and clears memory.
    Call this whenever documents are added/updated in the database.
    The index must be rebuilt to reflect new content.
    """
    global _bm25_index, _bm25_corpus
    _bm25_index = None
    _bm25_corpus = None
    if BM25_INDEX_PATH.exists():
        BM25_INDEX_PATH.unlink()
        print("BM25 cache invalidated.")


# ---------------------------------------------------------------------------
# 5. BM25 RETRIEVAL
# ---------------------------------------------------------------------------
def bm25_retrieve(query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """
    Given a query string, return the top_k most keyword-relevant chunks
    using BM25 scoring.

    Args:
        query:  The search query
        top_k:  How many results to return

    Returns:
        List of dicts with keys: chunk_id, content, source, score, rank
        (Same schema as dense_retrieve — hybrid_search.py expects both
        to return the same structure so it can merge them uniformly.)

    BM25 Score meaning:
        Higher score = more keyword overlap, weighted by how rare those
        keywords are across the corpus.
        Score of 0.0 = no keyword overlap at all.
        There's no fixed upper bound — depends on document lengths and corpus.
    """
    bm25, corpus = get_bm25_index()

    # Tokenize the query the same way we tokenized documents
    # CRITICAL: must use the same tokenizer for both — otherwise "NADRA"
    # in the query won't match "nadra" in the index
    query_tokens = tokenize(query)

    # BM25Okapi.get_scores() returns a score for EVERY document in the corpus
    # scores[i] = BM25 relevance score of corpus[i] for this query
    scores = bm25.get_scores(query_tokens)

    # Get the indices of the top_k highest scores
    # argsort() returns indices sorted by value ascending, so we reverse with [::-1]
    import numpy as np
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        # Skip documents with score 0 — zero means no keyword overlap at all
        # Including them just adds noise for the RRF fusion step
        if scores[idx] == 0.0:
            break
        results.append({
            "chunk_id": corpus[idx]["chunk_id"],
            "content":  corpus[idx]["content"],
            "source":   corpus[idx]["source"],
            "score":    float(scores[idx]),
            "rank":     rank,
        })

    return results


# ---------------------------------------------------------------------------
# 6. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_query = "CNIC B-Form NADRA registration requirements"
    print(f"\nBM25 Query: {test_query}\n")

    results = bm25_retrieve(test_query, top_k=5)

    if not results:
        print("No results — check that documents are loaded in the database.")
    for r in results:
        print(f"  Rank {r['rank']} | Score: {r['score']:.4f} | Source: {r['source']}")
        print(f"  {r['content'][:120]}...")
        print()
