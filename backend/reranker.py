import os
from typing import Any
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# 1. MODEL
# ---------------------------------------------------------------------------
# ms-marco-MiniLM-L-6-v2 is trained on the MS MARCO passage ranking dataset.
# It's specifically trained to score query-passage relevance, not just similarity.
# Output: a single float (higher = more relevant to query).
# "MiniLM" = a distilled (smaller, faster) version of a larger model.
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Minimum score to keep a chunk. Calibrated on real data during Day 3.
# Chunks below this score are not grounded enough to use in the answer.
#
# IMPORTANT: os.getenv(name, default) only uses `default` when the variable
# is NOT SET at all. If your .env file has a line like
#     MIN_RERANK_SCORE=2.8
# that value wins EVERY time, no matter what default you put in the code.
# If you're seeing "I could not find this information" for every question
# after changing this line, check .env first — an old MIN_RERANK_SCORE=2.8
# left over from before the real (non-sample) data was ingested will filter
# out every chunk, since real CrossEncoder scores on this corpus run lower.
MIN_RERANK_SCORE = float(os.getenv("MIN_RERANK_SCORE", "-5"))

_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    """Lazy-loads the CrossEncoder model (same pattern as retrieval.py)."""
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


# ---------------------------------------------------------------------------
# 2. RERANK
# ---------------------------------------------------------------------------
def rerank(
    question: str,
    chunks: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """
    Scores each chunk against the question using CrossEncoder, returns top_n.

    Args:
        question: The original user question (NOT the HyDE passage or sub-questions —
                  we rerank against the real question the user asked).
        chunks:   List of chunk dicts from hybrid_search (each has chunk_id, content, source).
        top_n:    How many to return after reranking (default 5).

    Returns:
        List of chunks, sorted by rerank_score descending, filtered by MIN_RERANK_SCORE.
        Each chunk gains a "rerank_score" field.

    Why use the original question for reranking?
        query_rewriter generates HyDE and sub-questions to FIND chunks.
        But relevance should be judged against what the user actually asked.
        Reranking against a HyDE passage would score "which chunk is most
        like the fake document" — not "which chunk best answers the question."
    """
    if not chunks:
        return []

    model = get_cross_encoder()

    # CrossEncoder expects a list of (query, passage) pairs
    # It scores each pair jointly — the model reads both together
    pairs = [(question, chunk["content"]) for chunk in chunks]

    # predict() returns a numpy array of floats, one per pair
    # This is the expensive step — O(n) model forward passes
    scores = model.predict(pairs)

    # Attach scores to chunks and sort
    scored_chunks = []
    for chunk, score in zip(chunks, scores):
        chunk_copy = chunk.copy()
        chunk_copy["rerank_score"] = float(score)
        scored_chunks.append(chunk_copy)

    # Sort by rerank_score descending
    scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Filter out chunks below the minimum threshold
    # These are not grounded enough — including them would introduce noise
    scored_chunks = [c for c in scored_chunks if c["rerank_score"] >= MIN_RERANK_SCORE]

    # Re-assign rank based on new ordering
    for i, chunk in enumerate(scored_chunks, start=1):
        chunk["rank"] = i

    return scored_chunks[:top_n]


# ---------------------------------------------------------------------------
# 3. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate what hybrid_search would return
    mock_chunks = [
        {"chunk_id": 1, "content": "CNIC registration requires original birth certificate, B-Form, and two passport photos.", "source": "nadra_guide.pdf", "rrf_score": 0.03, "rank": 1},
        {"chunk_id": 2, "content": "Pakistan was established in 1947 as an independent nation.", "source": "history.pdf",      "rrf_score": 0.02, "rank": 2},
        {"chunk_id": 3, "content": "NADRA issues the Computerized National Identity Card to Pakistani citizens above 18.", "source": "nadra_guide.pdf", "rrf_score": 0.02, "rank": 3},
    ]

    question = "What documents do I need for CNIC registration?"
    print(f"\nQuestion: {question}\n")

    results = rerank(question, mock_chunks, top_n=3)
    for r in results:
        print(f"  Rank {r['rank']} | Score: {r['rerank_score']:.4f} | {r['content'][:80]}...")
