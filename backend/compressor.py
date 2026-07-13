import os
import re
from typing import Any
from sentence_transformers import CrossEncoder
from reranker import get_cross_encoder   # reuse the already-loaded model

# ---------------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------------
# Minimum CrossEncoder score for a sentence to be kept
SENTENCE_MIN_SCORE = float(os.getenv("SENTENCE_MIN_SCORE", "0.0"))

# Maximum sentences to keep per chunk (prevents one chunk from dominating context)
MAX_SENTENCES_PER_CHUNK = int(os.getenv("MAX_SENTENCES_PER_CHUNK", "4"))

# Token budget: rough upper bound on total compressed context
# (we estimate tokens as words × 1.3)
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "2400"))


# ---------------------------------------------------------------------------
# 2. SENTENCE SPLITTER
# ---------------------------------------------------------------------------
def split_sentences(text: str) -> list[str]:
    """
    Splits a paragraph into individual sentences.

    We use a simple rule-based splitter rather than NLTK/spaCy to keep
    dependencies minimal. It handles:
    - "." "!" "?" as sentence endings
    - Abbreviations like "No." "Dr." "Mr." (don't split these)
    - Preserves the sentence text without stripping useful whitespace

    For government documents (formal, well-structured text) this is sufficient.
    """

    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital letter
    # The lookahead (?=[A-Z]) prevents splitting on abbreviations like "No. 47"
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())

    # Filter out very short fragments (likely artifacts)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

    return sentences


# ---------------------------------------------------------------------------
# 3. SCORE SENTENCES
# ---------------------------------------------------------------------------
def score_sentences(
    question: str,
    sentences: list[str],
    model: CrossEncoder,
) -> list[tuple[str, float]]:
    """
    Scores each sentence for relevance to the question using the CrossEncoder.

    Returns:
        List of (sentence, score) tuples.

    We reuse the CrossEncoder from reranker.py — no need to load a second model.
    At sentence level, the same model works: "is this sentence relevant to this question?"
    """
    if not sentences:
        return []

    pairs = [(question, sentence) for sentence in sentences]
    scores = model.predict(pairs)

    return list(zip(sentences, [float(s) for s in scores]))


# ---------------------------------------------------------------------------
# 4. COMPRESS ONE CHUNK
# ---------------------------------------------------------------------------
def compress_chunk(
    question: str,
    chunk: dict[str, Any],
    model: CrossEncoder,
) -> dict[str, Any]:
    """
    Compresses a single chunk by keeping only its most relevant sentences.

    Strategy:
        1. Split chunk content into sentences
        2. Score each sentence against the question
        3. Keep sentences above threshold OR top-N by score
        4. Reconstruct compressed content

    The compressed chunk retains all metadata (chunk_id, source, rerank_score).
    Only the "content" field changes — it becomes a shorter, focused excerpt.
    "compressed": True flag signals downstream that this content was modified.
    """
    # Defensive normalization: `content` should always be a real string by the
    # time a chunk reaches the compressor, but a NULL `content` column from a
    # bad ingestion row can slip through as None. We coalesce here and always
    # return the SAME chunk shape (content/compressed/original_length/
    # compressed_length) instead of returning the raw chunk unmodified — an
    # earlier debugging attempt did the latter, which left `content` as None
    # and crashed downstream in compress()'s token-budget loop with
    # "'NoneType' object has no attribute 'split'".
    original_content = chunk.get("content") or ""

    if not original_content:
        chunk_copy = chunk.copy()
        chunk_copy["content"] = ""
        chunk_copy["compressed"] = False
        chunk_copy["original_length"] = 0
        chunk_copy["compressed_length"] = 0
        return chunk_copy

    sentences = split_sentences(original_content)

    if len(sentences) <= 2:
        # Too short to compress meaningfully — return as-is
        chunk_copy = chunk.copy()
        chunk_copy["content"] = original_content
        chunk_copy["compressed"] = False
        chunk_copy["original_length"] = len(original_content)
        chunk_copy["compressed_length"] = len(original_content)
        return chunk_copy

    # Score all sentences
    scored = score_sentences(question, sentences, model)

    # Sort by score descending to find the most relevant
    scored.sort(key=lambda x: x[1], reverse=True)

    # Keep top sentences up to MAX_SENTENCES_PER_CHUNK
    kept = scored[:MAX_SENTENCES_PER_CHUNK]

    # Filter by minimum score if set
    if SENTENCE_MIN_SCORE > 0:
        kept = [(s, sc) for s, sc in kept if sc >= SENTENCE_MIN_SCORE]

    # If nothing passes the filter, keep the top 2 (don't return empty chunks)
    if not kept:
        kept = scored[:2]

    # Re-order kept sentences by their ORIGINAL position in the text
    # (not by score) so the compressed text reads naturally
    kept_sentences = {s for s, _ in kept}
    ordered_kept = [s for s in sentences if s in kept_sentences]

    compressed_content = " ".join(ordered_kept)

    chunk_copy = chunk.copy()
    chunk_copy["content"]           = compressed_content
    chunk_copy["compressed"]        = True
    chunk_copy["original_length"]   = len(original_content)
    chunk_copy["compressed_length"] = len(compressed_content)

    return chunk_copy


# ---------------------------------------------------------------------------
# 5. COMPRESS ALL CHUNKS
# ---------------------------------------------------------------------------
def compress(
    question: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Compresses all chunks and enforces the total token budget.

    Args:
        question: The original user question.
        chunks:   Reranked chunks from reranker.py.

    Returns:
        Compressed chunks, trimmed to stay within MAX_CONTEXT_TOKENS total.

    Token budget enforcement:
        We estimate tokens as len(text.split()) × 1.3
        (roughly 1.3 tokens per word for English text).
        We include chunks greedily (highest rerank_score first) until budget exhausted.
        This ensures the most relevant content is always included.
    """
    if not chunks:
        return []

    model = get_cross_encoder()

    # Compress each chunk individually
    compressed_chunks = [compress_chunk(question, chunk, model) for chunk in chunks]

    # Enforce total token budget — include chunks greedily by rank (already sorted)
    selected = []
    total_tokens = 0

    for chunk in compressed_chunks:
        content = chunk.get("content") or ""
        if not content:
            continue   # nothing usable in this chunk — skip rather than crash

        # Rough token estimate: words × 1.3
        chunk_tokens = len(content.split()) * 1.3

        if total_tokens + chunk_tokens <= MAX_CONTEXT_TOKENS:
            selected.append(chunk)
            total_tokens += chunk_tokens
        else:
            # Try to fit a truncated version if there's remaining budget
            remaining_words = int((MAX_CONTEXT_TOKENS - total_tokens) / 1.3)
            if remaining_words > 30:   # only bother if at least 30 words fit
                words = content.split()[:remaining_words]
                chunk_copy = chunk.copy()
                chunk_copy["content"] = " ".join(words) + "..."
                chunk_copy["truncated"] = True
                selected.append(chunk_copy)
            break   # budget exhausted

    return selected


# ---------------------------------------------------------------------------
# 6. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mock_chunks = [
        {
            "chunk_id": 1,
            "content": (
                "Pakistan has a long administrative history dating back to 1947. "
                "NADRA was established in 2000 under the Ministry of Interior. "
                "The authority is responsible for national registration and issuance of CNICs. "
                "NADRA also manages biometric data for over 120 million citizens. "
                "The weather in Islamabad is generally mild in spring."
            ),
            "source": "nadra_guide.pdf",
            "rerank_score": 8.5,
            "rank": 1,
        },
    ]

    question = "What is NADRA responsible for?"
    print(f"\nQuestion: {question}\n")

    result = compress(question, mock_chunks)
    for r in result:
        print(f"  Source: {r['source']}")
        print(f"  Original length: {r['original_length']} chars")
        print(f"  Compressed length: {r['compressed_length']} chars")
        print(f"  Content: {r['content']}")
