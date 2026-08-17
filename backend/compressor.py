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
# 3b. STRUCTURED (LIST / PROCEDURE) CHUNKS
# ---------------------------------------------------------------------------
# A line that is a bullet ("- x", "* x", "• x") or a numbered step ("1. x", "2) x")
LIST_LINE    = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s+\S')
HEADING_LINE = re.compile(r'^\s*#{1,6}\s+\S')

# Word budget for a structured chunk. Generous compared to the 4-sentence
# prose budget, because a requirements list is exactly the content the
# citizen asked for — trimming it defeats the retrieval.
MAX_STRUCTURED_WORDS = int(os.getenv("MAX_STRUCTURED_WORDS", "260"))


def is_structured(text: str) -> bool:
    """True if the text contains a real list (two or more list lines)."""
    return sum(1 for line in text.split("\n") if LIST_LINE.match(line)) >= 2


def split_blocks(text: str) -> list[str]:
    """
    Groups lines into blocks, keeping a list and its lead-in line together.

    "Requirements:\\n- CNIC number\\n- Original card" is ONE block, so the
    lead-in can never be kept while its items are dropped (or vice versa).
    A blank line, or a plain line following list items, ends a block.
    """
    blocks: list[str] = []
    current: list[str] = []
    in_list = False

    def flush():
        nonlocal current, in_list
        if current:
            blocks.append("\n".join(current))
        current, in_list = [], False

    for line in text.split("\n"):
        if not line.strip():
            flush()
            continue
        if LIST_LINE.match(line):
            current.append(line)
            in_list = True
        else:
            if in_list:       # prose after a list starts a new block
                flush()
            current.append(line)

    flush()
    return blocks


def compress_structured(
    question: str,
    chunk: dict[str, Any],
    model: CrossEncoder,
) -> dict[str, Any]:
    """
    Compresses a chunk that contains lists, WITHOUT breaking the lists.

    Why this path exists:
        The sentence-level compressor scores each sentence independently and
        keeps the top MAX_SENTENCES_PER_CHUNK. Applied to NADRA's CNIC renewal
        chunk, it kept process steps 1, 4 and 5 and deleted 2 and 3 — so the
        context handed to the LLM read:

            1. OIC Interview / Approval
            4. Payment
            5.

        The deleted step 2 was "Biometric Capture", the single most important
        thing a renewal applicant needs to know. Partial procedures are worse
        than omitted ones: they look complete, so neither the model nor the
        citizen can tell something is missing.

    Instead we keep or drop whole blocks, ranked by relevance to the question,
    up to a word budget. Section headings are always kept — "## CNIC RENEWAL"
    is what stops the model from answering with the new-registration rules.
    """
    content = chunk.get("content") or ""
    blocks  = split_blocks(content)

    if not blocks:
        chunk_copy = chunk.copy()
        chunk_copy["content"]           = content
        chunk_copy["compressed"]        = False
        chunk_copy["original_length"]   = len(content)
        chunk_copy["compressed_length"] = len(content)
        return chunk_copy

    scores = model.predict([(question, b) for b in blocks])

    kept = {i for i, b in enumerate(blocks) if HEADING_LINE.match(b.strip())}
    words = sum(len(blocks[i].split()) for i in kept)

    # Highest-scoring blocks first, fitting each whole block into the budget.
    for i in sorted(range(len(blocks)), key=lambda i: float(scores[i]), reverse=True):
        if i in kept:
            continue
        block_words = len(blocks[i].split())
        if words + block_words > MAX_STRUCTURED_WORDS:
            continue   # doesn't fit — try the next (smaller) block
        kept.add(i)
        words += block_words

    # Never return headings only — keep the single best block if nothing fit.
    if all(HEADING_LINE.match(blocks[i].strip()) for i in kept):
        kept.add(int(max(range(len(blocks)), key=lambda i: float(scores[i]))))

    compressed_content = "\n".join(blocks[i] for i in sorted(kept))

    chunk_copy = chunk.copy()
    chunk_copy["content"]           = compressed_content
    chunk_copy["compressed"]        = len(kept) < len(blocks)
    chunk_copy["structured"]        = True
    chunk_copy["original_length"]   = len(content)
    chunk_copy["compressed_length"] = len(compressed_content)
    return chunk_copy


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

    # Structured chunks (a "Requirements:" list, a numbered process) are kept
    # whole — see compress_structured() for why partial lists are worse than
    # no list at all.
    if is_structured(original_content):
        return compress_structured(question, chunk, model)

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
