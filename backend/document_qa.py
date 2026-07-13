"""
document_qa.py
---------------
PURPOSE: Let a user upload their OWN pdf/txt file and ask questions about
         just that file — separate from the 5 government-service corpora
         that live in Postgres/pgvector.

WHY A SEPARATE MODULE (not reusing service_chunks / pgvector):
    The 5 government corpora are curated, persistent, shared across all
    users. An uploaded document is:
        - user-specific (should NOT be mixed into NADRA/FBR/etc. search)
        - ephemeral (no reason to persist a stranger's tax return to a
          shared Postgres table)
    So this module keeps its own light in-memory store, keyed by doc_id,
    and reuses the SAME embedding model + reranker + generator as the main
    pipeline (consistency, and no second model to load).

STORAGE:
    In-memory only (a plain dict). This is intentional for a portfolio demo:
    - No new DB tables/migrations needed
    - Simple to reason about and demo
    - Trade-off: uploaded documents are lost on server restart, and this
      does NOT scale past a single process. In production this would move
      to a per-session pgvector table (or a lightweight vector store like
      Chroma/FAISS) with a TTL/cleanup job.

PIPELINE (mirrors the main RAG pipeline, minus the DB hop):
    upload  → extract_text → chunk_text → embed (BAAI/bge-large-en)
    ask     → embed question → cosine similarity over in-memory chunks
            → reranker.rerank (CrossEncoder, reused)
            → compressor.compress (reused)
            → generator.generate (reused, intent="document_qa")
"""

import io
import os
import re
import time
import uuid
from typing import Any

import numpy as np

from retrieval import get_model          # reuse the same BAAI/bge-large-en model
from reranker import rerank
from compressor import compress
from generator import generate, INTENT_INSTRUCTIONS, BASE_RULES

# ---------------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------------
MAX_FILE_BYTES   = int(os.getenv("DOC_QA_MAX_BYTES", str(15 * 1024 * 1024)))  # 15 MB
CHUNK_CHARS      = int(os.getenv("DOC_QA_CHUNK_CHARS", "1000"))
CHUNK_OVERLAP    = int(os.getenv("DOC_QA_CHUNK_OVERLAP", "150"))
MAX_CHUNKS       = int(os.getenv("DOC_QA_MAX_CHUNKS", "400"))   # guard against huge files
DOC_TTL_SECONDS  = int(os.getenv("DOC_QA_TTL_SECONDS", str(60 * 60 * 4)))  # 4 hours

# Register a document-specific generation prompt alongside the existing
# factual/comparison/procedural/out_of_scope ones in generator.py, so
# generate(..., intent="document_qa") "just works" without touching
# generator.py's core logic.
INTENT_INSTRUCTIONS["document_qa"] = BASE_RULES + """
Format: Write a clear, direct answer in 2-5 sentences. Cite every fact.
You are answering questions ONLY about the document the user uploaded —
not about Pakistani government services in general. If the uploaded
document doesn't contain the answer, say so plainly.
"""

# doc_id -> {
#   "filename": str, "chunks": list[str], "embeddings": np.ndarray (N, D),
#   "created_at": float, "chunk_count": int,
# }
_DOCUMENTS: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# 2. TEXT EXTRACTION
# ---------------------------------------------------------------------------
def extract_text(filename: str, raw_bytes: bytes) -> str:
    """
    Extracts plain text from an uploaded file.

    Supports .pdf (via pypdf) and .txt/.md (utf-8 decode).
    Raises ValueError for unsupported types or empty extraction —
    the caller (FastAPI route) turns that into a 400 response.
    """
    lower = filename.lower()

    if lower.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n".join(pages)
    elif lower.endswith((".txt", ".md")):
        text = raw_bytes.decode("utf-8", errors="ignore")
    else:
        raise ValueError(
            f"Unsupported file type for '{filename}'. Only .pdf, .txt, and .md are supported."
        )

    text = text.strip()
    if not text:
        raise ValueError(
            f"Could not extract any text from '{filename}'. "
            "It may be a scanned/image-only PDF (not supported without OCR)."
        )
    return text


# ---------------------------------------------------------------------------
# 3. CHUNKING
# ---------------------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_chars: int = CHUNK_CHARS,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Splits text into overlapping character-based chunks.

    Why character-based (not sentence/paragraph-based like ingest.py)?
        Uploaded documents are arbitrary and unstructured (no guaranteed
        "## Section" headers like the curated government docs). A simple
        sliding window is robust to any document shape. Overlap prevents
        losing context at chunk boundaries (a sentence split exactly at
        the cut point would otherwise lose its second half).
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    step = max(chunk_chars - overlap, 1)

    while start < n:
        end = min(start + chunk_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start += step

    return chunks[:MAX_CHUNKS]


# ---------------------------------------------------------------------------
# 4. UPLOAD (extract + chunk + embed + store)
# ---------------------------------------------------------------------------
def upload_document(filename: str, raw_bytes: bytes) -> dict[str, Any]:
    """
    Processes an uploaded file end-to-end and stores it for Q&A.

    Returns metadata (doc_id, filename, chunk_count) — never the raw text,
    to keep the upload response small.
    """
    if len(raw_bytes) > MAX_FILE_BYTES:
        raise ValueError(
            f"File too large ({len(raw_bytes) / 1_000_000:.1f} MB). "
            f"Max is {MAX_FILE_BYTES / 1_000_000:.0f} MB."
        )

    text = extract_text(filename, raw_bytes)
    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(f"'{filename}' produced no usable chunks after splitting.")

    model = get_model()
    embeddings = model.encode(chunks, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    doc_id = uuid.uuid4().hex[:12]
    _DOCUMENTS[doc_id] = {
        "filename":    filename,
        "chunks":      chunks,
        "embeddings":  embeddings,
        "created_at":  time.time(),
        "chunk_count": len(chunks),
    }

    _evict_expired()

    return {
        "doc_id":      doc_id,
        "filename":    filename,
        "chunk_count": len(chunks),
    }


def _evict_expired() -> None:
    """Drops documents older than DOC_TTL_SECONDS to bound memory growth."""
    now = time.time()
    expired = [
        doc_id for doc_id, doc in _DOCUMENTS.items()
        if now - doc["created_at"] > DOC_TTL_SECONDS
    ]
    for doc_id in expired:
        del _DOCUMENTS[doc_id]


def get_document_meta(doc_id: str) -> dict[str, Any] | None:
    doc = _DOCUMENTS.get(doc_id)
    if not doc:
        return None
    return {
        "doc_id":      doc_id,
        "filename":    doc["filename"],
        "chunk_count": doc["chunk_count"],
    }


# ---------------------------------------------------------------------------
# 5. RETRIEVAL OVER A SINGLE DOCUMENT (cosine similarity, in-memory)
# ---------------------------------------------------------------------------
def _search_document(doc: dict[str, Any], question: str, top_k: int = 10) -> list[dict[str, Any]]:
    """
    Embeds the question and ranks this document's chunks by cosine similarity.

    Since both the query and chunk embeddings are normalized (unit vectors),
    cosine similarity reduces to a plain dot product — no need for a vector
    DB here, numpy handles it in microseconds for a few hundred chunks.
    """
    model = get_model()
    query_vec = np.asarray(model.encode(question, normalize_embeddings=True), dtype=np.float32)

    scores = doc["embeddings"] @ query_vec   # (N,) dot products = cosine similarities
    order = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(order, start=1):
        results.append({
            "chunk_id": int(idx),
            "content":  doc["chunks"][int(idx)],
            "source":   doc["filename"],
            "score":    float(scores[idx]),
            "rank":     rank,
        })
    return results


# ---------------------------------------------------------------------------
# 6. MAIN ASK FUNCTION
# ---------------------------------------------------------------------------
def ask_document(doc_id: str, question: str) -> dict[str, Any]:
    """
    Full document-Q&A pipeline for one uploaded document:
        retrieve (cosine) → rerank (CrossEncoder) → compress → generate

    Raises KeyError if doc_id is unknown/expired — the FastAPI route turns
    that into a 404.
    """
    doc = _DOCUMENTS.get(doc_id)
    if doc is None:
        raise KeyError(f"Unknown or expired document id: {doc_id}")

    raw_hits = _search_document(doc, question, top_k=10)

    reranked = rerank(question, raw_hits, top_n=5)
    compressed = compress(question, reranked)

    result = generate(question=question, chunks=compressed, intent="document_qa")

    return {
        "answer":            result["answer"],
        "doc_id":            doc_id,
        "filename":          doc["filename"],
        "chunks_used":       compressed,
        "generation_meta": {
            "model":             result["model"],
            "prompt_tokens":     result["prompt_tokens"],
            "completion_tokens": result["completion_tokens"],
            "context_used":      result["context_used"],
        },
    }


# ---------------------------------------------------------------------------
# 7. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_text = (
        "This is a lease agreement between Landlord and Tenant. "
        "The monthly rent is Rs. 45,000, due on the 1st of each month. "
        "The security deposit is Rs. 90,000, refundable within 30 days "
        "of move-out, subject to a property inspection."
    ) * 3

    meta = upload_document("lease.txt", sample_text.encode("utf-8"))
    print(f"Uploaded: {meta}")

    answer = ask_document(meta["doc_id"], "How much is the monthly rent?")
    print(f"\nAnswer:\n{answer['answer']}")
