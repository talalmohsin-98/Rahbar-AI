"""
Regression + core-logic tests for the Rahbar AI backend.

Run with:  cd backend && pytest tests/ -v

These import the REAL project modules (reranker.py, compressor.py,
generator.py, citation_verifier.py, hallucination_eval.py, bm25_retrieval.py,
hybrid_search.py, document_qa.py) with the heavy external dependencies
(Groq, psycopg2, sentence-transformers, rank_bm25) stubbed out in
conftest.py — so what's being tested is the project's own logic, not the
stubs.
"""
import numpy as np
import pytest

import reranker
import compressor
import generator
import citation_verifier
import hallucination_eval
import bm25_retrieval
import hybrid_search
import intent_router
import query_rewriter
import document_qa

from tests.conftest import FAKE_GROQ_RESPONSES


# ---------------------------------------------------------------------------
# 1. reranker.py
# ---------------------------------------------------------------------------
def test_reranker_sorts_by_score_and_filters_below_threshold():
    chunks = [
        {"chunk_id": 1, "content": "passport renewal requires form and photo", "source": "a.pdf"},
        {"chunk_id": 2, "content": "completely unrelated weather forecast text", "source": "b.pdf"},
        {"chunk_id": 3, "content": "passport renewal form photo requirements documents list", "source": "c.pdf"},
    ]
    question = "what documents are required for passport renewal form photo requirements"

    results = reranker.rerank(question, chunks, top_n=5)

    assert len(results) >= 1
    # Highest-overlap chunk should be ranked first
    assert results[0]["chunk_id"] == 3
    # Every returned chunk must be at/above the configured minimum score
    assert all(c["rerank_score"] >= reranker.MIN_RERANK_SCORE for c in results)
    # rank field must be reassigned 1..N after filtering/sorting
    assert [c["rank"] for c in results] == list(range(1, len(results) + 1))


def test_reranker_empty_input_returns_empty_list():
    assert reranker.rerank("anything", [], top_n=5) == []


# ---------------------------------------------------------------------------
# 2. compressor.py — regression tests for the NoneType('split') crash
# ---------------------------------------------------------------------------
def test_compress_chunk_with_none_content_does_not_crash():
    """
    Regression test: a chunk with content=None (e.g. from a bad DB row)
    must not crash compress_chunk(), and must come back with a real string
    ("") in `content` rather than None, since compress() later calls
    .split() on every chunk's content unconditionally.
    """
    model = reranker.get_cross_encoder()
    bad_chunk = {"chunk_id": 1, "content": None, "source": "x.pdf"}

    result = compressor.compress_chunk("any question", bad_chunk, model)

    assert result["content"] == ""
    assert isinstance(result["content"], str)
    assert result["compressed"] is False


def test_compress_pipeline_with_mixed_none_and_real_chunks_does_not_crash():
    """
    End-to-end regression test matching the original crash: a list of
    chunks where one has content=None must not raise
    "'NoneType' object has no attribute 'split'" anywhere in compress().
    """
    chunks = [
        {"chunk_id": 1, "content": None, "source": "a.pdf", "rerank_score": 5.0, "rank": 1},
        {"chunk_id": 2, "content": "NADRA CNIC requires B-Form and two photographs.",
         "source": "b.pdf", "rerank_score": 4.0, "rank": 2},
    ]

    result = compressor.compress("What documents are required?", chunks)

    assert isinstance(result, list)
    for c in result:
        assert isinstance(c["content"], str)


def test_compress_respects_token_budget():
    long_text = " ".join([f"word{i}" for i in range(2000)])  # way over budget
    chunks = [{"chunk_id": 1, "content": long_text, "source": "big.pdf",
               "rerank_score": 1.0, "rank": 1}]

    result = compressor.compress("word5 word10", chunks)

    total_words = sum(len(c["content"].split()) for c in result)
    # MAX_CONTEXT_TOKENS=1200 at ~1.3 tokens/word => budget is ~920 words
    assert total_words * 1.3 <= compressor.MAX_CONTEXT_TOKENS + 5   # small slack for the "..." token


# ---------------------------------------------------------------------------
# 3. generator.py — regression test for the None `source` crash
# ---------------------------------------------------------------------------
def test_format_context_with_none_source_does_not_crash():
    """
    Regression test: chunk.get("source", "unknown") does NOT fall back to
    "unknown" when the key exists with value None (only when the key is
    missing entirely) — this crashed generator.py's format_context() with
    "'NoneType' object has no attribute 'split'" whenever a DB row's source
    column was NULL, which — before the retrieval.py/bm25_retrieval.py fix —
    was true for every single ingested chunk.
    """
    chunks = [{"chunk_id": 1, "content": "Some passage text.", "source": None}]

    context = generator.format_context(chunks)

    assert "unknown" in context
    assert "Some passage text." in context


def test_format_context_empty_list_returns_placeholder():
    assert generator.format_context([]) == "No context available."


def test_generate_out_of_scope_uses_small_model_and_returns_answer():
    FAKE_GROQ_RESPONSES.append("I can help with government services, but not with that.")
    result = generator.generate("Tell me a joke", chunks=[], intent="out_of_scope")

    assert result["model"] == "llama-3.1-8b-instant"
    assert result["context_used"] == 0
    assert "government services" in result["answer"]


def test_generate_factual_with_none_content_or_none_source_chunk_does_not_crash():
    FAKE_GROQ_RESPONSES.append("NADRA charges Rs. 750. [source: unknown]")
    chunks = [{"chunk_id": 1, "content": None, "source": None}]

    result = generator.generate("What is the fee?", chunks=chunks, intent="factual")

    assert isinstance(result["answer"], str)


# ---------------------------------------------------------------------------
# 4. citation_verifier.py
# ---------------------------------------------------------------------------
def test_parse_citations_extracts_claim_source_pairs():
    """
    NOTE: parse_citations() splits on ANY '.', '!', or '?' followed by
    optional whitespace and a '[source:' tag — it has no abbreviation
    handling (unlike compressor.split_sentences, which has a capital-letter
    lookahead). So a period inside "Rs." is treated as a sentence boundary:
    the regex backtracks past it and the captured claim starts *after* that
    stray period, not from the true sentence start. This is a known,
    pre-existing parser limitation (not something this test suite changes)
    — asserting the actual behavior here so a future fix is a deliberate,
    visible change instead of a silent regression.
    """
    answer = (
        "NADRA charges Rs. 750 for a new CNIC. [source: nadra_guide.pdf] "
        "Processing takes 30 days. [source: nadra_guide.pdf]"
    )
    claims = citation_verifier.parse_citations(answer)

    assert len(claims) == 2
    assert claims[0]["source"] == "nadra_guide.pdf"
    assert "750 for a new CNIC" in claims[0]["claim"]


def test_find_source_chunk_with_none_source_does_not_crash():
    """Regression test mirroring the generator.py None-source fix."""
    chunks = [{"chunk_id": 1, "content": "text", "source": None}]
    result = citation_verifier.find_source_chunk("nadra_guide.pdf", chunks)
    assert result is None   # no match, but must not raise


def test_verify_citations_flags_unsupported_claim():
    answer = "NADRA charges Rs. 750 for a CNIC. [source: nadra.pdf]"
    chunks = [{"chunk_id": 1, "content": "completely unrelated passage about weather",
               "source": "nadra.pdf"}]

    result = citation_verifier.verify_citations(answer, chunks)

    assert result["passed"] is False
    assert len(result["unverified_claims"]) == 1


def test_verify_citations_passes_supported_claim():
    answer = "NADRA charges Rs. 750 for a CNIC. [source: nadra.pdf]"
    chunks = [{"chunk_id": 1, "content": "NADRA charges Rs. 750 for a CNIC.",
               "source": "nadra.pdf"}]

    result = citation_verifier.verify_citations(answer, chunks)

    assert result["passed"] is True


# ---------------------------------------------------------------------------
# 5. hallucination_eval.py
# ---------------------------------------------------------------------------
def test_hallucination_eval_flags_ungrounded_sentence():
    chunks = [{"chunk_id": 1, "content": "NADRA registration requires B-Form and photographs.",
               "source": "nadra.pdf"}]
    answer = (
        "NADRA registration requires B-Form and photographs. "
        "Zebra quokka bicycle telescope pyramid volcano lighthouse forty."
    )

    result = hallucination_eval.evaluate_hallucination(answer, chunks)

    assert result["hallucinated_count"] >= 1
    assert 0.0 <= result["hallucination_rate"] <= 1.0


def test_hallucination_eval_empty_answer_passes_trivially():
    result = hallucination_eval.evaluate_hallucination("", [])
    assert result["passed"] is True


# ---------------------------------------------------------------------------
# 6. hybrid_search.py — RRF fusion math (from the module's own docstring example)
# ---------------------------------------------------------------------------
def test_rrf_fusion_matches_documented_example():
    dense = [
        {"chunk_id": "A", "content": "a", "source": "s", "rank": 1},
        {"chunk_id": "B", "content": "b", "source": "s", "rank": 2},
        {"chunk_id": "C", "content": "c", "source": "s", "rank": 3},
    ]
    bm25 = [
        {"chunk_id": "B", "content": "b", "source": "s", "rank": 1},
        {"chunk_id": "D", "content": "d", "source": "s", "rank": 2},
        {"chunk_id": "A", "content": "a", "source": "s", "rank": 3},
    ]

    fused = hybrid_search.reciprocal_rank_fusion([dense, bm25], k=60)
    order = [c["chunk_id"] for c in fused]

    assert order == ["B", "A", "D", "C"]


# ---------------------------------------------------------------------------
# 7. bm25_retrieval.py — tokenizer behavior
# ---------------------------------------------------------------------------
def test_bm25_tokenize_lowercases_and_keeps_hyphens():
    tokens = bm25_retrieval.tokenize("NADRA B-Form requirements!")
    assert tokens == ["nadra", "b-form", "requirements"]


# ---------------------------------------------------------------------------
# 8. intent_router.py — fallback behavior on unexpected LLM output
# ---------------------------------------------------------------------------
def test_intent_router_falls_back_to_factual_on_garbage_output():
    FAKE_GROQ_RESPONSES.append("blah unexpected nonsense")
    result = intent_router.route("some question")
    assert result == intent_router.Intent.FACTUAL


def test_intent_router_matches_known_category():
    FAKE_GROQ_RESPONSES.append("procedural")
    result = intent_router.route("How do I renew my passport?")
    assert result == intent_router.Intent.PROCEDURAL


# ---------------------------------------------------------------------------
# 9. query_rewriter.py — fallback on unparseable JSON
# ---------------------------------------------------------------------------
def test_sub_question_fallback_on_bad_json():
    FAKE_GROQ_RESPONSES.append("this is not json at all")
    result = query_rewriter.generate_sub_questions("What documents do I need?")
    assert result == ["What documents do I need?"]


def test_sub_question_parses_valid_json():
    FAKE_GROQ_RESPONSES.append('["q1", "q2", "q3"]')
    result = query_rewriter.generate_sub_questions("original question")
    assert result == ["q1", "q2", "q3"]


# ---------------------------------------------------------------------------
# 10. document_qa.py — new upload-your-own-document feature, end to end
# ---------------------------------------------------------------------------
def test_document_qa_upload_and_ask_end_to_end():
    text = (
        "This lease agreement is between Landlord and Tenant. "
        "The monthly rent is Rs. 45000, due on the first of each month. "
        "The security deposit is Rs. 90000, refundable within 30 days of move-out."
    )
    meta = document_qa.upload_document("lease.txt", text.encode("utf-8"))

    assert meta["chunk_count"] >= 1
    assert meta["filename"] == "lease.txt"

    FAKE_GROQ_RESPONSES.append("The monthly rent is Rs. 45000. [source: lease.txt]")
    answer = document_qa.ask_document(meta["doc_id"], "How much is the monthly rent?")

    assert "45000" in answer["answer"]
    assert answer["filename"] == "lease.txt"


def test_document_qa_unknown_doc_id_raises_keyerror():
    with pytest.raises(KeyError):
        document_qa.ask_document("does-not-exist", "any question")


def test_document_qa_rejects_unsupported_file_type():
    with pytest.raises(ValueError):
        document_qa.upload_document("image.png", b"\x89PNG\r\n")


def test_document_qa_rejects_empty_pdf_text():
    with pytest.raises(ValueError):
        document_qa.extract_text("empty.txt", b"   \n\n  ")
