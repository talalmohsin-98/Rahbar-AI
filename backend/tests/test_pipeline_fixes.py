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
import completeness_check
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

    assert result["model"] == generator.FAST_MODEL
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
    NOTE: parse_citations() no longer delimits claims by sentence punctuation
    — it takes the text from the previous tag (or the start of the line) up to
    each [source: ...] tag. That was changed deliberately to make bulleted
    answers verifiable, and it also removed the old "Rs." backtracking bug
    this test used to document. See
    test_parse_citations_keeps_full_sentence_through_an_abbreviation.
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


# ---------------------------------------------------------------------------
# 11. The "verified-looking bad answer" bug
# ---------------------------------------------------------------------------
# Reported from the UI: asking "What documents do I need to renew my CNIC?"
# returned the single bullet "- CNIC number" under a green "✓ Citations
# verified" badge and "Hal rate: 0%". Neither check had actually run — both
# stages silently no-op'd on list-formatted answers, which is the format
# generator.py mandates. These tests pin the failing shape directly.

LIST_ANSWER = (
    "To renew your CNIC you need to provide:\n"
    "- CNIC number [source: NADRA.txt]\n"
    "- Original expired CNIC [source: NADRA.txt]"
)


def test_parse_citations_reads_bullet_claims_without_terminal_punctuation():
    """The old sentence regex required '.'/'!'/'?' before the tag, so a
    bulleted answer parsed to ZERO citations."""
    claims = citation_verifier.parse_citations(LIST_ANSWER)

    assert len(claims) == 2
    assert claims[0]["claim"] == "CNIC number"          # list marker stripped
    assert claims[1]["claim"] == "Original expired CNIC"
    assert all(c["source"] == "NADRA.txt" for c in claims)


def test_parse_citations_claim_does_not_absorb_the_lead_in_line():
    claims = citation_verifier.parse_citations(LIST_ANSWER)
    assert "To renew your CNIC" not in claims[0]["claim"]


def test_parse_citations_keeps_full_sentence_through_an_abbreviation():
    """Regression on the old regex's 'Rs.' backtracking: the claim used to
    start mid-sentence, right after the abbreviation's period."""
    answer = "NADRA charges Rs. 750 for a new CNIC. [source: nadra.pdf]"
    claims = citation_verifier.parse_citations(answer)

    assert len(claims) == 1
    assert claims[0]["claim"].startswith("NADRA charges")


def test_uncited_answer_is_reported_unverifiable_not_verified():
    """An answer with no parseable citations must NOT come back passed=True —
    that is what let the UI paint a green 'citations verified' badge on an
    answer where nothing was checked."""
    chunks = [{"chunk_id": 1, "content": "CNIC renewal requires a CNIC number.",
               "source": "NADRA.txt"}]

    result = citation_verifier.verify_citations("Just some text, no tags.", chunks)

    assert result["status"] == "unverifiable"
    assert result["passed"] is False
    assert result["verification_rate"] is None


def test_verify_citations_with_no_chunks_is_not_applicable():
    """The out_of_scope path has no context to verify against — that is not a
    citation failure and must not trigger a regeneration."""
    result = citation_verifier.verify_citations("I can only help with X.", [])

    assert result["status"] == "not_applicable"
    assert result["passed"] is True


def test_verified_result_reports_status_and_count():
    answer = "NADRA charges Rs. 750 for a CNIC. [source: nadra.pdf]"
    chunks = [{"chunk_id": 1, "content": "NADRA charges Rs. 750 for a CNIC.",
               "source": "nadra.pdf"}]

    result = citation_verifier.verify_citations(answer, chunks)

    assert result["status"] == "verified"
    assert result["checked_count"] == 1


def test_claim_is_checked_against_every_chunk_of_the_cited_source():
    """
    A citation names a DOCUMENT, and one document contributes several chunks.
    Scoring only against the first/highest-ranked chunk of that document
    flagged claims the same document plainly supports one chunk over.
    """
    chunks = [
        {"chunk_id": 1, "source": "NADRA.txt",
         "content": "Eligibility falls under Rule 12 of the 2002 statute."},
        {"chunk_id": 2, "source": "NADRA.txt",
         "content": "Smart NIC Renewal costs Rs. 750 in the normal category."},
    ]
    answer = "Smart NIC Renewal costs Rs. 750 in the normal category [source: NADRA.txt]"

    result = citation_verifier.verify_citations(answer, chunks)

    assert result["passed"] is True
    assert result["status"] == "verified"


def test_split_claims_treats_each_bullet_as_its_own_claim():
    """compressor.split_sentences() collapsed this whole answer into ONE unit
    (bullets carry no terminal punctuation and 'CNIC number' is under the old
    15-char floor), which is how a 1-item sample produced 'Hal rate: 0%'."""
    claims = hallucination_eval.split_claims(
        "To renew your CNIC you need to provide:\n- CNIC number\n- Original expired CNIC"
    )

    assert claims == ["CNIC number", "Original expired CNIC"]   # lead-in dropped


def test_hallucination_eval_scores_each_bullet_separately():
    chunks = [{"chunk_id": 1, "content": "CNIC renewal requires your CNIC number.",
               "source": "NADRA.txt"}]
    answer = "You need:\n- CNIC number\n- Zebra quokka bicycle telescope pyramid volcano"

    result = hallucination_eval.evaluate_hallucination(answer, chunks)

    assert result["evaluated_count"] == 2
    assert result["status"] == "evaluated"
    assert result["hallucinated_count"] >= 1


def test_hallucination_rate_is_none_when_nothing_could_be_evaluated():
    """Must not report 0.0 — 'no claim was checked' is not 'no hallucination'."""
    result = hallucination_eval.evaluate_hallucination("Requirements:", [])

    assert result["hallucination_rate"] is None
    assert result["status"] == "not_evaluated"
    assert result["evaluated_count"] == 0


# ---------------------------------------------------------------------------
# 12. compressor.py — structured chunks must not lose list items
# ---------------------------------------------------------------------------
NADRA_RENEWAL_CHUNK = (
    "## CNIC RENEWAL\n"
    "\n"
    "Eligibility:\n"
    "As per Rule 12 of NADRA (NIC) Rules, 2002, a citizen shall at any time but not "
    "later than one month after the date of expiry or early termination of validity "
    "period, must apply for renewal of his/her card.\n"
    "\n"
    "Requirements:\n"
    "- CNIC number\n"
    "\n"
    "Process of applying through NADRA Registration Centre:\n"
    "1. Token Acquisition - Obtain a Queue Matic token upon arrival.\n"
    "2. Biometric Capture - Digital capture of live photograph and fingerprints.\n"
    "3. OIC Interview / Approval - A brief interview conducted by the Officer-In-Charge.\n"
    "4. Payment - Submission of the required processing fee.\n"
    "5. Card Printing - Card is printed.\n"
)


def test_compressor_keeps_numbered_procedure_intact():
    """
    Regression on the real failure: sentence-level compression kept process
    steps 1, 4 and 5 of this chunk and deleted 2 and 3 — silently dropping
    "Biometric Capture", the step that requires an in-person visit. A partial
    procedure is worse than none: it looks complete.
    """
    model = reranker.get_cross_encoder()
    chunk = {"chunk_id": 5, "content": NADRA_RENEWAL_CHUNK, "source": "NADRA.txt",
             "rerank_score": 3.0, "rank": 1}

    result = compressor.compress_chunk(
        "What documents do I need to renew my CNIC?", chunk, model)

    content = result["content"]
    kept_steps = [n for n in "12345" if f"\n{n}. " in "\n" + content]
    assert kept_steps in ([], list("12345")), \
        f"procedure was partially kept: steps {kept_steps}"
    assert "Biometric Capture" in content


def test_compressor_keeps_requirements_list_with_its_lead_in():
    model = reranker.get_cross_encoder()
    chunk = {"chunk_id": 5, "content": NADRA_RENEWAL_CHUNK, "source": "NADRA.txt",
             "rerank_score": 3.0, "rank": 1}

    result = compressor.compress_chunk(
        "What documents do I need to renew my CNIC?", chunk, model)

    assert "Requirements:" in result["content"]
    assert "- CNIC number" in result["content"]
    # The section heading must survive: without it the model cannot tell
    # renewal rules apart from the new-registration rules in the same file.
    assert "## CNIC RENEWAL" in result["content"]


def test_compressor_leaves_prose_chunks_on_the_sentence_path():
    """is_structured() must not hijack ordinary prose (no lists present)."""
    prose = (
        "NADRA was established in 2000 under the Ministry of Interior. "
        "The authority issues CNICs to Pakistani citizens. "
        "It also manages biometric records nationwide. "
        "The weather in Islamabad is mild in spring. "
        "Offices open at nine in the morning."
    )
    assert compressor.is_structured(prose) is False


# ---------------------------------------------------------------------------
# 12b. completeness_check.py — faithful but useless answers
# ---------------------------------------------------------------------------
COMPLETE_ANSWER = (
    "- CNIC number [source: NADRA.txt]\n"
    "At the centre:\n"
    "- Obtain a Queue Matic token upon arrival [source: NADRA.txt]\n"
    "- Biometric capture of live photograph and fingerprints [source: NADRA.txt]\n"
    "- Brief interview with the Officer-In-Charge [source: NADRA.txt]\n"
    "- Payment of the processing fee [source: NADRA.txt]\n"
    "- Card printing [source: NADRA.txt]"
)


def test_completeness_flags_the_reported_one_bullet_answer():
    """
    "- CNIC number" is a verbatim, grounded, correctly-cited quote of NADRA's
    Requirements line — it passes citation AND hallucination checks. It is
    still useless: the same chunk lists the five centre steps, including the
    biometric capture that forces an in-person visit.
    """
    chunks = [{"chunk_id": 5, "source": "NADRA.txt",
               "content": NADRA_RENEWAL_CHUNK, "rank": 1}]

    result = completeness_check.check_completeness(
        "- CNIC number [source: NADRA.txt]", chunks)

    assert result["passed"] is False
    assert result["status"] == "thin"
    assert any("Biometric Capture" in b["text"] for b in result["unused_blocks"])


def test_completeness_passes_an_answer_that_used_the_context():
    chunks = [{"chunk_id": 5, "source": "NADRA.txt",
               "content": NADRA_RENEWAL_CHUNK, "rank": 1}]

    result = completeness_check.check_completeness(COMPLETE_ANSWER, chunks)

    assert result["passed"] is True
    assert result["unused_blocks"] == []


def test_completeness_does_not_flag_a_short_answer_with_nothing_more_to_say():
    """No unused blocks in context => no retry, however short the answer."""
    chunks = [{"chunk_id": 1, "source": "NADRA.txt", "rank": 1,
               "content": "The CNIC renewal fee is Rs. 750 in the normal category."}]

    result = completeness_check.check_completeness(
        "- The renewal fee is Rs. 750 [source: NADRA.txt]", chunks)

    assert result["passed"] is True


def test_completeness_is_skipped_when_there_is_no_context():
    result = completeness_check.check_completeness("Some answer.", [])
    assert result["passed"] is True
    assert result["status"] == "not_applicable"


def test_retry_feedback_quotes_the_dropped_passage_verbatim():
    """A generic 'be more complete' retry reproduced the same short answer in
    testing — the omitted text has to be quoted back."""
    blocks = [{"source": "NADRA.txt",
               "text": "Process:\n1. Token Acquisition\n2. Biometric Capture"}]

    feedback = completeness_check.build_retry_feedback(blocks)

    assert "Biometric Capture" in feedback
    assert "NADRA.txt" in feedback


def test_generator_appends_retry_feedback_after_the_question():
    """Placement matters: a revision instruction above the context reads as
    background rather than as an instruction."""
    messages = generator.build_messages(
        "q", "some context", "factual", feedback="REVISE: add the steps.")

    user = messages[1]["content"]
    assert user.index("REVISE: add the steps.") > user.index("Question: q")


# ---------------------------------------------------------------------------
# 13. graph.py — the retry edge was unreachable
# ---------------------------------------------------------------------------
def test_route_after_verification_retries_once_then_stops():
    """
    generator_node used to increment retry_count on the FIRST generation, so
    route_after_verification always saw retry_count=1 and `1 < MAX_RETRIES(1)`
    was False — the retry edge could never be taken by any question.
    """
    import graph

    failed = {"citation_result": {"passed": False},
              "hallucination_result": {"passed": True}}

    assert graph.route_after_verification({**failed, "retry_count": 0}) == "generator"
    assert graph.route_after_verification({**failed, "retry_count": 1}) == "end"


def test_incomplete_answer_triggers_a_retry():
    """A thin answer passes citation and hallucination checks — completeness
    is the only stage that can catch it, so it must reach the router."""
    import graph

    state = {"citation_result": {"passed": True},
             "hallucination_result": {"passed": True},
             "completeness_result": {"passed": False},
             "retry_count": 0}

    assert graph.route_after_verification(state) == "generator"


def test_generator_node_passes_retry_feedback_only_on_a_retry():
    import graph

    FAKE_GROQ_RESPONSES.append("Short. [source: a.pdf]")
    captured = {}
    original = graph.generate

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    graph.generate = spy
    try:
        graph.generator_node({"question": "q", "intent": "factual",
                              "compressed_chunks": [], "generation_attempts": 0,
                              "retry_feedback": "ADD THE STEPS"})
        assert captured["feedback"] is None       # first attempt: no feedback

        FAKE_GROQ_RESPONSES.append("Longer. [source: a.pdf]")
        graph.generator_node({"question": "q", "intent": "factual",
                              "compressed_chunks": [], "generation_attempts": 1,
                              "retry_feedback": "ADD THE STEPS"})
        assert captured["feedback"] == "ADD THE STEPS"
    finally:
        graph.generate = original


def test_route_after_verification_survives_null_results():
    """verification_agent_node sets these to None when there is no answer;
    state.get(k, {}) returns None, and None.get() would crash the router."""
    import graph

    state = {"citation_result": None, "hallucination_result": None, "retry_count": 0}
    assert graph.route_after_verification(state) == "end"


def test_generator_node_counts_attempts_and_retries_separately():
    import graph

    FAKE_GROQ_RESPONSES.append("An answer. [source: a.pdf]")
    first = graph.generator_node({"question": "q", "intent": "factual",
                                  "compressed_chunks": []})

    assert first["retry_count"] == 0            # no retry has happened yet
    assert first["generation_attempts"] == 1
    assert first["generation_meta"]["is_retry"] is False

    FAKE_GROQ_RESPONSES.append("A better answer. [source: a.pdf]")
    second = graph.generator_node({"question": "q", "intent": "factual",
                                   "compressed_chunks": [],
                                   "generation_attempts": 1})

    assert second["retry_count"] == 1
    assert second["generation_attempts"] == 2
    assert second["generation_meta"]["is_retry"] is True


# ---------------------------------------------------------------------------
# 14. reranker.py — relative noise floor
# ---------------------------------------------------------------------------
def test_reranker_drops_negative_chunks_when_enough_relevant_ones_exist(monkeypatch):
    """Cross-domain noise (a driving-licence passage on a CNIC question) used
    to reach the prompt because MIN_RERANK_SCORE is an always-return-something
    floor of -5."""
    chunks = [{"chunk_id": i, "content": f"c{i}", "source": "s"} for i in range(5)]
    monkeypatch.setattr(reranker, "get_cross_encoder",
                        lambda: type("M", (), {"predict": staticmethod(
                            lambda pairs: np.array([3.0, 2.0, 1.0, -0.5, -3.0]))})())

    results = reranker.rerank("q", chunks, top_n=10)

    assert [c["chunk_id"] for c in results] == [0, 1, 2]


def test_reranker_keeps_weak_chunks_when_nothing_clears_the_floor(monkeypatch):
    """The floor must never empty the context — a thin answer beats none."""
    chunks = [{"chunk_id": i, "content": f"c{i}", "source": "s"} for i in range(3)]
    monkeypatch.setattr(reranker, "get_cross_encoder",
                        lambda: type("M", (), {"predict": staticmethod(
                            lambda pairs: np.array([-0.2, -1.0, -2.0]))})())

    results = reranker.rerank("q", chunks, top_n=10)

    assert len(results) == 3
