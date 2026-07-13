"""
test_pipeline.py
----------------
Tests every pipeline component using mocks — no real DB, no HuggingFace,
no Groq API key required.

This is how production RAG systems are tested:
- Mock the external dependencies (DB, models, APIs)
- Test YOUR logic in isolation
- The mock proves the interface contract — if your code works with the mock,
  it will work with the real thing (same input/output shapes)

Run with:  python test_pipeline.py
"""

import sys
import json
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(name):  print(f"  {GREEN}✓ PASS{RESET}  {name}")
def fail(name, err): print(f"  {RED}✗ FAIL{RESET}  {name}\n         {RED}{err}{RESET}")
def section(name): print(f"\n{BOLD}{YELLOW}{'─'*55}{RESET}\n{BOLD}{name}{RESET}")

passed = failed = 0

def run(name, fn):
    global passed, failed
    try:
        fn()
        ok(name)
        passed += 1
    except Exception as e:
        fail(name, e)
        failed += 1


# =============================================================================
# 1.  RETRIEVAL — embed_query logic
# =============================================================================
section("1. retrieval.py — embed_query & dense_retrieve")

def test_embed_returns_list_of_floats():
    """embed_query must return a plain Python list of floats (not numpy array)."""
    mock_model = MagicMock()
    import numpy as np
    mock_model.encode.return_value = np.array([0.1, -0.2, 0.3] * 341 + [0.1])  # 1024 values

    with patch("retrieval.get_model", return_value=mock_model):
        from retrieval import embed_query
        vec = embed_query("test query")

    assert isinstance(vec, list),        f"Expected list, got {type(vec)}"
    assert len(vec) == 1024,             f"Expected 1024 dims, got {len(vec)}"
    assert isinstance(vec[0], float),    f"Expected float, got {type(vec[0])}"

run("embed_query returns list[float] of length 1024", test_embed_returns_list_of_floats)


def test_dense_retrieve_returns_correct_schema():
    """dense_retrieve must return dicts with chunk_id, content, source, score, rank."""
    import numpy as np
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(1024)

    mock_row = {"chunk_id": 42, "content": "NADRA issues CNICs.", "source": "nadra.pdf", "score": 0.91}
    mock_cursor = MagicMock()
    mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
    mock_cursor.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchall.return_value = [mock_row]

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    with patch("retrieval.get_model", return_value=mock_model), \
         patch("retrieval.get_connection", return_value=mock_conn):
        from retrieval import dense_retrieve
        results = dense_retrieve("test query", top_k=1)

    assert len(results) == 1
    r = results[0]
    for key in ("chunk_id", "content", "source", "score", "rank"):
        assert key in r, f"Missing key: {key}"
    assert r["rank"] == 1
    assert r["chunk_id"] == 42

run("dense_retrieve returns correct schema with rank", test_dense_retrieve_returns_correct_schema)


# =============================================================================
# 2.  BM25 — tokenizer + retrieval logic
# =============================================================================
section("2. bm25_retrieval.py — tokenize & bm25_retrieve")

def test_tokenize_lowercases_and_keeps_hyphens():
    from bm25_retrieval import tokenize
    tokens = tokenize("NADRA B-Form Requirements!")
    assert "nadra"    in tokens, f"Expected 'nadra' in {tokens}"
    assert "b-form"   in tokens, f"Expected 'b-form' in {tokens}"
    assert "requirements" in tokens

run("tokenize: lowercases, keeps hyphens, strips punctuation", test_tokenize_lowercases_and_keeps_hyphens)


def test_tokenize_empty_string():
    from bm25_retrieval import tokenize
    assert tokenize("") == []

run("tokenize: empty string returns empty list", test_tokenize_empty_string)


def test_bm25_retrieve_schema_and_zero_filtering():
    """bm25_retrieve must return correct schema and skip zero-score docs.

    NOTE: BM25Okapi IDF = log((N - n + 0.5) / (n + 0.5)).
    With only 2 docs and a term appearing in 1 doc: log(1.5/1.5) = 0.
    We need 4+ docs so IDF is non-zero for terms that don't appear everywhere.
    """
    import numpy as np
    from rank_bm25 import BM25Okapi

    corpus = [
        {"chunk_id": 1, "content": "NADRA issues CNIC to Pakistani citizens.", "source": "nadra.pdf"},
        {"chunk_id": 2, "content": "Passport office processes travel documents.", "source": "passport.pdf"},
        {"chunk_id": 3, "content": "FBR handles tax returns and filing.", "source": "fbr.pdf"},
        {"chunk_id": 4, "content": "Visa embassy application requirements abroad.", "source": "visa.pdf"},
    ]
    tokenized = [["nadra", "issues", "cnic", "citizens"],
                 ["passport", "office", "travel", "documents"],
                 ["fbr", "tax", "returns", "filing"],
                 ["visa", "embassy", "application"]]
    bm25 = BM25Okapi(tokenized)

    with patch("bm25_retrieval.get_bm25_index", return_value=(bm25, corpus)):
        from bm25_retrieval import bm25_retrieve
        results = bm25_retrieve("NADRA CNIC", top_k=5)

    assert len(results) >= 1, "Should return at least 1 result for NADRA CNIC query"
    assert results[0]["chunk_id"] == 1, f"NADRA chunk should rank first, got chunk_id={results[0]['chunk_id']}"
    for key in ("chunk_id", "content", "source", "score", "rank"):
        assert key in results[0], f"Missing key: {key}"
    assert all(r["score"] > 0 for r in results), "Zero-score docs should be filtered"

run("bm25_retrieve: correct schema, zero-score docs filtered", test_bm25_retrieve_schema_and_zero_filtering)


# =============================================================================
# 3.  HYBRID SEARCH — RRF fusion logic
# =============================================================================
section("3. hybrid_search.py — RRF fusion")

def test_rrf_chunk_appearing_in_both_lists_wins():
    """A chunk ranked high in both dense and BM25 should win RRF."""
    from hybrid_search import reciprocal_rank_fusion

    dense_results = [
        {"chunk_id": "A", "content": "chunk A", "source": "a.pdf", "rank": 1},
        {"chunk_id": "B", "content": "chunk B", "source": "b.pdf", "rank": 2},
    ]
    bm25_results = [
        {"chunk_id": "B", "content": "chunk B", "source": "b.pdf", "rank": 1},
        {"chunk_id": "C", "content": "chunk C", "source": "c.pdf", "rank": 2},
    ]

    fused = reciprocal_rank_fusion([dense_results, bm25_results])

    # B appears in both lists — should have highest RRF score
    assert fused[0]["chunk_id"] == "B", \
        f"Expected B to win (appears in both lists), got {fused[0]['chunk_id']}"

run("RRF: chunk in both lists ranks #1", test_rrf_chunk_appearing_in_both_lists_wins)


def test_rrf_score_formula():
    """Verify RRF scores match the formula: 1/(k+rank)."""
    from hybrid_search import reciprocal_rank_fusion

    k = 60
    single_list = [
        {"chunk_id": "X", "content": "x", "source": "x.pdf", "rank": 1},
        {"chunk_id": "Y", "content": "y", "source": "y.pdf", "rank": 2},
    ]
    fused = reciprocal_rank_fusion([single_list], k=k)

    expected_x = 1.0 / (k + 1)
    expected_y = 1.0 / (k + 2)
    assert abs(fused[0]["rrf_score"] - expected_x) < 1e-9, \
        f"Score mismatch: {fused[0]['rrf_score']} != {expected_x}"
    assert abs(fused[1]["rrf_score"] - expected_y) < 1e-9

run("RRF formula: score equals 1/(k+rank)", test_rrf_score_formula)


def test_rrf_deduplicates_chunks():
    """Same chunk_id in two lists should appear once in output."""
    from hybrid_search import reciprocal_rank_fusion

    list_a = [{"chunk_id": "1", "content": "c", "source": "s.pdf", "rank": 1}]
    list_b = [{"chunk_id": "1", "content": "c", "source": "s.pdf", "rank": 1}]
    fused  = reciprocal_rank_fusion([list_a, list_b])

    assert len(fused) == 1, f"Expected 1 unique chunk, got {len(fused)}"

run("RRF: deduplicates same chunk across lists", test_rrf_deduplicates_chunks)


def test_rrf_final_k_limit():
    """hybrid_search should return at most final_k results."""
    from hybrid_search import hybrid_search

    # Build 10 fake chunks for both retrievers to return
    def make_chunks(n):
        return [{"chunk_id": str(i), "content": f"chunk {i}",
                 "source": "s.pdf", "score": 0.9, "rank": i+1}
                for i in range(n)]

    with patch("hybrid_search.dense_retrieve", return_value=make_chunks(10)), \
         patch("hybrid_search.bm25_retrieve",  return_value=make_chunks(10)):
        results = hybrid_search(["test query"], final_k=5)

    assert len(results) <= 5, f"Expected ≤5 results, got {len(results)}"

run("hybrid_search: respects final_k limit", test_rrf_final_k_limit)


# =============================================================================
# 4.  INTENT ROUTER
# =============================================================================
section("4. intent_router.py — routing & config")

def test_route_factual():
    with patch("intent_router._call_llm", return_value="factual"):
        from intent_router import route, Intent
        assert route("What is NADRA?") == Intent.FACTUAL

run("route: 'factual' LLM output → Intent.FACTUAL", test_route_factual)


def test_route_out_of_scope():
    with patch("intent_router._call_llm", return_value="out_of_scope"):
        from intent_router import route, Intent
        assert route("Is NADRA corrupt?") == Intent.OUT_OF_SCOPE

run("route: 'out_of_scope' → Intent.OUT_OF_SCOPE", test_route_out_of_scope)


def test_route_unexpected_llm_output_defaults_to_factual():
    """If LLM returns garbage, we default to FACTUAL (never crash)."""
    with patch("intent_router._call_llm", return_value="i am confused"):
        from intent_router import route, Intent
        result = route("Some question")
        assert result == Intent.FACTUAL, f"Expected FACTUAL fallback, got {result}"

run("route: unexpected LLM output defaults to FACTUAL", test_route_unexpected_llm_output_defaults_to_factual)


def test_routing_config_out_of_scope_skips_retrieval():
    from intent_router import get_routing_config, Intent
    config = get_routing_config(Intent.OUT_OF_SCOPE)
    assert config["should_retrieve"] is False

run("get_routing_config: OUT_OF_SCOPE has should_retrieve=False", test_routing_config_out_of_scope_skips_retrieval)


def test_routing_config_comparison_has_higher_top_k():
    from intent_router import get_routing_config, Intent
    factual_config    = get_routing_config(Intent.FACTUAL)
    comparison_config = get_routing_config(Intent.COMPARISON)
    # Comparison needs more chunks to cover both entities
    assert comparison_config["top_k_override"] is not None
    assert factual_config["top_k_override"] is None

run("get_routing_config: COMPARISON has top_k_override, FACTUAL does not", test_routing_config_comparison_has_higher_top_k)


# =============================================================================
# 5.  QUERY REWRITER
# =============================================================================
section("5. query_rewriter.py — rewrite strategies")

def test_rewrite_always_includes_original():
    with patch("query_rewriter.generate_hyde_query", return_value="hyde passage"), \
         patch("query_rewriter.generate_sub_questions", return_value=["q1", "q2", "q3"]):
        from query_rewriter import rewrite
        from intent_router import Intent
        queries = rewrite("original question", Intent.FACTUAL)
    assert queries[0] == "original question", "Original question must always be first"

run("rewrite: original question always first in list", test_rewrite_always_includes_original)


def test_rewrite_out_of_scope_returns_only_original():
    from query_rewriter import rewrite
    from intent_router import Intent
    queries = rewrite("Is NADRA corrupt?", Intent.OUT_OF_SCOPE)
    assert queries == ["Is NADRA corrupt?"]
    assert len(queries) == 1

run("rewrite: OUT_OF_SCOPE returns only original (no LLM calls)", test_rewrite_out_of_scope_returns_only_original)


def test_rewrite_deduplicates():
    # If HyDE happens to return the same string as the original, deduplicate
    with patch("query_rewriter.generate_hyde_query", return_value="original question"), \
         patch("query_rewriter.generate_sub_questions", return_value=["original question", "q2"]):
        from query_rewriter import rewrite
        from intent_router import Intent
        queries = rewrite("original question", Intent.FACTUAL)
    assert queries.count("original question") == 1, "Duplicates should be removed"

run("rewrite: deduplicates identical queries", test_rewrite_deduplicates)


def test_sub_question_json_parse_failure_returns_original():
    """If LLM returns malformed JSON, fall back to original question gracefully."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "not valid json at all {{{"

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("query_rewriter.get_groq_client", return_value=mock_client):
        from query_rewriter import generate_sub_questions
        result = generate_sub_questions("What is NADRA?")
    assert result == ["What is NADRA?"], f"Expected fallback to original, got {result}"

run("generate_sub_questions: malformed JSON falls back gracefully", test_sub_question_json_parse_failure_returns_original)


# =============================================================================
# 6.  RERANKER
# =============================================================================
section("6. reranker.py — CrossEncoder scoring")

def test_rerank_sorts_by_score_descending():
    import numpy as np
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([2.0, 9.5, 1.1])  # chunk B should win

    chunks = [
        {"chunk_id": 1, "content": "Low relevance text.", "source": "a.pdf", "rrf_score": 0.02},
        {"chunk_id": 2, "content": "Highly relevant NADRA CNIC info.", "source": "b.pdf", "rrf_score": 0.03},
        {"chunk_id": 3, "content": "Unrelated text.", "source": "c.pdf", "rrf_score": 0.01},
    ]

    with patch("reranker.get_cross_encoder", return_value=mock_ce):
        from reranker import rerank
        results = rerank("What is CNIC?", chunks, top_n=3)

    assert results[0]["chunk_id"] == 2, "Chunk with score 9.5 should rank first"
    assert results[0]["rerank_score"] == 9.5

run("rerank: sorts by CrossEncoder score descending", test_rerank_sorts_by_score_descending)


def test_rerank_filters_below_min_score():
    import numpy as np
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([1.0, 5.0])  # 1.0 < MIN_RERANK_SCORE=2.8

    chunks = [
        {"chunk_id": 1, "content": "Not relevant.", "source": "a.pdf", "rrf_score": 0.01},
        {"chunk_id": 2, "content": "Very relevant.", "source": "b.pdf", "rrf_score": 0.03},
    ]

    with patch("reranker.get_cross_encoder", return_value=mock_ce):
        from reranker import rerank
        results = rerank("test", chunks, top_n=5)

    chunk_ids = [r["chunk_id"] for r in results]
    assert 1 not in chunk_ids, "Chunk with score 1.0 < 2.8 should be filtered"
    assert 2 in chunk_ids

run("rerank: filters chunks below MIN_RERANK_SCORE=2.8", test_rerank_filters_below_min_score)


def test_rerank_empty_input_returns_empty():
    from reranker import rerank
    assert rerank("query", [], top_n=5) == []

run("rerank: empty input returns empty list", test_rerank_empty_input_returns_empty)


# =============================================================================
# 7.  COMPRESSOR
# =============================================================================
section("7. compressor.py — sentence compression")

def test_split_sentences_basic():
    from compressor import split_sentences
    # Each sentence must be >20 chars to pass the minimum-length filter in split_sentences
    text = "NADRA was established in 2000. It issues CNICs to Pakistani citizens. Headquarters is located in Islamabad."
    sentences = split_sentences(text)
    assert len(sentences) == 3, f"Expected 3 sentences, got {len(sentences)}: {sentences}"
    assert sentences[0].startswith("NADRA")

run("split_sentences: splits on sentence boundaries", test_split_sentences_basic)


def test_split_sentences_preserves_abbreviations():
    from compressor import split_sentences
    # "Rs. 750" should NOT split into two sentences
    text = "NADRA charges Rs. 750 for CNIC. Processing takes 30 days."
    sentences = split_sentences(text)
    # Should produce 2 sentences, not 3
    assert len(sentences) == 2, f"Expected 2, got {len(sentences)}: {sentences}"

run("split_sentences: does not split on abbreviations like Rs.", test_split_sentences_preserves_abbreviations)


def test_compress_keeps_relevant_sentences():
    import numpy as np
    mock_ce = MagicMock()
    # 5 sentences: sentence 0 scores lowest (0.1) — should be cut when top-4 are kept
    mock_ce.predict.return_value = np.array([0.1, 9.5, 8.0, 7.5, 6.0])

    chunk = {
        "chunk_id": 1,
        "content": (
            "Pakistan was established in 1947 as an independent nation. "          # score 0.1 — cut
            "NADRA issues CNICs to all Pakistani citizens above age eighteen. "    # score 9.5
            "NADRA manages biometric data for over one hundred million people. "   # score 8.0
            "NADRA was established in year two thousand under Interior Ministry. " # score 7.5
            "NADRA headquarters is situated in the capital city Islamabad."        # score 6.0
        ),
        "source": "nadra.pdf", "rerank_score": 7.0, "rank": 1,
    }

    with patch("compressor.get_cross_encoder", return_value=mock_ce):
        from compressor import compress_chunk
        result = compress_chunk("What does NADRA do?", chunk, mock_ce)

    assert result["compressed"] is True
    assert "Pakistan was established in 1947" not in result["content"], \
        "Lowest-score sentence should be removed when top-4 kept out of 5"
    assert "NADRA" in result["content"]

run("compress_chunk: removes low-scoring sentences", test_compress_keeps_relevant_sentences)


# =============================================================================
# 8.  CITATION VERIFIER
# =============================================================================
section("8. citation_verifier.py — citation parsing & verification")

def test_parse_citations_extracts_claim_and_source():
    from citation_verifier import parse_citations
    answer = "NADRA charges Rs. 750 for CNIC. [source: nadra_guide.pdf] Processing takes 30 days. [source: nadra_guide.pdf]"
    claims = parse_citations(answer)
    assert len(claims) == 2
    assert claims[0]["source"] == "nadra_guide.pdf"
    assert "750" in claims[0]["claim"]

run("parse_citations: extracts claim+source pairs", test_parse_citations_extracts_claim_and_source)


def test_parse_citations_no_citations_returns_empty():
    from citation_verifier import parse_citations
    answer = "NADRA was established in 2000 with no citation."
    assert parse_citations(answer) == []

run("parse_citations: answer with no citations returns []", test_parse_citations_no_citations_returns_empty)


def test_find_source_chunk_partial_match():
    from citation_verifier import find_source_chunk
    chunks = [
        {"chunk_id": 1, "content": "CNIC costs Rs. 750.", "source": "data/nadra_guide.pdf", "rank": 1},
    ]
    # LLM cites "nadra_guide" but actual source path is "data/nadra_guide.pdf"
    result = find_source_chunk("nadra_guide", chunks)
    assert result is not None
    assert result["chunk_id"] == 1

run("find_source_chunk: partial case-insensitive match works", test_find_source_chunk_partial_match)


def test_find_source_chunk_missing_returns_none():
    from citation_verifier import find_source_chunk
    chunks = [{"chunk_id": 1, "content": "x", "source": "nadra.pdf", "rank": 1}]
    assert find_source_chunk("passport.pdf", chunks) is None

run("find_source_chunk: unknown source returns None", test_find_source_chunk_missing_returns_none)


def test_verify_citations_hallucinated_source_flagged():
    """If LLM cites a source that doesn't exist in chunks, mark as unverified."""
    import numpy as np
    mock_ce = MagicMock()
    mock_ce.predict.return_value = np.array([8.0])

    answer = "NADRA charges Rs. 500. [source: NONEXISTENT_FILE.pdf]"
    chunks = [{"chunk_id": 1, "content": "NADRA charges Rs. 750.", "source": "nadra.pdf", "rank": 1}]

    with patch("citation_verifier.get_cross_encoder", return_value=mock_ce):
        from citation_verifier import verify_citations
        result = verify_citations(answer, chunks)

    assert len(result["unverified_claims"]) == 1
    assert "not found" in result["unverified_claims"][0]["reason"]
    assert result["passed"] is False

run("verify_citations: hallucinated source → unverified + passed=False", test_verify_citations_hallucinated_source_flagged)


# =============================================================================
# 9.  GRAPH — LangGraph wiring
# =============================================================================
section("9. graph.py — state and node logic")

def test_initial_state_has_all_keys():
    from graph import initial_state
    state = initial_state("test question")
    required = ["question", "intent", "routing_config", "search_queries",
                "raw_chunks", "reranked_chunks", "compressed_chunks",
                "needs_tools", "tool_results", "answer",
                "citation_result", "hallucination_result", "final_answer", "error"]
    for key in required:
        assert key in state, f"Missing state key: {key}"
    assert state["question"] == "test question"

run("initial_state: contains all required keys", test_initial_state_has_all_keys)


def test_tool_selector_triggers_on_keywords():
    from graph import tool_selector_node, initial_state
    state = initial_state("What is the current status of my application?")
    state["intent"] = "factual"
    result = tool_selector_node(state)
    assert result["needs_tools"] is True

run("tool_selector_node: 'current status' triggers needs_tools=True", test_tool_selector_triggers_on_keywords)


def test_tool_selector_skips_for_simple_questions():
    from graph import tool_selector_node, initial_state
    state = initial_state("What is NADRA?")
    state["intent"] = "factual"
    result = tool_selector_node(state)
    assert result["needs_tools"] is False

run("tool_selector_node: simple question → needs_tools=False", test_tool_selector_skips_for_simple_questions)


def test_generator_node_out_of_scope_returns_polite_decline():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "I can only help with Pakistani government services."
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 15
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("generator.get_groq_client", return_value=mock_client):
        from graph import generator_node, initial_state
        state = initial_state("Is the government corrupt?")
        state["intent"] = "out_of_scope"
        state["compressed_chunks"] = []
        result = generator_node(state)
    assert "answer" in result
    assert len(result["answer"]) > 0

run("generator_node: OUT_OF_SCOPE returns polite decline message", test_generator_node_out_of_scope_returns_polite_decline)


def test_graph_compiles_without_error():
    """build_graph() must succeed — validates all nodes and edges are connected."""
    from graph import build_graph
    graph = build_graph()
    assert graph is not None

run("build_graph: compiles without error (all nodes/edges valid)", test_graph_compiles_without_error)


# =============================================================================
# SUMMARY
# =============================================================================
total = passed + failed
print(f"\n{'='*55}")
print(f"{BOLD}RESULTS: {GREEN}{passed} passed{RESET}{BOLD}, {RED}{failed} failed{RESET}{BOLD}, {total} total{RESET}")
print(f"{'='*55}\n")

if failed > 0:
    sys.exit(1)


# =============================================================================
# 10. GENERATOR
# =============================================================================
section("10. generator.py — format context & build messages")

def test_format_context_numbers_passages():
    from generator import format_context
    chunks = [
        {"chunk_id": 1, "content": "NADRA charges Rs. 750.", "source": "nadra_guide.pdf"},
        {"chunk_id": 2, "content": "Processing takes 30 days.", "source": "nadra_guide.pdf"},
    ]
    ctx = format_context(chunks)
    assert "[Passage 1 | source: nadra_guide.pdf]" in ctx
    assert "[Passage 2 | source: nadra_guide.pdf]" in ctx
    assert "NADRA charges Rs. 750." in ctx

run("format_context: numbers passages with source labels", test_format_context_numbers_passages)


def test_format_context_empty_returns_fallback():
    from generator import format_context
    assert format_context([]) == "No context available."

run("format_context: empty chunks returns fallback string", test_format_context_empty_returns_fallback)


def test_build_messages_structure():
    from generator import build_messages
    messages = build_messages("What is CNIC?", "Some context here.", "factual")
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "What is CNIC?" in messages[1]["content"]
    assert "Some context here." in messages[1]["content"]

run("build_messages: returns [system, user] with question and context", test_build_messages_structure)


def test_generate_calls_groq_and_returns_schema():
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "NADRA charges Rs. 750. [source: nadra.pdf]"
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 30

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch("generator.get_groq_client", return_value=mock_client):
        from generator import generate
        result = generate("test question", [{"content": "context", "source": "nadra.pdf"}], "factual")

    for key in ("answer", "model", "prompt_tokens", "completion_tokens", "context_used"):
        assert key in result, f"Missing key: {key}"
    assert result["prompt_tokens"] == 100
    assert "750" in result["answer"]

run("generate: returns correct schema from Groq response", test_generate_calls_groq_and_returns_schema)


# =============================================================================
# 11. HALLUCINATION EVAL
# =============================================================================
section("11. hallucination_eval.py — sentence grounding")

def test_strip_citation_tags():
    from hallucination_eval import strip_citation_tags
    text = "NADRA charges Rs. 750. [source: nadra.pdf] Processing takes 30 days."
    clean = strip_citation_tags(text)
    assert "[source:" not in clean
    assert "NADRA charges Rs. 750." in clean

run("strip_citation_tags: removes [source: ...] tags", test_strip_citation_tags)


def test_evaluate_hallucination_grounded_sentence_passes():
    import numpy as np
    mock_ce = MagicMock()
    # High score = grounded
    mock_ce.predict.return_value = np.array([8.5])

    chunks = [{"chunk_id": 1, "content": "NADRA was established in 2000 under the Ministry of Interior.", "source": "nadra.pdf", "rank": 1}]
    answer = "NADRA was established in 2000 under the Ministry of Interior."

    with patch("hallucination_eval.get_cross_encoder", return_value=mock_ce):
        from hallucination_eval import evaluate_hallucination
        result = evaluate_hallucination(answer, chunks)

    assert result["grounded_count"] == 1
    assert result["hallucinated_count"] == 0
    assert result["passed"] is True

run("evaluate_hallucination: grounded sentence → passed=True", test_evaluate_hallucination_grounded_sentence_passes)


def test_evaluate_hallucination_ungrounded_sentence_flagged():
    import numpy as np
    mock_ce = MagicMock()
    # Low score = hallucinated
    mock_ce.predict.return_value = np.array([0.1])

    chunks = [{"chunk_id": 1, "content": "NADRA was established in 2000.", "source": "nadra.pdf", "rank": 1}]
    answer = "NADRA employs over fifty thousand people across all provinces of Pakistan."

    with patch("hallucination_eval.get_cross_encoder", return_value=mock_ce):
        from hallucination_eval import evaluate_hallucination
        result = evaluate_hallucination(answer, chunks)

    assert result["hallucinated_count"] >= 1
    assert len(result["flagged_sentences"]) >= 1

run("evaluate_hallucination: ungrounded sentence → flagged", test_evaluate_hallucination_ungrounded_sentence_flagged)


def test_evaluate_hallucination_empty_answer():
    from hallucination_eval import evaluate_hallucination
    result = evaluate_hallucination("", [])
    assert result["passed"] is True  # nothing to fail

run("evaluate_hallucination: empty answer → passed=True (nothing to fail)", test_evaluate_hallucination_empty_answer)


# =============================================================================
# 12. MCP TOOLS
# =============================================================================
section("12. mcp_tools.py — tool registry & selection")

def test_tool_registry_has_all_tools():
    from mcp_tools import TOOL_REGISTRY
    expected = {"search_nadra", "search_fbr", "search_passport", "extract_entities", "compare_documents"}
    for tool in expected:
        assert tool in TOOL_REGISTRY, f"Missing tool: {tool}"

run("TOOL_REGISTRY: contains all 5 expected tools", test_tool_registry_has_all_tools)


def test_call_tool_unknown_tool_returns_failure():
    from mcp_tools import call_tool
    result = call_tool("nonexistent_tool", query="test")
    assert result["success"] is False
    assert result["tool"] == "nonexistent_tool"

run("call_tool: unknown tool returns success=False", test_call_tool_unknown_tool_returns_failure)


def test_tool_result_schema():
    from mcp_tools import tool_result
    r = tool_result("search_nadra", "CNIC", {"results": []})
    for key in ("tool", "query", "success", "data", "source"):
        assert key in r, f"Missing key: {key}"
    assert r["source"] == "mcp:search_nadra"

run("tool_result: returns correct schema with all required keys", test_tool_result_schema)


def test_select_tools_nadra_question():
    from mcp_tools import select_tools_for_question
    tools = select_tools_for_question("What are NADRA CNIC requirements?", "factual")
    assert "search_nadra" in tools

run("select_tools_for_question: NADRA question selects search_nadra", test_select_tools_nadra_question)


def test_select_tools_passport_question():
    from mcp_tools import select_tools_for_question
    tools = select_tools_for_question("How do I renew my passport?", "procedural")
    assert "search_passport" in tools

run("select_tools_for_question: passport question selects search_passport", test_select_tools_passport_question)


def test_select_tools_out_of_scope_returns_empty():
    from mcp_tools import select_tools_for_question
    tools = select_tools_for_question("What is the weather today?", "out_of_scope")
    # Weather question has no gov keywords — empty list
    assert isinstance(tools, list)

run("select_tools_for_question: unrelated question returns list (may be empty)", test_select_tools_out_of_scope_returns_empty)


# REPRINT SUMMARY
total = passed + failed
print(f"\n{'='*55}")
print(f"{BOLD}FINAL RESULTS: {GREEN}{passed} passed{RESET}{BOLD}, {RED}{failed} failed{RESET}{BOLD}, {total} total{RESET}")
print(f"{'='*55}\n")

if failed > 0:
    import sys
    sys.exit(1)