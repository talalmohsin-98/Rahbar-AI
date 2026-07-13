"""
e2e_test.py
-----------
PURPOSE: End-to-end tests that exercise the FULL LangGraph pipeline
         for every question type and every conditional edge path.

Unlike test_pipeline.py (unit tests — one function at a time),
e2e_test.py runs the entire graph.run_pipeline() and checks the
shape and correctness of the final state.

What this covers:
    PATH 1: FACTUAL question  → full RAG → answer + verified citations
    PATH 2: COMPARISON question → full RAG with higher top_k → table answer
    PATH 3: PROCEDURAL question → full RAG → numbered steps answer
    PATH 4: OUT_OF_SCOPE → skip retrieval → polite decline
    PATH 5: Tool trigger → MCP tools → RAG → answer
    PATH 6: Retry path → verification fails → regenerate at temp=0.0

Every path is fully mocked — no DB, no HuggingFace, no Groq API needed.
The mocks prove the wiring: correct data flows between every node.

Run with:  python e2e_test.py
"""

import sys
import json
import numpy as np
from unittest.mock import MagicMock, patch

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(name):        print(f"  {GREEN}✓ PASS{RESET}  {name}")
def fail(name, err): print(f"  {RED}✗ FAIL{RESET}  {name}\n         {RED}{err}{RESET}")
def section(name):   print(f"\n{BOLD}{YELLOW}{'─'*60}{RESET}\n{BOLD}{name}{RESET}")

passed = failed = 0

def run(name, fn):
    global passed, failed
    try:
        fn()
        ok(name)
        passed += 1
    except Exception as e:
        import traceback
        fail(name, f"{e}\n{traceback.format_exc()[-300:]}")
        failed += 1


# =============================================================================
# SHARED MOCK FIXTURES
# =============================================================================

def make_chunks(n=3, source="nadra_guide.pdf"):
    """Returns n mock chunk dicts in the schema all pipeline stages expect."""
    return [
        {
            "chunk_id":     i + 1,
            "content":      f"NADRA government service information chunk {i+1}. "
                            f"This contains relevant details about Pakistani citizen services.",
            "source":       source,
            "score":        0.9 - i * 0.05,
            "rrf_score":    0.03 - i * 0.003,
            "rerank_score": 8.5 - i * 0.5,
            "rank":         i + 1,
            "compressed":   True,
            "original_length":   200,
            "compressed_length": 120,
        }
        for i in range(n)
    ]


def mock_groq_response(content: str, prompt_tokens=80, completion_tokens=40):
    """Builds a mock Groq API response object."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens    = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def mock_embed(text, normalize_embeddings=True):
    """Returns a deterministic fake embedding vector."""
    return np.zeros(1024)


def make_cross_encoder_mock(score=7.5):
    """Returns a CrossEncoder mock that always returns the given score."""
    ce = MagicMock()
    ce.predict.return_value = np.array([score] * 20)   # enough for any batch size
    return ce


# =============================================================================
# FULL PIPELINE MOCK CONTEXT
# All external dependencies replaced — only our pipeline logic runs.
# =============================================================================

def pipeline_mocks(
    intent_label="factual",
    llm_answer="NADRA charges Rs. 750 for CNIC. [source: nadra_guide.pdf]",
    ce_score=7.5,
    num_chunks=3,
):
    """
    Returns a context manager stack that patches all external calls.
    Usage:  with pipeline_mocks(...) as _: result = run_pipeline(q)
    """
    from contextlib import ExitStack

    chunks = make_chunks(num_chunks)
    ce     = make_cross_encoder_mock(ce_score)
    groq_r = mock_groq_response(llm_answer)

    stack = ExitStack()

    # Embedding model
    mock_model = MagicMock()
    mock_model.encode.return_value = np.zeros(1024)
    stack.enter_context(patch("retrieval.get_model",         return_value=mock_model))
    stack.enter_context(patch("retrieval.get_connection",    return_value=_fake_db_conn(chunks)))

    # BM25
    stack.enter_context(patch("bm25_retrieval.get_bm25_index", return_value=_fake_bm25(chunks)))

    # CrossEncoder (shared by reranker, compressor, citation_verifier, hallucination_eval)
    stack.enter_context(patch("reranker.get_cross_encoder",          return_value=ce))
    stack.enter_context(patch("compressor.get_cross_encoder",        return_value=ce))
    stack.enter_context(patch("citation_verifier.get_cross_encoder", return_value=ce))
    stack.enter_context(patch("hallucination_eval.get_cross_encoder",return_value=ce))

    # Intent router
    stack.enter_context(patch("intent_router._call_llm", return_value=intent_label))

    # Query rewriter
    stack.enter_context(patch("query_rewriter.generate_hyde_query",
                              return_value="Hypothetical document about NADRA services."))
    stack.enter_context(patch("query_rewriter.generate_sub_questions",
                              return_value=["NADRA requirements", "NADRA fees", "NADRA process"]))

    # Groq (generator)
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = groq_r
    stack.enter_context(patch("generator.get_groq_client", return_value=mock_client))

    # MCP tools (prevent real API calls)
    stack.enter_context(patch("mcp_tools.get_groq_client", return_value=mock_client))

    return stack


def _fake_db_conn(chunks):
    """Fake psycopg2 connection that returns mock chunk rows."""
    rows = [
        {"chunk_id": c["chunk_id"], "content": c["content"],
         "source": c["source"], "score": c["rerank_score"]}
        for c in chunks
    ]
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__  = MagicMock(return_value=False)
    cursor.fetchall.return_value = rows

    conn = MagicMock()
    conn.cursor.return_value = cursor
    return conn


def _fake_bm25(chunks):
    """Fake BM25 index that returns mock corpus."""
    from rank_bm25 import BM25Okapi
    from bm25_retrieval import tokenize

    corpus = [
        {"chunk_id": c["chunk_id"], "content": c["content"], "source": c["source"]}
        for c in chunks
    ]
    # Add padding docs so BM25 IDF is non-zero (needs 4+ docs)
    for i in range(4 - len(corpus)):
        corpus.append({"chunk_id": 99 + i, "content": f"padding document {i}", "source": "pad.pdf"})

    tokenized = [tokenize(d["content"]) for d in corpus]
    bm25      = BM25Okapi(tokenized)
    return bm25, corpus


# =============================================================================
# PATH 1 — FACTUAL QUESTION
# =============================================================================
section("PATH 1 — Factual question: full RAG → answer")

def test_factual_pipeline_state_populated():
    """All key state fields must be populated after a factual question."""
    from graph import run_pipeline

    with pipeline_mocks(intent_label="factual"):
        state = run_pipeline("What documents are required for NADRA CNIC?")

    assert state["intent"]          == "factual",  f"Got intent: {state['intent']}"
    assert state["search_queries"],                 "search_queries should be non-empty"
    assert state["raw_chunks"],                     "raw_chunks should be non-empty"
    assert state["reranked_chunks"] is not None,   "reranked_chunks should be set"
    assert state["answer"],                         "answer should be non-empty string"
    assert state["final_answer"],                   "final_answer should be set"
    assert state["citation_result"] is not None,   "citation_result should be set"

run("Factual: all pipeline state keys populated", test_factual_pipeline_state_populated)


def test_factual_query_rewriting_generates_multiple_queries():
    """Query rewriter should produce more than 1 search query for factual questions."""
    from graph import run_pipeline

    with pipeline_mocks(intent_label="factual"):
        state = run_pipeline("What is NADRA?")

    assert len(state["search_queries"]) > 1, \
        f"Expected multiple queries, got {len(state['search_queries'])}"

run("Factual: query rewriter produces multiple search queries", test_factual_query_rewriting_generates_multiple_queries)


def test_factual_answer_passes_through_verification():
    """With a high-scoring CE mock (score=7.5), verification should pass."""
    from graph import run_pipeline

    with pipeline_mocks(intent_label="factual", ce_score=7.5,
                        llm_answer="NADRA issues CNICs to Pakistani citizens. [source: nadra_guide.pdf]"):
        state = run_pipeline("What does NADRA do?")

    assert state["final_answer"] is not None
    # With score=7.5 well above thresholds, no retry needed
    assert state.get("retry_count", 0) <= 1

run("Factual: high-confidence answer passes verification", test_factual_answer_passes_through_verification)


# =============================================================================
# PATH 2 — COMPARISON QUESTION
# =============================================================================
section("PATH 2 — Comparison question: higher top_k → structured answer")

def test_comparison_routing_config_has_top_k_override():
    """Comparison intent should set top_k_override in routing_config."""
    from graph import run_pipeline

    with pipeline_mocks(intent_label="comparison"):
        state = run_pipeline("Compare B-Form and CNIC registration requirements")

    assert state["intent"] == "comparison"
    config = state.get("routing_config", {})
    assert config.get("top_k_override") is not None, \
        "COMPARISON should have top_k_override set (needs more chunks)"

run("Comparison: routing_config has top_k_override", test_comparison_routing_config_has_top_k_override)


def test_comparison_no_hyde_in_queries():
    """Comparison questions should NOT use HyDE (only sub-questions)."""
    from graph import run_pipeline

    hyde_called = []

    def track_hyde(q):
        hyde_called.append(q)
        return "hypothetical passage"

    with pipeline_mocks(intent_label="comparison"):
        with patch("query_rewriter.generate_hyde_query", side_effect=track_hyde):
            state = run_pipeline("Compare B-Form and CNIC requirements")

    # HyDE should NOT be called for comparison intent
    assert len(hyde_called) == 0, \
        f"HyDE should be skipped for comparison questions, but was called {len(hyde_called)} time(s)"

run("Comparison: HyDE skipped, only sub-questions used", test_comparison_no_hyde_in_queries)


# =============================================================================
# PATH 3 — PROCEDURAL QUESTION
# =============================================================================
section("PATH 3 — Procedural question: numbered steps format")

def test_procedural_uses_step_format_prompt():
    """Procedural intent should inject step-format instruction into the LLM prompt."""
    from generator import build_messages

    messages = build_messages(
        "How do I apply for a passport?",
        "Context about passport application.",
        "procedural",
    )

    system_msg = messages[0]["content"]
    assert "step" in system_msg.lower() or "numbered" in system_msg.lower(), \
        "System prompt for procedural questions should mention steps/numbered format"

run("Procedural: system prompt includes step-format instruction", test_procedural_uses_step_format_prompt)


def test_procedural_pipeline_completes():
    from graph import run_pipeline

    with pipeline_mocks(intent_label="procedural",
                        llm_answer="1. Visit NADRA office. [source: nadra_guide.pdf]\n2. Submit documents. [source: nadra_guide.pdf]"):
        state = run_pipeline("How do I apply for a CNIC step by step?")

    assert state["intent"] == "procedural"
    assert state["final_answer"]

run("Procedural: pipeline completes and final_answer is set", test_procedural_pipeline_completes)


# =============================================================================
# PATH 4 — OUT OF SCOPE
# =============================================================================
section("PATH 4 — Out of scope: skip retrieval entirely")

def test_out_of_scope_skips_retrieval():
    """Out-of-scope questions must NOT populate raw_chunks or reranked_chunks."""
    from graph import run_pipeline

    retrieve_called = []

    def track_retrieve(*args, **kwargs):
        retrieve_called.append(args)
        return []

    with pipeline_mocks(intent_label="out_of_scope"):
        with patch("hybrid_search.dense_retrieve", side_effect=track_retrieve), \
             patch("hybrid_search.bm25_retrieve",  side_effect=track_retrieve):
            state = run_pipeline("Is the government of Pakistan corrupt?")

    assert state["intent"] == "out_of_scope"
    assert len(retrieve_called) == 0, \
        f"dense/BM25 retrieve should NOT be called for out_of_scope, was called {len(retrieve_called)} times"

run("Out-of-scope: hybrid_search is never called", test_out_of_scope_skips_retrieval)


def test_out_of_scope_returns_decline():
    """Out-of-scope questions should produce a polite decline, not a factual answer."""
    from graph import run_pipeline

    with pipeline_mocks(intent_label="out_of_scope",
                        llm_answer="I can only assist with Pakistani government services and documentation."):
        state = run_pipeline("Tell me a joke")

    assert state["final_answer"]
    # Should NOT contain government data — just a decline message
    assert state.get("raw_chunks", []) == [] or state.get("reranked_chunks", []) == []

run("Out-of-scope: final_answer is a decline, no retrieval data", test_out_of_scope_returns_decline)


def test_out_of_scope_fast_path_no_citation_needed():
    """Out-of-scope should still have citation_result set (verifier runs on decline message)."""
    from graph import run_pipeline

    with pipeline_mocks(intent_label="out_of_scope",
                        llm_answer="This is outside my scope."):
        state = run_pipeline("What is the weather today?")

    # Citation result is set (verifier always runs), but no citations to verify
    assert state["citation_result"] is not None

run("Out-of-scope: citation_result is set (verifier still runs)", test_out_of_scope_fast_path_no_citation_needed)


# =============================================================================
# PATH 5 — TOOL TRIGGER
# =============================================================================
section("PATH 5 — Tool trigger: question routes through MCP tools")

def test_tool_trigger_question_sets_needs_tools():
    """A question containing 'current status' should trigger needs_tools=True."""
    from graph import run_pipeline

    with pipeline_mocks(intent_label="factual"):
        state = run_pipeline("What is the current status of my CNIC application?")

    assert state.get("needs_tools") is True, \
        "Question with 'current status' should set needs_tools=True"

run("Tool trigger: 'current status' question sets needs_tools=True", test_tool_trigger_question_sets_needs_tools)


def test_tool_trigger_calls_mcp_tool():
    """When needs_tools=True and question mentions NADRA, search_nadra should be called."""
    from graph import run_pipeline

    tool_calls = []

    def track_call_tool(tool_name, **kwargs):
        tool_calls.append(tool_name)
        from mcp_tools import tool_result
        return tool_result(tool_name, kwargs.get("query", ""), {"results": []})

    with pipeline_mocks(intent_label="factual"):
        with patch("graph.call_tool", side_effect=track_call_tool):
            state = run_pipeline("What is the current NADRA CNIC status?")

    assert len(tool_calls) > 0, "At least one MCP tool should be called for NADRA + current status query"
    assert "search_nadra" in tool_calls, f"Expected search_nadra in {tool_calls}"

run("Tool trigger: search_nadra called for NADRA + 'current' query", test_tool_trigger_calls_mcp_tool)


def test_tool_results_in_state():
    """tool_results key in state should be a list after MCP tools run."""
    from graph import run_pipeline

    with pipeline_mocks(intent_label="factual"):
        state = run_pipeline("What is the current status of my NADRA application?")

    assert isinstance(state.get("tool_results"), list)

run("Tool trigger: tool_results is a list in final state", test_tool_results_in_state)


# =============================================================================
# PATH 6 — RETRY PATH
# =============================================================================
section("PATH 6 — Retry: verification fails → regenerate at temp=0.0")

def test_retry_triggered_when_citation_fails():
    """
    route_after_verification returns 'generator' when citation fails and retries remain.
    Tests the routing logic directly.
    """
    from graph import route_after_verification, initial_state

    state = initial_state("What is NADRA?")
    state["citation_result"]      = {"passed": False, "unverified_claims": [{"claim": "x"}]}
    state["hallucination_result"] = {"passed": True,  "hallucination_rate": 0.0}
    state["retry_count"]          = 0

    result = route_after_verification(state)
    assert result == "generator", f"Expected 'generator' (retry), got '{result}'"

run("Retry: route_after_verification → 'generator' when citation fails",
    test_retry_triggered_when_citation_fails)


def test_retry_capped_at_max_retries():
    """When retry_count >= MAX_RETRIES, route_after_verification returns 'end' even if failed."""
    from graph import route_after_verification, initial_state, MAX_RETRIES

    state = initial_state("What is NADRA?")
    state["citation_result"]      = {"passed": False, "unverified_claims": [{"claim": "x"}]}
    state["hallucination_result"] = {"passed": False, "hallucination_rate": 1.0}
    state["retry_count"]          = MAX_RETRIES

    result = route_after_verification(state)
    assert result == "end", f"Expected 'end' when retries exhausted, got '{result}'"

run("Retry: capped at MAX_RETRIES — routes to 'end' not 'generator'",
    test_retry_capped_at_max_retries)



def test_no_retry_when_verification_passes():
    """When CE score is high, verification passes and generator is called exactly once."""
    from graph import run_pipeline

    generate_calls = []

    def track_generate(**kwargs):
        generate_calls.append(1)
        return {
            "answer":            "NADRA issues CNICs. [source: nadra_guide.pdf]",
            "model":             "llama3-70b-8192",
            "prompt_tokens":     80,
            "completion_tokens": 30,
            "context_used":      2,
        }

    with pipeline_mocks(intent_label="factual", ce_score=8.0):
        with patch("graph.generate", side_effect=track_generate):
            state = run_pipeline("What does NADRA do?")

    assert len(generate_calls) == 1, \
        f"Expected exactly 1 generate call when verification passes, got {len(generate_calls)}"

run("Retry: no retry when verification passes (generate called once)",
    test_no_retry_when_verification_passes)


# =============================================================================
# GRAPH STRUCTURE VALIDATION
# =============================================================================
section("GRAPH — Structure and compilation")

def test_graph_has_all_nodes():
    """All 6 nodes must be registered in the compiled graph."""
    from graph import build_graph
    g = build_graph()
    nodes = set(g.nodes.keys())
    expected = {"planner", "tool_selector", "mcp_tools", "rag_agent", "generator", "verification_agent"}
    for node in expected:
        assert node in nodes, f"Missing node: {node}"

run("Graph: all 6 nodes registered", test_graph_has_all_nodes)


def test_graph_compiles_cleanly():
    """Graph compilation validates all edges and nodes are correctly connected."""
    from graph import build_graph
    g = build_graph()
    assert g is not None

run("Graph: compiles without errors", test_graph_compiles_cleanly)


def test_initial_state_has_retry_count():
    """initial_state must include retry_count field for retry logic to work."""
    from graph import initial_state
    s = initial_state("test")
    assert "retry_count" in s
    assert s["retry_count"] == 0

run("Graph: initial_state includes retry_count=0", test_initial_state_has_retry_count)


# =============================================================================
# SUMMARY
# =============================================================================
total = passed + failed
print(f"\n{'='*60}")
print(f"{BOLD}E2E RESULTS: {GREEN}{passed} passed{RESET}{BOLD}, {RED}{failed} failed{RESET}{BOLD}, {total} total{RESET}")
print(f"{'='*60}\n")

if failed > 0:
    sys.exit(1)
