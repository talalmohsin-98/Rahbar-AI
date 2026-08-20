"""
graph.py  (SKELETON — Day 5)
----------------------------
PURPOSE: Wire all pipeline components as a LangGraph StateGraph.
         This is the orchestration layer — it defines HOW the nodes connect,
         what data flows between them, and which paths are conditional.

Pipeline position:
    This IS the pipeline. Every other file is a node inside this graph.

    User → Planner (intent_router)
         → Tool Selector (conditional edge — MCP tools or straight to RAG)
         → MCP Tools (Day 6)
         → RAG Agent (query_rewriter + hybrid_search + reranker + compressor)
         → Verification Agent (citation_verifier)
         → Response Agent (generator + hallucination_eval)
         → Answer

DAY 5 STATUS:
    ✅ State definition
    ✅ All node function stubs
    ✅ Graph wiring (add_node, add_edge, add_conditional_edges)
    ✅ Conditional routing logic
    ⏳ Generator node → Day 6
    ⏳ Hallucination eval node → Day 6
    ⏳ MCP tool nodes → Day 6

WHY LANGGRAPH:
    Standard function calls (A calls B calls C) work fine for linear pipelines.
    But we need:
    - Conditional routing (tool selector: does this question need MCP tools?)
    - Shared state passed between all nodes without argument threading
    - Easy visualization of the graph
    - Future extensibility (cycles, human-in-the-loop, retries)

    LangGraph models the pipeline as a directed graph where nodes are functions
    and edges define the flow. State is a typed dict passed to every node.
"""

import operator
import sys
from typing import Any, Literal, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END

# Import all pipeline components
from intent_router      import route, get_routing_config, Intent
from query_rewriter     import rewrite, build_rewrite_result
from hybrid_search      import hybrid_search
from reranker           import rerank
from compressor         import compress
from citation_verifier  import verify_citations
from completeness_check import check_completeness, build_retry_feedback
from generator          import generate
from hallucination_eval import evaluate_hallucination
from mcp_tools          import call_tool, select_tools_for_question
from input_analysis     import analyse as analyse_input
from verification_verdict import (build_verdict, check_term_attestation,
                                  is_refusal, refusal_results)


# Every node below prints the question and answer it is working on. On a
# Windows console those streams default to cp1252, which cannot encode Urdu
# script — so a code-switched question would take down the whole request from
# inside a print(), long before retrieval got a chance to be bad at it.
# Degrade unencodable characters instead of raising.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass  # already-wrapped or non-reconfigurable stream; nothing to do


# ---------------------------------------------------------------------------
# 1. PIPELINE STATE
# ---------------------------------------------------------------------------
class PipelineState(TypedDict, total=False):
    """
    Typed state dict passed between every LangGraph node.
    total=False means all fields are optional — nodes only set the keys they own.

    Fields populated progressively as nodes execute:
        question:             original user question
        intent:               from intent_router
        routing_config:       from intent_router (answer_format, top_k, etc.)
        search_queries:       from query_rewriter
        raw_chunks:           from hybrid_search (RRF-ranked)
        reranked_chunks:      from reranker
        compressed_chunks:    from compressor
        needs_tools:          from tool_selector
        tool_results:         from MCP tool nodes
        answer:               from generator
        generation_meta:      token counts, model used
        citation_result:      from citation_verifier
        hallucination_result: from hallucination_eval
        retry_count:          how many generation retries so far
        final_answer:         delivered to user
        error:                populated if any node fails
    """
    question:             str
    intent:               str
    routing_config:       dict
    input_analysis:       dict
    search_queries:       list
    raw_chunks:           list
    reranked_chunks:      list
    compressed_chunks:    list
    needs_tools:          bool
    tool_results:         list
    answer:               str
    generation_meta:      dict
    citation_result:      dict
    hallucination_result: dict
    completeness_result:  dict
    attestation_result:   dict
    verification_verdict: dict
    retry_feedback:       str
    generation_attempts:  int
    retry_count:          int
    final_answer:         str
    error:                str


def initial_state(question: str) -> dict:
    """Creates the starting state from a user question."""
    return {
        "question":             question,
        "intent":               None,
        "routing_config":       None,
        "input_analysis":       None,
        "search_queries":       [],
        "raw_chunks":           [],
        "reranked_chunks":      [],
        "compressed_chunks":    [],
        "needs_tools":          False,
        "tool_results":         [],
        "answer":               None,
        "generation_meta":      None,
        "citation_result":      None,
        "hallucination_result": None,
        "completeness_result":  None,
        "attestation_result":   None,
        "verification_verdict": None,
        "retry_feedback":       None,
        "generation_attempts":  0,
        "retry_count":          0,
        "final_answer":         None,
        "error":                None,
    }


# ---------------------------------------------------------------------------
# 2. NODE FUNCTIONS
# ---------------------------------------------------------------------------
# Each node is a plain Python function that:
#   - Receives the full state dict
#   - Returns a dict of ONLY the keys it wants to update
#   - Never modifies the state directly (treat it as read-only)

def planner_node(state: PipelineState) -> dict:
    """
    NODE: Planner
    Wraps intent_router.py — classifies the question and sets routing config.
    This is the first node every query hits.

    Returns updates to: intent, routing_config
    """
    print(f"\n[Planner] Question: {state['question'][:80]}")

    # Deterministic, LLM-free inspection of the raw question. Runs first
    # because both the rewriter (which needs to know it should translate and
    # fan out per domain) and the final verdict (which must not claim
    # confidence on a query the embedder cannot represent) depend on it.
    analysis = analyse_input(state["question"])
    if analysis["code_switched"]:
        print(f"[Planner] Code-switched input ({analysis['code_switch_kind']})"
              f" - retrieval confidence downgraded")
    if analysis["compound"]:
        print(f"[Planner] Compound question spans {analysis['domains']}")

    intent = route(state["question"])
    config = get_routing_config(intent)

    print(f"[Planner] Intent: {intent.value}")

    return {
        "intent":         intent.value,
        "routing_config": config,
        "input_analysis": analysis,
    }


def tool_selector_node(state: PipelineState) -> dict:
    """
    NODE: Tool Selector
    Decides whether this question needs MCP tools before RAG retrieval.

    Current logic (Day 5 skeleton):
        - OUT_OF_SCOPE → no tools, no retrieval
        - Questions containing keywords suggesting live data → tools
        - Everything else → straight to RAG

    Day 6 will expand this with actual tool selection logic.

    Returns updates to: needs_tools
    """
    intent = state["intent"]
    question = state["question"].lower()

    # Keywords suggesting we need live/external data beyond the document corpus
    tool_trigger_keywords = [
        "current", "today", "latest", "status", "track",
        "check", "verify my", "application number"
    ]

    needs_tools = (
        intent != Intent.OUT_OF_SCOPE.value
        and any(kw in question for kw in tool_trigger_keywords)
    )

    print(f"[ToolSelector] Needs tools: {needs_tools}")
    return {"needs_tools": needs_tools}


def rag_agent_node(state: PipelineState) -> dict:
    """
    NODE: RAG Agent
    The core retrieval pipeline: rewrite → hybrid search → rerank → compress.
    Skipped entirely if intent is OUT_OF_SCOPE.

    Returns updates to: search_queries, raw_chunks, reranked_chunks, compressed_chunks
    """
    if state["intent"] == Intent.OUT_OF_SCOPE.value:
        print("[RAGAgent] Skipping — out of scope")
        return {}

    question = state["question"]
    intent   = Intent(state["intent"])
    config   = state["routing_config"]

    # Step 1: Rewrite query
    print("[RAGAgent] Rewriting query...")
    queries = rewrite(question, intent, state.get("input_analysis"))
    print(f"[RAGAgent] {len(queries)} search queries generated")

    # Step 2: Hybrid search across all query variants
    print("[RAGAgent] Running hybrid search (dense + BM25 + RRF)...")
    top_k = config.get("top_k_override") or 20
    raw_chunks = hybrid_search(queries, top_k=top_k, final_k=10)
    print(f"[RAGAgent] {len(raw_chunks)} chunks after RRF fusion")

    # Step 3: Rerank with CrossEncoder
    print("[RAGAgent] Reranking with CrossEncoder...")
    # top_n=10 (the full RRF pool), not 5: the CrossEncoder favours chunks that
    # keyword-match the question ("documents required" + "CNIC") even when they
    # belong to another domain (SECP, driving license), so the truly relevant
    # chunk can land at rank 7-9. Cutting at 5 dropped it and the generator
    # answered "I could not find this information" despite the data existing.
    # The compressor's token budget still bounds the final context size.
    reranked = rerank(question, raw_chunks, top_n=10)
    print(f"[RAGAgent] {len(reranked)} chunks after reranking (above MIN_RERANK_SCORE)")

    # Step 4: Compress to fit context window
    print("[RAGAgent] Compressing context...")
    compressed = compress(question, reranked)
    print(f"[RAGAgent] {len(compressed)} chunks after compression")

    return {
        "search_queries":    queries,
        "raw_chunks":        raw_chunks,
        "reranked_chunks":   reranked,
        "compressed_chunks": compressed,
    }


def mcp_tools_node(state: PipelineState) -> dict:
    """
    NODE: MCP Tools  ✅ Implemented Day 6
    Selects and calls appropriate MCP tools based on the question and intent.
    Tool results are added to state and later merged with RAG context.

    Returns updates to: tool_results
    """
    question = state["question"]
    intent   = state["intent"]

    # Decide which tools to call
    tools_to_call = select_tools_for_question(question, intent)
    print(f"[MCPTools] Tools selected: {tools_to_call or 'none'}")

    tool_results = []
    for tool_name in tools_to_call:
        print(f"[MCPTools] Calling {tool_name}...")
        result = call_tool(tool_name, query=question)
        tool_results.append(result)
        print(f"[MCPTools] {tool_name} → success={result['success']}")

    return {"tool_results": tool_results}


def generator_node(state: PipelineState) -> dict:
    """
    NODE: Response Agent — Generator  ✅ Implemented Day 6
    Calls generator.py which calls Groq LLM with compressed chunks.
    Intent controls the answer format (factual/comparison/procedural/out_of_scope).

    Returns updates to: answer, generation_meta
    """
    question = state["question"]
    intent   = state["intent"]
    chunks   = state.get("compressed_chunks", [])

    # `attempts` counts generations already made; `retry_count` counts
    # REGENERATIONS. They must be separate fields.
    #
    # The old code incremented retry_count on the very first generation, so
    # by the time route_after_verification tested `retry_count < MAX_RETRIES`
    # the count was already 1 and 1 < 1 is False — the retry edge could never
    # be taken, for any question, ever. The retry path was dead code that the
    # pipeline reported as available.
    attempts = state.get("generation_attempts", 0)
    is_retry = attempts > 0

    if is_retry:
        print(f"[Generator] RETRY attempt {attempts} — using stricter temperature")

    # Set by verification_agent_node when it can say exactly what the previous
    # attempt got wrong. A retry without it just re-rolls the same dice.
    feedback = state.get("retry_feedback") if is_retry else None
    if feedback:
        print("[Generator] Retry carries revision feedback from verification")

    print(f"[Generator] Calling Groq ({intent} format, {len(chunks)} chunks)...")

    result = generate(
        question=question,
        chunks=chunks,
        intent=intent,
        # On retry: lower temperature = more conservative, less hallucination
        temperature=0.0 if is_retry else 0.1,
        feedback=feedback,
    )

    print(f"[Generator] {result['prompt_tokens']} prompt + {result['completion_tokens']} completion tokens")

    return {
        "answer":             result["answer"],
        "generation_attempts": attempts + 1,
        "retry_count":        attempts,   # 0 after the first generation
        "generation_meta": {
            "model":              result["model"],
            "prompt_tokens":      result["prompt_tokens"],
            "completion_tokens":  result["completion_tokens"],
            "context_used":       result["context_used"],
            "finish_reason":      result.get("finish_reason"),
            "truncated":          result.get("truncated", False),
            "is_retry":           is_retry,
        },
    }


def verification_agent_node(state: PipelineState) -> dict:
    """
    NODE: Verification Agent  ✅ Implemented Day 6
    Runs BOTH citation_verifier.py AND hallucination_eval.py.

    Citation verifier:    checks cited sentences against their sources
    Hallucination eval:   checks ALL sentences (cited or not) against all chunks

    Returns updates to: citation_result, hallucination_result, final_answer
    """
    answer = state.get("answer")
    chunks = state.get("compressed_chunks", [])

    if not answer:
        return {"final_answer": "No answer was generated.",
                "citation_result": None, "hallucination_result": None,
                "completeness_result": None, "attestation_result": None,
                "verification_verdict": build_verdict(
                    None, None, None, state.get("input_analysis"), None)}

    # An explicit refusal asserts nothing, so there is nothing to verify. QA
    # found the checkers scoring the decline sentence itself as an unsupported
    # claim, which showed the citizen a warning about the absence of support
    # for a sentence that says there is no support - and sent a correct refusal
    # back through the retry loop. Both are fixed by not asking the question.
    if is_refusal(answer):
        print("[VerificationAgent] Answer is an explicit refusal - checks not applicable")
        results = refusal_results()
        verdict = build_verdict(
            results["citation_result"], results["hallucination_result"],
            results["completeness_result"], state.get("input_analysis"),
            results["attestation_result"], refused=True)
        print(f"[VerificationAgent] Verdict: {verdict['status'].upper()}"
              f" - {verdict['headline']}")
        return {**results, "verification_verdict": verdict,
                "retry_feedback": None, "final_answer": answer}

    # Step 1: Citation verification
    print("[VerificationAgent] Verifying citations...")
    citation_result = verify_citations(answer, chunks)
    print(f"[VerificationAgent] Citations: {citation_result['summary']}")

    # Step 2: Hallucination evaluation
    print("[VerificationAgent] Evaluating hallucinations...")
    hallucination_result = evaluate_hallucination(answer, chunks)
    print(f"[VerificationAgent] Hallucination: {hallucination_result['summary']}")

    # Step 3: Completeness — steps 1 and 2 only ask whether the answer is TRUE.
    # This asks whether it is USEFUL: did we retrieve material that answers the
    # question and then drop it? (See completeness_check.py.)
    print("[VerificationAgent] Checking completeness...")
    completeness_result = check_completeness(answer, chunks)
    print(f"[VerificationAgent] Completeness: {completeness_result['summary']}")

    # Step 4: Term attestation. The three checks above all interrogate the
    # answer; this one asks whether the sources mention what was asked about at
    # all. Deterministic and free — no model call. See verification_verdict.py
    # for the "Gamma Family FRC" case that motivated it.
    attestation_result = check_term_attestation(state.get("question", ""), chunks)
    print(f"[VerificationAgent] Attestation: {attestation_result['summary']}")

    # Step 5: One verdict, one owner. Before this existed the badges derived
    # their own answer from citations+grounding only, ignoring completeness
    # entirely, while the inspector panel derived a different one inline. Two
    # surfaces, two rules. Both now render this field instead of re-deriving.
    verdict = build_verdict(citation_result, hallucination_result,
                            completeness_result, state.get("input_analysis"),
                            attestation_result)
    print(f"[VerificationAgent] Verdict: {verdict['status'].upper()}"
          f" - {verdict['headline']}")
    for _reason in verdict["reasons"]:
        print(f"[VerificationAgent]   . {_reason}")

    return {
        "citation_result":      citation_result,
        "hallucination_result": hallucination_result,
        "completeness_result":  completeness_result,
        "attestation_result":   attestation_result,
        "verification_verdict": verdict,
        # Quote the dropped passages back to the generator if we regenerate.
        "retry_feedback": (
            build_retry_feedback(completeness_result["unused_blocks"])
            if completeness_result["unused_blocks"] else None
        ),
        "final_answer":         answer,
    }


# ---------------------------------------------------------------------------
# 3. CONDITIONAL ROUTING FUNCTIONS
# ---------------------------------------------------------------------------
def route_after_tool_selector(state: PipelineState) -> Literal["mcp_tools", "rag_agent"]:
    """
    Conditional edge: after tool_selector, go to MCP tools or straight to RAG.

    LangGraph conditional edges work by calling a function that returns
    a string matching one of the registered edge targets.
    """
    if state.get("needs_tools"):
        return "mcp_tools"
    return "rag_agent"


MAX_RETRIES = 1   # allow one regeneration attempt before accepting the answer


def route_after_verification(state: PipelineState) -> Literal["generator", "end"]:
    """
    Conditional edge: after verification_agent, decide whether to retry generation.

    Retry if any verification stage failed — citations unverified, claims
    ungrounded, or the answer left retrieved material unused — and we haven't
    already retried (MAX_RETRIES cap).

    Why cap retries at 1?
        If the LLM hallucinated once, a second attempt usually fixes it
        because we pass a stricter prompt. A third attempt rarely adds value
        and doubles latency. One retry is the pragmatic sweet spot.
    """
    # `or {}` — not just a default arg: verification_agent_node sets these keys
    # to None when there was no answer to check, and None.get() would crash.
    # NOTE: attestation_result is deliberately NOT consulted here. The other
    # three failures describe an answer that could have been written better
    # from the same context, so regenerating is worth one shot. An unattested
    # term is a fact about the CORPUS, not the answer — "Gamma" will still be
    # absent on the second attempt, so a retry burns a generation to arrive at
    # the identical verdict. It downgrades the badge; it does not re-roll.
    citation_ok      = (state.get("citation_result") or {}).get("passed", True)
    hallucination_ok = (state.get("hallucination_result") or {}).get("passed", True)
    complete_ok      = (state.get("completeness_result") or {}).get("passed", True)
    retry_count      = state.get("retry_count", 0)

    if (not citation_ok or not hallucination_ok or not complete_ok) and retry_count < MAX_RETRIES:
        print(f"[Router] Verification failed — retrying generation (attempt {retry_count + 1})")
        return "generator"

    return "end"


def route_after_intent(state: PipelineState) -> Literal["tool_selector", "generator"]:
    """
    Conditional edge: if OUT_OF_SCOPE, skip straight to generator (which returns a polite decline).
    Otherwise proceed to tool_selector → RAG.
    """
    if state.get("intent") == Intent.OUT_OF_SCOPE.value:
        return "generator"
    return "tool_selector"


# ---------------------------------------------------------------------------
# 4. BUILD THE GRAPH
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph StateGraph.

    Graph structure:
        START
          ↓
        planner
          ↓ (conditional)
        ┌─────────────────────┐
        │  tool_selector      │  ← only for in-scope questions
        │      ↓ (conditional) │
        │  mcp_tools ──┐      │
        │  rag_agent ◄─┘      │
        └─────────────────────┘
          ↓
        generator
          ↓
        verification_agent
          ↓
        END

    Nodes are registered with add_node(name, function).
    Edges are added with add_edge(from, to) for unconditional flow,
    or add_conditional_edges(from, routing_fn, {return_value: node_name}) for branching.
    """
    # Create the graph with our state class
    graph = StateGraph(PipelineState)

    # Register all nodes
    graph.add_node("planner",             planner_node)
    graph.add_node("tool_selector",       tool_selector_node)
    graph.add_node("mcp_tools",           mcp_tools_node)
    graph.add_node("rag_agent",           rag_agent_node)
    graph.add_node("generator",           generator_node)
    graph.add_node("verification_agent",  verification_agent_node)

    # Entry point
    graph.set_entry_point("planner")

    # Planner → conditional: OUT_OF_SCOPE goes to generator, everything else to tool_selector
    graph.add_conditional_edges(
        "planner",
        route_after_intent,
        {
            "tool_selector": "tool_selector",
            "generator":     "generator",
        }
    )

    # Tool selector → conditional: needs tools → MCP, else → RAG
    graph.add_conditional_edges(
        "tool_selector",
        route_after_tool_selector,
        {
            "mcp_tools": "mcp_tools",
            "rag_agent": "rag_agent",
        }
    )

    # MCP tools always flows to RAG after fetching external data
    graph.add_edge("mcp_tools",  "rag_agent")

    # RAG always flows to generator
    graph.add_edge("rag_agent",  "generator")

    # Generator always flows to verification
    graph.add_edge("generator",  "verification_agent")

    # Verification → conditional: if failed and retries remain → back to generator
    graph.add_conditional_edges(
        "verification_agent",
        route_after_verification,
        {
            "generator": "generator",   # retry path
            "end":       END,           # happy path
        }
    )

    # Compile: validates the graph (catches disconnected nodes, missing edges, etc.)
    return graph.compile()


# ---------------------------------------------------------------------------
# 5. RUN FUNCTION (entry point for callers)
# ---------------------------------------------------------------------------
def run_pipeline(question: str) -> dict:
    """
    Main entry point. Takes a question, returns the full final state.

    The final state contains everything: the answer, all intermediate results,
    verification scores, etc. Useful for debugging and the demo_runner.py (Day 8).
    """
    graph = build_graph()
    state = initial_state(question)

    print(f"\n{'='*60}")
    print(f"PIPELINE START: {question[:80]}")
    print(f"{'='*60}")

    final_state = graph.invoke(state)

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    # ascii-safe: Windows consoles often run cp1252, and a non-ASCII char in
    # the LLM answer (e.g. U+2011 non-breaking hyphen) makes print() raise
    # UnicodeEncodeError — failing the whole request after a successful answer.
    preview = (final_state.get("final_answer") or "None")[:200]
    print("Answer: " + preview.encode("ascii", "replace").decode("ascii"))
    print(f"{'='*60}\n")

    return final_state


# ---------------------------------------------------------------------------
# 6. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run_pipeline("What documents do I need for NADRA CNIC registration?")

    print("\nFull state keys populated:")
    for key, val in result.items():
        if val:
            preview = str(val)[:60] if not isinstance(val, list) else f"[{len(val)} items]"
            print(f"  {key}: {preview}")