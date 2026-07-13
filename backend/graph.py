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
from generator          import generate
from hallucination_eval import evaluate_hallucination
from mcp_tools          import call_tool, select_tools_for_question


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
    retry_count:          int
    final_answer:         str
    error:                str


def initial_state(question: str) -> dict:
    """Creates the starting state from a user question."""
    return {
        "question":             question,
        "intent":               None,
        "routing_config":       None,
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

    intent = route(state["question"])
    config = get_routing_config(intent)

    print(f"[Planner] Intent: {intent.value}")

    return {
        "intent":         intent.value,
        "routing_config": config,
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
    queries = rewrite(question, intent)
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

    retry_count = state.get("retry_count", 0)
    is_retry    = retry_count > 0

    if is_retry:
        print(f"[Generator] RETRY attempt {retry_count} — using stricter temperature")

    print(f"[Generator] Calling Groq ({intent} format, {len(chunks)} chunks)...")

    result = generate(
        question=question,
        chunks=chunks,
        intent=intent,
        # On retry: lower temperature = more conservative, less hallucination
        temperature=0.0 if is_retry else 0.1,
    )

    print(f"[Generator] {result['prompt_tokens']} prompt + {result['completion_tokens']} completion tokens")

    return {
        "answer":          result["answer"],
        "retry_count":     retry_count + 1,
        "generation_meta": {
            "model":              result["model"],
            "prompt_tokens":      result["prompt_tokens"],
            "completion_tokens":  result["completion_tokens"],
            "context_used":       result["context_used"],
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
                "citation_result": None, "hallucination_result": None}

    # Step 1: Citation verification
    print("[VerificationAgent] Verifying citations...")
    citation_result = verify_citations(answer, chunks)
    print(f"[VerificationAgent] Citations: {citation_result['summary']}")

    # Step 2: Hallucination evaluation
    print("[VerificationAgent] Evaluating hallucinations...")
    hallucination_result = evaluate_hallucination(answer, chunks)
    print(f"[VerificationAgent] Hallucination: {hallucination_result['summary']}")

    # Day 7 will add: if not passed → retry with stricter prompt
    return {
        "citation_result":      citation_result,
        "hallucination_result": hallucination_result,
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

    Retry if:
        - Citation verification failed (unverified claims exist) AND
        - Hallucination rate is above threshold AND
        - We haven't already retried (MAX_RETRIES cap)

    Why cap retries at 1?
        If the LLM hallucinated once, a second attempt usually fixes it
        because we pass a stricter prompt. A third attempt rarely adds value
        and doubles latency. One retry is the pragmatic sweet spot.
    """
    citation_ok    = state.get("citation_result", {}).get("passed", True)
    hallucination_ok = state.get("hallucination_result", {}).get("passed", True)
    retry_count    = state.get("retry_count", 0)

    if (not citation_ok or not hallucination_ok) and retry_count < MAX_RETRIES:
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