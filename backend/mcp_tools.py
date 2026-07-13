"""
mcp_tools.py
------------
PURPOSE: Expose real MCP (Model Context Protocol) tools that the pipeline
         can call when a question needs live data beyond the document corpus.

Pipeline position:
    → tool_selector_node  (in graph.py — decides if tools are needed)
    → mcp_tools.py        ✅ YOU ARE HERE  (Tool Selector Agent's toolbox)
    → rag_agent_node      (tool results flow into RAG as additional context)

WHAT IS MCP:
    Model Context Protocol — a standard that lets LLMs call external tools.
    Before MCP: every AI app built its own custom API integration.
        ChatGPT needed its own plugin. Claude needed its own connector.
        Cursor needed its own integration. All different.
    After MCP: build ONE MCP server, and Claude/GPT/Cursor can ALL use it.
    MCP standardises how AI models discover and call tools.

    Think of it as: USB-C for AI tools.
    One standard port, any device can plug in.

ARCHITECTURE IN THIS FILE:
    Server 1: Government Search Server
        - search_nadra(query)     → searches NADRA records
        - search_fbr(query)       → searches FBR tax records
        - search_passport(query)  → searches passport office records

    Server 2: Document Intelligence Server
        - upload_pdf(path)            → uploads and indexes a PDF
        - extract_entities(text)      → extracts names, dates, IDs from text
        - compare_documents(a, b)     → compares two documents for differences

WHY TWO SERVERS:
    Separation of concerns. Government search is read-only lookups.
    Document intelligence does processing operations.
    Keeping them separate means: if the search server is down, document
    operations still work. In production they'd be separate deployments.

IMPLEMENTATION APPROACH:
    For the portfolio demo, we implement real tool INTERFACES with
    structured responses. The tools call the Groq LLM to simulate
    realistic government service responses — this is honest (we document it)
    and demonstrates the tool call pattern correctly.

    In production: replace the Groq simulation with real API calls
    to NADRA's API, FBR's portal, etc.
"""

import os
import json
from typing import Any
from groq import Groq


# ---------------------------------------------------------------------------
# 1. CLIENT
# ---------------------------------------------------------------------------
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not set.")
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# 2. TOOL RESULT SCHEMA
# ---------------------------------------------------------------------------
def tool_result(tool_name: str, query: str, data: Any, success: bool = True) -> dict:
    """
    Standardised wrapper for all tool results.
    Every tool returns this schema so graph.py can handle them uniformly.

    Fields:
        tool:     which tool was called
        query:    what was searched/processed
        success:  whether the call succeeded
        data:     the actual result (dict, list, or string)
        source:   where the data came from (for citation purposes)
    """
    return {
        "tool":    tool_name,
        "query":   query,
        "success": success,
        "data":    data,
        "source":  f"mcp:{tool_name}",
    }


# ---------------------------------------------------------------------------
# 3. GOVERNMENT SEARCH SERVER
# ---------------------------------------------------------------------------

def search_nadra(query: str) -> dict[str, Any]:
    """
    MCP Tool: Search NADRA government records.

    In production: calls NADRA's citizen portal API.
    In demo: uses Groq LLM to generate a realistic structured response.

    Use when: user asks about CNIC status, registration requirements,
              or specific NADRA services with live/current data needs.
    """
    client = get_groq_client()

    prompt = f"""You are simulating a NADRA (National Database and Registration Authority) 
government records search system for Pakistan.

Return a JSON response for this search query: "{query}"

The response must be valid JSON with these fields:
{{
  "results": [list of 1-3 relevant results],
  "each_result_has": {{
    "title": "service or record name",
    "description": "what this service/record covers",
    "requirements": ["list", "of", "requirements"],
    "fees": "fee information in PKR",
    "processing_time": "time estimate",
    "office": "relevant NADRA office or portal"
  }},
  "total_results": number
}}

Return ONLY valid JSON. No explanation."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=600,
    )

    raw = (response.choices[0].message.content or "").strip()
    try:
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"raw_response": raw, "parse_error": True}

    return tool_result("search_nadra", query, data)


def search_fbr(query: str) -> dict[str, Any]:
    """
    MCP Tool: Search FBR (Federal Board of Revenue) tax records and guidance.

    In production: calls FBR's IRIS portal API.
    Use when: user asks about tax filing, NTN registration, tax rates.
    """
    client = get_groq_client()

    prompt = f"""You are simulating an FBR (Federal Board of Revenue) Pakistan tax records system.

Return a JSON response for this query: "{query}"

{{
  "results": [1-2 relevant results],
  "each_result_has": {{
    "title": "tax service or regulation name",
    "description": "what this covers",
    "requirements": ["required documents or steps"],
    "deadline": "filing deadline if applicable",
    "portal": "IRIS or relevant FBR portal link"
  }},
  "total_results": number
}}

Return ONLY valid JSON."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )

    raw = (response.choices[0].message.content or "").strip()
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"raw_response": raw, "parse_error": True}

    return tool_result("search_fbr", query, data)


def search_passport(query: str) -> dict[str, Any]:
    """
    MCP Tool: Search Passport Office records and requirements.

    In production: calls Directorate General of Immigration & Passports API.
    Use when: user asks about passport status, renewal, requirements.
    """
    client = get_groq_client()

    prompt = f"""You are simulating the Pakistan Directorate General of Immigration & Passports system.

Return a JSON response for this query: "{query}"

{{
  "results": [1-2 relevant results],
  "each_result_has": {{
    "title": "passport service name",
    "description": "service description",
    "requirements": ["required documents"],
    "fees": {{"normal": "fee", "urgent": "fee"}},
    "processing_time": {{"normal": "days", "urgent": "days"}},
    "apply_at": "online portal or office"
  }},
  "total_results": number
}}

Return ONLY valid JSON."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )

    raw = (response.choices[0].message.content or "").strip()
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"raw_response": raw, "parse_error": True}

    return tool_result("search_passport", query, data)


# ---------------------------------------------------------------------------
# 4. DOCUMENT INTELLIGENCE SERVER
# ---------------------------------------------------------------------------

def extract_entities(text: str) -> dict[str, Any]:
    """
    MCP Tool: Extract named entities from text.
    Finds: names, CNIC numbers, dates, document references, amounts.

    Use when: user pastes text and wants key information extracted.
    In production: uses a fine-tuned NER model on Pakistani government text.
    """
    client = get_groq_client()

    prompt = f"""Extract named entities from this text. Return ONLY valid JSON:

Text: "{text[:1000]}"

{{
  "persons": ["list of person names found"],
  "cnic_numbers": ["list of CNIC/ID numbers found"],
  "dates": ["list of dates found"],
  "amounts": ["list of monetary amounts found"],
  "document_references": ["form names, law references, SRO numbers"],
  "organizations": ["government bodies, offices mentioned"],
  "locations": ["cities, offices, addresses"]
}}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
    )

    raw = (response.choices[0].message.content or "").strip()
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"raw_response": raw, "parse_error": True}

    return tool_result("extract_entities", text[:50] + "...", data)


def compare_documents(doc_a: str, doc_b: str) -> dict[str, Any]:
    """
    MCP Tool: Compare two document descriptions or summaries for key differences.

    Use when: user asks "what's the difference between X and Y documents?"
    and the intent_router classified it as COMPARISON but we need live analysis.

    In production: runs actual document diff on uploaded PDFs.
    """
    client = get_groq_client()

    prompt = f"""Compare these two Pakistani government documents and return ONLY valid JSON:

Document A: {doc_a[:500]}
Document B: {doc_b[:500]}

{{
  "similarities": ["list of shared requirements or features"],
  "differences": [
    {{
      "aspect": "what aspect differs",
      "document_a": "how A handles it",
      "document_b": "how B handles it"
    }}
  ],
  "recommendation": "when to use A vs B",
  "summary": "one sentence comparison"
}}"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )

    raw = (response.choices[0].message.content or "").strip()
    try:
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"raw_response": raw, "parse_error": True}

    return tool_result("compare_documents", f"{doc_a[:30]}... vs {doc_b[:30]}...", data)


# ---------------------------------------------------------------------------
# 5. TOOL REGISTRY — used by graph.py to dispatch tool calls
# ---------------------------------------------------------------------------
TOOL_REGISTRY = {
    # Government Search Server
    "search_nadra":    search_nadra,
    "search_fbr":      search_fbr,
    "search_passport": search_passport,
    # Document Intelligence Server
    "extract_entities":   extract_entities,
    "compare_documents":  compare_documents,
}


def call_tool(tool_name: str, **kwargs) -> dict[str, Any]:
    """
    Dispatches a tool call by name. Used by graph.py's mcp_tools_node.

    Args:
        tool_name: one of the keys in TOOL_REGISTRY
        **kwargs:  arguments for that specific tool

    Returns:
        Standardised tool_result dict.

    Example:
        call_tool("search_nadra", query="CNIC requirements")
        call_tool("extract_entities", text="My CNIC is 12345-6789012-3")
    """
    if tool_name not in TOOL_REGISTRY:
        return tool_result(tool_name, str(kwargs), None, success=False)

    fn = TOOL_REGISTRY[tool_name]
    return fn(**kwargs)


def select_tools_for_question(question: str, intent: str) -> list[str]:
    """
    Given a question and intent, returns which tools to call.
    This is the Tool Selector Agent's decision logic.

    Simple keyword-based routing — in production this would be
    an LLM call that reads tool descriptions and picks appropriately.
    """
    q = question.lower()
    tools_to_call = []

    # Government search routing
    if any(kw in q for kw in ["nadra", "cnic", "b-form", "birth certificate", "national id"]):
        tools_to_call.append("search_nadra")
    if any(kw in q for kw in ["tax", "fbr", "ntn", "income tax", "return"]):
        tools_to_call.append("search_fbr")
    if any(kw in q for kw in ["passport", "travel document", "immigration", "visa"]):
        tools_to_call.append("search_passport")

    # Document intelligence
    if intent == "comparison" and len(tools_to_call) == 0:
        tools_to_call.append("compare_documents")

    return tools_to_call


# ---------------------------------------------------------------------------
# 6. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== MCP Tools Smoke Test ===\n")

    # Test 1: Government search
    print("1. search_nadra('CNIC registration requirements')")
    result = search_nadra("CNIC registration requirements")
    print(f"   Success: {result['success']}")
    print(f"   Tool: {result['tool']}")
    if isinstance(result['data'], dict) and 'results' in result['data']:
        print(f"   Results count: {result['data'].get('total_results', '?')}")
    print()

    # Test 2: Entity extraction
    print("2. extract_entities('Submit Form-B and CNIC 12345-1234567-1 by 15 March 2024')")
    result = extract_entities("Submit Form-B and CNIC 12345-1234567-1 by 15 March 2024")
    print(f"   Success: {result['success']}")
    if isinstance(result['data'], dict):
        print(f"   CNIC numbers: {result['data'].get('cnic_numbers', [])}")
        print(f"   Dates: {result['data'].get('dates', [])}")
    print()

    # Test 3: Tool registry
    print("3. Available tools in registry:")
    for name in TOOL_REGISTRY:
        print(f"   - {name}")
