import json
import uvicorn
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp import types
from starlette.applications import Starlette
from starlette.routing import Route, Mount

# ── Shared service data (single source of truth) ──────────────────────────────
# Defined here so mcp_server.py is self-contained.
# main.py has the same list — if you add a service, update both.
# In production this would be a shared DB query.
SERVICES: list[dict] = [
    {
        "id":          "nadra",
        "name":        "NADRA / CNIC",
        "department":  "NADRA",
        "description": "National identity card registration, renewal, correction, and replacement.",
        "tags":        ["CNIC", "B-Form", "Registration", "Biometrics"],
        "doc_count":   12,
        "chunk_count": 148,
        "last_ingested": "2025-01-10",
        "source_url":  "https://nadra.gov.pk",
    },
    {
        "id":          "fbr",
        "name":        "FBR Tax Filing",
        "department":  "Federal Board of Revenue",
        "description": "NTN registration, income tax returns, ATL status, and refund claims.",
        "tags":        ["NTN", "Tax Return", "ATL", "IRIS", "Refund"],
        "doc_count":   10,
        "chunk_count": 124,
        "last_ingested": "2025-01-08",
        "source_url":  "https://fbr.gov.pk",
    },
    {
        "id":          "license",
        "name":        "Driving License",
        "department":  "Provincial Excise & Taxation",
        "description": "New license applications, renewals, and international driving permits.",
        "tags":        ["License", "Learner Permit", "IDP", "Renewal"],
        "doc_count":   7,
        "chunk_count": 86,
        "last_ingested": "2025-01-05",
        "source_url":  "https://peto.punjab.gov.pk",
    },
    {
        "id":          "secp",
        "name":        "SECP Registration",
        "department":  "Securities & Exchange Commission",
        "description": "Company registration for Pvt Ltd, SMC-Pvt, and sole proprietorships.",
        "tags":        ["Pvt Ltd", "SMC", "Company", "SECP Portal"],
        "doc_count":   8,
        "chunk_count": 96,
        "last_ingested": "2025-01-07",
        "source_url":  "https://eservices.secp.gov.pk",
    },
    {
        "id":          "passport",
        "name":        "Passport (DGIP)",
        "department":  "Directorate General of Immigration & Passports",
        "description": "New passports, renewals, urgent processing, and minor passports.",
        "tags":        ["Passport", "Renewal", "Urgent", "Immigration"],
        "doc_count":   9,
        "chunk_count": 110,
        "last_ingested": "2025-01-09",
        "source_url":  "https://onlinemrs.dgip.gov.pk",
    },
]

CATEGORY_MAP = {
    "identity": ["nadra"],
    "tax":      ["fbr"],
    "travel":   ["passport", "license"],
    "business": ["secp"],
}


def _run_pipeline(question: str) -> dict:
    """Calls the LangGraph pipeline. Same helper as main.py."""
    from graph import run_pipeline
    return run_pipeline(question)


# ── Build the MCP server ──────────────────────────────────────────────────────
server = Server("citizen-services-assistant")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="search_services",
            description="Semantic search across the Pakistani government services knowledge base. Returns the most relevant document chunks.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query in plain English"}
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get_service_details",
            description="Structured lookup for a specific government service: fees, required documents, office locations, and processing time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "service_id": {
                        "type": "string",
                        "enum": ["nadra", "fbr", "license", "secp", "passport"],
                        "description": "The service identifier",
                    }
                },
                "required": ["service_id"],
            },
        ),
        types.Tool(
            name="answer_with_citations",
            description="Full RAG pipeline: retrieves relevant chunks, reranks, compresses, generates a grounded answer, and verifies citations. Use this for any factual question about government services.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The question to answer"}
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="list_services",
            description="Enumerate available Pakistani government services, optionally filtered by category (identity, tax, travel, business).",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter: identity, tax, travel, business",
                    }
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── search_services ───────────────────────────────────────────────────────
    if name == "search_services":
        query = arguments.get("query", "")
        try:
            from retrieval import dense_retrieve
            results = dense_retrieve(query, top_k=5)
            # Return only what's useful to an MCP client
            clean = [
                {"rank": r["rank"], "content": r["content"],
                 "source": r["source"], "score": round(r["score"], 4)}
                for r in results
            ]
            return [types.TextContent(type="text", text=json.dumps(clean, indent=2))]
        except Exception as e:
            return [types.TextContent(type="text",
                    text=json.dumps({"error": str(e), "query": query}))]

    # ── get_service_details ───────────────────────────────────────────────────
    elif name == "get_service_details":
        svc_id = arguments.get("service_id", "")
        svc = next((s for s in SERVICES if s["id"] == svc_id), None)
        if not svc:
            return [types.TextContent(type="text",
                    text=json.dumps({"error": f"Unknown service_id: '{svc_id}'. "
                                              f"Valid values: {[s['id'] for s in SERVICES]}"}))]
        return [types.TextContent(type="text", text=json.dumps(svc, indent=2))]

    # ── answer_with_citations ─────────────────────────────────────────────────
    elif name == "answer_with_citations":
        query = arguments.get("query", "")
        try:
            state = _run_pipeline(query)
            result = {
                "answer":             state.get("final_answer") or state.get("answer", ""),
                "intent":             state.get("intent"),
                "citations_verified": state.get("citation_result", {}).get("passed", True),
                "hallucination_rate": state.get("hallucination_result", {}).get("hallucination_rate", 0),
                "sources": list({
                    c.get("source") for c in state.get("compressed_chunks", [])
                    if c.get("source")
                }),
                "retry_count": state.get("retry_count", 0),
            }
        except Exception as e:
            result = {"error": str(e), "query": query}
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    # ── list_services ─────────────────────────────────────────────────────────
    elif name == "list_services":
        category = arguments.get("category", "").lower().strip()
        if category and category in CATEGORY_MAP:
            filtered = [s for s in SERVICES if s["id"] in CATEGORY_MAP[category]]
        else:
            filtered = SERVICES
        result = [
            {"id": s["id"], "name": s["name"], "department": s["department"],
             "description": s["description"]}
            for s in filtered
        ]
        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]

    # ── unknown tool ──────────────────────────────────────────────────────────
    return [types.TextContent(type="text",
            text=json.dumps({"error": f"Unknown tool: '{name}'"}))]


# ── SSE transport wiring ──────────────────────────────────────────────────────
sse = SseServerTransport("/messages/")


async def handle_sse(request):
    async with sse.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1],
            server.create_initialization_options(),
        )


starlette_app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Mount("/messages/", app=sse.handle_post_message),
])


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Starting MCP server on http://localhost:8001")
    print("SSE endpoint: http://localhost:8001/sse")
    print("Test with: npx @modelcontextprotocol/inspector http://localhost:8001/sse")
    uvicorn.run(starlette_app, host="0.0.0.0", port=8001, log_level="info")