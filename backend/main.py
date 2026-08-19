import os
import json
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Citizen Services Assistant API",
    description="RAG-powered Pakistani government services assistant",
    version="2.0.0",
)

# CORS. The Vite dev server (5173) and a plain `serve dist` (3000) are the two
# origins this runs behind locally. Starlette compares allow_origins by exact
# string — a glob like "http://localhost:*" matches nothing — so any other
# origin (a LAN IP for testing on a phone, say) has to be listed explicitly via
# FRONTEND_URL rather than pattern-matched.
_allowed_origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]
for _origin in os.getenv("FRONTEND_URL", "").split(","):
    _origin = _origin.strip().rstrip("/")
    if _origin:
        _allowed_origins.append(_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Static service data ───────────────────────────────────────────────────────
# In production this would come from the `services` DB table.
# For the demo, we serve it statically — the pipeline logic is unchanged.
SERVICES: list[dict] = [
    {
        "id":          "nadra",
        "name":        "NADRA / CNIC",
        "department":  "NADRA",
        "description": "National identity card registration, renewal, correction, and replacement for Pakistani citizens.",
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
        "description": "NTN registration, annual income tax returns, ATL status checks, and refund claims via IRIS portal.",
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
        "description": "New driving license applications, renewals, and international driving permits across Pakistani provinces.",
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
        "description": "Company registration for Pvt Ltd, SMC-Pvt, and sole proprietorships via SECP e-Services portal.",
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
        "description": "New passport applications, renewals, urgent processing, and minor passports via DGIP offices and portal.",
        "tags":        ["Passport", "Renewal", "Urgent", "Immigration"],
        "doc_count":   9,
        "chunk_count": 110,
        "last_ingested": "2025-01-09",
        "source_url":  "https://onlinemrs.dgip.gov.pk",
    },
]


# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    history:  list[dict] = []

class DocumentAskRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    final_answer:         str
    answer:               str | None = None
    intent:               str | None = None
    routing_config:       dict | None = None
    search_queries:       list  = []
    raw_chunks:           list  = []
    reranked_chunks:      list  = []
    compressed_chunks:    list  = []
    tool_results:         list  = []
    generation_meta:      dict | None = None
    citation_result:      dict | None = None
    hallucination_result: dict | None = None
    retry_count:          int   = 0


# ── Pipeline runner ───────────────────────────────────────────────────────────
def _run_pipeline(question: str) -> dict:
    """
    Imports and invokes the LangGraph pipeline.
    Wrapped in a function so the import happens at request time,
    not at startup — keeps startup fast and allows hot-reload.
    """
    from graph import run_pipeline
    return run_pipeline(question)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/services")
def list_services():
    """Returns all 5 service domains."""
    return SERVICES


@app.get("/services/{service_id}")
def get_service(service_id: str):
    """Returns detail for a single service."""
    svc = next((s for s in SERVICES if s["id"] == service_id), None)
    if not svc:
        raise HTTPException(status_code=404, detail=f"Service '{service_id}' not found")
    return svc


@app.get("/about/ingestion")
def ingestion_metadata():
    """
    Returns ingestion metadata for the About page.
    In production this queries the ingestion_log table.
    For demo, computed from the static SERVICES data.
    """
    return {
        "total_chunks": sum(s["chunk_count"] for s in SERVICES),
        "total_documents": sum(s["doc_count"] for s in SERVICES),
        "services": [
            {
                "id":           s["id"],
                "name":         s["name"],
                "department":   s["department"],
                "doc_count":    s["doc_count"],
                "chunk_count":  s["chunk_count"],
                "last_ingested": s["last_ingested"],
                "source_url":   s["source_url"],
            }
            for s in SERVICES
        ],
        "embedding_model": "BAAI/bge-large-en",
        "embedding_dim":   1024,
        "last_updated":    max(s["last_ingested"] for s in SERVICES),
    }


@app.post("/chat", response_model=None)
def chat(req: ChatRequest):
    """
    Main RAG endpoint. Runs the full LangGraph pipeline and returns the
    complete state so the frontend can display pipeline inspector data.

    The history parameter is available for multi-turn context (future use).
    Currently the pipeline is stateless per request.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        state = _run_pipeline(req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # Serialise — remove any non-JSON-serialisable fields
    def safe(obj):
        if isinstance(obj, dict):
            return {k: safe(v) for k, v in obj.items()
                    if not callable(v)}
        if isinstance(obj, list):
            return [safe(i) for i in obj]
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    return safe(dict(state))


@app.post("/documents/upload")
async def upload_document_endpoint(file: UploadFile = File(...)):
    """
    Uploads a .pdf/.txt/.md file, extracts + chunks + embeds it in memory,
    and returns a doc_id used for subsequent /documents/{doc_id}/ask calls.

    This is separate from the 5 government-service corpora in Postgres —
    see document_qa.py for why (user-specific, ephemeral, in-memory).
    """
    from document_qa import upload_document

    raw_bytes = await file.read()
    try:
        meta = upload_document(file.filename, raw_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return meta


@app.get("/documents/{doc_id}")
def get_document_endpoint(doc_id: str):
    """Returns metadata for a previously uploaded document."""
    from document_qa import get_document_meta

    meta = get_document_meta(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found or expired")
    return meta


@app.post("/documents/{doc_id}/ask")
def ask_document_endpoint(doc_id: str, req: DocumentAskRequest):
    """Asks a question about a single previously-uploaded document."""
    from document_qa import ask_document

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        result = ask_document(doc_id, req.question)
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found or expired")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document Q&A error: {str(e)}")

    def safe(obj):
        if isinstance(obj, dict):
            return {k: safe(v) for k, v in obj.items() if not callable(v)}
        if isinstance(obj, list):
            return [safe(i) for i in obj]
        try:
            json.dumps(obj)
            return obj
        except Exception:
            return str(obj)

    return safe(result)


# MCP server lives in mcp_server.py — run it as a separate process:
#   python mcp_server.py
# This keeps FastAPI and MCP independently restartable and deployable.


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)