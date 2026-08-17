# 🇵🇰 Rahbar AI — Citizen Services Assistant

A RAG-powered assistant that answers questions about Pakistani government services — **NADRA/CNIC, FBR tax filing, Driving License, SECP company registration, and Passport (DGIP)** — by retrieving from a curated, hand-ingested corpus of official documentation and generating **cited, grounded** answers. Every claim is verified twice (citation check + hallucination check) before it reaches the user, and the assistant declines rather than guesses when the source material doesn't support an answer.

> ⚠️ **Not affiliated with NADRA, FBR, SECP, or any Pakistani government body.** This is a research/portfolio demonstration. Always verify information with official sources.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-1C3C3C)
![Groq](https://img.shields.io/badge/Groq-Inference-F55036)
![pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite&logoColor=white)

---

## Table of contents

- [Scope](#scope)
- [Screenshots](#screenshots)
- [What makes this different](#what-makes-this-different)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
- [API reference](#api-reference)
- [Testing](#testing)
- [Known limitations](#known-limitations--honest-tradeoffs)
- [Roadmap](#roadmap)

---

## Scope

Rahbar AI does exactly one thing: **help a Pakistani citizen carry out a government
procedure**, across five service domains.

| Domain | Covers |
|---|---|
| **NADRA / CNIC** | Identity card registration, renewal, correction, replacement, B-Form |
| **FBR** | NTN registration, annual income tax returns, ATL status, refunds via IRIS |
| **Driving License** | New applications, renewals, and international permits, by province |
| **SECP** | Company name reservation, incorporation, annual filings |
| **Passport (DGIP)** | New passports, renewals, urgent/executive categories, fees |

Anything else is deliberately out of scope. In particular, **there is no
"upload your own document and chat with it" surface** — that is a different
product from guiding a citizen through an official procedure, and answering
questions about a user's private file adds a data-handling burden this project
has no reason to take on. The corpus is fixed, public, and curated: the five
official-documentation files in [`data/`](data/).

> The upload-your-own-file code still exists in the repo (`backend/document_qa.py`,
> the `/documents/*` endpoints, `frontend/src/pages/DocumentQA.jsx`) but is
> **switched off at the UI layer** behind `FEATURES.documentQA` in
> [`frontend/src/config.js`](frontend/src/config.js). No nav entry, no footer
> link, no route — `/documents` redirects home. The flag is the single place to
> flip if that scope decision is ever revisited.

## Screenshots

Images live in [`data/images/`](data/images/) — see that folder's README for exact filenames expected.

### Home

<img src="data/images/landing.png" alt="Landing page" width="100%">

### Services directory

<img src="data/images/services.png" alt="Services grid" width="100%">

### Get Help — deterministic wizard

<img src="data/images/recommend.png" alt="Recommend wizard result" width="100%">

### Assistant — grounded, cited answer

<img src="data/images/assistant.png" alt="Assistant giving a cited answer" width="100%">

### Assistant — Pipeline Inspector

See every stage of the RAG pipeline (intent, rewritten queries, retrieval, reranking, verification) for the answer above.

<img src="data/images/assistant-pipeline.png" alt="Pipeline inspector panel" width="100%">

### About — methodology & tech stack

<img src="data/images/about.png" alt="About page" width="100%">

---

## What makes this different

Most "chat with your docs" demos stop at embed → retrieve → generate. This project treats **groundedness as a first-class requirement**, not an afterthought:

- **Hybrid retrieval** — dense (pgvector cosine similarity) *and* BM25 keyword search, fused with Reciprocal Rank Fusion, so exact terms (form names, fee tables) aren't lost to pure semantic search.
- **Query rewriting** — HyDE (hypothetical document embeddings) + sub-question decomposition, generated concurrently, so a single vague question still finds the right chunk.
- **Cross-encoder reranking + compression** — a second, more expensive model re-scores the retrieved pool against the *original* question before a sentence-level compressor fits everything into the token budget.
- **Two independent grounding checks** *after* generation — a citation verifier (does the cited chunk actually support this specific claim?) and a hallucination evaluator (does *every* sentence, cited or not, have support anywhere in the context?). A failed check triggers one stricter regeneration at `temperature=0`.
- **Explicit decline path** — if the retrieved context doesn't support an answer, the assistant says so instead of fabricating one.
- **An LLM-free deterministic path** — the "Get Help" wizard answers the ~20 most common procedural questions (fees, required docs, timelines) from a static lookup table, with zero hallucination risk, and hands off to the RAG assistant only for open-ended follow-ups.

## Architecture

```mermaid
flowchart TD
    U[User question] --> P[Planner<br/>Intent Router]
    P -->|out_of_scope| G[Generator]
    P -->|in scope| TS{Tool Selector}
    TS -->|needs live/tracking data| MCP[MCP Tools<br/>simulated gov-service calls]
    TS -->|no| RAG
    MCP --> RAG[RAG Agent]

    subgraph RAG[RAG Agent]
        direction TB
        QR[Query Rewriter<br/>HyDE + sub-questions] --> HS[Hybrid Search<br/>dense + BM25 → RRF]
        HS --> RR[Cross-Encoder Reranker]
        RR --> CP[Context Compressor<br/>sentence-level, token budget]
    end

    RAG --> G
    G --> V[Verification Agent]

    subgraph V[Verification Agent]
        direction TB
        CV[Citation Verifier] --> HE[Hallucination Evaluator]
        HE --> CC[Completeness Check<br/>was retrieved material dropped?]
    end

    V -->|failed & retries left| G
    V -->|passed| A[Final answer + citations]
```

**Pipeline stages, in order:**

1. **Intent Router** (`intent_router.py`) — classifies the question as `factual` / `comparison` / `procedural` / `out_of_scope` using Groq `openai/gpt-oss-20b`, `temperature=0`. Drives retrieval depth and answer format downstream.
2. **Tool Selector** (`mcp_tools.py`) — keyword-routes to simulated "live data" tools (`search_nadra`, `search_fbr`, `search_passport`, `extract_entities`, `compare_documents`) when the question implies tracking/status/current data.
3. **Query Rewriter** (`query_rewriter.py`) — expands the question into 3-5 search variants: HyDE passage + 3 sub-questions (concurrent Groq calls), plus deterministic keyword-boost queries for known problem phrasings (fee questions, "required documents" questions).
4. **Hybrid Search** (`hybrid_search.py`, `retrieval.py`, `bm25_retrieval.py`) — every query variant runs dense (pgvector, `BAAI/bge-large-en`) and BM25 search concurrently; all ranked lists are fused with Reciprocal Rank Fusion (`k=60`).
5. **Cross-Encoder Reranker** (`reranker.py`) — `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the fused pool against the original question; chunks below a minimum score are dropped.
6. **Context Compressor** (`compressor.py`) — re-scores individual *sentences* within surviving chunks and greedily packs the highest-value sentences into a fixed token budget. Chunks containing lists (a requirements list, a numbered procedure) take a separate path that keeps whole blocks: sentence-level trimming used to delete steps 2-3 of a five-step procedure, which reads as complete but isn't.
7. **Generator** (`generator.py`) — Groq `openai/gpt-oss-120b` (`temperature=0.1`) writes the answer with mandatory inline `[source: filename]` citations, using an intent-specific system prompt (factual / comparison / procedural / out-of-scope / document Q&A).
8. **Verification Agent** (`citation_verifier.py` + `hallucination_eval.py` + `completeness_check.py`) — three independent checks:
   - *citations*: each cited claim is scored against the chunks of the document it cites. Claims are delimited per line, so bulleted answers are verifiable (they carry no terminal punctuation).
   - *grounding*: every claim, cited or not, is scored against the whole context pool — again per bullet, not per prose sentence.
   - *completeness*: asks the question the other two can't — was retrieved material that answers the question left unused? A verbatim, correctly-cited one-line answer passes every truthfulness check and can still be useless.

   A failure in any of the three loops back to the generator once, at `temperature=0`. A completeness failure quotes the dropped passages into the retry prompt: a generic "be more complete" retry measurably reproduced the same short answer.

   Where a check *cannot* run (no citations parsed, no evaluable claims, no retrieved context) it reports `unverifiable` / `not_evaluated` rather than a pass — the UI shows a neutral "not verified" badge instead of a green tick.

All state flows through a single `PipelineState` TypedDict inside a compiled **LangGraph** `StateGraph` (`graph.py`), so every intermediate value (rewritten queries, raw/reranked/compressed chunks, tool results, verification scores) is available to the frontend's **Pipeline Inspector** panel for full transparency.

## Tech stack

| Layer | Choice | Notes |
|---|---|---|
| LLM inference | **Groq** — `openai/gpt-oss-120b` (generation), `openai/gpt-oss-20b` (routing, query rewriting, tool simulation, out-of-scope decline) | Split by cost/latency: cheap, fast model for structured/short tasks; larger model only for the final cited answer. Both IDs live in `backend/config.py` and are env-overridable via `GROQ_FAST_MODEL` / `GROQ_ANSWER_MODEL` |
| Embeddings | **`BAAI/bge-large-en`** (HuggingFace, `sentence-transformers`), 1024-dim | Free, runs locally, no per-call cost |
| Vector store | **PostgreSQL + pgvector** | Cosine distance (`<=>` operator) |
| Keyword search | **`rank_bm25`** (BM25Okapi), disk-cached index | Custom tokenizer preserves hyphenated government codes (e.g. `b-form`) |
| Reranker | **`cross-encoder/ms-marco-MiniLM-L-6-v2`** | Reused for reranking, compression, and citation/hallucination scoring |
| Orchestration | **LangGraph** `StateGraph` | Conditional edges + a bounded retry loop |
| Tool protocol | **MCP** (`mcp` Python SDK, SSE transport) | Standalone server on port 8001 |
| Backend | **FastAPI** (Railway) | |
| Frontend | **React 19 + Vite + React Router 7** (Vercel) | Mixed JS/TSX, `oxlint` |

## Project structure

```
Rahbar AI - Citizen Services Assistant/
├── backend/
│   ├── main.py                 FastAPI app — /chat, /services, /documents/*
│   ├── graph.py                LangGraph StateGraph — the pipeline itself
│   ├── intent_router.py        Question classification
│   ├── query_rewriter.py       HyDE + sub-question decomposition
│   ├── hybrid_search.py        Concurrent dense+BM25 fan-out → RRF fusion
│   ├── retrieval.py            Dense (pgvector) retrieval
│   ├── bm25_retrieval.py       BM25 keyword retrieval (disk-cached)
│   ├── reranker.py             Cross-encoder reranking
│   ├── compressor.py           Sentence-level context compression
│   ├── generator.py            Groq LLM answer generation + prompts
│   ├── citation_verifier.py    Per-claim citation grounding check
│   ├── hallucination_eval.py   Whole-answer grounding check
│   ├── completeness_check.py   Did the answer use what was retrieved?
│   ├── mcp_tools.py            Simulated gov-service tool functions
│   ├── mcp_server.py           Standalone MCP server (SSE, port 8001)
│   ├── document_qa.py          Upload-your-own-document Q&A (in-memory) — backend only, no UI (see Scope)
│   ├── ingest.py               One-time corpus chunk+embed+ingest script
│   ├── test_db.py              pgvector connectivity check
│   ├── test_pipeline.py        Legacy mock-based smoke test
│   ├── e2e_test.py             Full-pipeline mocked end-to-end test
│   └── tests/                  pytest regression suite (stubs all external deps)
├── frontend/
│   └── src/
│       ├── pages/              Landing, Services, Recommend, Assistant, About
│       │                       (+ DocumentQA — present but unrouted, see Scope)
│       ├── components/         Nav, Footer, PipelinePanel, CitationChip, ServiceCard, Disclaimer
│       ├── lib/                Answer formatting / citation stripping
│       ├── config.js           Feature flags (documentQA: false)
│       └── api.js              Backend fetch wrapper
├── data/
│   ├── NADRA.txt, FBR.txt, Passport.txt, SECP.txt, DrivingLicense.txt   (source corpus)
│   └── images/                 README screenshots (see data/images/README.md)
└── requirements.txt
```

## Getting started

### Prerequisites

- Python 3.13
- Node.js 18+
- PostgreSQL with the `pgvector` extension enabled
- A [Groq](https://console.groq.com) API key
- A [HuggingFace](https://huggingface.co/settings/tokens) token (avoids anonymous rate limits when downloading the embedding/reranker models)

### 1. Clone and configure environment

```bash
git clone <your-repo-url>
cd "Rahbar AI - Citizen Services Assistant"
```

Create a `.env` file in the project root:

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rahbar_ai
DB_USER=your_db_user
DB_PASSWORD=your_db_password
GROQ_API_KEY=your_groq_key
HF_TOKEN=your_huggingface_token
VITE_API_URL=http://localhost:8000
```

### 2. Backend setup

```bash
python -m venv rahbar_ai
rahbar_ai\Scripts\activate        # Windows
# source rahbar_ai/bin/activate   # macOS/Linux

pip install -r requirements.txt

cd backend
python test_db.py                 # confirms pgvector is enabled
python ingest.py                  # chunks + embeds the 5 corpora into Postgres
python main.py                    # FastAPI on http://localhost:8000
```

### 3. MCP server (optional, separate process)

```bash
cd backend
python mcp_server.py              # SSE server on http://localhost:8001
```

### 4. Frontend setup

```bash
cd frontend
npm install
npm run dev                       # Vite dev server on http://localhost:5173
```

Full command reference — including every command already used to build this project and every command still needed to finish, commit, and deploy it — lives in **[COMMANDS.md](COMMANDS.md)**.

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/services` | List all 5 service domains |
| `GET` | `/services/{service_id}` | Single service detail |
| `GET` | `/about/ingestion` | Corpus ingestion stats (doc/chunk counts, embedding model) |
| `POST` | `/chat` | Run the full pipeline: `{ question, history }` → full pipeline state (answer, citations, verification, pipeline trace) |

The FastAPI app also still serves `POST /documents/upload`, `GET /documents/{doc_id}`
and `POST /documents/{doc_id}/ask` from `document_qa.py`. **No part of the UI calls
them** — they are out of scope (see [Scope](#scope)) and reachable only by hitting
the API directly.

## Testing

```bash
cd backend
pytest tests/ -v            # regression suite — stubs sentence-transformers/groq/psycopg2/rank_bm25, no live deps needed
python test_pipeline.py     # legacy mock-based component smoke test
python e2e_test.py          # full pipeline, mocked, covers all 6 routing paths incl. retry
```

## Known limitations & honest tradeoffs

This is a portfolio/research project, and the README won't pretend otherwise:

- **MCP tools are simulated.** `search_nadra`, `search_fbr`, etc. call Groq to generate *realistic-looking* structured responses — they do not hit real government APIs (which mostly don't expose public APIs). The tool-call *pattern* is real; the data behind it isn't.
- **The corpus is hand-curated and static.** Five official-documentation files, ingested once. Fees, forms, and procedures change; nothing here re-crawls or re-validates them, so an answer is only as current as the last ingest.
- **The dormant document-Q&A endpoints are in-memory only** — lost on server restart, auto-evicted after 4 hours, and don't scale past a single process. They're left in the codebase rather than deleted, but they're not part of the product surface.
- **No connection pooling** — each retrieval opens a fresh `psycopg2` connection. Fine at demo scale; would need `pgbouncer` in production.
- **Citation regex has a known limitation** with abbreviations like "Rs." inside a sentence — documented and deliberately left as-is rather than papered over (see `tests/test_pipeline_fixes.py`).
- **Citation/hallucination checks use a general-purpose cross-encoder**, not a dedicated NLI model (e.g. `facebook/bart-large-mnli`) — a reasonable scope tradeoff, called out explicitly in the code.

## Roadmap

- [ ] Wire real government data sources behind the MCP tool interfaces where public APIs exist
- [ ] Widen the corpus within the five domains (provincial variations, updated fee schedules)
- [ ] Add connection pooling (pgbouncer) for the retrieval path
- [ ] Multi-turn conversational memory (currently stateless per request)
- [ ] Swap the cross-encoder citation check for a dedicated NLI model
- [ ] CI (GitHub Actions) running `pytest` + `npm run build` on every PR

---

Built by **Muhammad Talal Mohsin** · Portfolio demonstration project · Not affiliated with any Pakistani government body.
