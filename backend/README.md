# Rahbar AI — Backend API

FastAPI + LangGraph RAG pipeline answering questions about five Pakistani
government service domains: NADRA/CNIC, FBR tax filing, driving licence,
SECP registration, and passports (DGIP).

## Endpoints

| Method | Path                    | Purpose                                  |
| ------ | ----------------------- | ---------------------------------------- |
| GET    | `/health`               | Liveness check                           |
| GET    | `/services`             | The five service domains                 |
| GET    | `/services/{id}`        | One service domain                       |
| GET    | `/about/ingestion`      | Corpus / ingestion metadata              |
| POST   | `/chat`                 | Full RAG pipeline, returns pipeline state |
| POST   | `/documents/upload`     | Upload a file for ad-hoc Q&A             |
| POST   | `/documents/{id}/ask`   | Ask about an uploaded file               |
| GET    | `/docs`                 | Interactive OpenAPI docs                 |

The `/documents/*` endpoints are live but unrouted in the UI — see **Scope** in
the [root README](../README.md).

## Required environment

Copy [`.env.example`](../.env.example) to `.env` in the project root and fill it
in. Nothing works without these, and none of them belong in git.

| Name           | Value                                        |
| -------------- | -------------------------------------------- |
| `GROQ_API_KEY` | Groq API key                                 |
| `DB_HOST`      | Postgres host                                |
| `DB_PORT`      | `5432`                                       |
| `DB_NAME`      | Database name                                |
| `DB_USER`      | Database role                                |
| `DB_PASSWORD`  | Database password                            |
| `DATABASE_URL` | Full connection string (used by `ingest.py`) |

## Local run

```bash
pip install -r requirements.txt
python test_db.py                 # confirms pgvector is enabled
python ingest.py                  # one-time: chunk + embed the corpus
uvicorn main:app --reload --port 8000
```

The MCP server is a separate process:

```bash
python mcp_server.py              # SSE on http://localhost:8001
```

## Tests

```bash
pytest tests/ -v            # regression suite, all external deps stubbed
python test_pipeline.py     # legacy mock-based component smoke test
python e2e_test.py          # full pipeline, mocked, all 6 routing paths
```

First run downloads `BAAI/bge-large-en` and `cross-encoder/ms-marco-MiniLM-L-6-v2`
from HuggingFace (~1.4 GB, cached afterwards). Set `HF_TOKEN` to avoid anonymous
rate limits.
