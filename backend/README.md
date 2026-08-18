---
title: Rahbar AI Backend
emoji: 🇵🇰
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: RAG API for Pakistani government service guidance
---

# Rahbar AI — Backend API

FastAPI + LangGraph RAG pipeline answering questions about five Pakistani
government service domains: NADRA/CNIC, FBR tax filing, driving licence,
SECP registration, and passports (DGIP).

The YAML block above is not documentation — it is the Hugging Face Space
configuration. `sdk: docker` tells Spaces to build `Dockerfile`, and
`app_port: 7860` tells it which port to route public traffic to. Editing
those two lines changes how the Space runs.

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

## Required secrets

Set these in **Settings → Variables and secrets** on the Space. Nothing works
without them, and none of them belong in git.

| Name             | Kind   | Value                                      |
| ---------------- | ------ | ------------------------------------------ |
| `GROQ_API_KEY`   | Secret | Groq API key                               |
| `DB_HOST`        | Secret | Neon host, e.g. `ep-xxx.neon.tech`         |
| `DB_PORT`        | Secret | `5432`                                     |
| `DB_NAME`        | Secret | Neon database name                         |
| `DB_USER`        | Secret | Neon role                                  |
| `DB_PASSWORD`    | Secret | Neon password                              |
| `DATABASE_URL`   | Secret | Full Neon connection string (used by ingest)|
| `FRONTEND_URL`   | Variable | Deployed frontend origin, no trailing slash |

## Local run

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## Deploying an update

This directory is pushed to the Space as a git subtree from the main repo:

```bash
git subtree push --prefix=backend hf main
```

Spaces rebuilds the image on every push. Build takes roughly 10–15 minutes
because the model weights are baked into the image.
