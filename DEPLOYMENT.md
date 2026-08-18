# Deploying Rahbar AI

Target architecture, all on free tiers:

```
Browser
  │
  ├─► Vercel (Hobby)          static React/Vite SPA
  │      rahbar-ai.vercel.app
  │
  └─► HF Spaces (CPU basic)   FastAPI + LangGraph + sentence-transformers
         *.hf.space
           │
           ├─► Neon Postgres (free)   pgvector chunk store
           └─► Groq API (free)        LLM inference
```

Nothing here costs money. The trade-offs that come with that are in
[Free-tier reality check](#free-tier-reality-check) — read it before you demo
this to anyone.

---

## Step 0 — Get your latest code onto a branch (do this first)

> Already done if `git log --oneline -1 origin/main` shows the deployment
> config commit. Skip to Step 1.

V1.4 (`d9c5bc3`) was committed on a **detached HEAD** — no branch, never
pushed — while `origin/main` sat at `4c5a07e` (V1.3, a different line of
history). Pointing Vercel at `main` in that state would have deployed V1.3, and
an unreferenced commit is garbage-collectable.

The fix, for reference if it ever recurs:

```bash
git tag v1.3-main main                  # keep the old line, permanently
git push origin v1.3-main

git switch -c v1.4 d9c5bc3              # V1.4 now has a name
git push -u origin v1.4                 # and a second copy, on GitHub

git switch main
git reset --hard v1.4                   # main == V1.4
git push --force-with-lease origin main
```

`--force-with-lease` refuses to push if GitHub moved since you last fetched —
it is the safe form of `--force`.

---

## Step 1 — Confirm Neon and Groq are live

The backend is useless without both. Check them from your machine before you
spend fifteen minutes on a Docker build.

```bash
# Neon: are the chunks actually there?
cd backend
python -c "import os,psycopg2; from dotenv import load_dotenv; load_dotenv('../.env'); c=psycopg2.connect(os.getenv('DATABASE_URL')); k=c.cursor(); k.execute('select count(*) from chunks'); print('chunks:', k.fetchone()[0])"
```

```bash
# Groq: is the key valid and are the model IDs still reachable?
curl -s https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
```

If `gpt-oss-20b` / `gpt-oss-120b` are missing from that list, Groq retired
them. Set `GROQ_FAST_MODEL` / `GROQ_ANSWER_MODEL` to current IDs rather than
editing `config.py` — that is exactly why those env overrides exist.

---

## Step 2 — Deploy the backend to Hugging Face Spaces

### 2a. Create the Space

huggingface.co → **New Space**:

| Field      | Value                                                 |
| ---------- | ----------------------------------------------------- |
| Owner      | your account                                          |
| Space name | `rahbar-ai-backend`                                   |
| SDK        | **Docker → Blank**                                    |
| Hardware   | **CPU basic — free**                                  |
| Visibility | **Public**                                            |

Visibility must be Public. A private Space demands an auth token on every
request, and your frontend is public JavaScript — it has nowhere safe to keep
one.

Your API will live at `https://<username>-rahbar-ai-backend.hf.space`.

### 2b. Add the Space secrets

Space → **Settings → Variables and secrets**. Add each of these as a **Secret**,
copying values from your local `.env`:

```
GROQ_API_KEY, DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DATABASE_URL
```

### 2c. Wire up automatic deploys

`.github/workflows/deploy-backend.yml` pushes `backend/` to the Space on every
change to it. It exists because a Space needs the `Dockerfile` at its repo
**root**, not in a `backend/` subfolder — so the workflow uses `git subtree
split` to rewrite `backend/` as the root before pushing.

On GitHub → **Settings → Secrets and variables → Actions**, add three
repository secrets:

| Secret        | Value                                                          |
| ------------- | -------------------------------------------------------------- |
| `HF_TOKEN`    | A **write**-scoped token from huggingface.co/settings/tokens    |
| `HF_USERNAME` | Your Hugging Face username                                      |
| `HF_SPACE`    | `rahbar-ai-backend`                                             |

Then trigger the first deploy: **Actions → Deploy backend to Hugging Face →
Run workflow**. Every later push that touches `backend/` deploys on its own.

<details>
<summary>Manual fallback, if you would rather not use Actions</summary>

```bash
git remote add hf https://huggingface.co/spaces/<username>/rahbar-ai-backend
git subtree split --prefix=backend -b hf-deploy
git push hf hf-deploy:main --force
git branch -D hf-deploy
```

Username at the credential prompt is your HF username; password is the
write-scoped access token, not your account password.
</details>

### 2d. Watch the build

Space → **Logs → Build**. Expect **10–15 minutes** on the first build; most of
it is pulling torch and baking the two models into the image. Later pushes that
do not touch `requirements.txt` reuse cached layers and finish much faster.

Then, in **App logs**, wait for `Uvicorn running on http://0.0.0.0:7860`.

### 2e. Test it

```bash
curl https://<username>-rahbar-ai-backend.hf.space/health
curl https://<username>-rahbar-ai-backend.hf.space/services
```

```bash
curl -X POST https://<username>-rahbar-ai-backend.hf.space/chat -H "Content-Type: application/json" -d "{\"question\":\"How do I renew my CNIC?\"}" -w "\n\nTook %{time_total}s\n"
```

Note that `time_total`. It is the number that decides whether this feels like a
product or a science project — see the reality check below.

The first `/chat` is slower than the rest: `bm25_retrieval.get_bm25_index()`
finds no pickle, so it reads every chunk out of Neon, builds the BM25 index
once, and caches it to disk.

---

## Step 3 — Deploy the frontend to Vercel

vercel.com → **Add New → Project** → import `talalmohsin-98/Rahbar-AI`.

| Setting          | Value                                                     |
| ---------------- | --------------------------------------------------------- |
| Root Directory   | **`frontend`** ← must be set, it is not the repo root      |
| Framework Preset | Vite (auto-detected)                                       |
| Build Command    | `npm run build` (from `vercel.json`)                       |
| Output Directory | `dist` (from `vercel.json`)                                |

Add one environment variable, for **all** environments:

```
VITE_API_URL = https://<username>-rahbar-ai-backend.hf.space
```

No trailing slash — `api.js` concatenates `${BASE}${path}`, so a trailing slash
produces `//chat`.

**Vite bakes `VITE_*` variables into the bundle at build time.** They are not
read at runtime. Changing this value later does nothing until you redeploy:
Deployments → ⋯ → **Redeploy**.

Deploy. You get `https://rahbar-ai-<hash>.vercel.app`.

`frontend/vercel.json` already rewrites all paths to `index.html`, so deep links
like `/services` and `/about` work instead of 404ing — React Router needs that
on any static host.

---

## Step 4 — CORS

**Nothing to do while you are on a `.vercel.app` URL.** `main.py` matches
`https://.*\.vercel\.app` via `allow_origin_regex`, which covers the production
URL and every preview deploy.

Only when you move to a custom domain: add a Space **Variable** (not a secret —
it is not sensitive) and restart the Space.

```
FRONTEND_URL = https://rahbar.example.com
```

Comma-separate multiple origins. No trailing slashes.

> The original code had `"https://*.vercel.app"` in `allow_origins`. Starlette
> compares that list by **exact string** — the glob matched nothing, and would
> have blocked the frontend with a CORS error that reads like a backend
> outage. That is fixed; the glob now lives in `allow_origin_regex`, which is
> the field that is actually pattern-matched.

---

## Step 5 — Custom domain (optional, later)

Not needed for a demo, and adding one later is non-destructive — the
`.vercel.app` URL keeps working alongside it.

Vercel → Project → **Settings → Domains** → add the subdomain, then at your DNS
host:

```
Type    Name      Value
CNAME   rahbar    cname.vercel-dns.com
```

If the domain is on Cloudflare DNS, set that record to **DNS only** (grey
cloud). Proxying in front of Vercel is the most common cause of cert-issuance
failures. Then do Step 4 to add the new origin to CORS.

---

## Step 6 — Verify end to end

1. Open the site. The landing page renders — that alone only proves Vercel works.
2. Go to **Services**. The cards load, so `GET /services` crossed CORS.
3. Ask a real question: *"What documents do I need for a new passport?"*
4. DevTools → Network. Confirm `/chat` returns 200, and note how long it took.
5. Open the pipeline inspector panel. It should show retrieval, reranking, and
   citations — that proves the whole graph ran, not just a fallback path.
6. Hard-refresh on a deep link like `/about`. It must render, not 404.

---

## Free-tier reality check

Four things will bite you, in descending order of how much they matter.

**Per-request latency is the real constraint.** The pipeline runs the MiniLM
cross-encoder four separate times per question — reranking, compression,
citation verification, hallucination evaluation — on two shared vCPUs with no
GPU. Groq is fast; the CPU scoring passes are not. Measure the `time_total`
from Step 2e before deciding anything. If it is unpleasant, the honest fixes in
ascending order of effort: shrink the reranked pool from 10 chunks, then skip
the hallucination-eval pass on short answers, then move to a smaller embedding
model. That last one means re-ingesting Neon — the stored vectors are
1024-dimensional and pinned to `bge-large-en`, so the model cannot be swapped
without rebuilding the corpus.

**Free Spaces sleep when idle** (roughly 48 hours) and cold-start on the next
request. With the models baked into the image, a wake-up is startup time rather
than a 1.5GB download, but the first visitor after a quiet weekend still waits.
If you are sending this link to recruiters or clients, keep it warm: a free
UptimeRobot monitor hitting `/health` every 5 minutes, or a GitHub Actions cron.

**Groq free-tier rate limits are per-minute and per-day.** One user question is
not one API call — it is roughly four to six (intent routing, query rewriting,
tool simulation, generation, plus retries). A live demo to a room of ten people
can trip the limit. Check your dashboard's limits against that multiplier
before any presentation.

**Neon free tier auto-suspends after about 5 minutes idle.** It wakes on
connect in under a second, so this mostly shows up as a slightly slower first
query. `retrieval.py` opens a fresh connection per query with no pooling, which
is fine at demo traffic and is the first thing to revisit if this ever takes
real load.

---

## Redeploying after a change

```bash
git push origin main
```

That is the whole thing. Vercel rebuilds the frontend on any push; the GitHub
Action redeploys the Space when the push touched `backend/`.
