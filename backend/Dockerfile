# Rahbar AI backend — Hugging Face Spaces (Docker SDK, free CPU tier).
#
# Two things drive every decision in this file:
#   1. There is no GPU, so torch must be the CPU-only wheel (~200MB, not ~2.5GB).
#   2. Spaces sleep when idle, so the models are baked into the image at build
#      time. Downloading 1.5GB of weights on the first request after a wake-up
#      would time out the user's very first question.

FROM python:3.13-slim

# Spaces runs the container as uid 1000. Creating that user here means the
# HF cache, the app directory, and the BM25 pickle are all writable at runtime.
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    BM25_INDEX_PATH=/home/user/app/bm25_index.pkl

WORKDIR /home/user/app

# CPU-only torch first, from PyTorch's own index. sentence-transformers then
# sees the requirement already satisfied and does not pull the CUDA build.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

USER user
COPY --chown=user:user . .

# Bake the weights in. bge-large-en (~1.3GB) is the embedding model the Neon
# vectors were built with — it cannot be swapped without re-ingesting, because
# the stored vectors are 1024-dimensional. MiniLM is the cross-encoder used by
# the reranker, compressor, citation verifier, and hallucination evaluator.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-large-en'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('models cached')"

# Spaces routes external traffic to app_port from README.md, which is 7860.
EXPOSE 7860

# Single worker on purpose: two workers would each load their own ~1.5GB copy
# of the models. One worker, one model set, on 2 free vCPUs.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
