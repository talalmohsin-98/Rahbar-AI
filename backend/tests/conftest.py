import os, sys
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Stubs the heavy/external dependencies (groq, psycopg2, sentence_transformers,
rank_bm25, langgraph) with lightweight fakes BEFORE any project module is
imported, so we can exercise the REAL project logic in reranker.py,
compressor.py, generator.py, citation_verifier.py, hallucination_eval.py,
bm25_retrieval.py, hybrid_search.py, document_qa.py etc. without needing a
live Groq API key, a live Postgres/pgvector instance, or downloading real
embedding/CrossEncoder models.
"""
import sys
import types
import numpy as np


# ---------------------------------------------------------------------------
# Fake CrossEncoder (used by reranker.py, compressor.py, citation_verifier.py,
# hallucination_eval.py). Scores pairs by lexical word-overlap so tests are
# deterministic and meaningful (relevant text really does score higher).
#
# The overlap count is mapped onto a SIGNED range, because the real
# ms-marco-MiniLM CrossEncoder emits roughly -11 (irrelevant) to +10 (strongly
# relevant), and every threshold in the pipeline (MIN_RERANK_SCORE,
# CITATION_MIN_SCORE, HALLUCINATION_THRESHOLD) is calibrated in that space.
# A stub returning only 0..n made "completely unrelated" score 0 — above every
# negative threshold — so a test could not tell rejection from acceptance.
# ---------------------------------------------------------------------------
class FakeCrossEncoder:
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, pairs):
        scores = []
        for a, b in pairs:
            wa = set(str(a).lower().split())
            wb = set(str(b).lower().split())
            overlap = len(wa & wb)
            scores.append(float(overlap) * 2.0 - 4.0)   # 0 overlap => -4.0
        return np.array(scores)


# ---------------------------------------------------------------------------
# Fake SentenceTransformer (used by retrieval.py, document_qa.py). Produces a
# deterministic "embedding" from a hash of the words, so semantically similar
# strings that share words end up with similar (though not identical) vectors.
# ---------------------------------------------------------------------------
class FakeSentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def encode(self, text, normalize_embeddings=True):
        texts = [text] if isinstance(text, str) else list(text)
        vecs = []
        for t in texts:
            v = np.zeros(32, dtype=np.float32)
            for word in str(t).lower().split():
                v[hash(word) % 32] += 1.0
            norm = np.linalg.norm(v)
            if normalize_embeddings and norm > 0:
                v = v / norm
            vecs.append(v)
        result = np.array(vecs, dtype=np.float32)
        return result[0] if isinstance(text, str) else result


# ---------------------------------------------------------------------------
# Fake Groq client (used by generator.py, intent_router.py, query_rewriter.py,
# mcp_tools.py). Returns a canned response; individual tests monkeypatch
# `_response_text` via the module-level FAKE_GROQ_RESPONSES queue when they
# need specific content.
# ---------------------------------------------------------------------------
FAKE_GROQ_RESPONSES = []  # tests push strings here; Groq pops them in order


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5


class _FakeCompletionResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    def create(self, **kwargs):
        content = FAKE_GROQ_RESPONSES.pop(0) if FAKE_GROQ_RESPONSES else "factual"
        return _FakeCompletionResponse(content)


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class FakeGroq:
    def __init__(self, *args, **kwargs):
        self.chat = _FakeChat()


def _install_stub_modules():
    # sentence_transformers
    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = FakeSentenceTransformer
    st_mod.CrossEncoder = FakeCrossEncoder
    sys.modules["sentence_transformers"] = st_mod

    # groq
    groq_mod = types.ModuleType("groq")
    groq_mod.Groq = FakeGroq
    sys.modules["groq"] = groq_mod

    # psycopg2 (+ .extras) — only needs to exist for import; DB-hitting tests
    # monkeypatch get_connection()/get_bm25_index() directly instead of
    # calling through this stub.
    psycopg2_mod = types.ModuleType("psycopg2")
    extras_mod = types.ModuleType("psycopg2.extras")
    extras_mod.RealDictCursor = object
    extensions_mod = types.ModuleType("psycopg2.extensions")
    extensions_mod.connection = object
    psycopg2_mod.extras = extras_mod
    psycopg2_mod.extensions = extensions_mod
    psycopg2_mod.connect = lambda *a, **k: None
    sys.modules["psycopg2"] = psycopg2_mod
    sys.modules["psycopg2.extras"] = extras_mod
    sys.modules["psycopg2.extensions"] = extensions_mod

    # rank_bm25
    bm25_mod = types.ModuleType("rank_bm25")

    class _FakeBM25Okapi:
        def __init__(self, tokenized_corpus):
            self.corpus = tokenized_corpus

        def get_scores(self, query_tokens):
            qset = set(query_tokens)
            return np.array([
                float(len(qset & set(doc))) for doc in self.corpus
            ])

    bm25_mod.BM25Okapi = _FakeBM25Okapi
    sys.modules["rank_bm25"] = bm25_mod


_install_stub_modules()
