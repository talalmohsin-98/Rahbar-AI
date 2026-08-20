import os
import json
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from config import FAST_MODEL
from intent_router import Intent  # we use intent to choose rewriting strategy


# ---------------------------------------------------------------------------
# 1. GROQ CLIENT (same pattern as intent_router.py)
# ---------------------------------------------------------------------------
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set.")
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# 2. HyDE — HYPOTHETICAL DOCUMENT EMBEDDING
# ---------------------------------------------------------------------------
HYDE_SYSTEM_PROMPT = """You are an expert on Pakistani government services and documentation.

The user will ask you a question. Write a SHORT, FACTUAL paragraph (3-5 sentences)
that DIRECTLY ANSWERS the question — as if it were an excerpt from an official
Pakistani government document or guide.

Use formal, document-like language. Include specific terms, document names, and
procedural details that would appear in real government documentation.

Do not say "According to..." or "I think...". Write as if it IS the document.
Do not include any preamble or explanation. Just write the hypothetical document excerpt."""


def generate_hyde_query(question: str) -> str:
    """
    Generates a hypothetical document passage that would answer the question.

    This passage is then used AS the search query instead of (or alongside) the question.

    Why this works:
        User question: "What documents do I need for CNIC?"
        → Embedding of this question might not match well with document excerpts
          because question-style text (interrogative) has different embedding
          characteristics than statement-style text (declarative).

        HyDE output: "CNIC registration requires the applicant to submit the original
          B-Form or birth certificate, two passport photographs, and proof of address..."
        → This is statement-style, matches the style of real document chunks.
        → Cosine similarity between this and actual document chunks is much higher.

    The HyDE document doesn't need to be factually correct — it just needs to use
    the right vocabulary and style to pull the right chunks from the vector space.
    """
    client = get_groq_client()
    response = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[
            {"role": "system", "content": HYDE_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        temperature=0.3,   # slight creativity helps vocabulary diversity
        max_tokens=200,    # short passage — don't need a full essay
    )
    return (response.choices[0].message.content or "").strip()


TRANSLATE_SYSTEM_PROMPT = """You translate Pakistani citizens' questions about
government services into plain English so they can be matched against
English-language official documentation.

The input may be Urdu, romanized Urdu (Urdu written in English letters), or a
mix of Urdu and English in one sentence.

Rules:
- Output ONLY the English translation. No preamble, no notes, no quotes.
- Keep official terms, acronyms and document names exactly as written:
  CNIC, NICOP, NADRA, FRC, B-Form, NTN, FBR, IRIS, ATL, SECP, SMC, DGIP, MRP.
- Preserve the question form. A question stays a question.
- Do not answer the question and do not add information that is not there."""


def translate_to_english(question: str) -> str:
    """
    Renders a code-switched or Urdu question in English.

    WHY: the embedder is `BAAI/bge-large-en`. It has no Urdu vocabulary, so
    "CNIC renewal ka process kya hai" contributes almost nothing beyond the
    token "CNIC" — the vector lands nowhere near the renewal chunk. Searching
    the English rendering ALONGSIDE the original recovers that signal without
    betting the whole retrieval on the translation being faithful: the original
    is still query #1, so a bad translation can only add noise, never replace
    the baseline.

    On any failure this returns "" and the caller simply skips the variant.
    """
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[
                {"role": "system", "content": TRANSLATE_SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            temperature=0.0,   # translation is not a creative task
            max_tokens=150,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[QueryRewriter] Translation failed ({e}) - using original only")
        return ""


# ---------------------------------------------------------------------------
# 3. SUB-QUESTION DECOMPOSITION
# ---------------------------------------------------------------------------
SUBQ_SYSTEM_PROMPT = """You are a search query optimizer for a Pakistani government services system.

Given a user question, generate exactly 3 focused sub-questions that together cover
all the information needed to answer the original question.

Each sub-question should:
- Be self-contained and searchable on its own
- Focus on ONE specific aspect of the original question
- Use vocabulary likely to appear in official government documents

Return ONLY a JSON array of 3 strings. No explanation. No other text.
Example output format: ["sub-question 1", "sub-question 2", "sub-question 3"]"""


def generate_sub_questions(question: str) -> list[str]:
    """
    Decomposes a question into 3 focused sub-questions for retrieval.

    Example:
        Input:  "What are the NADRA CNIC requirements and how long does it take?"
        Output: [
            "What documents are required for NADRA CNIC registration?",
            "What is the processing time for NADRA CNIC application?",
            "What is the NADRA CNIC application procedure?"
        ]

    Each sub-question is then passed separately to hybrid_search.
    More focused queries → more precise retrieval per aspect of the question.

    We parse the LLM output as JSON for reliability.
    If parsing fails, we return just the original question (safe fallback).
    """
    client = get_groq_client()
    response = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[
            {"role": "system", "content": SUBQ_SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        temperature=0.2,   # low temperature for consistent structured output
        max_tokens=300,
    )

    raw = (response.choices[0].message.content or "").strip()

    try:
        # Remove markdown code fences if present (LLMs sometimes add ```json ... ```)
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        sub_questions = json.loads(raw)

        # Validate: should be a list of strings
        if isinstance(sub_questions, list) and all(isinstance(q, str) for q in sub_questions):
            return sub_questions[:3]  # cap at 3
    except (json.JSONDecodeError, IndexError):
        pass

    # Fallback: LLM output wasn't parseable — return original question
    print(f"[query_rewriter] Sub-question parsing failed, using original query.")
    return [question]


# ---------------------------------------------------------------------------
# 4. MAIN REWRITE FUNCTION
# ---------------------------------------------------------------------------
# Vocabulary that pulls each domain's own chunks into a compound query's
# candidate pool. Keys match input_analysis._DOMAIN_KEYWORDS.
_DOMAIN_QUERY_HINTS = {
    "nadra":    "NADRA CNIC NICOP identity card registration requirements",
    "fbr":      "FBR NTN IRIS income tax return filing requirements",
    "license":  "driving licence learner permit excise office requirements",
    "secp":     "SECP company incorporation registration requirements",
    "passport": "DGIP passport application renewal requirements",
}


def rewrite(question: str, intent: Intent, analysis: dict | None = None) -> list[str]:
    """
    Produces a list of search queries from the original question.
    The list always starts with the original question (so it's never dropped).

    Strategy depends on intent:
        FACTUAL:    original + HyDE passage + 2 sub-questions = up to 4 queries
        PROCEDURAL: original + HyDE passage + sub-questions (steps-focused)
        COMPARISON: original + sub-questions per entity (no HyDE — comparison
                    needs targeted retrieval for each side, not one blended passage)
        OUT_OF_SCOPE: return just [question] — caller should skip retrieval anyway

    Why always include the original?
        HyDE and sub-questions are improvements, but they can drift.
        The original query is always a valid search — keeping it ensures
        we never do worse than just the baseline query.

    `analysis` is input_analysis.analyse()'s output, or None when a caller
    (tests, the MCP layer) has not run it. Two things change when it is present:

        code-switched input  -> add an English translation as an extra query,
                                because the embedder is English-only
        compound question    -> add one query per named service domain, because
                                a single vector spanning NADRA and Passport
                                retrieves the best of neither

    Returns:
        List of query strings. hybrid_search.py will run all of them.
        Duplicates are removed while preserving order.
    """
    analysis = analysis or {}
    if intent == Intent.OUT_OF_SCOPE:
        # No point rewriting — caller will skip retrieval
        return [question]

    queries = [question]  # always start with the original

    if intent in (Intent.FACTUAL, Intent.PROCEDURAL):
        # HyDE and sub-questions are independent Groq calls — running them
        # sequentially needlessly doubles the network round-trip latency
        # here. Run both concurrently instead.
        with ThreadPoolExecutor(max_workers=2) as executor:
            hyde_future = executor.submit(generate_hyde_query, question)
            subq_future = executor.submit(generate_sub_questions, question)
            hyde_passage = hyde_future.result()
            sub_qs = subq_future.result()

        queries.append(hyde_passage)
        queries.extend(sub_qs)

    elif intent == Intent.COMPARISON:
        # For comparisons, skip HyDE (a blended passage confuses retrieval)
        # Instead: focus sub-questions on each entity being compared
        sub_qs = generate_sub_questions(question)
        queries.extend(sub_qs)

    # Deterministic keyword boost for fee/cost questions. Phrasings like
    # "fee of a new CNIC" also keyword-match the many procedural chunks about
    # a "new CNIC", pushing the actual "## FEE STRUCTURE" table chunk out of
    # the top-k and making the generator wrongly decline. Appending the fee
    # table's own vocabulary as an extra BM25 query pulls it into the pool
    # regardless of how the citizen phrases the question. Domain-agnostic.
    if any(w in question.lower() for w in ("fee", "cost", "charge", "price", "how much")):
        queries.append(f"{question} fee structure charges normal urgent executive")

    # Same idea for "what documents / what do I need" questions. Phrasings like
    # "CNIC registration" keyword-match the many chunks mentioning "registration"
    # (centres, policy, process) and bury the actual "Requirements:" chunk, so
    # the generator wrongly declines. Appending the requirements vocabulary pulls
    # that chunk into the pool regardless of phrasing. Domain-agnostic.
    if any(w in question.lower() for w in ("document", "papers", "require", "need", "eligib", "checklist")):
        queries.append(f"{question} required documents requirements eligibility checklist")

    # Code-switched or Urdu input: search the English rendering too. This is
    # additive - the original stays first in the list, so a poor translation
    # can only dilute the pool, never replace the baseline query.
    if analysis.get("needs_translation"):
        english = translate_to_english(question)
        if english and english.lower() != question.lower():
            print(f"[QueryRewriter] Translated variant: {english[:70]}")
            queries.append(english)

    # Compound cross-domain question. generate_sub_questions() already
    # decomposes, but it is an LLM with no knowledge that these five domains
    # exist, so nothing guarantees it produces one sub-question per domain -
    # it frequently splits along some other axis and leaves a domain
    # unretrieved. These deterministic per-domain queries guarantee each named
    # domain gets its own shot at the index, in the same spirit as the fee and
    # documents keyword boosts below.
    if analysis.get("compound"):
        for domain in analysis.get("domains", []):
            boost = _DOMAIN_QUERY_HINTS.get(domain)
            if boost:
                queries.append(f"{question} {boost}")

    # Deduplicate while preserving order
    # We use dict.fromkeys() which keeps insertion order (Python 3.7+)
    queries = list(dict.fromkeys(queries))

    return queries


# ---------------------------------------------------------------------------
# 5. PIPELINE STATE OBJECT (passed between nodes in LangGraph later)
# ---------------------------------------------------------------------------
def build_rewrite_result(question: str, intent: Intent, queries: list[str]) -> dict:
    """
    Packages the rewriter's output into a dict that LangGraph state can carry.

    This isn't just nice organization — in LangGraph, each node receives and
    returns a state dict. Building this structure here prepares us for
    clean graph integration in graph.py (Day 5).
    """
    return {
        "original_question": question,
        "intent":            intent.value,
        "search_queries":    queries,
        "query_count":       len(queries),
    }


# ---------------------------------------------------------------------------
# 6. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from intent_router import route

    test_questions = [
        ("What documents do I need for NADRA CNIC?", Intent.FACTUAL),
        ("Compare B-Form and CNIC registration requirements", Intent.COMPARISON),
        ("How do I apply for a Pakistani passport?", Intent.PROCEDURAL),
    ]

    for question, intent in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {question}")
        print(f"Intent: {intent.value}")
        queries = rewrite(question, intent)
        result = build_rewrite_result(question, intent, queries)
        print(f"\nGenerated {result['query_count']} search queries:")
        for i, q in enumerate(result['search_queries'], 1):
            prefix = "ORIGINAL" if i == 1 else f"VARIANT {i}"
            print(f"  [{prefix}] {q[:100]}...")
