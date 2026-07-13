"""
citation_verifier.py
--------------------
PURPOSE: After the LLM generates an answer, verify that each cited claim
         actually appears in (or is entailed by) the source chunks.
         Catches hallucinations that passed groundedness but still crept
         into the final answer.

Pipeline position:
    Question
        → intent_router.py
        → query_rewriter.py
        → hybrid_search.py
        → reranker.py
        → compressor.py
        → [generator.py]        (Day 6 — generates the answer WITH citations)
        → citation_verifier.py  ✅ YOU ARE HERE
        → hallucination_eval.py (Day 6)
        → Answer

WHY THIS EXISTS — the subtle problem:
    groundedness.py (MIN_RERANK_SCORE=2.8) filters out chunks that aren't
    relevant to the question. That prevents retrieving bad chunks.

    But the LLM can still hallucinate WITHIN a valid chunk's topic.
    Example:
        Chunk says: "NADRA charges Rs. 750 for CNIC."
        LLM answers: "NADRA charges Rs. 500 for CNIC." [source: nadra_guide.pdf]

    The chunk is grounded (score > 2.8), but the LLM stated the wrong number.
    The citation looks legitimate but is factually wrong.

    citation_verifier.py checks: is the specific claim in the answer
    actually supported by the cited chunk? Not just "is the chunk relevant"
    but "does the chunk actually say this?"

APPROACH:
    1. Parse the answer to extract (claim, source) pairs
    2. For each claim, find the cited source chunk
    3. Use CrossEncoder to score: does the chunk ENTAIL this claim?
    4. Flag claims with low entailment scores as unverified
"""

import os
import re
from typing import Any
from sentence_transformers import CrossEncoder
from reranker import get_cross_encoder   # reuse loaded model

# ---------------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------------
# Minimum CrossEncoder score for a claim to be considered supported by its source.
# We use a lower threshold than reranking (2.8) because entailment scoring
# has a different distribution than passage reranking scoring.
# This is a tunable parameter — calibrate on your domain.
CITATION_MIN_SCORE = float(os.getenv("CITATION_MIN_SCORE", "1.5"))


# ---------------------------------------------------------------------------
# 2. CITATION PARSER
# ---------------------------------------------------------------------------
def parse_citations(answer: str) -> list[dict[str, str]]:
    """
    Extracts (claim, source) pairs from the LLM's answer.

    We instruct the LLM (in generator.py, Day 6) to format citations like:
        "NADRA charges Rs. 750 for CNIC. [source: nadra_guide.pdf]"

    This function finds those patterns and returns structured data.

    Returns:
        List of dicts: [{"claim": "...", "source": "nadra_guide.pdf"}, ...]

    Why this format?
        Square bracket citations are common in academic and document-style text.
        They're easy to parse with regex without requiring complex NLP.
        The LLM can be reliably instructed to produce them.

    Sentences WITHOUT a citation tag are not verified here — they're handled
    by hallucination_eval.py which checks the whole answer against all chunks.
    """
    cited_claims = []

    # Pattern: any text followed by [source: filename]
    # The .*? is non-greedy — stops at the first [source: ...] it finds
    pattern = r'([^.!?]*[.!?])\s*\[source:\s*([^\]]+)\]'
    matches = re.finditer(pattern, answer, re.IGNORECASE)

    for match in matches:
        claim  = match.group(1).strip()
        source = match.group(2).strip()
        if claim and source:
            cited_claims.append({"claim": claim, "source": source})

    return cited_claims


# ---------------------------------------------------------------------------
# 3. FIND SOURCE CHUNK
# ---------------------------------------------------------------------------
def find_source_chunk(
    source_name: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """
    Finds the chunk whose source matches the cited source name.

    We do partial matching (source_name in chunk["source"]) because the LLM
    might cite "nadra_guide" while the actual source is "nadra_guide.pdf".
    Case-insensitive for robustness.

    Returns:
        The matching chunk dict, or None if not found.
        If multiple chunks from the same source exist, returns the highest-ranked one
        (already sorted by rank from reranker.py).
    """
    source_lower = source_name.lower()
    for chunk in chunks:
        if source_lower in (chunk.get("source") or "").lower():
            return chunk
    return None


# ---------------------------------------------------------------------------
# 4. VERIFY ONE CLAIM
# ---------------------------------------------------------------------------
def verify_claim(
    claim: str,
    chunk: dict[str, Any],
    model: CrossEncoder,
) -> dict[str, Any]:
    """
    Scores whether a chunk entails (supports) a specific claim.

    We use the CrossEncoder for this. It was trained on passage relevance,
    but high relevance between a claim and a chunk means the chunk supports
    the claim — which is a reasonable proxy for entailment.

    For a production system, you'd use a dedicated NLI (Natural Language Inference)
    model (e.g. facebook/bart-large-mnli) which is trained specifically for
    entailment/contradiction/neutral classification. For our scope, CrossEncoder is sufficient.

    Returns:
        Dict with: claim, source, score, verified (bool), reason
    """
    score = float(model.predict([(claim, chunk.get("content") or "")])[0])
    verified = score >= CITATION_MIN_SCORE

    return {
        "claim":    claim,
        "source":   chunk["source"],
        "score":    score,
        "verified": verified,
        "reason":   "supported" if verified else f"low entailment score ({score:.2f} < {CITATION_MIN_SCORE})",
    }


# ---------------------------------------------------------------------------
# 5. MAIN VERIFICATION FUNCTION
# ---------------------------------------------------------------------------
def verify_citations(
    answer: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Full citation verification pipeline for a complete LLM answer.

    Args:
        answer: The LLM-generated answer string (with [source: ...] tags).
        chunks: The compressed chunks that were passed to the LLM as context.

    Returns:
        Dict containing:
            - verified_claims:    list of claims that ARE supported by sources
            - unverified_claims:  list of claims that are NOT supported
            - verification_rate:  float (0.0 to 1.0) — % of citations verified
            - passed:             bool — True if all citations verified
            - summary:            human-readable summary string

    The LangGraph Verification Agent node will check `passed`.
    If False, it can trigger answer regeneration with a stricter prompt.
    """
    if not answer:
        return {
            "verified_claims": [], "unverified_claims": [],
            "verification_rate": 1.0, "passed": True,
            "summary": "No answer to verify."
        }

    model = get_cross_encoder()

    # Step 1: Extract all (claim, source) pairs from the answer
    cited_claims = parse_citations(answer)

    if not cited_claims:
        # No citations found — either the LLM didn't cite, or the format is off
        # This is itself a signal worth logging; we pass but flag it
        return {
            "verified_claims": [], "unverified_claims": [],
            "verification_rate": 1.0, "passed": True,
            "summary": "No citations found in answer. Cannot verify. Check LLM citation format."
        }

    # Step 2: Verify each cited claim
    verified   = []
    unverified = []

    for cited in cited_claims:
        # Find the chunk the LLM cited
        chunk = find_source_chunk(cited["source"], chunks)

        if chunk is None:
            # LLM cited a source that doesn't exist in our context — hallucinated citation
            unverified.append({
                "claim":    cited["claim"],
                "source":   cited["source"],
                "score":    0.0,
                "verified": False,
                "reason":   "cited source not found in context chunks",
            })
            continue

        # Score the claim against the chunk
        result = verify_claim(cited["claim"], chunk, model)

        if result["verified"]:
            verified.append(result)
        else:
            unverified.append(result)

    # Step 3: Compute overall verification rate
    total = len(cited_claims)
    verification_rate = len(verified) / total if total > 0 else 1.0
    passed = len(unverified) == 0

    summary = (
        f"{len(verified)}/{total} citations verified. "
        + (f"Unverified: {[u['claim'][:50] for u in unverified]}" if unverified else "All citations supported.")
    )

    return {
        "verified_claims":   verified,
        "unverified_claims": unverified,
        "verification_rate": verification_rate,
        "passed":            passed,
        "summary":           summary,
    }


# ---------------------------------------------------------------------------
# 6. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mock_answer = (
        "NADRA charges Rs. 750 for a new CNIC. [source: nadra_guide.pdf] "
        "The processing time is 30 days. [source: nadra_guide.pdf] "
        "NADRA was founded in 1995. [source: nadra_guide.pdf]"
    )

    mock_chunks = [
        {
            "chunk_id": 1,
            "content": "NADRA charges Rs. 750 for issuance of a new Computerized National Identity Card. Standard processing takes 30 working days.",
            "source":  "nadra_guide.pdf",
            "rerank_score": 9.1,
            "rank": 1,
        }
    ]

    print(f"\nAnswer:\n{mock_answer}\n")
    result = verify_citations(mock_answer, mock_chunks)

    print(f"Verification rate: {result['verification_rate']:.0%}")
    print(f"Passed: {result['passed']}")
    print(f"Summary: {result['summary']}")
    print(f"\nVerified ({len(result['verified_claims'])}):")
    for c in result['verified_claims']:
        print(f"  ✓ [{c['score']:.2f}] {c['claim'][:70]}")
    print(f"\nUnverified ({len(result['unverified_claims'])}):")
    for c in result['unverified_claims']:
        print(f"  ✗ [{c['score']:.2f}] {c['claim'][:70]} — {c['reason']}")
