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
#
# CALIBRATED ON THIS CORPUS (backend/, 60 sampled chunks — rerun before trusting
# it on new data). Scoring a claim against the chunk it cites:
#
#   verbatim claims lifted from the chunk      min  +0.62   median +7.77
#   LLM paraphrases of the chunk (6 observed)  min  -2.62   max    +6.36
#   bullets from an unrelated document         max  -3.91
#   invented requirements/fees                 max  -6.61
#
# The old 1.5 was calibrated for full-sentence claims. Once parse_citations()
# started (correctly) reading short bullet claims, 1.5 rejected 4 of 6
# genuinely-supported claims on the CNIC renewal answer — the verifier cried
# wolf on a faithful answer. -3.0 keeps every observed true claim and still
# rejects every observed fabrication.
#
# Note the margin between a paraphrase (-2.62) and an unrelated bullet (-3.91)
# is thin: this is a relevance model doing entailment's job. A dedicated NLI
# model (e.g. bart-large-mnli) would separate these far more cleanly and is the
# right upgrade if false flags matter more than the extra model.
CITATION_MIN_SCORE = float(os.getenv("CITATION_MIN_SCORE", "-3.0"))


# ---------------------------------------------------------------------------
# 2. CITATION PARSER
# ---------------------------------------------------------------------------
CITATION_TAG = re.compile(r'\[\s*source\s*:\s*([^\]]+)\]', re.IGNORECASE)

# Leading list marker on a bullet or numbered line: "- ", "* ", "• ", "1. ", "2) "
LIST_MARKER = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s*')


def parse_citations(answer: str) -> list[dict[str, str]]:
    """
    Extracts (claim, source) pairs from the LLM's answer.

    We instruct the LLM (in generator.py) to format citations like:
        "NADRA charges Rs. 750 for CNIC. [source: nadra_guide.pdf]"

    Returns:
        List of dicts: [{"claim": "...", "source": "nadra_guide.pdf"}, ...]

    HOW A CLAIM IS DELIMITED — and why it is not sentence-based:
        The old pattern was r'([^.!?]*[.!?])\\s*\\[source:...\\]' — it required
        the claim to END in '.', '!' or '?' immediately before the tag. But
        generator.py MANDATES list formatting for any answer that enumerates
        documents, requirements, steps or fees, and list items don't carry
        terminal punctuation:

            - CNIC number [source: NADRA.txt]

        So the regex matched nothing, verify_citations() reported "no citations
        found", and — because that path returned passed=True — the UI showed a
        green "citations verified" badge on an answer where NOTHING had been
        checked. Every list answer (i.e. most of them) verified vacuously.

        A claim now runs from the end of the previous citation tag (or the
        start of the line, whichever is later) up to this tag. Anchoring on
        the line break is what makes bullets work, and it also fixes the
        "Rs. 750" backtracking bug the old sentence regex had — an
        abbreviation's period no longer truncates the claim.

    Sentences WITHOUT a citation tag are not verified here — they're handled
    by hallucination_eval.py which checks the whole answer against all chunks.
    """
    cited_claims = []
    cursor = 0

    for match in CITATION_TAG.finditer(answer):
        segment = answer[cursor:match.start()]
        cursor  = match.end()

        # A claim never spans a line break — keep only the last line of the
        # segment so a bullet doesn't absorb the lead-in sentence above it.
        claim = segment.rsplit("\n", 1)[-1]
        claim = LIST_MARKER.sub("", claim).strip()

        source = match.group(1).strip()
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
    chunks_for_source = find_source_chunks(source_name, chunks)
    return chunks_for_source[0] if chunks_for_source else None


def find_source_chunks(
    source_name: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    ALL chunks from the cited source, in rank order.

    A citation names a document ("NADRA.txt"), not a passage — and one document
    contributes several chunks to the context (renewal, modification, fees...).
    Scoring a claim only against the highest-RANKED chunk of that document asks
    the wrong question: "does the best-retrieved passage support this?" instead
    of "does the cited document support this?". A renewal-fee claim cited to
    NADRA.txt would be checked against the renewal chunk and flagged, even
    though the fee table chunk — same document, same context — states it.
    """
    source_lower = source_name.lower()
    return [c for c in chunks if source_lower in (c.get("source") or "").lower()]


# ---------------------------------------------------------------------------
# 4. VERIFY ONE CLAIM
# ---------------------------------------------------------------------------
def verify_claim(
    claim: str,
    chunks: dict[str, Any] | list[dict[str, Any]],
    model: CrossEncoder,
) -> dict[str, Any]:
    """
    Scores whether the cited source supports a specific claim.

    We use the CrossEncoder for this. It was trained on passage relevance,
    but high relevance between a claim and a chunk means the chunk supports
    the claim — which is a reasonable proxy for entailment.

    For a production system, you'd use a dedicated NLI (Natural Language Inference)
    model (e.g. facebook/bart-large-mnli) which is trained specifically for
    entailment/contradiction/neutral classification. For our scope, CrossEncoder is sufficient.

    Args:
        chunks: every chunk from the cited source (a single chunk is also
                accepted, for callers that already narrowed it down). The
                claim is scored against each and the BEST score wins — the
                claim only has to be supported SOMEWHERE in the document it cites.

    Returns:
        Dict with: claim, source, score, verified (bool), reason
    """
    candidates = [chunks] if isinstance(chunks, dict) else list(chunks)
    scores = model.predict([(claim, c.get("content") or "") for c in candidates])

    best_idx = int(max(range(len(candidates)), key=lambda i: float(scores[i])))
    score    = float(scores[best_idx])
    verified = score >= CITATION_MIN_SCORE

    return {
        "claim":    claim,
        "source":   candidates[best_idx]["source"],
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
            - verification_rate:  float (0.0 to 1.0) — % of citations verified,
                                  or None when nothing could be checked
            - passed:             bool — True if all citations verified
            - status:             "verified" | "flagged" | "unverifiable" | "not_applicable"
            - summary:            human-readable summary string

    The LangGraph Verification Agent node will check `passed`.
    If False, it can trigger answer regeneration with a stricter prompt.

    WHY `status` EXISTS SEPARATELY FROM `passed`:
        "every citation checked out" and "there was nothing to check" are very
        different facts, and collapsing both into passed=True is how the UI
        ended up stamping "✓ Citations verified" on an unchecked answer.
        `passed` drives the retry decision; `status` is what the UI must render.
    """
    if not answer:
        return {
            "verified_claims": [], "unverified_claims": [],
            "verification_rate": None, "passed": True,
            "status": "not_applicable",
            "summary": "No answer to verify."
        }

    # No context chunks means there is nothing to verify against — this is the
    # out_of_scope path (a polite decline), not a citation failure.
    if not chunks:
        return {
            "verified_claims": [], "unverified_claims": [],
            "verification_rate": None, "passed": True,
            "status": "not_applicable",
            "summary": "No retrieved context — nothing to verify against."
        }

    model = get_cross_encoder()

    # Step 1: Extract all (claim, source) pairs from the answer
    cited_claims = parse_citations(answer)

    if not cited_claims:
        # The answer cites nothing, but it WAS built from retrieved context —
        # so this is a real failure of the citation contract, not a free pass.
        # passed=False lets route_after_verification regenerate it once.
        return {
            "verified_claims": [], "unverified_claims": [],
            "verification_rate": None, "passed": False,
            "status": "unverifiable",
            "summary": "No citations found in answer — nothing could be verified."
        }

    # Step 2: Verify each cited claim
    verified   = []
    unverified = []

    for cited in cited_claims:
        # Every chunk the cited document contributed to this context
        source_chunks = find_source_chunks(cited["source"], chunks)

        if not source_chunks:
            # LLM cited a source that doesn't exist in our context — hallucinated citation
            unverified.append({
                "claim":    cited["claim"],
                "source":   cited["source"],
                "score":    0.0,
                "verified": False,
                "reason":   "cited source not found in context chunks",
            })
            continue

        # Score the claim against the cited document's chunks (best match wins)
        result = verify_claim(cited["claim"], source_chunks, model)

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
        "status":            "verified" if passed else "flagged",
        "checked_count":     total,
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
