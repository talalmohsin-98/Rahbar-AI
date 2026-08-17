"""
hallucination_eval.py
---------------------
PURPOSE: After generation, evaluate the FULL answer sentence by sentence
         against ALL available chunks — catching hallucinations that
         citation_verifier.py doesn't cover (uncited sentences).

Pipeline position:
    ...
    → generator.py          (answer produced)
    → citation_verifier.py  (cited claims checked)
    → hallucination_eval.py ✅ YOU ARE HERE
    → final_answer

DIFFERENCE FROM citation_verifier.py:
    citation_verifier.py:   only checks sentences that HAVE a [source: ...] tag
    hallucination_eval.py:  checks EVERY sentence in the answer, cited or not

The failure mode this catches:
    Answer: "NADRA was established in 1998. [source: nadra.pdf]
             The authority employs 12,000 staff members."   ← no citation, could be hallucinated

    citation_verifier:  checks the first sentence ✓ (if score is ok)
    hallucination_eval: also checks "The authority employs 12,000 staff members"
                        → if no chunk supports it, flag it.

APPROACH:
    For each sentence in the answer:
        1. Score it against ALL compressed chunks using CrossEncoder
        2. Take the MAX score (the best-matching chunk)
        3. If max score < threshold → this sentence is not grounded in context
        4. Flag it as a potential hallucination

    We use the CrossEncoder again — same model, third context:
    high score = the sentence is entailed by / consistent with a chunk
    low score  = the sentence has no support in the context
"""

import os
import re
from typing import Any
from reranker import get_cross_encoder
from compressor import split_sentences

# ---------------------------------------------------------------------------
# 1. CONSTANTS
# ---------------------------------------------------------------------------
# Minimum CrossEncoder score for a claim to be considered grounded.
# A claim with no chunk scoring above this is flagged as potential hallucination.
#
# Shares the calibration measured for CITATION_MIN_SCORE (see citation_verifier.py
# for the numbers): the same model scores the same kind of short claim against
# the same chunks, so the same floor applies. The old 0.5 was set for
# full-sentence claims and flags ordinary paraphrases like "A brief interview
# with the Officer-In-Charge" (real score -2.62) as hallucinations.
HALLUCINATION_THRESHOLD = float(os.getenv("HALLUCINATION_THRESHOLD", "-3.0"))

# Claims shorter than this are skipped (transition phrases, connectives).
# Kept deliberately low: a real requirement line can be as short as
# "CNIC number" (11 chars), and the old 15-char floor silently discarded
# exactly those — leaving list answers with zero evaluable claims and an
# undeserved "0% hallucination" score.
MIN_CLAIM_LENGTH = int(os.getenv("MIN_CLAIM_LENGTH", "8"))

# Leading list marker on a bullet or numbered line: "- ", "* ", "• ", "1. ", "2) "
LIST_MARKER = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s*')


# ---------------------------------------------------------------------------
# 2. STRIP CITATION TAGS
# ---------------------------------------------------------------------------
def strip_citation_tags(text: str) -> str:
    """
    Removes [source: ...] tags from answer text before sentence-level evaluation.

    We evaluate the claim content, not the citation metadata.
    "NADRA charges Rs. 750. [source: nadra.pdf]" → "NADRA charges Rs. 750."
    """
    return re.sub(r'\[source:[^\]]+\]', '', text).strip()


# ---------------------------------------------------------------------------
# 2b. SPLIT AN ANSWER INTO CHECKABLE CLAIMS
# ---------------------------------------------------------------------------
def split_claims(text: str) -> list[str]:
    """
    Splits an answer into the units we actually score for grounding.

    Why not reuse compressor.split_sentences() directly?
        That splitter is built for prose: it breaks on ".!?" followed by a
        capital letter and drops anything under 20 characters. Run it over
        the list-formatted answers this system is prompted to produce:

            To renew your CNIC you need to provide:
            - CNIC number

        ...and the whole answer comes back as ONE unit. It then scores 4.87
        against the chunk it was copied from and the UI reports "0%
        hallucination" — a 0 out of a sample of 1, measured on a blob.
        Each list item is a separate factual claim and has to be scored as one.

    Rules:
        - Every line is its own claim boundary (that's what makes bullets work)
        - List markers are stripped so "- CNIC number" scores as "CNIC number"
        - Prose lines are further split into sentences
        - Lead-in lines ending in ':' are structural, not claims — skipped
        - Units with no letters or digits (separators, stray punctuation) are skipped
    """
    claims: list[str] = []

    for raw_line in (text or "").split("\n"):
        line = LIST_MARKER.sub("", raw_line).strip()
        if not line:
            continue

        # "You will need the following documents:" states nothing checkable.
        if line.endswith(":"):
            continue

        # A bullet is one claim; a prose line may hold several sentences.
        units = split_sentences(line) or [line]

        for unit in units:
            unit = unit.strip()
            if len(unit) < MIN_CLAIM_LENGTH:
                continue
            if not re.search(r'[A-Za-z0-9]', unit):
                continue
            claims.append(unit)

    return claims


# ---------------------------------------------------------------------------
# 3. EVALUATE ONE SENTENCE
# ---------------------------------------------------------------------------
def evaluate_sentence(
    sentence: str,
    chunks: list[dict[str, Any]],
    model,
) -> dict[str, Any]:
    """
    Scores one answer sentence against all available chunks.
    Returns the best-matching chunk and score, plus a grounded flag.

    Strategy: score the sentence against every chunk, keep the maximum.
    If the sentence IS grounded, at least one chunk should score high.
    We don't need all chunks to support it — just one is enough.
    """
    if not chunks:
        return {
            "sentence":       sentence,
            "max_score":      0.0,
            "best_source":    None,
            "grounded":       False,
            "reason":         "no context chunks available",
        }

    # Score sentence against every chunk
    pairs  = [(sentence, chunk.get("content") or "") for chunk in chunks]
    scores = model.predict(pairs)

    # Find the best-matching chunk
    best_idx   = int(scores.argmax())
    max_score  = float(scores[best_idx])
    best_chunk = chunks[best_idx]

    grounded = max_score >= HALLUCINATION_THRESHOLD

    return {
        "sentence":    sentence,
        "max_score":   max_score,
        "best_source": best_chunk.get("source", "unknown"),
        "grounded":    grounded,
        "reason":      "supported" if grounded else f"no chunk scores above {HALLUCINATION_THRESHOLD} (max={max_score:.2f})",
    }


# ---------------------------------------------------------------------------
# 4. MAIN EVALUATION FUNCTION
# ---------------------------------------------------------------------------
def evaluate_hallucination(
    answer: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Full hallucination evaluation of the generated answer.

    Args:
        answer:  The LLM-generated answer string.
        chunks:  Compressed chunks that were sent to the LLM as context.

    Returns:
        Dict containing:
            sentence_results:    list — per-sentence grounding evaluation
            grounded_count:      int — sentences with chunk support
            hallucinated_count:  int — sentences without chunk support
            hallucination_rate:  float — % of sentences flagged
            passed:              bool — True if hallucination_rate below threshold
            flagged_sentences:   list — the problematic sentences
            summary:             str — human-readable summary

    What "passed" means:
        We allow up to 20% of sentences to be ungrounded before failing.
        Some sentences are structural/transitional ("In summary...", "Therefore...")
        and won't match any chunk — that's acceptable noise.
        If > 20% of sentences lack chunk support, something is wrong.
    """
    if not answer:
        return {
            "sentence_results": [], "grounded_count": 0,
            "hallucinated_count": 0, "hallucination_rate": None,
            "passed": True, "flagged_sentences": [],
            "evaluated_count": 0, "status": "not_evaluated",
            "summary": "No answer to evaluate."
        }

    # No context means the answer is a decline ("I could not find this
    # information...") or an out-of-scope reply. Scoring it against nothing
    # marks it 100% ungrounded, fails verification, and burns a regeneration
    # to produce the identical decline. There is nothing here to hallucinate.
    if not chunks:
        return {
            "sentence_results": [], "grounded_count": 0,
            "hallucinated_count": 0, "hallucination_rate": None,
            "passed": True, "flagged_sentences": [],
            "evaluated_count": 0, "status": "not_evaluated",
            "summary": "No retrieved context — nothing to evaluate against."
        }

    model = get_cross_encoder()

    # Strip citation tags before splitting (they confuse sentence splitting)
    clean_answer = strip_citation_tags(answer)

    # Split into individually checkable claims (bullets included — see split_claims)
    sentences = split_claims(clean_answer)

    if not sentences:
        # rate stays None, NOT 0.0: "nothing was measured" must never render
        # as "0% hallucination", which is what the old return value caused.
        return {
            "sentence_results": [], "grounded_count": 0,
            "hallucinated_count": 0, "hallucination_rate": None,
            "passed": True, "flagged_sentences": [],
            "evaluated_count": 0, "status": "not_evaluated",
            "summary": "No evaluable claims found in answer."
        }

    # Evaluate each sentence
    results = [evaluate_sentence(s, chunks, model) for s in sentences]

    # Aggregate
    grounded_count     = sum(1 for r in results if r["grounded"])
    hallucinated_count = len(results) - grounded_count
    hallucination_rate = hallucinated_count / len(results)

    # Allow up to 20% ungrounded before failing
    passed = hallucination_rate <= 0.20

    flagged = [r for r in results if not r["grounded"]]

    summary = (
        f"{grounded_count}/{len(results)} sentences grounded. "
        f"Hallucination rate: {hallucination_rate:.0%}. "
        + ("PASSED." if passed else f"FAILED — {hallucinated_count} sentence(s) lack chunk support.")
    )

    return {
        "sentence_results":   results,
        "grounded_count":     grounded_count,
        "hallucinated_count": hallucinated_count,
        "hallucination_rate": hallucination_rate,
        "passed":             passed,
        "flagged_sentences":  flagged,
        "evaluated_count":    len(results),
        "status":             "evaluated",
        "summary":            summary,
    }


# ---------------------------------------------------------------------------
# 5. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mock_chunks = [
        {
            "chunk_id": 1,
            "content": "NADRA was established in 2000 under the Ministry of Interior. It issues CNICs to Pakistani citizens.",
            "source": "nadra_guide.pdf", "rerank_score": 9.0, "rank": 1,
        },
    ]

    # Answer with one grounded sentence and one hallucinated sentence
    answer = (
        "NADRA was established in 2000 under the Ministry of Interior. [source: nadra_guide.pdf] "
        "The authority has over 50,000 employees across Pakistan."   # hallucinated — not in chunk
    )

    print(f"\nAnswer:\n{answer}\n")
    result = evaluate_hallucination(answer, mock_chunks)

    print(f"Summary: {result['summary']}")
    print(f"Passed:  {result['passed']}")
    print(f"\nPer-sentence results:")
    for r in result["sentence_results"]:
        icon = "✓" if r["grounded"] else "✗"
        print(f"  {icon} [{r['max_score']:.2f}] {r['sentence'][:70]}")
        if not r["grounded"]:
            print(f"      → {r['reason']}")
