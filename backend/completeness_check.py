"""
completeness_check.py
---------------------
PURPOSE: Catch answers that are faithful but useless — the ones that quote a
         two-word requirement and drop the procedure, fee or timeline sitting
         right beside it in the same retrieved passage.

Pipeline position:
    ...
    → generator.py
    → citation_verifier.py    (are the cited claims supported?)
    → hallucination_eval.py   (is every claim supported by something?)
    → completeness_check.py   ✅ YOU ARE HERE  (did we USE what we retrieved?)
    → final_answer

WHY THIS EXISTS:
    Both existing verifiers only ask "is what the answer says true?". Neither
    asks "is what the answer says enough?". A real reported failure:

        Q: "What documents do I need to renew my CNIC?"
        A: "- CNIC number"

    That is a verbatim, fully-grounded, correctly-cited quote of NADRA's
    published Requirements line — and it passed every check the pipeline had.
    It is still a bad answer: the very same retrieved chunk lists the five
    steps at the centre, including the biometric capture that forces the
    citizen to attend in person. Nothing in the pipeline noticed we retrieved
    that and threw it away.

WHY NOT JUST PROMPT THE MODEL:
    Tried first, and measured: with an explicit "do not stop at one item"
    instruction the model included the steps in 1-2 runs out of 3 at
    temperature 0.1. A rule the model follows a third of the time is not a
    fix. So the prompt keeps the instruction, and this module verifies the
    outcome and feeds a concrete, quoted retry when the instruction was
    ignored — deterministic detection, model-independent.

DELIBERATELY NARROW:
    Only thin answers are checked (MAX_THIN_CLAIMS), and only the top few
    chunks are examined. A normal multi-item answer never triggers a retry,
    so this costs nothing on the common path.
"""

import os
import re
from typing import Any

from compressor import split_blocks, LIST_LINE

# An answer with more than this many claim lines is not "thin" — leave it alone.
#
# Kept at 2 deliberately. At 3, "What is the fee for a new CNIC?" — correctly
# and completely answered with normal/urgent/executive rows — was judged thin,
# and the retry bolted the requirements list and the whole centre procedure
# onto a pure fee question. This check exists to rescue one-line answers, not
# to maximise answer length.
MAX_THIN_CLAIMS = int(os.getenv("MAX_THIN_CLAIMS", "2"))

# How many top-ranked chunks to examine for unused material.
CHECK_TOP_CHUNKS = int(os.getenv("CHECK_TOP_CHUNKS", "2"))

# A block must hold at least this many list items to be worth reporting —
# one stray bullet is not a missing procedure.
MIN_BLOCK_ITEMS = 2

# Fraction of a block's items that must appear in the answer for the block
# to count as used.
BLOCK_USED_RATIO = 0.34

# Words too common to prove that an item was actually used.
STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "your", "you", "are",
    "any", "all", "not", "was", "will", "must", "may", "can", "per", "his",
    "her", "their", "its", "upon", "into", "onto", "out", "off", "over",
    "required", "requirement", "requirements", "document", "documents",
    "applicant", "application", "apply", "case", "centre", "center", "office",
}


def _content_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9][a-z0-9\-/]{3,}", text.lower())
        if w not in STOPWORDS
    }


def _strip_citations(text: str) -> str:
    return re.sub(r'\[\s*source\s*:[^\]]*\]', '', text or '', flags=re.IGNORECASE)


def count_claim_lines(answer: str) -> int:
    """Non-empty answer lines that assert something (lead-ins excluded)."""
    lines = []
    for raw in _strip_citations(answer).split("\n"):
        line = raw.strip()
        if not line or line.endswith(":"):
            continue
        lines.append(line)
    return len(lines)


def _block_items(block: str) -> list[str]:
    """The list items in a block (bullets or numbered steps)."""
    return [m.group(1).strip() for line in block.split("\n")
            if (m := LIST_LINE_CAPTURE.match(line))]


# LIST_LINE (imported) only tests a line; this variant captures its text.
LIST_LINE_CAPTURE = re.compile(r'^\s*(?:[-*•]|\d+[.)])\s+(.*\S)\s*$')


def _item_is_covered(item: str, answer_words: set[str]) -> bool:
    item_words = _content_words(item)
    if not item_words:
        return True          # nothing distinctive to look for — don't report it
    hits = len(item_words & answer_words)
    return hits / len(item_words) >= 0.4


def find_unused_blocks(
    answer: str,
    chunks: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """
    List blocks in the top chunks that the answer makes no use of.

    Returns a list of {"source": ..., "text": ...}, best chunk first, so the
    caller can quote them back to the model verbatim on a retry.
    """
    answer_words = _content_words(_strip_citations(answer))
    unused: list[dict[str, str]] = []

    for chunk in chunks[:CHECK_TOP_CHUNKS]:
        for block in split_blocks(chunk.get("content") or ""):
            items = _block_items(block)
            if len(items) < MIN_BLOCK_ITEMS:
                continue

            covered = sum(1 for it in items if _item_is_covered(it, answer_words))
            if covered / len(items) < BLOCK_USED_RATIO:
                unused.append({
                    "source": chunk.get("source") or "unknown",
                    "text":   block,
                })

    return unused


def check_completeness(
    answer: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Decides whether a thin answer left usable retrieved material on the floor.

    Returns:
        passed:        False only when the answer is thin AND relevant blocks
                       were retrieved but unused
        claim_lines:   how many assertions the answer made
        unused_blocks: the blocks to quote back on a retry
        status:        "complete" | "thin" | "not_applicable"
        summary:       human-readable
    """
    if not answer or not chunks:
        return {"passed": True, "claim_lines": 0, "unused_blocks": [],
                "status": "not_applicable",
                "summary": "No answer or no context — nothing to check."}

    claim_lines = count_claim_lines(answer)

    if claim_lines > MAX_THIN_CLAIMS:
        return {"passed": True, "claim_lines": claim_lines, "unused_blocks": [],
                "status": "complete",
                "summary": f"Answer states {claim_lines} claims — not thin."}

    unused = find_unused_blocks(answer, chunks)

    if not unused:
        return {"passed": True, "claim_lines": claim_lines, "unused_blocks": [],
                "status": "complete",
                "summary": f"Answer is short ({claim_lines} claims) but the retrieved context held nothing more."}

    return {
        "passed":        False,
        "claim_lines":   claim_lines,
        "unused_blocks": unused,
        "status":        "thin",
        "summary": (
            f"Answer states only {claim_lines} claim(s) while {len(unused)} "
            f"retrieved block(s) went unused: "
            + ", ".join(b["text"].split("\n")[0][:45] for b in unused)
        ),
    }


def build_retry_feedback(unused_blocks: list[dict[str, str]]) -> str:
    """
    The revision instruction handed back to the generator.

    The omitted passages are quoted verbatim: a generic "be more complete"
    retry reproduces the same short answer (measured), whereas naming the
    exact text to incorporate leaves nothing to interpret.
    """
    quoted = "\n\n".join(
        f"[from {b['source']}]\n{b['text']}" for b in unused_blocks
    )
    return (
        "YOUR PREVIOUS ANSWER WAS TOO SHORT AND IS BEING REGENERATED.\n"
        "It quoted only a bare requirement and ignored the passages below, "
        "which were retrieved for this same question and describe what the "
        "citizen actually has to do.\n\n"
        f"{quoted}\n\n"
        "Write the answer again. Keep what you had, and add this material as "
        "its own labelled list (for example \"At the centre:\" or \"Fee:\"), "
        "one item per line, each with its [source: ...] tag. Add nothing that "
        "is not in the passages."
    )


# ---------------------------------------------------------------------------
# SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    chunk = {
        "source": "NADRA.txt",
        "content": (
            "## CNIC RENEWAL\n"
            "Requirements:\n"
            "- CNIC number\n"
            "Process of applying through NADRA Registration Centre:\n"
            "1. Token Acquisition - Obtain a Queue Matic token upon arrival.\n"
            "2. Biometric Capture - Digital capture of live photograph and fingerprints.\n"
            "3. OIC Interview / Approval - A brief interview by the Officer-In-Charge.\n"
            "4. Payment - Submission of the required processing fee.\n"
            "5. Card Printing - Card is printed.\n"
        ),
    }

    thin = "- CNIC number [source: NADRA.txt]"
    full = (
        "- CNIC number [source: NADRA.txt]\n"
        "At the centre:\n"
        "- Obtain a Queue Matic token upon arrival [source: NADRA.txt]\n"
        "- Biometric capture of live photograph and fingerprints [source: NADRA.txt]\n"
        "- Brief interview with the Officer-In-Charge [source: NADRA.txt]\n"
        "- Payment of the processing fee [source: NADRA.txt]\n"
        "- Card printing [source: NADRA.txt]"
    )

    for label, ans in (("THIN", thin), ("FULL", full)):
        result = check_completeness(ans, [chunk])
        print(f"\n{label}: passed={result['passed']} status={result['status']}")
        print(f"  {result['summary']}")
