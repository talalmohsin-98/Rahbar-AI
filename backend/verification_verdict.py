"""
verification_verdict.py
-----------------------
Collapses the three independent verification checks into ONE verdict that the
UI renders.

    citation_verifier.py    → is each cited claim supported by the source it cites?
    hallucination_eval.py   → is every claim supported by something in context?
    completeness_check.py   → did we retrieve material and then fail to use it?
    verification_verdict.py ✅ YOU ARE HERE — what do we tell the citizen?

WHY THIS FILE EXISTS
    Each checker already reports honestly. The bug was downstream of them: the
    answer badges read citation_result and hallucination_result and never read
    completeness_result at all, so an answer the pipeline had FLAGGED still
    rendered as full green "2/2 claims grounded · 2 citations verified".

    Asking "What is Gamma Family FRC?" is what exposed it. Depending on how
    long the generated answer runs, completeness either flags it (`thin`) or
    is skipped entirely (>2 claim lines) — and when it flagged, the badges
    showed green anyway, because they never read that field. Wiring alone
    fixes that half.

    The other half is that on the runs where completeness is skipped, NO check
    objects at all, and the answer still closes with an invented sentence about
    "the Gamma family". That is what check_term_attestation below exists for.

    The deeper problem was that "overall status" had no owner. PipelinePanel
    recomputed it inline from the three results; the badges used their own
    per-check logic. Two surfaces, two rules, guaranteed to drift. This module
    is the single owner: the graph writes its output to state once, and every
    consumer renders that field instead of re-deriving a verdict.

`passed` VS `status`, AGAIN
    citation_verifier.py already draws this distinction and it matters here
    too: `passed` drives the retry decision inside the graph, `status` records
    whether a check could run at all. A check that could not run is not a check
    that succeeded, so "nothing was verifiable" must never collapse into the
    same verdict as "everything verified".
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# TERM ATTESTATION — the fourth check
# ---------------------------------------------------------------------------
# The three existing checks all interrogate the ANSWER. None of them asks the
# prior question: does the retrieved context mention the thing that was asked
# about at all?
#
# "What is Gamma Family FRC?" is the case that exposed the gap. The corpus
# lists `FRC (Alpha / Beta / Gama family)` in one fee-table row and never
# defines the categories. The generator answered with two true, correctly
# cited sentences about FRCs generally and then closed with:
#
#     "Therefore, a 'Gamma Family FRC' would be the FRC document specifically
#      reflecting the family composition of the Gamma family."
#
# It invented a surname. Every existing check passed it: the citation verifier
# only inspects cited sentences and that one carries no citation; the
# hallucination evaluator scored it as supported because it is a paraphrase of
# the FRC definition sitting right there in context; and completeness never
# ran, because at 3 claim lines the answer is not "thin" (MAX_THIN_CLAIMS=2).
#
# The signal all three miss is trivially available: the word "Gamma" does not
# occur anywhere in what we retrieved. An answer cannot be about a term the
# sources never mention. This check is deterministic, costs no LLM call, and
# generalises to any question built on a term the corpus does not attest.
#
# It is a heuristic and it is scoped like one: it downgrades the verdict to
# "partial", never suppresses the answer, and it only considers reasonably
# distinctive words so ordinary phrasing does not trip it.

_MIN_TERM_LEN = 4
# Was 5, which left 4-character terms with exact-substring matching only and no
# morphological tolerance. Lowering it to 4 makes the check slightly more
# willing to accept a term as attested — the safe direction: a missed flag
# costs one uncaught weak answer, a false flag tells a citizen we did not
# understand their question. "gamma" still does not match "gama" at any prefix
# length, which is the case that matters.
_PREFIX_LEN   = 4

# Question scaffolding and service-generic vocabulary. A word here is never
# treated as a distinctive term, because its absence from context says nothing
# about whether the question was answerable.
_GENERIC_TERMS = {
    # question scaffolding
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "does", "do", "did", "is", "are", "was", "were", "can", "could", "should",
    "would", "will", "shall", "may", "might", "must", "have", "has", "had",
    "the", "and", "for", "with", "from", "that", "this", "these", "those",
    "there", "their", "they", "them", "you", "your", "yours", "mine", "our",
    "about", "into", "onto", "than", "then", "also", "any", "all", "some",
    "much", "many", "more", "most", "less", "need", "needs", "needed",
    "want", "wants", "get", "gets", "getting", "give", "take", "make",
    "please", "tell", "explain", "know", "like", "just", "only", "still",
    # generic service vocabulary — present in every domain, proves nothing
    "apply", "application", "applying", "process", "procedure", "step",
    "steps", "document", "documents", "requirement", "requirements",
    "required", "fee", "fees", "cost", "costs", "charge", "charges", "price",
    "time", "days", "day", "week", "weeks", "month", "months", "year",
    "years", "office", "offices", "online", "portal", "form", "forms",
    "card", "certificate", "number", "status", "renew", "renewal", "new",
    "old", "valid", "validity", "eligible", "eligibility", "information",
    # Definitional vocabulary. QA caught this the hard way: "What does
    # Decline on Processed application MEAN...?" reported '"mean" never
    # appears in the sources' — gibberish to a citizen, and it hijacked the
    # headline on an answer that had correctly said the corpus does not define
    # the term. These are precisely the words people use to ask about an
    # undefined term, i.e. this check's own target scenario.
    "mean", "means", "meaning", "meant", "define", "defines", "defined",
    "definition", "term", "terms", "called", "call", "refer", "refers",
    "stand", "stands", "difference", "between", "example", "examples",
}


# ---------------------------------------------------------------------------
# REFUSAL DETECTION
# ---------------------------------------------------------------------------
# The generator is instructed to answer "I could not find this information in
# the available documents." when the context genuinely does not cover the
# question. That is the system working correctly — the explicit decline path
# this project is built around.
#
# Every checker nonetheless treated it as a claim. QA hit this on "What is the
# Smart NICOP fee for Zone C countries?" (NADRA.txt defines only Zones A and
# B): the hallucination evaluator scored the refusal sentence as unsupported,
# so the citizen was shown an AMBER badge reading "1 statement in this answer
# had no support anywhere in the retrieved sources" — about a sentence whose
# entire content is that there is no support. Worse, `passed: False` sent it
# back through the retry loop to regenerate a refusal that was already right.
#
# A refusal makes no claims, so there is nothing to verify: every check is
# not-applicable and the honest verdict is "unverified" (neutral grey), not a
# warning. This mirrors the reasoning that keeps attestation out of the retry
# gate — a second generation cannot conjure a Zone C the corpus never had.
_REFUSAL_PATTERNS = (
    r"could\s*not\s+find",
    r"couldn'?t\s+find",
    r"do(?:es)?\s+not\s+contain",
    r"no\s+information\s+(?:is\s+)?available",
    r"not\s+available\s+in\s+the\s+(?:provided|available|retrieved)\s+document",
)


def is_refusal(answer: str | None) -> bool:
    """
    True when the answer is ONLY a decline, not an answer that happens to
    mention a gap. The length bound matters: an answer that gives three real
    bullets and then notes one detail is missing is still an answer, and its
    claims must still be checked.
    """
    raw = answer or ""

    # A refusal asserts nothing, so it cites nothing. This is the discriminator
    # that matters: an answer giving three cited bullets and then noting one
    # missing detail matches "do not contain" too, and it is emphatically NOT a
    # refusal - its claims still need checking. Length alone conflates the two,
    # which my first attempt at this function did.
    if re.search(r"\[\s*source\s*:", raw, flags=re.IGNORECASE):
        return False

    text = " ".join(raw.split()).strip()
    if not text or len(text) > 300:
        return False
    if len([ln for ln in raw.splitlines() if ln.strip()]) > 2:
        return False

    lowered = text.lower()
    return any(re.search(p, lowered) for p in _REFUSAL_PATTERNS)


def refusal_results() -> dict[str, dict]:
    """The four check results to record for a refusal: nothing was checkable."""
    return {
        "citation_result": {
            "status": "not_applicable", "passed": True, "checked_count": 0,
            "verification_rate": None, "verified_claims": [], "unverified_claims": [],
            "summary": "Answer is an explicit refusal — no claims to verify.",
        },
        "hallucination_result": {
            "status": "not_evaluated", "passed": True, "grounded_count": 0,
            "hallucinated_count": 0, "evaluated_count": 0, "hallucination_rate": None,
            "sentence_results": [], "flagged_sentences": [],
            "summary": "Answer is an explicit refusal — no claims to ground.",
        },
        "completeness_result": {
            "status": "not_applicable", "passed": True, "claim_lines": 0,
            "unused_blocks": [],
            "summary": "Answer is an explicit refusal — nothing was asserted.",
        },
        "attestation_result": {
            "status": "not_applicable", "passed": True, "missing": [], "salient": [],
            "summary": "Answer is an explicit refusal — no claim to attest.",
        },
    }


def _context_words(chunks: list[dict] | None) -> tuple[str, set[str]]:
    text = " ".join((c or {}).get("content", "") for c in (chunks or [])).lower()
    return text, set(re.findall(r"[a-z0-9]+", text))


def _attested(term: str, context_text: str, context_words: set[str]) -> bool:
    t = term.lower()
    if t in context_text:
        return True
    # Prefix match absorbs ordinary morphology — licence/license,
    # register/registration, document/documents — without pulling in a
    # fuzzy-matching dependency. Note it deliberately does NOT rescue
    # "gamma" from "gama": those are different strings, and treating them as
    # the same is precisely the guess this check exists to refuse.
    if len(t) >= _PREFIX_LEN:
        prefix = t[:_PREFIX_LEN]
        return any(w.startswith(prefix) for w in context_words)
    return False


def check_term_attestation(question: str, chunks: list[dict] | None) -> dict[str, Any]:
    """
    Does the retrieved context mention the distinctive terms of the question?

    Returns the same shape as the other checkers:
        passed:  False when a distinctive term appears nowhere in context
        status:  "attested" | "missing" | "not_applicable"
        missing: the unattested terms, in question order
    """
    context_text, context_words = _context_words(chunks)
    if not context_text:
        return {"passed": True, "status": "not_applicable", "missing": [],
                "summary": "No retrieved context — nothing to attest against."}

    seen: set[str] = set()
    terms: list[str] = []
    all_words = re.findall(r"[A-Za-z][A-Za-z0-9-]*", question or "")
    # The FIRST WORD OF THE QUESTION, not the first surviving term. Sentence-
    # initial capitalisation carries no information, but "What is Gamma..."
    # filters "What" out as generic, which would leave "Gamma" looking
    # sentence-initial and cost it its salience.
    first_word = all_words[0] if all_words else None
    for raw in all_words:
        low = raw.lower()
        if (len(low) >= _MIN_TERM_LEN and low not in _GENERIC_TERMS and low not in seen):
            seen.add(low)
            terms.append(raw)

    if not terms:
        return {"passed": True, "status": "not_applicable", "missing": [],
                "summary": "No distinctive terms in the question."}

    missing = [t for t in terms if not _attested(t, context_text, context_words)]

    # Which missing terms look like the SUBJECT of the question rather than
    # part of its grammar? A capitalised word mid-sentence, or an acronym, is
    # the shape of a named category ("Gamma", "NICOP"). The distinction does
    # not change whether we flag — any unattested distinctive term is still
    # worth surfacing — but it decides whether attestation gets to claim the
    # alarming headline. One unanticipated lowercase verb should never promote
    # an answer to "the sources never mention what you asked about".
    salient = [t for t in missing
               if t != first_word and (t[:1].isupper() or t.isupper())]

    if missing:
        return {
            "passed": False, "status": "missing", "missing": missing,
            "salient": salient,
            "summary": (
                f"{len(missing)} of {len(terms)} distinctive term(s) never appear "
                f"in the retrieved sources: {', '.join(missing)}."
            ),
        }
    return {"passed": True, "status": "attested", "missing": [], "salient": [],
            "summary": f"All {len(terms)} distinctive term(s) appear in the retrieved sources."}

# Statuses that mean the check actually evaluated something. Anything else
# ("unverifiable", "not_applicable", "not_evaluated") means the check had
# nothing to work with — which is not the same as passing.
_ATTESTATION_RAN  = {"attested", "missing"}
_CITATION_RAN     = {"verified", "flagged"}
_GROUNDING_RAN    = {"evaluated"}
_COMPLETENESS_RAN = {"complete", "thin"}


def _check(result: dict | None, ran_statuses: set[str]) -> dict[str, Any]:
    r = result or {}
    status = r.get("status")
    return {
        "status": status,
        "ran":    status in ran_statuses,
        # `passed` is None when the check never ran, so a consumer cannot
        # mistake "not checked" for "checked and fine".
        "passed": r.get("passed") if status in ran_statuses else None,
    }


def build_verdict(
    citation_result:      dict | None,
    hallucination_result: dict | None,
    completeness_result:  dict | None,
    input_analysis:       dict | None = None,
    attestation_result:   dict | None = None,
    refused:              bool = False,
) -> dict[str, Any]:
    """
    Returns the one verdict the UI renders.

        status:   "verified" | "partial" | "unverified"
        headline: the sentence shown next to the answer
        reasons:  why it is not "verified" — empty when it is
        checks:   per-check {status, ran, passed}, for the inspector panel

    "partial" is deliberately not "failed". The answer is still shown, because
    a partially grounded answer about FRCs is more useful to a citizen than
    nothing — it just must not wear a green badge that says we confirmed it
    answers their question when we did not.
    """
    cit = _check(citation_result,      _CITATION_RAN)
    hal = _check(hallucination_result, _GROUNDING_RAN)
    com = _check(completeness_result,  _COMPLETENESS_RAN)
    att = _check(attestation_result,   _ATTESTATION_RAN)

    reasons: list[str] = []

    if att["passed"] is False:
        missing = (attestation_result or {}).get("missing") or []
        quoted = ", ".join(f'"{m}"' for m in missing[:3])
        reasons.append(
            f"the sources found never mention {quoted} — this answer may be "
            f"about something related rather than what you asked"
        )

    if cit["passed"] is False:
        n = len((citation_result or {}).get("unverified_claims") or [])
        reasons.append(
            f"{n} cited claim{'' if n == 1 else 's'} could not be traced back to "
            f"the source {'it cites' if n == 1 else 'they cite'}"
        )

    if hal["passed"] is False:
        n = (hallucination_result or {}).get("hallucinated_count", 0)
        reasons.append(
            f"{n} statement{'' if n == 1 else 's'} in this answer had no support "
            f"anywhere in the retrieved sources"
        )

    if com["passed"] is False:
        # The Gamma Family FRC shape: we found related material, used little of
        # it, and cannot show the answer addresses what was actually asked.
        reasons.append(
            "the sources found are related but may not cover your exact "
            "question — relevant retrieved material went unused"
        )

    # Code-switched input degrades the embedding badly enough that a confident
    # verdict is not defensible even when every downstream check passes. See
    # input_analysis.py.
    analysis = input_analysis or {}
    if analysis.get("code_switched"):
        reasons.append(
            "your question mixes Urdu and English, which weakens document "
            "search — the sources found may not be the best match"
        )

    if refused:
        # Neutral, never a warning. The system declined on purpose and said so;
        # dressing that up as a verification problem punishes the one behaviour
        # this pipeline most wants to encourage. Any input caveat still rides
        # along, because "your question mixed languages" is often WHY nothing
        # was found.
        return {
            "status":   "unverified",
            "headline": "No answer found — the sources do not cover this question",
            "reasons":  reasons,
            "checks":   {"citations": cit, "grounding": hal, "completeness": com,
                         "attestation": att},
        }

    if reasons:
        status = "partial"
        # Lead with the failure the citizen can act on. A coverage gap means
        # "ask more specifically"; an unsupported claim means "do not rely on
        # this line". Those warrant different sentences.
        if att["passed"] is False and (attestation_result or {}).get("salient"):
            headline = "Unconfirmed — the sources never mention what you asked about"
        elif com["passed"] is False and cit["passed"] is not False and hal["passed"] is not False:
            headline = "Partially grounded — sources may not cover your exact question"
        elif analysis.get("code_switched") and len(reasons) == 1:
            headline = "Lower confidence — mixed-language question weakens search"
        else:
            headline = "Partially grounded — some claims could not be verified"

    elif not (cit["ran"] or hal["ran"]):
        # Nothing was checkable. Neutral, never green.
        status = "unverified"
        headline = "Not verified — no claim in this answer could be checked"

    else:
        status = "verified"
        headline = "Grounded — every checked claim traced to a source"

    return {
        "status":   status,
        "headline": headline,
        "reasons":  reasons,
        "checks":   {"citations": cit, "grounding": hal, "completeness": com,
                     "attestation": att},
    }


if __name__ == "__main__":
    # The Gamma Family FRC case: citations and grounding pass, completeness
    # does not. This must NOT come back "verified".
    v = build_verdict(
        {"status": "verified",  "passed": True,  "checked_count": 2},
        {"status": "evaluated", "passed": True,  "grounded_count": 2, "evaluated_count": 2},
        {"status": "thin",      "passed": False, "unused_blocks": [{"source": "NADRA.txt"}]},
    )
    print(v["status"], "|", v["headline"])
    assert v["status"] == "partial", v

    # A clean answer stays green.
    v = build_verdict(
        {"status": "verified",  "passed": True,  "checked_count": 3},
        {"status": "evaluated", "passed": True,  "grounded_count": 4, "evaluated_count": 4},
        {"status": "complete",  "passed": True,  "unused_blocks": []},
    )
    print(v["status"], "|", v["headline"])
    assert v["status"] == "verified", v

    # Nothing checkable is neutral, not green.
    v = build_verdict(
        {"status": "not_applicable", "passed": True},
        {"status": "not_evaluated",  "passed": True},
        {"status": "not_applicable", "passed": True},
    )
    print(v["status"], "|", v["headline"])
    assert v["status"] == "unverified", v

    # Term attestation: the Gamma Family FRC case as it actually behaved.
    # All three original checks pass; only attestation catches it.
    chunks = [{"content": "Family Registration Certificate means a certificate "
                          "issued by the Authority which reflects the family "
                          "composition data of a registered person."}]
    att = check_term_attestation("What is Gamma Family FRC?", chunks)
    print("attestation:", att["summary"])
    assert att["passed"] is False and att["missing"] == ["Gamma"], att

    v = build_verdict(
        {"status": "verified",  "passed": True,  "checked_count": 2},
        {"status": "evaluated", "passed": True,  "grounded_count": 3, "evaluated_count": 3},
        {"status": "complete",  "passed": True,  "unused_blocks": []},
        None, att,
    )
    print(v["status"], "|", v["headline"])
    assert v["status"] == "partial", v

    # And it must not fire on an ordinary well-covered question.
    good_chunks = [{"content": "Requirements for CNIC registration: original "
                               "B-Form or birth certificate issued by the Union "
                               "Council, two photographs, biometric verification."}]
    att_ok = check_term_attestation(
        "What documents do I need for CNIC registration?", good_chunks)
    print("attestation:", att_ok["summary"])
    assert att_ok["passed"] is True, att_ok

    # QA regression 1: an ordinary verb must not be reported as a missing
    # subject, and must not own the headline.
    att_verb = check_term_attestation(
        "What does Decline on Processed application mean for a succession certificate?",
        [{"content": "Fee for a Decline on Processed application: Rs. 15,000/-."}])
    print("attestation:", att_verb["summary"])
    assert "mean" not in att_verb["missing"], att_verb

    # Even if some unanticipated lowercase word slips through, it may flag but
    # must not promote the headline to the alarming variant.
    att_lower = {"status": "missing", "passed": False,
                 "missing": ["widget"], "salient": []}
    v = build_verdict(
        {"status": "verified",  "passed": True,  "checked_count": 1},
        {"status": "evaluated", "passed": True,  "grounded_count": 1, "evaluated_count": 1},
        {"status": "complete",  "passed": True}, None, att_lower)
    print(v["status"], "|", v["headline"])
    assert v["status"] == "partial" and "never mention what you asked" not in v["headline"], v

    # QA regression 2: an explicit refusal is neutral, not a warning.
    assert is_refusal("I could not find this information in the available documents.")
    assert not is_refusal(
        "- CNIC fee is Rs. 750 [source: NADRA.txt]\n"
        "- Urgent is Rs. 1,500 [source: NADRA.txt]\n"
        "- The documents do not contain the executive-category fee.")
    r = refusal_results()
    v = build_verdict(r["citation_result"], r["hallucination_result"],
                      r["completeness_result"], None, r["attestation_result"],
                      refused=True)
    print(v["status"], "|", v["headline"])
    assert v["status"] == "unverified" and not v["reasons"], v

    print("verification_verdict self-checks passed")
