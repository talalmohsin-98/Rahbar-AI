"""
input_analysis.py
-----------------
Cheap, deterministic inspection of the question BEFORE it reaches the
embedding model. No LLM call, no network — this runs on every query.

It answers two questions the retrieval stack was silently getting wrong:

1. IS THIS CODE-SWITCHED URDU-ENGLISH?
   The embedder is `BAAI/bge-large-en` — an English-only model. A question
   like "CNIC renewal ka process kya hai?" embeds as mostly noise: the Urdu
   tokens carry no useful signal and actively drag the vector away from the
   English chunks that hold the answer. Retrieval degrades badly and nothing
   downstream notices, because every later stage just trusts the chunks it
   was handed.

   Two forms have to be caught, and only one is obvious:
     - script-mixed: actual Urdu script alongside Latin ("CNIC کا طریقہ")
     - romanized:    Urdu written in Latin letters ("CNIC ka tareeqa kya hai")
   The romanized form is what Pakistani users actually type, and no script
   check will ever see it, so it needs a function-word heuristic instead.

2. DOES IT SPAN MULTIPLE SERVICE DOMAINS?
   "Do I need a CNIC before applying for a passport?" is two retrievals
   wearing a trench coat. A single vector lands between the NADRA and
   Passport regions of the space and retrieves the best of neither. The
   rewriter already decomposes questions into sub-questions, but it does so
   with an LLM that has no idea these five domains exist, so nothing
   guarantees a sub-question per domain. Naming the domains here lets
   query_rewriter.py add one deterministic query per domain on top.

Both flags are advisory. This module never blocks a question — it annotates
it, and the rewriter and the verdict decide what to do about it.
"""

import re

# ---------------------------------------------------------------------------
# SCRIPT DETECTION
# ---------------------------------------------------------------------------
# Arabic block (Urdu is written in Nastaliq using Arabic-derived codepoints),
# plus the Arabic Supplement / Presentation Forms ranges Urdu text commonly
# lands in after normalisation.
_URDU_SCRIPT = re.compile('[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]')
_LATIN       = re.compile(r'[A-Za-z]')

# ---------------------------------------------------------------------------
# ROMANIZED URDU DETECTION
# ---------------------------------------------------------------------------
# Function words, not content words. Content words ("passport", "CNIC") are
# identical in both languages here and prove nothing; grammar words are what
# actually mark the sentence as Urdu. Matched on word boundaries so "ka" does
# not fire inside "Karachi", and two DISTINCT markers are required so a stray
# token in an otherwise English sentence stays English.
_ROMAN_URDU_MARKERS = {
    "kya", "kia", "kaise", "kaisay", "kese", "kitna", "kitni", "kitne",
    "kahan", "kahaan", "kab", "kyun", "kyu", "kiya",
    "hai", "hain", "hay", "ho", "hoga", "hogi", "tha", "thi",
    "ka", "ki", "ke", "ko", "se", "mein", "men", "par",
    "mujhe", "mujhay", "mera", "meri", "mere", "apna", "apni", "apne",
    "chahiye", "chahiyay", "chaiye", "zaroori", "zarurat", "zaroorat",
    "karna", "karne", "karwana", "banwana", "banane", "lena", "lene",
    "liye", "liay", "wala", "wali", "walay", "nahi", "nahin", "nai",
    "tareeqa", "tarika", "kagzat", "kaghzat", "darkhast", "shanakhti",
}
_MIN_ROMAN_MARKERS = 2

# ---------------------------------------------------------------------------
# DOMAIN DETECTION
# ---------------------------------------------------------------------------
# Substring match, lowercased. Deliberately generous: a false positive costs
# one extra BM25 query, a false negative costs the whole half of a compound
# answer. Keys match the service ids used in main.py's SERVICES list.
_DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    "nadra": (
        "cnic", "nadra", "nicop", "b-form", "b form", "bform", "frc", "crc",
        "identity card", "family registration", "shanakhti", "smart card",
    ),
    "fbr": (
        "fbr", "ntn", "iris", "atl", "income tax", "tax return", "taxpayer",
        "filer", "tax refund", "withholding", "tax",
    ),
    "license": (
        "driving licence", "driving license", "learner", "excise", "ldc",
        "international driving", "driving permit", "driving test",
    ),
    "secp": (
        "secp", "company", "incorporat", "pvt ltd", "private limited",
        "smc", "sole proprietor", "business registration", "memorandum",
    ),
    "passport": (
        "passport", "dgip", "mrp", "travel document", "immigration",
    ),
}


def _detect_domains(question: str) -> list[str]:
    q = question.lower()
    return [d for d, kws in _DOMAIN_KEYWORDS.items() if any(k in q for k in kws)]


def _romanized_markers(question: str) -> list[str]:
    words = set(re.findall(r"[a-z']+", question.lower()))
    return sorted(words & _ROMAN_URDU_MARKERS)


def analyse(question: str) -> dict:
    """
    Returns:
        scripts:           which writing systems appear ("latin" / "urdu")
        code_switched:     the question mixes Urdu and English
        code_switch_kind:  "script_mixed" | "romanized" | None
        needs_translation: there is non-English content the English-only
                           embedder cannot use — includes pure Urdu script,
                           which is not "mixed" but is just as unusable
        markers:           the romanized markers that fired, for the inspector
        domains:           service domains named in the question
        compound:          spans two or more domains
    """
    q = question or ""

    has_urdu_script  = bool(_URDU_SCRIPT.search(q))
    has_latin        = bool(_LATIN.search(q))
    markers          = _romanized_markers(q)
    is_romanized     = len(markers) >= _MIN_ROMAN_MARKERS

    scripts = [s for s, present in (("latin", has_latin), ("urdu", has_urdu_script)) if present]

    if has_urdu_script and has_latin:
        kind = "script_mixed"
    elif is_romanized:
        kind = "romanized"
    else:
        kind = None

    domains = _detect_domains(q)

    return {
        "scripts":           scripts,
        "code_switched":     kind is not None,
        "code_switch_kind":  kind,
        "needs_translation": kind is not None or (has_urdu_script and not has_latin),
        "markers":           markers,
        "domains":           domains,
        "compound":          len(domains) >= 2,
    }


if __name__ == "__main__":
    cases = [
        # question, code_switched, kind, domains, compound
        ("What documents do I need for CNIC registration?", False, None,           ["nadra"],             False),
        ("CNIC renewal ka process kya hai?",                True,  "romanized",    ["nadra"],             False),
        ("CNIC ke liye کیا کاغذات chahiye?", True, "script_mixed", ["nadra"], False),
        ("Do I need a CNIC before applying for a passport?", False, None,          ["nadra", "passport"], True),
        ("How do I file my income tax return?",             False, None,           ["fbr"],               False),
        ("Passport renewal fee kitni hai aur kahan jama karna hai?", True, "romanized", ["passport"],      False),
    ]

    for q, cs, kind, domains, compound in cases:
        r = analyse(q)
        assert r["code_switched"]    == cs,       (q, r)
        assert r["code_switch_kind"] == kind,     (q, r)
        assert r["domains"]          == domains,  (q, r)
        assert r["compound"]         == compound, (q, r)
        # The console this runs on may be cp1252; never let a demo print
        # be the thing that fails on an Urdu-script question.
        safe_q = q[:44].encode("ascii", "replace").decode("ascii")
        print(f"  ok  switched={str(r['code_switched']):<5} "
              f"kind={str(r['code_switch_kind']):<13} "
              f"domains={','.join(r['domains']) or '-':<16} {safe_q}")

    print("input_analysis self-checks passed")
