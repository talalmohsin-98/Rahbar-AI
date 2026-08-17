import os
from typing import Any
from groq import Groq

from config import ANSWER_MODEL, FAST_MODEL


# ---------------------------------------------------------------------------
# 1. CLIENT
# ---------------------------------------------------------------------------
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set.")
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# 2. SYSTEM PROMPTS PER INTENT
# ---------------------------------------------------------------------------
# We write different system prompts per intent because the answer FORMAT
# should match what the user actually asked for.

BASE_RULES = """
You are a knowledgeable assistant for Pakistani government services and documentation.

STRICT RULES:
1. Answer ONLY from the provided context passages. Do not use outside knowledge.
2. After EVERY factual claim, add a citation tag in plain ASCII square brackets exactly like this: [source: filename]. Never use full-width brackets (like 【 】) or any other citation style.
3. Only say "I could not find this information in the available documents." when the context is genuinely irrelevant. If the context contains a fee/charge table or any partial answer, USE it — do not decline when relevant data is present.
4. Never invent facts, numbers, dates, or document names.
5. Keep your answer concise and directly responsive to the question.

FORMATTING RULES:
6. Do NOT use markdown emphasis characters — no asterisks (* or **) and no bullet symbols like •.
7. Whenever your answer enumerates multiple documents, requirements, steps, options, or fees, you MUST format them as a list — put EACH item on its OWN line beginning with "- " using a real line break. Never pack multiple items into one sentence or paragraph. A short lead-in sentence before the list is fine.
8. The modern CNIC is issued as a "Smart NIC" (a smart card). Smart NIC fees (e.g. Rs. 750 normal / Rs. 1,500 urgent) are the current CNIC card fees; the plain "New CNIC" row (Rs. 0 normal) is the legacy card. Treat a citizen asking about "CNIC fee" as asking about the Smart NIC/CNIC card fees.

COMPLETENESS RULES:
9. A one-line answer is almost never a useful answer. If the passages give only one or two items for what was asked, keep going and report the rest of what THOSE SAME passages say about that service: the steps carried out at the office, the fee, and the processing time. Give each its own "- " line under a short label such as "At the centre:" or "Fee:".
   Worked example — the citizen asks which documents a CNIC renewal needs, and the passage lists "Requirements: CNIC number" plus a five-step centre process (token, biometric capture, interview, payment, card printing). A correct answer states the CNIC number requirement AND all five steps. An answer that stops at "CNIC number" is wrong, because it hides from the citizen that they must attend in person for biometrics.
   Everything you add must still come from a passage: never write what "typically" or "usually" happens and never fill a gap from your own knowledge. That restriction is about invented material only — it is never a reason to leave out something the passages do contain.
10. Never present a requirement list as exhaustive when the passages show only a bare item or two. In that case end with one plain line, with NO citation tag on it (it is your advice to the citizen, not a statement from the documents): "The published requirements list is brief — confirm the exact documents for your case at a NADRA Registration Centre (or the relevant office)." Adapt the office name to the service being asked about.
11. If the passages describe a numbered procedure with steps missing (for example the numbering jumps), describe only the steps actually present and do not invent the gaps.
"""

INTENT_INSTRUCTIONS = {
    "factual": BASE_RULES + """
Format: If the answer lists documents, requirements, fees, or options, present them as a "- " list with ONE item per line (not a paragraph). Use 2-4 plain sentences only when the answer is genuinely not a list. Cite every fact.
""",

    "comparison": BASE_RULES + """
Format: Present each item being compared under its own short label, with each detail on its own line starting with "- ".
Cover: requirements, process, timeline, and any key differences. Cite every point.
""",

    "procedural": BASE_RULES + """
Format: Numbered steps only. Each step on its own line.
Start each step with an action verb (e.g., "Visit", "Submit", "Collect").
Cite the source after each step.
""",

    "out_of_scope": """
You are an assistant for Pakistani government services.
The user's question is outside your scope. Politely explain what you CAN help with.
Do not answer the question itself.
""",
}


# ---------------------------------------------------------------------------
# 3. FORMAT CONTEXT PASSAGES
# ---------------------------------------------------------------------------
def format_context(chunks: list[dict[str, Any]]) -> str:
    """
    Formats compressed chunks into a numbered context block for the LLM prompt.

    Each passage is labelled with its source filename so the LLM knows
    what to put in the [source: ...] citation tag.

    Example output:
        [Passage 1 | source: nadra_guide.pdf]
        NADRA charges Rs. 750 for a new CNIC. Processing takes 30 working days.

        [Passage 2 | source: passport_guide.pdf]
        Passport applications require Form-1, two photos, and CNIC copy.

    The LLM is instructed to use the source name from the passage label.
    This ensures citations are traceable back to real chunks.
    """
    if not chunks:
        return "No context available."

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        # Use just the filename, not full path — cleaner in citations.
        # NOTE: `chunk.get("source", "unknown")` only falls back to "unknown"
        # when the "source" key is MISSING. If the DB row has source=NULL,
        # the key is still present with value None, so the old code crashed
        # with "'NoneType' object has no attribute 'split'". `or "unknown"`
        # catches both cases.
        source = (chunk.get("source") or "unknown").split("/")[-1]
        content = chunk.get("content") or ""
        if not content:
            continue   # nothing to show for this chunk — skip it
        parts.append(f"[Passage {i} | source: {source}]\n{content}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 4. BUILD MESSAGES
# ---------------------------------------------------------------------------
def build_messages(
    question: str,
    context: str,
    intent: str,
    feedback: str | None = None,
) -> list[dict[str, str]]:
    """
    Constructs the messages list for the Groq chat completion API.

    We use a two-message structure:
        - system: instructions + rules (who you are, what to do)
        - user:   the context passages + the question

    Why put context in the user message, not system?
        System messages are for persistent instructions.
        Context changes every query — it belongs in the user turn.
        Some models also attend better to content in the user role.
    """
    system_prompt = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS["factual"])

    user_content = f"""Here are the relevant context passages:

{context}

---

Question: {question}

Answer based strictly on the passages above. Cite every claim."""

    # Retry feedback goes LAST, after the question — a revision instruction
    # buried above the context gets treated as background. See
    # completeness_check.build_retry_feedback for what this contains.
    if feedback:
        user_content += f"\n\n---\n\n{feedback}"

    return [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": user_content},
    ]


# ---------------------------------------------------------------------------
# 5. MAIN GENERATION FUNCTION
# ---------------------------------------------------------------------------
def _reasoning_tokens(response: Any) -> int:
    """
    Reasoning tokens spent by the model, or 0 if it doesn't report them.

    Groq returns these under usage.completion_tokens_details.reasoning_tokens
    for the gpt-oss models. The tests' stub Groq client returns a plain object
    without those attributes, so every access is guarded.
    """
    details = getattr(getattr(response, "usage", None), "completion_tokens_details", None)
    return int(getattr(details, "reasoning_tokens", 0) or 0)
def generate(
    question: str,
    chunks: list[dict[str, Any]],
    intent: str = "factual",
    temperature: float = 0.1,
    max_tokens: int = 1400,
    feedback: str | None = None,
) -> dict[str, Any]:
    """
    Generates a cited answer from compressed chunks.

    Args:
        question:    Original user question.
        chunks:      Compressed chunks from compressor.py.
        intent:      From intent_router — controls answer format.
        temperature: Low (0.1) for factual accuracy. Higher = more creative but less accurate.
        feedback:    Revision instruction for a regeneration — what was wrong
                     with the previous attempt, with the omitted passages
                     quoted. None on a first attempt.
        max_tokens:  Cap on completion length. NOTE this budget is shared with
                     the model's hidden reasoning: ANSWER_MODEL (gpt-oss-120b)
                     is a reasoning model and spends 150-300 completion tokens
                     thinking before it emits a single visible character. At the
                     old 800 cap a long list answer could be cut off mid-item
                     with no error raised anywhere — the truncated text just
                     became the final answer.

    Returns:
        Dict with:
            answer:          The raw LLM response string (with citation tags)
            model:           Which model was used
            prompt_tokens:   Input token count (for cost tracking)
            completion_tokens: Output token count
            reasoning_tokens:  Of which spent on hidden reasoning
            finish_reason:   "stop" (complete) or "length" (hit the cap)
            truncated:       True if the answer was cut off mid-generation
            context_used:    How many chunks were sent to the LLM
    """
    # Handle out-of-scope early — no context needed, different prompt
    if intent == "out_of_scope":
        client = get_groq_client()
        response = client.chat.completions.create(
            model=FAST_MODEL,   # small model fine for a polite decline
            messages=[
                {"role": "system", "content": INTENT_INSTRUCTIONS["out_of_scope"]},
                {"role": "user",   "content": question},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        return {
            "answer":             (response.choices[0].message.content or "").strip(),
            "model":              FAST_MODEL,
            "prompt_tokens":      response.usage.prompt_tokens,
            "completion_tokens":  response.usage.completion_tokens,
            "reasoning_tokens":   _reasoning_tokens(response),
            "finish_reason":      getattr(response.choices[0], "finish_reason", "stop"),
            "truncated":          getattr(response.choices[0], "finish_reason", "stop") == "length",
            "context_used":       0,
        }

    # Format context from compressed chunks
    context = format_context(chunks)

    # Build messages
    messages = build_messages(question, context, intent, feedback=feedback)

    # Call Groq — larger model for final answer quality
    client = get_groq_client()
    response = client.chat.completions.create(
        model=ANSWER_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    choice = response.choices[0]
    answer = (choice.message.content or "").strip()

    # A truncated answer is a broken answer, and nothing downstream can detect
    # that from the text alone — the citation verifier and hallucination eval
    # both happily score a half-written list. Surface it instead of hiding it.
    finish_reason = getattr(choice, "finish_reason", "stop")
    truncated     = finish_reason == "length"
    if truncated:
        print(f"[Generator] WARNING: hit the {max_tokens}-token cap — answer is cut off.")

    return {
        "answer":             answer,
        "model":              ANSWER_MODEL,
        "prompt_tokens":      response.usage.prompt_tokens,
        "completion_tokens":  response.usage.completion_tokens,
        "reasoning_tokens":   _reasoning_tokens(response),
        "finish_reason":      finish_reason,
        "truncated":          truncated,
        "context_used":       len(chunks),
    }


# ---------------------------------------------------------------------------
# 6. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mock_chunks = [
        {
            "chunk_id": 1,
            "content": "NADRA charges Rs. 750 for a new CNIC. Processing takes 30 working days for standard delivery.",
            "source": "nadra_guide.pdf",
            "rerank_score": 9.1,
            "rank": 1,
        },
        {
            "chunk_id": 2,
            "content": "Required documents: original B-Form or birth certificate, two passport-size photographs, and proof of address.",
            "source": "nadra_guide.pdf",
            "rerank_score": 8.5,
            "rank": 2,
        },
    ]

    question = "What documents and fees are required for NADRA CNIC?"
    print(f"\nQuestion: {question}\n")

    result = generate(question, mock_chunks, intent="factual")
    print(f"Answer:\n{result['answer']}")
    print(f"\nModel: {result['model']}")
    print(f"Tokens: {result['prompt_tokens']} prompt + {result['completion_tokens']} completion")
