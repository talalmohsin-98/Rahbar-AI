import os
from enum import Enum
from groq import Groq

# ---------------------------------------------------------------------------
# 1. INTENT CATEGORIES
# ---------------------------------------------------------------------------
class Intent(str, Enum):
    """
    The four question types this system handles.

    We use str Enum so Intent.FACTUAL == "factual" (useful for JSON serialization
    and prompt construction — you can just use the string value directly).
    """
    FACTUAL     = "factual"      # What is X? Who is responsible for Y?
    COMPARISON  = "comparison"   # Compare X and Y. What's the difference between X and Y?
    PROCEDURAL  = "procedural"   # How do I do X? What are the steps for Y?
    OUT_OF_SCOPE = "out_of_scope" # Politics, opinions, things outside the document corpus


# ---------------------------------------------------------------------------
# 2. GROQ CLIENT
# ---------------------------------------------------------------------------
# We use Groq (not OpenAI) for LLM calls — fast, cheap, good enough for
# classification tasks that don't need frontier model capability.
def get_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable not set.")
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# 3. CLASSIFICATION PROMPT
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an intent classifier for a Pakistani government services AI assistant.

Classify the user's question into exactly ONE of these categories:

- factual: The user wants a specific fact, definition, or piece of information.
  Examples: "What is NADRA?", "What documents do I need for CNIC?", "Who issues passports in Pakistan?"

- comparison: The user wants to compare two or more things.
  Examples: "What is the difference between B-Form and CNIC?", "Compare manual vs online NADRA registration."

- procedural: The user wants step-by-step instructions or a process explained.
  Examples: "How do I apply for a passport?", "What are the steps to register a birth?"

- out_of_scope: The question is not about Pakistani government services,
  asks for opinions, is political, or cannot be answered from official documents.
  Examples: "Is NADRA corrupt?", "What do you think about the government?", "Tell me a joke."

Respond with ONLY the category name — one of: factual, comparison, procedural, out_of_scope
No explanation. No punctuation. Just the word."""


def _call_llm(question: str) -> str:
    """
    Sends the question to the LLM and returns its raw text response.
    Separated from route() so it can be mocked in tests.
    """
    client = get_groq_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # Small, fast model — classification doesn't need GPT-4
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        temperature=0.0,   # 0 = deterministic — always the same answer for the same question
        max_tokens=10,     # We only need one word — "factual", "comparison", etc.
    )
    return (response.choices[0].message.content or "").strip().lower()


# ---------------------------------------------------------------------------
# 4. MAIN ROUTING FUNCTION
# ---------------------------------------------------------------------------
def route(question: str) -> Intent:
    """
    Classifies the question and returns an Intent enum value.

    Args:
        question: The raw user question string.

    Returns:
        An Intent enum value. Downstream functions check this to decide
        how to handle retrieval and generation.

    Fallback behavior:
        If the LLM returns something unexpected (not one of our 4 categories),
        we default to Intent.FACTUAL. Better to attempt retrieval than to fail.
        This is a deliberate defensive choice — in production you'd log these
        for model improvement.

    Temperature=0 explained:
        LLM temperature controls randomness. At 0, the model always picks
        the highest-probability next token — fully deterministic.
        For classification, we want the same question to always get the same
        category. Randomness here would be a bug, not a feature.
    """
    raw = _call_llm(question)

    # Map the raw string to an Intent enum
    # We try to match each known intent value
    for intent in Intent:
        if intent.value in raw:
            return intent

    # LLM returned something unexpected — default to factual
    # Log in production: this might indicate prompt drift or model issues
    print(f"[intent_router] Unexpected LLM output: '{raw}' — defaulting to FACTUAL")
    return Intent.FACTUAL


# ---------------------------------------------------------------------------
# 5. ROUTING METADATA (downstream needs this to adjust behavior)
# ---------------------------------------------------------------------------
def get_routing_config(intent: Intent) -> dict:
    """
    Returns configuration that downstream components use based on intent.

    This is the "so what" of classification — not just what category it is,
    but what should change in the pipeline because of that category.

    Fields:
        retrieval_strategy: hint for hybrid_search (may expand later)
        answer_format:      hint for the LLM prompt in generator.py
        top_k_override:     comparison needs more chunks (covers multiple topics)
        should_retrieve:    out_of_scope skips retrieval entirely
    """
    configs = {
        Intent.FACTUAL: {
            "retrieval_strategy": "standard",
            "answer_format":      "concise factual answer with citation",
            "top_k_override":     None,     # use default
            "should_retrieve":    True,
        },
        Intent.COMPARISON: {
            "retrieval_strategy": "multi_topic",
            "answer_format":      "structured comparison, use a table if appropriate",
            "top_k_override":     15,       # need more chunks to cover both topics
            "should_retrieve":    True,
        },
        Intent.PROCEDURAL: {
            "retrieval_strategy": "standard",
            "answer_format":      "numbered step-by-step instructions",
            "top_k_override":     None,
            "should_retrieve":    True,
        },
        Intent.OUT_OF_SCOPE: {
            "retrieval_strategy": None,
            "answer_format":      "polite decline, explain scope",
            "top_k_override":     None,
            "should_retrieve":    False,    # skip retrieval entirely
        },
    }
    return configs[intent]


# ---------------------------------------------------------------------------
# 6. SMOKE TEST
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_questions = [
        "What documents do I need for NADRA CNIC registration?",
        "What is the difference between B-Form and Form-B?",
        "How do I apply for a Pakistani passport step by step?",
        "Is the government doing a good job with NADRA?",
    ]

    print("\nIntent Router Test\n" + "="*40)
    for q in test_questions:
        intent = route(q)
        config = get_routing_config(intent)
        print(f"\nQ: {q}")
        print(f"   Intent:  {intent.value}")
        print(f"   Format:  {config['answer_format']}")
        print(f"   Retrieve: {config['should_retrieve']}")
