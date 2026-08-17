import os

# ---------------------------------------------------------------------------
# MODEL IDS
# ---------------------------------------------------------------------------
# Every Groq model ID used by the pipeline lives here. Providers decommission
# models on their own schedule (the Llama 3.1 family was retired out from under
# this project, which 404'd every stage at once), so keep these in one place and
# allow an env override to patch a retirement without a code change.
#
# Check what the current key can actually reach:
#   curl -s https://api.groq.com/openai/v1/models \
#     -H "Authorization: Bearer $GROQ_API_KEY"

# Small, fast model — routing, query rewriting, tool simulation, short declines.
FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")

# Larger model — the final cited answer, where quality matters most.
ANSWER_MODEL = os.getenv("GROQ_ANSWER_MODEL", "openai/gpt-oss-120b")
