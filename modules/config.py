"""
config.py — Application-wide constants and configuration.

This module centralises every tuneable constant so that contributors
can change behaviour without hunting through 2 500+ lines of app code.
"""

# Default Gemini model used for intelligence summaries (Set to 2.5 Flash for safety/speed)
DEFAULT_LLM_MODEL = "models/gemini-2.5-flash"

# How many trials to send to the LLM in a single summarisation call
MAX_TRIALS_FOR_SUMMARY = 10

# Google Generative AI REST base URL
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

# Models exposed in the Streamlit sidebar selector
GEMINI_MODEL_OPTIONS = {
    "Gemini 2.5 Flash (Ultra-Fast & Safe)": "models/gemini-2.5-flash",
    "Gemini 2.5 Pro (Deep Analysis)": "models/gemini-2.5-pro",
    "Robotics 1.5 Preview (Experimental)": "models/gemini-robotics-er-1.5-preview",
}

# ── Model output token limits ──
# Max output tokens each model supports (conservative estimates for safe usage)
GEMINI_MODEL_TOKEN_LIMITS = {
    "models/gemini-2.5-flash": 8192,
    "models/gemini-2.5-pro": 8192,
    "models/gemini-robotics-er-1.5-preview": 8192,
}
DEFAULT_TOKEN_LIMIT = 8192  # fallback for unknown models

# Per-prompt output budgets as fraction of model limit
PROMPT_TOKEN_BUDGETS = {
    "executive_summary": 0.45,   # ~3600 tokens
    "single_trial":     0.30,    # ~2500 tokens
    "vaccine_intel":    0.45,    # ~3600 tokens
    "comparison":       0.55,    # ~4500 tokens — enough for all 4 sections
    "default":          0.25,    # ~2000 tokens
}

# Appended to every long-form prompt to prevent mid-section truncation
ANTI_TRUNCATION_SUFFIX = (
    "\n\nIMPORTANT: You MUST complete ALL sections listed above. Do NOT leave any "
    "section unfinished or cut off. Be concise within each section to ensure every "
    "section is fully addressed within the output limit."
)

# Local file paths for persistent analyst state
ANALYST_NOTES_FILE = "analyst_notes.json"
LEARNED_MFR_FILE = "learned_manufacturers.json"
WATCHLIST_FILE = "watchlist.json"
LAST_SEEN_FILE = "last_seen.json"
VACCINE_ONTOLOGY_FILE = "vaccine_ontology.json"