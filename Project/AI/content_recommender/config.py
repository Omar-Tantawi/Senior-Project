import os
from pathlib import Path

# Auto-load .env file if it exists next to this file
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Backend auth ──────────────────────────────────────────────────────────────
AI_API_KEY = os.getenv("AI_API_KEY", "change-me-shared-secret")

# ── Google API key (YouTube Data API v3) ──────────────────────────────────────
GOOGLE_API_KEY  = os.getenv("GOOGLE_API_KEY", "AIzaSyCA-2mUd5fmbWUl7w237owshKx_2R88eKw")
YOUTUBE_API_KEY = GOOGLE_API_KEY

# ── Ollama (Arabic intent extraction) ─────────────────────────────────────────
OLLAMA_URL     = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "command-r7b-arabic")
OLLAMA_TIMEOUT = 120

# ── Trained model — relative path so it works on any machine ─────────────────
_DEFAULT_MODEL = Path(__file__).parent / "training" / "checkpoints" / "edu_ranker_ar"
EMBED_MODEL = os.getenv("EMBED_MODEL", str(_DEFAULT_MODEL))

# ── Search settings ───────────────────────────────────────────────────────────
MAX_RESULTS_PER_SOURCE = 10
FINAL_RESULTS_COUNT    = 15
SEARCH_TIMEOUT         = 8
