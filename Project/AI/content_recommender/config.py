import os

AI_API_KEY   = os.getenv("AI_API_KEY",   "change-me-shared-secret")
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "command-r7b-arabic")

EMBED_MODEL            = r"C:\Users\ASUS\Desktop\content_recommender\.claude\worktrees\practical-black-7169a2\Project\AI\content_recommender\training\checkpoints\edu_ranker_ar"
MAX_RESULTS_PER_SOURCE = 10
FINAL_RESULTS_COUNT    = 15
SEARCH_TIMEOUT         = 8
OLLAMA_TIMEOUT         = 120
