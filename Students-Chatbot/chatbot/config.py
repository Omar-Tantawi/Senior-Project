from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "Data"
QDRANT_DIR = BASE_DIR / "qdrant_storage"
BM25_PATH = BASE_DIR / "qdrant_storage" / "bm25.pkl"

COLLECTION_NAME = "curriculum"
EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
EMBEDDING_DIM = 1024

# Ollama model — run `ollama pull <name>` before ingesting
# Default: deepseek-r1:7b. Swap for arabic-deepseek-r1:7b once available.
LLM_MODEL = "command-r7b-arabic:7b"

SIMILARITY_THRESHOLD = 0.10
TOP_K_DENSE = 10
TOP_K_BM25 = 10
TOP_K_FINAL = 5

# Origins allowed to call the API — set to the school LMS URL before deployment
# e.g. "https://lms.school.edu.sy"
ALLOWED_ORIGINS: list[str] = ["*"]

CHUNK_SIZE = 400       # approximate tokens per chunk
CHUNK_OVERLAP = 60     # token overlap between consecutive chunks

# Minimum extracted text per page before falling back to Surya OCR
OCR_FALLBACK_THRESHOLD = 80
