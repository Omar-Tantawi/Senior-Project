"""
Content Recommender microservice — FastAPI app.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8005 --reload

Environment variables (copy .env.example to .env and fill in):
    AI_API_KEY       Shared secret for X-API-Key header (between this service and Laravel)
    GOOGLE_API_KEY   Google Cloud key for YouTube Data API v3
    OLLAMA_URL       Ollama base URL (default: http://localhost:11434)
    OLLAMA_MODEL     Arabic model to use (default: command-r7b-arabic)
    EMBED_MODEL      Path to trained edu_ranker_ar model (auto-detected if kept next to this file)
"""

import time
import uuid
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from config            import AI_API_KEY, FINAL_RESULTS_COUNT, MAX_RESULTS_PER_SOURCE
from models            import (
    RecommendRequest, RecommendResponse, ExtractRequest, Intent,
)
from intent_extractor  import IntentExtractor
from ranker            import Ranker
from searcher          import Searcher


app = FastAPI(
    title="Content Recommender Service",
    version="1.0.0",
    description="Finds educational videos, articles, PDFs and images for teachers from Arabic input.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

extractor = IntentExtractor()
ranker    = Ranker()
searcher  = Searcher()


# ── Auth ──────────────────────────────────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_key(api_key: str = Security(_api_key_header)):
    if api_key != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def warmup():
    try:
        extractor.warmup()
    except Exception as e:
        print(f"[startup] Ollama warmup skipped: {e}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "content-recommender"}


@app.get("/reference/content-types")
async def content_types_ref():
    return {
        "content_types": ["video", "article", "pdf", "image"],
        "language_preference": ["ar", "en", "both"],
    }


@app.post("/recommend/extract-only", response_model=Intent, dependencies=[Depends(verify_key)])
async def extract_only(body: ExtractRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty.")
    intent = extractor.extract(body.query, body.refinement)
    return Intent(**intent)


@app.post("/recommend", response_model=RecommendResponse, dependencies=[Depends(verify_key)])
async def recommend(body: RecommendRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty.")
    if not body.content_types:
        raise HTTPException(status_code=400, detail="content_types must not be empty.")

    start = time.time()
    session_id = body.session_id or str(uuid.uuid4())

    intent = extractor.extract(body.query, body.refinement)

    per_source = max(MAX_RESULTS_PER_SOURCE, body.max_results * 2)
    raw_items = await searcher.search(
        intent=intent,
        content_types=body.content_types,
        language_pref=body.language_preference,
        per_source=per_source,
        exclude_urls=set(body.exclude_urls),
    )

    top_n = body.max_results or FINAL_RESULTS_COUNT
    ranked = ranker.rank(body.query, raw_items, top_n)

    elapsed_ms = int((time.time() - start) * 1000)

    return RecommendResponse(
        query_original=body.query,
        topic_ar=intent["topic_ar"],
        topic_en=intent["topic_en"],
        keywords_ar=intent["keywords_ar"],
        keywords_en=intent["keywords_en"],
        results=ranked,
        total_found=len(ranked),
        search_time_ms=elapsed_ms,
        session_id=session_id,
    )
