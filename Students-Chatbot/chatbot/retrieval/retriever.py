"""
Hybrid retrieval: dense (Qdrant cosine) + lexical (BM25).
Results merged with Reciprocal Rank Fusion (RRF).
Chunks below SIMILARITY_THRESHOLD are dropped before returning.
"""
import pickle
from functools import lru_cache
from typing import Optional

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from chatbot.config import (
    QDRANT_DIR, BM25_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, SIMILARITY_THRESHOLD,
    TOP_K_DENSE, TOP_K_BM25, TOP_K_FINAL,
)
from chatbot.indexing.normalizer import normalize

_RRF_K = 60  # standard RRF constant


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)


@lru_cache(maxsize=1)
def _load_bm25() -> dict:
    with open(BM25_PATH, "rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _load_qdrant() -> QdrantClient:
    return QdrantClient(path=str(QDRANT_DIR))


def retrieve(
    query: str,
    grade: Optional[str] = None,
    subject: Optional[str] = None,
) -> list[dict]:
    """
    Returns up to TOP_K_FINAL chunks above SIMILARITY_THRESHOLD,
    ranked by RRF-combined dense + BM25 scores.
    Each result dict includes the chunk payload plus a `score` field.

    grade / subject are applied as Qdrant payload filters before scoring,
    so they can never silently empty results that were already retrieved.
    """
    normalized_query = normalize(query)

    model = _load_model()
    query_vec = model.encode(
        normalized_query,
        task="retrieval.query",
        normalize_embeddings=True,
    ).tolist()

    # Build payload filter for grade / subject if provided.
    must = []
    if grade:
        must.append(models.FieldCondition(key="grade", match=models.MatchValue(value=grade)))
    if subject:
        must.append(models.FieldCondition(key="subject", match=models.MatchValue(value=subject)))
    query_filter = models.Filter(must=must) if must else None

    # Fetch top-K dense candidates without a score cutoff so we can read their
    # raw scores.  We use the best score as the guardrail signal — if even the
    # closest curriculum chunk is below SIMILARITY_THRESHOLD the query is
    # off-curriculum and we return nothing.  Using score_threshold=0 here
    # (instead of the actual threshold) is intentional: the threshold is applied
    # below when building the RRF table, not at retrieval time.
    dense_hits_all = _load_qdrant().query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=TOP_K_DENSE,
        with_payload=True,
        score_threshold=0.0,
        query_filter=query_filter,
    ).points

    # Guardrail: block if the best-matching chunk doesn't clear the threshold.
    if not dense_hits_all or dense_hits_all[0].score < SIMILARITY_THRESHOLD:
        return []

    # Only pass chunks that cleared the threshold into RRF so low-quality
    # dense hits don't dilute the ranking.
    dense_hits = [h for h in dense_hits_all if h.score >= SIMILARITY_THRESHOLD]

    bm25_data = _load_bm25()
    bm25_scores = bm25_data["bm25"].get_scores(normalized_query.split())
    bm25_ranked = sorted(
        enumerate(bm25_scores), key=lambda x: x[1], reverse=True
    )[:TOP_K_BM25]

    # RRF merge
    rrf: dict[int, float] = {}
    for rank, hit in enumerate(dense_hits):
        rrf[hit.id] = rrf.get(hit.id, 0.0) + 1 / (_RRF_K + rank + 1)
    for rank, (idx, _) in enumerate(bm25_ranked):
        rrf[idx] = rrf.get(idx, 0.0) + 1 / (_RRF_K + rank + 1)

    top_ids = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:TOP_K_FINAL]

    client = _load_qdrant()
    results = []
    for doc_id in top_ids:
        hit = client.retrieve(COLLECTION_NAME, ids=[doc_id], with_payload=True)
        if not hit:
            continue
        payload = hit[0].payload
        # Apply grade/subject filter to BM25-only hits (dense hits are
        # already filtered by the Qdrant payload filter above).
        if grade and payload.get("grade") != grade:
            continue
        if subject and subject.lower() not in payload.get("subject", "").lower():
            continue
        dense_score = next((h.score for h in dense_hits if h.id == doc_id), None)
        if dense_score is None or dense_score >= SIMILARITY_THRESHOLD:
            results.append({**payload, "score": rrf[doc_id]})

    return results
