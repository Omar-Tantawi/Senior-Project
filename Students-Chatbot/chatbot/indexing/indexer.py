"""
Builds and persists the Qdrant dense index and the BM25 lexical index.
Run via scripts/ingest.py — not called at query time.
"""
import pickle
from pathlib import Path
from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from chatbot.config import (
    QDRANT_DIR, BM25_PATH, COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIM,
)


def _get_client() -> QdrantClient:
    QDRANT_DIR.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_DIR))


def build_index(chunks: list[dict]) -> None:
    client = _get_client()

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)

    texts = [c["normalized_text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks…")
    import torch
    batch_size = 128 if torch.cuda.is_available() else 32
    embeddings = model.encode(
        texts,
        task="retrieval.passage",
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    points = [
        PointStruct(
            id=i,
            vector=embeddings[i].tolist(),
            payload={k: v for k, v in chunks[i].items() if k != "normalized_text"},
        )
        for i in range(len(chunks))
    ]

    batch_size = 256
    for start in tqdm(range(0, len(points), batch_size), desc="Upserting to Qdrant"):
        client.upsert(COLLECTION_NAME, points=points[start : start + batch_size])

    _build_bm25(chunks)
    print(f"Index complete. {len(chunks)} chunks stored.")


def _build_bm25(chunks: list[dict]) -> None:
    tokenized = [c["normalized_text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    BM25_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "ids": list(range(len(chunks)))}, f)
    print(f"BM25 index saved → {BM25_PATH}")
