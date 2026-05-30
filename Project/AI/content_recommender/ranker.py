"""
Re-rank fetched ContentItems by semantic similarity to the teacher's original
query, using a multilingual sentence-transformer (Arabic + English in one model).
"""

import numpy as np

from config import EMBED_MODEL
from models import ContentItem


class Ranker:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        print(f"[Ranker] Loading embedding model: {EMBED_MODEL}")
        self.model = SentenceTransformer(EMBED_MODEL)
        print("[Ranker] Embedding model ready.")

    def rank(self, query: str, items: list[ContentItem], top_n: int) -> list[ContentItem]:
        if not items:
            return []

        texts = [f"{it.title}. {it.description}".strip() for it in items]

        q_vec = self.model.encode([query], normalize_embeddings=True)[0]
        m_vec = self.model.encode(texts,  normalize_embeddings=True)

        sims = m_vec @ q_vec   # cosine because vectors are normalized

        for it, s in zip(items, sims):
            it.relevance_score = float(max(0.0, min(1.0, (s + 1) / 2)))

        items.sort(key=lambda x: x.relevance_score, reverse=True)
        return items[:top_n]
