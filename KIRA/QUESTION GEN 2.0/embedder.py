"""
Multilingual embeddings + in-memory vector store.

Model: paraphrase-multilingual-MiniLM-L12-v2
  - Supports 50+ languages including Arabic and English
  - 384-dim embeddings
  - Fast enough to run on CPU

Uses a simple numpy cosine-similarity store instead of ChromaDB so
the service works on Python 3.9 / Windows without any sqlite3 version
constraints.  Data lives only in memory and resets on server restart —
this matches the original design (see handoff doc, section 7).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class Embedder:
    def __init__(self, persist_dir: str = "./chroma_store"):
        print("[Embedder] Loading multilingual embedding model...")
        self.model = SentenceTransformer(EMBED_MODEL)
        # { document_id: {"chunks": list[str], "embeddings": np.ndarray} }
        self._store: dict = {}
        print("[Embedder] Ready.")

    # ── Document management ───────────────────────────────────────────────────

    def add_document(self, document_id: str, chunks: list[str], metadata: dict = None):
        """Embed and store all chunks for a document."""
        embeddings = self.model.encode(chunks, show_progress_bar=True)
        self._store[document_id] = {
            "chunks":     chunks,
            "embeddings": embeddings,   # shape: (n_chunks, dim)
        }
        print(f"[Embedder] Stored {len(chunks)} chunks for document '{document_id}'.")

    def delete_document(self, document_id: str):
        self._store.pop(document_id, None)

    def document_exists(self, document_id: str) -> bool:
        return document_id in self._store

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, document_id: str, query: str, top_k: int = 5) -> list[str]:
        """Return the top-k most relevant chunks using cosine similarity."""
        entry       = self._store[document_id]
        query_emb   = self.model.encode([query])                  # (1, dim)
        scores      = self._cosine(query_emb, entry["embeddings"]) # (n,)
        indices     = np.argsort(scores)[::-1][:top_k]
        return [entry["chunks"][i] for i in indices]

    def get_random_chunks(self, document_id: str, n: int = 5) -> list[str]:
        """Return n evenly-spaced chunks — used for broad generation."""
        chunks = self._store[document_id]["chunks"]
        step   = max(1, len(chunks) // n)
        return [chunks[i * step] for i in range(min(n, len(chunks)))]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Cosine similarity between one vector a (1, d) and matrix b (n, d)."""
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return (a_norm @ b_norm.T).flatten()
