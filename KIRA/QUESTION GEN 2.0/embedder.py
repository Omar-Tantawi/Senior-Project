"""
In-memory TF-IDF retriever — no internet, no model downloads required.

Replaces sentence-transformers + ChromaDB with a pure numpy/stdlib
TF-IDF implementation that works offline on any Python 3.9+ environment.
Quality is sufficient for selecting relevant PDF chunks to feed to Ollama.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-word chars, keep Arabic and Latin tokens."""
    return re.findall(r"[\w؀-ۿ]+", text.lower())


class Embedder:
    def __init__(self, persist_dir: str = "./chroma_store"):
        print("[Embedder] Ready (offline TF-IDF mode).")
        # { document_id: {"chunks": list[str], "tfidf": np.ndarray, "vocab": dict} }
        self._store: dict = {}

    # ── Document management ───────────────────────────────────────────────────

    def add_document(self, document_id: str, chunks: list[str], metadata: dict = None):
        """Index all chunks with TF-IDF vectors."""
        tokenized = [_tokenize(c) for c in chunks]

        # Build vocabulary from all chunks
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for t in tokens:
                if t not in vocab:
                    vocab[t] = len(vocab)

        V = len(vocab)
        N = len(chunks)

        # Document frequency per term
        df = np.zeros(V)
        for tokens in tokenized:
            for t in set(tokens):
                df[vocab[t]] += 1

        # IDF (smoothed)
        idf = np.log((N + 1) / (df + 1)) + 1.0

        # TF-IDF matrix  (N x V)
        matrix = np.zeros((N, V), dtype=np.float32)
        for i, tokens in enumerate(tokenized):
            tf = Counter(tokens)
            total = max(len(tokens), 1)
            for t, cnt in tf.items():
                matrix[i, vocab[t]] = (cnt / total) * idf[vocab[t]]

        # L2-normalise rows
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
        matrix /= norms

        self._store[document_id] = {"chunks": chunks, "tfidf": matrix, "vocab": vocab, "idf": idf}
        print(f"[Embedder] Indexed {N} chunks for document '{document_id}'.")

    def delete_document(self, document_id: str):
        self._store.pop(document_id, None)

    def document_exists(self, document_id: str) -> bool:
        return document_id in self._store

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, document_id: str, query: str, top_k: int = 5) -> list[str]:
        """Return the top-k most relevant chunks for a query."""
        entry  = self._store[document_id]
        q_vec  = self._query_vector(query, entry["vocab"], entry["idf"], entry["tfidf"].shape[1])
        scores = entry["tfidf"] @ q_vec
        indices = np.argsort(scores)[::-1][:top_k]
        return [entry["chunks"][i] for i in indices]

    def get_random_chunks(self, document_id: str, n: int = 5) -> list[str]:
        """Return n evenly-spaced chunks — used for broad generation."""
        chunks = self._store[document_id]["chunks"]
        step   = max(1, len(chunks) // n)
        return [chunks[i * step] for i in range(min(n, len(chunks)))]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _query_vector(query: str, vocab: dict, idf: np.ndarray, V: int) -> np.ndarray:
        tokens = _tokenize(query)
        tf     = Counter(tokens)
        total  = max(len(tokens), 1)
        vec    = np.zeros(V, dtype=np.float32)
        for t, cnt in tf.items():
            if t in vocab:
                vec[vocab[t]] = (cnt / total) * idf[vocab[t]]
        norm = np.linalg.norm(vec) + 1e-9
        return vec / norm
