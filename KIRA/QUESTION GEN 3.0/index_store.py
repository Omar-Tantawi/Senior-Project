"""
index_store.py — single source of truth for the prebuilt curriculum index.

Loads once at startup and holds:
  • the BM25 model + the 12k chunk list (text + metadata)   ← bm25_v2.pkl
  • lookups: book → chunk indices, (book, page) → indices, known book names
  • a lazy Qdrant client + embedding model for dense search  ← qdrant_storage/

Every chunk is a dict: {text, book, subject, grade, page_num, used_ocr}.
Chunk index i lines up positionally with Qdrant point id i.

Config (qdrant_storage/index_config.json, written by tools/build_index.py):
  { "bm25_file", "embed_model", "embed_prefix", "collection", "dense_enabled", "dim" }
If dense_enabled is false/absent, the store runs BM25-only (still fully functional).
"""

import os
import re
import json
import pickle
import threading

ROOT        = os.path.dirname(os.path.abspath(__file__))
QDRANT_PATH = os.path.join(ROOT, "qdrant_storage")
CONFIG_PATH = os.path.join(QDRANT_PATH, "index_config.json")


def _norm_book(name: str) -> str:
    """Normalize a book/filename stem for matching: lowercase, drop extension & spaces."""
    name = os.path.splitext(os.path.basename(name))[0]
    return re.sub(r"[\s_]+", "-", name.strip().lower())


class IndexStore:
    def __init__(self):
        cfg = {}
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)

        bm25_file = cfg.get("bm25_file", "bm25_v2.pkl")
        pkl_path  = os.path.join(QDRANT_PATH, bm25_file)
        if not os.path.exists(pkl_path):                 # fall back to original
            pkl_path = os.path.join(QDRANT_PATH, "bm25.pkl")

        print(f"[IndexStore] loading {os.path.basename(pkl_path)} ...")
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)
        self.bm25   = obj["bm25"]
        self.chunks = obj["chunks"]
        print(f"[IndexStore] {len(self.chunks)} chunks, BM25 corpus={self.bm25.corpus_size}")

        # dense config
        self.collection    = cfg.get("collection", "curriculum")
        self.embed_model   = cfg.get("embed_model", "BAAI/bge-m3")
        self.doc_prefix    = cfg.get("embed_prefix", "")
        self.dense_enabled = bool(cfg.get("dense_enabled", False))
        self._client = None
        self._model  = None
        self._lock   = threading.Lock()

        # ── build lookups ────────────────────────────────────────────────────
        self.book_to_idxs:  dict[str, list[int]] = {}
        self._norm_to_book: dict[str, str]        = {}
        subjects, grades = set(), set()
        for i, c in enumerate(self.chunks):
            book = c["book"]
            self.book_to_idxs.setdefault(book, []).append(i)
            self._norm_to_book[_norm_book(book)] = book
            subjects.add(c.get("subject"))
            grades.add(c.get("grade"))
        # keep each book's indices in page order
        for book, idxs in self.book_to_idxs.items():
            idxs.sort(key=lambda i: (self.chunks[i].get("page_num", 0), i))
        self.subjects = sorted(s for s in subjects if s)
        self.grades   = sorted(g for g in grades if g is not None)
        print(f"[IndexStore] {len(self.book_to_idxs)} books, "
              f"{len(self.subjects)} subjects, grades {self.grades} | "
              f"dense={'on' if self.dense_enabled else 'off'} ({self.collection})")

    # ── basic access ──────────────────────────────────────────────────────────

    def chunk(self, idx: int) -> dict:
        return self.chunks[idx]

    def text(self, idx: int) -> str:
        return self.chunks[idx]["text"]

    def list_books(self) -> list[dict]:
        out = []
        for book, idxs in sorted(self.book_to_idxs.items()):
            pages = {self.chunks[i]["page_num"] for i in idxs}
            c0 = self.chunks[idxs[0]]
            out.append({
                "book": book, "subject": c0.get("subject"), "grade": c0.get("grade"),
                "chunks": len(idxs), "pages": len(pages),
            })
        return out

    def resolve_book(self, name: str) -> str | None:
        """Map an uploaded filename / book label to a known book, or None."""
        if name in self.book_to_idxs:
            return name
        return self._norm_to_book.get(_norm_book(name))

    def book_indices(self, book: str,
                     page_start: int | None = None,
                     page_end:   int | None = None) -> list[int]:
        """Chunk indices for a book, optionally limited to a page range (1-based)."""
        idxs = self.book_to_idxs.get(book, [])
        if page_start is None and page_end is None:
            return list(idxs)
        lo = page_start or 1
        hi = page_end   or 10**9
        return [i for i in idxs if lo <= self.chunks[i].get("page_num", 0) <= hi]

    # ── dense resources (lazy) ─────────────────────────────────────────────────

    def _ensure_dense(self) -> bool:
        if not self.dense_enabled:
            return False
        if self._model is not None and self._client is not None:
            return True
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    # Default to CPU so the 6 GB GPU stays free for the Ollama LLM.
                    # Query embedding is one short string at a time — CPU latency is fine.
                    device = os.getenv("QG_EMBED_DEVICE", "cpu")
                    print(f"[IndexStore] loading embedder {self.embed_model} on {device} ...")
                    self._model = SentenceTransformer(self.embed_model, device=device)
                except Exception as e:
                    print(f"[IndexStore] dense disabled (model load failed: {e})")
                    self.dense_enabled = False
                    return False
            if self._client is None:
                try:
                    from qdrant_client import QdrantClient
                    self._client = QdrantClient(path=QDRANT_PATH)
                except Exception as e:
                    print(f"[IndexStore] dense disabled (qdrant open failed: {e})")
                    self.dense_enabled = False
                    return False
        return True

    def embed_query(self, query: str):
        """Embed a search query (handles e5 'query:' prefix). Returns 1-D list or None."""
        if not self._ensure_dense():
            return None
        prefix = "query: " if "e5" in self.embed_model.lower() else ""
        vec = self._model.encode([prefix + query], normalize_embeddings=True)[0]
        return vec.tolist()

    def vectors_for(self, idxs: list[int]) -> dict[int, list[float]]:
        """Fetch stored dense vectors for the given chunk indices (point ids)."""
        if not self._ensure_dense() or not idxs:
            return {}
        recs = self._client.retrieve(
            collection_name=self.collection, ids=list(idxs), with_vectors=True
        )
        return {r.id: r.vector for r in recs if r.vector is not None}


# ── module singleton ───────────────────────────────────────────────────────────

_store: IndexStore | None = None
_store_lock = threading.Lock()


def get_store() -> IndexStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = IndexStore()
    return _store
