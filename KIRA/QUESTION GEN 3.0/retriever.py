"""
retriever.py — book-scoped hybrid retrieval + grounding sources.

The core relevance fix: questions can only be grounded in CHUNKS THAT BELONG TO
THE SUBJECT the teacher uploaded — never a random chunk from another book.

Two grounding sources expose the same small interface to the question engine:
  • CurriculumSource — a known textbook already in the prebuilt index (hybrid BM25⊕dense)
  • MemSource        — an unknown uploaded PDF, indexed in-memory (BM25 only)

Interface used by question_engine:
  .book                      label for logging / titles
  .indices                   candidate chunk indices
  .text(i) -> str            chunk text
  .hybrid(query, k) -> [i]   best chunks for a query
  .even_coverage(n) -> [i]   query-free spread across the whole source
  .context_for(i) -> str     chunk + adjacent neighbors, capped for the prompt
  .outline_samples(m) -> [str]  lead snippets sampled across the source (for LLM outline)
"""

import re
import numpy as np
from rank_bm25 import BM25Okapi

from arabic_norm import tokenize
from index_store import IndexStore

_RRF_K = 60


# ── shared scoring helpers ──────────────────────────────────────────────────────

def _informativeness(text: str) -> float:
    """Content richness: real word tokens, penalized for symbol/equation noise."""
    toks = tokenize(text)
    if not toks:
        return 0.0
    words = [t for t in toks if not t.isdigit()]
    word_ratio = len(words) / len(toks)
    symbols = len(re.findall(r"[^\w\s؀-ۿ]", text))
    sym_penalty = min(1.0, symbols / max(len(text), 1) * 4)
    return len(words) * word_ratio * (1 - 0.5 * sym_penalty)


def _pick_even(indices, text_fn, n: int) -> list[int]:
    """Split informative chunks into n contiguous segments; pick the richest in each."""
    scored = [(i, _informativeness(text_fn(i))) for i in indices]
    pool = [i for i, s in scored if s >= 8] or [i for i, _ in scored]
    if not pool:
        return []
    if n >= len(pool):
        return pool
    seg, out = len(pool) / n, []
    for k in range(n):
        lo = int(k * seg)
        hi = int((k + 1) * seg) if k < n - 1 else len(pool)
        window = pool[lo:hi] or [pool[min(lo, len(pool) - 1)]]
        out.append(max(window, key=lambda i: _informativeness(text_fn(i))))
    seen, uniq = set(), []
    for i in out:
        if i not in seen:
            seen.add(i); uniq.append(i)
    return uniq


def _sample_snippets(indices, text_fn, m: int, n_chars: int = 180) -> list[str]:
    """Evenly sample m chunks; return their leading text (for the LLM topic outline)."""
    if not indices:
        return []
    step = max(1, len(indices) // m)
    out = []
    for i in indices[::step][:m]:
        snippet = " ".join(text_fn(i).split())[:n_chars]
        if snippet:
            out.append(snippet)
    return out


# ── low-level retriever over the global curriculum store ────────────────────────

class Retriever:
    def __init__(self, store: IndexStore):
        self.store = store

    def bm25_search(self, query: str, allowed: list[int], top_k: int) -> list[int]:
        toks = tokenize(query)
        if not toks or not allowed:
            return []
        scores = self.store.bm25.get_scores(toks)
        ranked = sorted(allowed, key=lambda i: scores[i], reverse=True)
        positive = [i for i in ranked if scores[i] > 0]
        return (positive or ranked)[:top_k]

    def dense_search(self, query: str, allowed: list[int], top_k: int) -> list[int]:
        qvec = self.store.embed_query(query)
        if qvec is None or not allowed:
            return []
        vecs = self.store.vectors_for(allowed)
        if not vecs:
            return []
        q = np.asarray(qvec, dtype="float32")
        qn = np.linalg.norm(q) or 1.0
        sims = {i: float(q @ np.asarray(v, dtype="float32") /
                         (qn * (np.linalg.norm(v) or 1.0)))
                for i, v in vecs.items()}
        return sorted(sims, key=sims.get, reverse=True)[:top_k]

    def hybrid(self, query: str, allowed: list[int], top_k: int = 3) -> list[int]:
        pool = max(top_k * 3, 10)
        lex = self.bm25_search(query, allowed, pool)
        sem = self.dense_search(query, allowed, pool)
        if not sem:
            return lex[:top_k]
        if not lex:
            return sem[:top_k]
        rrf: dict[int, float] = {}
        for ranklist in (lex, sem):
            for rank, idx in enumerate(ranklist):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (_RRF_K + rank)
        return sorted(rrf, key=rrf.get, reverse=True)[:top_k]


# ── grounding sources (what the engine talks to) ────────────────────────────────

class CurriculumSource:
    """A known textbook already in the prebuilt hybrid index."""
    kind = "curriculum"

    def __init__(self, store: IndexStore, book: str,
                 page_start: int | None = None, page_end: int | None = None):
        self.store = store
        self.retriever = Retriever(store)
        self.book = book
        self.indices = store.book_indices(book, page_start, page_end)
        self._book_order = store.book_indices(book)      # full order, for neighbors

    def text(self, i: int) -> str:
        return self.store.text(i)

    def hybrid(self, query: str, k: int = 3) -> list[int]:
        return self.retriever.hybrid(query, self.indices, k)

    def even_coverage(self, n: int) -> list[int]:
        return _pick_even(self.indices, self.text, n)

    def outline_samples(self, m: int) -> list[str]:
        return _sample_snippets(self.indices, self.text, m)

    def context_for(self, idx: int, neighbor_radius: int = 1, max_chars: int = 2400) -> str:
        order = self._book_order
        if idx in order:
            pos = order.index(idx)
            window = order[max(0, pos - neighbor_radius): pos + neighbor_radius + 1]
        else:
            window = [idx]
        ordered = [idx] + [i for i in window if i != idx]
        parts, total = [], 0
        for i in ordered:
            t = self.text(i).strip()
            if not t:
                continue
            if total + len(t) > max_chars and parts:
                break
            parts.append(t); total += len(t)
        return "\n\n".join(parts)


class MemSource:
    """An unknown uploaded PDF — indexed in-memory (BM25 only)."""
    kind = "memory"

    def __init__(self, chunks: list[dict], book: str = "uploaded"):
        self.book = book
        self.texts = [c["text"] for c in chunks]
        self._tok = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self._tok) if self.texts else None
        self.indices = list(range(len(self.texts)))

    def text(self, i: int) -> str:
        return self.texts[i]

    def hybrid(self, query: str, k: int = 3) -> list[int]:
        toks = tokenize(query)
        if not toks or self.bm25 is None:
            return self.indices[:k]
        scores = self.bm25.get_scores(toks)
        ranked = sorted(self.indices, key=lambda i: scores[i], reverse=True)
        positive = [i for i in ranked if scores[i] > 0]
        return (positive or ranked)[:k]

    def even_coverage(self, n: int) -> list[int]:
        return _pick_even(self.indices, self.text, n)

    def outline_samples(self, m: int) -> list[str]:
        return _sample_snippets(self.indices, self.text, m)

    def context_for(self, idx: int, neighbor_radius: int = 1, max_chars: int = 2400) -> str:
        lo = max(0, idx - neighbor_radius)
        hi = min(len(self.texts), idx + neighbor_radius + 1)
        ordered = [idx] + [i for i in range(lo, hi) if i != idx]
        parts, total = [], 0
        for i in ordered:
            t = self.texts[i].strip()
            if not t:
                continue
            if total + len(t) > max_chars and parts:
                break
            parts.append(t); total += len(t)
        return "\n\n".join(parts)
