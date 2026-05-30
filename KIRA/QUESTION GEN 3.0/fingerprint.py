"""
fingerprint.py — content-based identification of an uploaded PDF.

Goal: recognize a PDF we've ALREADY indexed even if the teacher renamed it.
Three signals, strongest first:

  1. sha256(file bytes)      → exact same file, any name        (instant, 100% sure)
  2. text_hash              → same text content, re-saved PDF   (rename + re-export safe)
  3. token Jaccard          → near-duplicate / partial match    (robust fallback)

The known-books store (qdrant_storage/fingerprints.json) is built once from the
Data/ PDFs by tools/build_fingerprints.py and maps each canonical book name to
its fingerprint.
"""

import os
import re
import json
import hashlib
from collections import Counter

import fitz  # PyMuPDF

from arabic_norm import tokenize, normalize_text

FP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "qdrant_storage", "fingerprints.json")

_TOP_TOKENS = 300          # how many frequent tokens to keep per document
_JACCARD_MATCH = 0.55      # token-overlap threshold for a fuzzy match
_MIN_FUZZY_TOKENS = 120    # below this (image-based covers) fuzzy match is unreliable


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _first_pages_text(data: bytes, n: int = 6) -> str:
    """Fast text-only read of the first n pages (no OCR)."""
    doc = fitz.open(stream=data, filetype="pdf")
    parts = [doc[i].get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
             for i in range(min(n, len(doc)))]
    doc.close()
    return "\n".join(parts)


def fingerprint(data: bytes) -> dict:
    """Compute the fingerprint of a PDF (bytes)."""
    raw = _first_pages_text(data)
    norm = re.sub(r"\s+", " ", normalize_text(raw.lower())).strip()
    toks = tokenize(raw)
    top = [t for t, _ in Counter(toks).most_common(_TOP_TOKENS)]
    return {
        "sha256":    file_sha256(data),
        "text_hash": hashlib.sha256(norm.encode("utf-8")).hexdigest(),
        "tokens":    top,
        "n_tokens":  len(toks),
    }


def _jaccard(a: list, b: list) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def match(fp: dict, known: dict) -> tuple[str | None, float, str]:
    """
    Match a fingerprint against the known-books store.
    Returns (book | None, score, method).
    """
    # 1. exact bytes
    for book, kfp in known.items():
        if kfp.get("sha256") and kfp["sha256"] == fp["sha256"]:
            return book, 1.0, "sha256"
    # 2. identical extracted text
    for book, kfp in known.items():
        if kfp.get("text_hash") and kfp["text_hash"] == fp["text_hash"]:
            return book, 1.0, "text_hash"
    # 3. fuzzy token overlap — only when BOTH sides have enough real text.
    # (Image-based covers yield ~30 boilerplate tokens shared across books, which
    #  would false-match; sha256/text_hash above already handle the rename case.)
    if fp.get("n_tokens", 0) < _MIN_FUZZY_TOKENS:
        return None, 0.0, "none"
    best, best_score = None, 0.0
    for book, kfp in known.items():
        if kfp.get("n_tokens", 0) < _MIN_FUZZY_TOKENS:
            continue
        s = _jaccard(fp["tokens"], kfp.get("tokens", []))
        if s > best_score:
            best, best_score = book, s
    if best and best_score >= _JACCARD_MATCH:
        return best, round(best_score, 3), "jaccard"
    return None, round(best_score, 3), "none"


def load_store(path: str = FP_PATH) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_store(store: dict, path: str = FP_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)
