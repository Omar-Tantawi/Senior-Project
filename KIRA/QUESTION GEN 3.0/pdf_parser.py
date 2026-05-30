"""
PDF / text extraction with Arabic and English support.

Used ONLY for unknown uploads that are NOT already in the curriculum index
(known textbooks reuse the prebuilt chunks and skip this entirely).

Key fixes vs. the old version
  • No more blind word-order reversal of Arabic lines. PyMuPDF already returns
    logical reading order; reversing it scrambled the text and produced nonsense
    questions. We only clean whitespace now.
  • Page-aware, sentence-aware chunking (~100 words) so each chunk carries its
    page number and stays on one coherent idea — matching the prebuilt index.
"""

import re
import numpy as np
import fitz   # PyMuPDF
from langdetect import detect


# ── Arabic detection ────────────────────────────────────────────────────────────

def _is_arabic(text: str) -> bool:
    arabic_chars = len(re.findall(r'[؀-ۿ]', text))
    return arabic_chars / max(len(text), 1) > 0.3


def _clean_text(text: str) -> str:
    """Whitespace cleanup only — NO word reordering (that corrupts Arabic)."""
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned = [ln for ln in lines if ln]
    result = "\n".join(cleaned)
    return re.sub(r'\n{3,}', '\n\n', result).strip()


# ── Sentence-aware chunking ──────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r'(?<=[.!?؟۔\n])\s+')


def _chunk_page(text: str, target_words: int = 100, min_words: int = 12) -> list[str]:
    """Split one page into ~target_words chunks on sentence boundaries (1-sentence overlap)."""
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return []

    chunks, cur, count = [], [], 0
    for sent in sentences:
        w = len(sent.split())
        if count + w > target_words and cur:
            chunks.append(" ".join(cur))
            cur = cur[-1:] if len(cur) > 1 else []   # carry last sentence as overlap
            count = sum(len(s.split()) for s in cur)
        cur.append(sent)
        count += w
    if cur:
        chunks.append(" ".join(cur))

    # merge a tiny trailing chunk into the previous one
    if len(chunks) > 1 and len(chunks[-1].split()) < min_words:
        chunks[-2] = chunks[-2] + " " + chunks.pop()
    return [c for c in chunks if len(c.split()) >= 4]


# ── OCR (lazy) ────────────────────────────────────────────────────────────────

_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("[PDFParser] Loading OCR model (first time only)...")
        _ocr_reader = easyocr.Reader(['ar', 'en'], gpu=True, verbose=False)
        print("[PDFParser] OCR model ready.")
    return _ocr_reader


def _ocr_page(page: fitz.Page) -> str:
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    results = _get_ocr_reader().readtext(img, detail=0, paragraph=True)
    return " ".join(results)


_TEXT_WORD_THRESHOLD = 20    # fewer words than this on a page ⇒ treat as image ⇒ OCR


class PDFParser:
    def parse(
        self,
        file_bytes: bytes,
        filename:   str = "",
        page_start: int | None = None,
        page_end:   int | None = None,
        use_ocr:    bool = True,
    ) -> dict:
        """
        Extract text from a PDF, optionally limited to a 1-based page range.

        Returns:
          {
            full_text, chunks: [{text, page_num, used_ocr}], pages, pages_used,
            language, filename, ocr_pages
          }
        """
        doc         = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)

        start_idx = (page_start - 1) if page_start else 0
        end_idx   = (page_end)       if page_end   else total_pages
        start_idx = max(0, min(start_idx, total_pages - 1))
        end_idx   = max(start_idx + 1, min(end_idx, total_pages))

        chunks: list[dict] = []
        page_text_parts: list[str] = []
        ocr_count = 0

        for page_num in range(start_idx, end_idx):
            page = doc[page_num]
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES)
            used_ocr = False
            if len(text.split()) < _TEXT_WORD_THRESHOLD and use_ocr:
                print(f"[PDFParser] Page {page_num + 1}: image detected, running OCR...")
                text = _ocr_page(page)
                used_ocr = True
                ocr_count += 1

            text = _clean_text(text)
            page_text_parts.append(text)
            for ch in _chunk_page(text):
                chunks.append({"text": ch, "page_num": page_num + 1, "used_ocr": used_ocr})

        doc.close()

        full_text = _clean_text("\n\n".join(page_text_parts))
        if not full_text.strip():
            raise ValueError(
                "Could not extract text from the selected pages. "
                "If these are scanned images, make sure OCR is enabled."
            )

        return {
            "full_text":  full_text,
            "chunks":     chunks,
            "pages":      total_pages,
            "pages_used": (end_idx - start_idx),
            "language":   self._detect_language(full_text),
            "filename":   filename,
            "ocr_pages":  ocr_count,
        }

    def quick_text(self, file_bytes: bytes, max_pages: int = 4) -> str:
        """Fast text-only read of the first pages (no OCR) — used for book fingerprinting."""
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        parts = []
        for i in range(min(max_pages, len(doc))):
            parts.append(doc[i].get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES))
        doc.close()
        return _clean_text("\n".join(parts))

    def _detect_language(self, text: str) -> str:
        sample        = text[:2000]
        arabic_ratio  = len(re.findall(r'[؀-ۿ]', sample)) / max(len(sample), 1)
        english_ratio = len(re.findall(r'[a-zA-Z]', sample))        / max(len(sample), 1)
        if arabic_ratio > 0.3 and english_ratio > 0.2:
            return "mixed"
        if arabic_ratio > 0.2:
            return "ar"
        if english_ratio > 0.2:
            return "en"
        try:
            return detect(text[:1000])
        except Exception:
            return "unknown"
