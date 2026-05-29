"""
Splits extracted page text into overlapping chunks with curriculum metadata.
Arabic-aware: splits on sentence boundaries before grouping into chunks.
"""
import re
from pathlib import Path
from typing import Iterator

from chatbot.config import CHUNK_SIZE, CHUNK_OVERLAP
from chatbot.indexing.normalizer import normalize

# Arabic + Latin sentence boundaries
_SENTENCE_RE = re.compile(r'(?<=[.!?؟।\n])\s+')

SUBJECT_MAP = {
    "algebra": "الجبر",
    "geometry": "الهندسة",
    "math1": "الرياضيات 1",
    "math2": "الرياضيات 2",
    "sci-math-1": "الرياضيات 1",
    "sci-math-2": "الرياضيات 2",
    "sci-physics": "الفيزياء",
    "physics-sci": "الفيزياء",
    "sci-chemistry": "الكيمياء",
    "chemistry-sci": "الكيمياء",
    "sci-science": "العلوم",
    "science": "العلوم",
    "sci-history": "التاريخ",
    "arabic": "اللغة العربية",
    "sci-arabic": "اللغة العربية",
    "arabic-sci": "اللغة العربية",
    "english-sb": "اللغة الإنجليزية",
    "english-wb": "اللغة الإنجليزية",
    "english-sci-sb": "اللغة الإنجليزية",
    "english-sci-wb": "اللغة الإنجليزية",
}


def _parse_metadata(pdf_path: Path) -> dict:
    stem = pdf_path.stem.lower()
    # strip leading grade prefix (e.g. "10-", "11-")
    key = re.sub(r"^\d+-", "", stem)
    grade = re.match(r"^(\d+)", pdf_path.stem)
    return {
        "grade": grade.group(1) if grade else "unknown",
        "subject": SUBJECT_MAP.get(key, stem),
        "source": pdf_path.name,
    }


def _sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_pages(pages: list[dict], pdf_path: Path) -> Iterator[dict]:
    """
    Yields chunk dicts with keys: text, normalized_text, grade, subject,
    source, page_num, chunk_index.
    """
    meta = _parse_metadata(pdf_path)
    chunk_index = 0
    buffer: list[str] = []
    buffer_pages: list[int] = []
    buffer_words = 0

    def _emit(sentences: list[str], pages: list[int]) -> dict:
        nonlocal chunk_index
        text = " ".join(sentences)
        result = {
            **meta,
            "text": text,
            "normalized_text": normalize(text),
            "page_num": pages[0] if pages else 0,
            "chunk_index": chunk_index,
        }
        chunk_index += 1
        return result

    for page in pages:
        for sent in _sentences(page["text"]):
            wc = _word_count(sent)
            if buffer_words + wc > CHUNK_SIZE and buffer:
                yield _emit(buffer, buffer_pages)
                # keep overlap
                overlap_buf, overlap_pages, overlap_words = [], [], 0
                for s, p in zip(reversed(buffer), reversed(buffer_pages)):
                    if overlap_words + _word_count(s) > CHUNK_OVERLAP:
                        break
                    overlap_buf.insert(0, s)
                    overlap_pages.insert(0, p)
                    overlap_words += _word_count(s)
                buffer, buffer_pages, buffer_words = overlap_buf, overlap_pages, overlap_words

            buffer.append(sent)
            buffer_pages.append(page["page_num"])
            buffer_words += wc

    if buffer:
        yield _emit(buffer, buffer_pages)
