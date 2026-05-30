"""
chapters.py — detect chapter / unit boundaries per book from the indexed content.

Powers two things:
  • the frontend's chapter checkboxes (instead of typing page ranges), and
  • picking clean chapter-aligned page ranges (so a range never starts mid-chapter).

Detection is heuristic: a chunk whose start contains a unit/chapter cue
(Arabic الوحدة/الفصل/الباب/الدرس + ordinal, or English Unit/Chapter/Lesson + number)
marks a boundary at its page. Works on OCR text and is much cleaner on Marker text.
"""

import re

_AR_UNIT = r"(?:الوحدة|الوحده|الفصل|الباب|الدرس|الوحدةُ|الفصلُ)"
_AR_ORD  = (r"(?:الأولى|الاولى|الأولي|الثانية|الثانيه|الثالثة|الثالثه|الرابعة|الرابعه|"
            r"الخامسة|الخامسه|السادسة|السادسه|السابعة|السابعه|الثامنة|الثامنه|"
            r"التاسعة|التاسعه|العاشرة|العاشره|الحادية|الثانية عشرة|\d+)")
_EN_UNIT = r"(?:unit|chapter|lesson)"

_AR_RE = re.compile(_AR_UNIT + r"\s*" + _AR_ORD)
_EN_RE = re.compile(_EN_UNIT + r"\s+\d+", re.IGNORECASE)

_AR_ORD_NUM = {
    "الاولى": 1, "الأولى": 1, "الأولي": 1, "الاولي": 1,
    "الثانية": 2, "الثانيه": 2, "الثالثة": 3, "الثالثه": 3,
    "الرابعة": 4, "الرابعه": 4, "الخامسة": 5, "الخامسه": 5,
    "السادسة": 6, "السادسه": 6, "السابعة": 7, "السابعه": 7,
    "الثامنة": 8, "الثامنه": 8, "التاسعة": 9, "التاسعه": 9,
    "العاشرة": 10, "العاشره": 10,
}


def _chapter_cue(text: str):
    """Return (title, ordinal_number | None) if the text starts a unit/chapter."""
    head = re.sub(r"[#*_>|]", " ", text.strip())[:70]
    m = _AR_RE.search(head)
    if m:
        cue = m.group(0)
        for word, num in _AR_ORD_NUM.items():
            if word in cue:
                return cue.strip(), num
        d = re.search(r"\d+", cue)
        return cue.strip(), (int(d.group()) if d else None)
    m = _EN_RE.search(head)
    if m:
        d = re.search(r"\d+", m.group(0))
        return m.group(0).strip(), (int(d.group()) if d else None)
    return None, None


def detect_chapters(store, book: str) -> list[dict]:
    """Return [{index, title, start_page, end_page}] for a book (best-effort)."""
    idxs = store.book_indices(book)               # page-ordered
    if not idxs:
        return []
    pages = sorted({store.chunk(i)["page_num"] for i in idxs})
    last_page = pages[-1]

    boundaries: list[tuple[int, str]] = []
    last_ord = None
    for i in idxs:
        c = store.chunk(i)
        cue, ordn = _chapter_cue(c["text"])
        if not cue:
            continue
        pg = c["page_num"]
        # New boundary only when the unit/chapter number actually CHANGES (a repeated
        # running header keeps the same ordinal → collapsed into one chapter).
        if ordn is not None and ordn != last_ord and (not boundaries or pg > boundaries[-1][0]):
            boundaries.append((pg, f"{cue}"))
            last_ord = ordn

    # Fallback: if we couldn't find headings, split into ~6 equal page bands.
    if len(boundaries) < 2:
        n = 6
        span = max(1, (last_page - pages[0] + 1) // n)
        boundaries = [(pages[0] + k * span, f"Part {k+1}") for k in range(n)
                      if pages[0] + k * span <= last_page]

    chapters = []
    for k, (pg, title) in enumerate(boundaries):
        end = (boundaries[k + 1][0] - 1) if k + 1 < len(boundaries) else last_page
        if end < pg:
            end = pg
        chapters.append({"index": k + 1, "title": title,
                         "start_page": pg, "end_page": end})
    return chapters


def pages_for_chapters(store, book: str, chapter_indices: list[int]) -> tuple[int, int] | None:
    """Map selected chapter indices to a (min_start, max_end) page range."""
    chs = {c["index"]: c for c in detect_chapters(store, book)}
    picked = [chs[i] for i in chapter_indices if i in chs]
    if not picked:
        return None
    return min(c["start_page"] for c in picked), max(c["end_page"] for c in picked)
