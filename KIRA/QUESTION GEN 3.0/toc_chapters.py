"""
toc_chapters.py — accurate chapters from each PDF's printed table of contents.

The PDFs have no embedded bookmarks, so we parse the فهرس/المحتويات/Contents page
(using PyMuPDF text, which is much cleaner than the OCR index), extract
(title, page-range), and convert the printed BOOK page numbers to PDF page indices
via a detected offset. Falls back to the keyword heuristic (chapters.detect_chapters)
when a book has no parseable ToC.
"""
import os
import re
import glob
import fitz

from chapters import detect_chapters as _heuristic

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "Data")

_TOC_KW   = re.compile(r"الفهرس|المحتويات|المحتوى|contents", re.IGNORECASE)
_INTRO_KW = re.compile(r"المقدمة|introduction", re.IGNORECASE)
# a ToC entry line ending in a page range "end - start" (RTL) or a single page
_RANGE_RE = re.compile(r"(.+?)(\d{1,3})\s*[-–—]\s*(\d{1,3})\s*$")
_SINGLE_RE = re.compile(r"(.+?)(\d{1,3})\s*$")


def _find_pdf(book: str) -> str | None:
    hits = glob.glob(os.path.join(DATA, "**", f"{book}.pdf"), recursive=True)
    return hits[0] if hits else None


def _toc_text(doc, scan: int = 14):
    for i in range(min(scan, len(doc))):
        t = doc[i].get_text("text")
        if _TOC_KW.search(t):
            txt = t
            j = i + 1
            while j < len(doc) and j < i + 3 and not _INTRO_KW.search(doc[j].get_text("text")[:120]):
                nxt = doc[j].get_text("text")
                if _RANGE_RE.search(nxt) or _TOC_KW.search(nxt):
                    txt += "\n" + nxt
                    j += 1
                else:
                    break
            return txt
    return None


def _clean_title(s: str) -> str:
    s = re.sub(r"^[\s\d.()\-–—:]+", "", s)          # strip leading week/numbers/punct
    s = re.sub(r"\s+", " ", s).strip(" .:-–—")
    return s


_UNIT_RE = re.compile(r"^(unit|chapter|lesson|الوحدة|الفصل|الدرس)\b", re.IGNORECASE)
_NUM_ONLY = re.compile(r"^\d{1,3}$")
_DOTTED   = re.compile(r"^(\d{1,3})[®.․‥…\s]{3,}(.+)$|^(.+?)[®.․‥…\s]{3,}(\d{1,3})$")


def _parse_range(toc_text: str):
    """Format A: 'title  end - start' (e.g. science) → full ranges."""
    entries = []
    for line in toc_text.splitlines():
        line = line.strip()
        if not line or _TOC_KW.search(line):
            continue
        m = _RANGE_RE.search(line)
        if m:
            title = _clean_title(m.group(1))
            a, b = int(m.group(2)), int(m.group(3))
            start, end = min(a, b), max(a, b)
            if len(title) >= 4 and end > start and (end - start) <= 60:
                entries.append((title, start, end))
    return entries


def _parse_starts(toc_text: str):
    """Formats B/C: start-page-only ('Unit N / title / page' or 'page…dots…title')."""
    lines = [l.strip() for l in toc_text.splitlines() if l.strip()]
    starts = []
    # B: English "Unit N" → title → page
    i = 0
    while i < len(lines):
        if _UNIT_RE.search(lines[i]):
            title, page, j = None, None, i + 1
            while j < len(lines) and j < i + 5:
                if _NUM_ONLY.match(lines[j]):
                    if title:
                        page = int(lines[j]); break
                elif not _UNIT_RE.search(lines[j]) and title is None:
                    title = lines[j]
                j += 1
            if title and page and len(title) >= 3:
                starts.append((title, page))
            i = j + 1
        else:
            i += 1
    # C: dotted "page®®®title" / "title®®®page"
    if len(starts) < 2:
        starts = []
        for l in lines:
            m = _DOTTED.match(l)
            if not m:
                continue
            if m.group(1):
                page, title = int(m.group(1)), m.group(2)
            else:
                title, page = m.group(3), int(m.group(4))
            title = _clean_title(title)
            if len(title) >= 4:
                starts.append((title, page))
    # ascending pages only, then compute end = next start - 1
    starts = [(t, p) for t, p in starts]
    out = []
    for k, (t, s) in enumerate(starts):
        e = (starts[k + 1][1] - 1) if k + 1 < len(starts) else s + 20
        if e >= s:
            out.append((t, s, e))
    return out


def _parse_entries(toc_text: str):
    """Return [(title, book_start, book_end)] — tries range format, then start-only."""
    entries = _parse_range(toc_text)
    if len(entries) >= 2:
        return entries
    return _parse_starts(toc_text)


def _detect_offset(doc, title: str, book_start: int) -> int:
    """Find the PDF page where this chapter actually begins → offset = pdf - book."""
    key = _clean_title(title).split("-")[0].split("،")[0].strip()[:14]
    if len(key) < 4:
        return 0
    # search a window around the expected page first, then the whole doc
    order = list(range(max(0, book_start - 3), min(len(doc), book_start + 12))) + list(range(len(doc)))
    for i in order:
        if key in re.sub(r"\s+", " ", doc[i].get_text("text")[:400]):
            return (i + 1) - book_start
    return 0


def toc_chapters(pdf_path: str):
    doc = fitz.open(pdf_path)
    try:
        text = _toc_text(doc)
        if not text:
            return None
        entries = _parse_entries(text)
        if len(entries) < 2:
            return None
        offset = _detect_offset(doc, entries[0][0], entries[0][1])
        last = len(doc)
        out = []
        for k, (title, s, e) in enumerate(entries):
            sp = max(1, s + offset)
            ep = min(last, e + offset)
            if ep < sp:
                ep = sp
            out.append({"index": k + 1, "title": title,
                        "start_page": sp, "end_page": ep})
        return out
    finally:
        doc.close()


def chapters_for_book(store, book: str) -> dict:
    """Best source of chapters for a book: ToC if parseable, else heuristic."""
    pdf = _find_pdf(book)
    if pdf:
        try:
            toc = toc_chapters(pdf)
            if toc and len(toc) >= 2:
                return {"book": book, "source": "toc", "chapters": toc}
        except Exception:
            pass
    return {"book": book, "source": "heuristic", "chapters": _heuristic(store, book)}
