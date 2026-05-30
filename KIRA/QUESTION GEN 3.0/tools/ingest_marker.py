"""
Ingest Marker markdown (text + LaTeX) into the curriculum index.

Converts each Marker .md into page-tagged chunks that KEEP equations ($...$, $$...$$)
and tables intact, then rebuilds bm25_v2.pkl by replacing the old (garbled) chunks
for the math/physics books with these clean ones. Also enriches those books'
content fingerprints (so re-saved copies match too).

Runs in the normal project env (needs rank_bm25 + the project modules):
  # ingest whatever .md files exist in output/marker/
  python tools/ingest_marker.py

  # just test the chunker on one file (no index changes)
  python tools/ingest_marker.py --test output/marker_physics_test.md
"""
import os
import re
import sys
import shutil
import pickle
import argparse
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from arabic_norm import tokenize
import fingerprint

QDRANT = os.path.join(ROOT, "qdrant_storage")
PKL    = os.path.join(QDRANT, "bm25_v2.pkl")
MARKER_DIR = os.path.join(ROOT, "output", "marker")

# Equation-heavy books to upgrade with Marker (math + physics + chemistry), grades
# 10-12. (names match the Data/ filenames / index book names)
EQUATION_BOOKS = [
    # math
    "10-Algebra", "10-Geometry", "11-math1", "11-math2", "12-sci-math-1", "12-sci-math-2",
    # physics
    "10-sci-physics", "11-sci-physics", "12-physics-Sci",
    # chemistry (redox, nuclear eqns, scientific notation — OCR mangles these too)
    "10-sci-chemistry", "11-sci-chemistry", "12-chemistry-Sci",
]

_PAGE_RE  = re.compile(r"_page_(\d+)_")
_IMG_RE   = re.compile(r"^\s*!\[\]\([^)]*\)\s*$")


def markdown_to_chunks(md: str, book: str, subject: str, grade,
                       target_words: int = 110) -> list[dict]:
    """Markdown → page-tagged chunks; equations and tables are kept as whole blocks."""
    lines = md.split("\n")
    blocks: list[tuple[str, int]] = []     # (text, page_num)
    buf: list[str] = []
    _first = _PAGE_RE.search(md)            # start at the first real page seen
    page = (int(_first.group(1)) + 1) if _first else 1
    i = 0

    def flush():
        if buf:
            t = "\n".join(buf).strip()
            if t:
                blocks.append((t, page))
            buf.clear()

    while i < len(lines):
        line = lines[i]
        m = _PAGE_RE.search(line)
        if m:
            page = int(m.group(1)) + 1          # marker page is 0-indexed
        if _IMG_RE.match(line):                  # drop standalone image refs
            i += 1
            continue
        s = line.strip()
        if s.startswith("$$"):                    # display equation (atomic)
            flush()
            eq = [line]
            if s.count("$$") < 2:
                i += 1
                while i < len(lines) and "$$" not in lines[i]:
                    eq.append(lines[i]); i += 1
                if i < len(lines):
                    eq.append(lines[i])
            blocks.append(("\n".join(eq).strip(), page)); i += 1; continue
        if s.startswith("|"):                     # markdown table (atomic)
            flush()
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i]); i += 1
            blocks.append(("\n".join(tbl).strip(), page)); continue
        if s.startswith("#"):                      # heading → boundary
            flush()
            blocks.append((s.lstrip("#").strip(), page)); i += 1; continue
        if not s:                                  # blank → soft boundary
            flush(); i += 1; continue
        buf.append(line); i += 1
    flush()

    # group blocks into ~target_words chunks (never split an equation/table block)
    chunks, cur, cur_words, cur_page = [], [], 0, page
    def emit():
        if cur:
            text = "\n\n".join(cur).strip()
            if len(text.split()) >= 4:
                chunks.append({"text": text, "book": book, "subject": subject,
                               "grade": grade, "page_num": cur_page, "used_ocr": True})
    for text, pg in blocks:
        w = len(text.split())
        if cur and cur_words + w > target_words:
            emit(); cur, cur_words = [], 0
        if not cur:
            cur_page = pg
        cur.append(text); cur_words += w
    emit()
    return chunks


def _subject_grade(store, book):
    idxs = store.book_to_idxs.get(book, [])
    if idxs:
        c = store.chunk(idxs[0])
        return c.get("subject"), c.get("grade")
    # fall back to parsing the book name: "12-physics-Sci" → grade 12, subject physics-Sci
    m = re.match(r"(\d+)-(.+)", book)
    return (m.group(2) if m else book), (m.group(1) if m else None)


def rebuild(books=EQUATION_BOOKS, marker_dir=MARKER_DIR):
    from rank_bm25 import BM25Okapi
    from index_store import get_store
    store = get_store()

    new_chunks, done = [], []
    for book in books:
        md_path = os.path.join(marker_dir, f"{book}.md")
        if not os.path.exists(md_path):
            print(f"  [skip] no markdown for {book}")
            continue
        subject, grade = _subject_grade(store, book)
        md = open(md_path, encoding="utf-8").read()
        ch = markdown_to_chunks(md, book, subject, grade)
        new_chunks += ch; done.append(book)
        print(f"  {book:18} → {len(ch):4} chunks  ({sum('$' in c['text'] for c in ch)} with equations)")

    if not done:
        print("nothing to ingest (no markdown files found).")
        return

    # PAGE-LEVEL merge (hybrid): for each upgraded book, replace ONLY the pages that
    # Marker actually covered; every other page keeps its existing OCR chunk. This
    # supports both full-book scans and selective (equation-pages-only) scans.
    with open(PKL, "rb") as f:
        cur = pickle.load(f)["chunks"]
    marker_pages = {}                       # book -> set(pages Marker produced)
    for c in new_chunks:
        marker_pages.setdefault(c["book"], set()).add(c["page_num"])
    kept = [c for c in cur
            if c["book"] not in marker_pages
            or c.get("page_num") not in marker_pages[c["book"]]]
    merged = kept + new_chunks
    replaced = len(cur) - len(kept)
    print(f"\nrebuilding BM25: kept {len(kept)} (replaced {replaced} OCR chunks) "
          f"+ {len(new_chunks)} Marker = {len(merged)} chunks")
    print(f"  pages upgraded per book: " +
          ", ".join(f"{b}:{len(p)}" for b, p in marker_pages.items()))

    tokenized = [tokenize(c["text"]) for c in merged]
    bm25 = BM25Okapi(tokenized)
    shutil.copy2(PKL, PKL + ".prebmarker.bak")
    with open(PKL, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": merged}, f)
    print(f"saved {os.path.basename(PKL)} (backup: bm25_v2.pkl.prebmarker.bak)")

    # enrich fingerprints for the upgraded books (rich tokens now)
    fps = fingerprint.load_store()
    for book in done:
        text = " ".join(c["text"] for c in new_chunks if c["book"] == book)
        toks = tokenize(text)
        if book in fps:
            fps[book]["tokens"]   = [t for t, _ in Counter(toks).most_common(300)]
            fps[book]["n_tokens"] = len(toks)
    fingerprint.save_store(fps)
    print(f"updated fingerprints for {len(done)} books")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="", help="chunk a single .md and print samples (no changes)")
    args = ap.parse_args()

    if args.test:
        path = args.test if os.path.isabs(args.test) else os.path.join(ROOT, args.test)
        md = open(path, encoding="utf-8").read()
        chunks = markdown_to_chunks(md, "TEST", "physics", "12")
        eq = sum(1 for c in chunks if "$" in c["text"])
        print(f"{len(chunks)} chunks, {eq} contain equations\n")
        for c in chunks[:6]:
            print(f"--- p{c['page_num']} ({len(c['text'].split())} words) ---")
            print(c["text"][:300]); print()
    else:
        rebuild()


if __name__ == "__main__":
    main()
