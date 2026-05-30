"""
OCR + index the Data/ PDFs that are NOT yet in the curriculum index
(religion / philosophy / french / geography / technology / class-10 English…).

These are purely additive — new books, nothing replaced. Parses each with the
fixed pdf_parser (PyMuPDF text + EasyOCR fallback for image pages), tags
book/subject/grade, then refits BM25 and refreshes fingerprints.

  python tools/index_remaining.py
"""
import os
import sys
import glob
import shutil
import pickle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pdf_parser import PDFParser
from index_store import get_store
from arabic_norm import tokenize
import fingerprint

PKL  = os.path.join(ROOT, "qdrant_storage", "bm25_v2.pkl")
DATA = os.path.join(ROOT, "Data")


def main():
    store = get_store()
    parser = PDFParser()
    pdfs = glob.glob(os.path.join(DATA, "**", "*.pdf"), recursive=True)
    todo = [p for p in sorted(pdfs)
            if not store.resolve_book(os.path.splitext(os.path.basename(p))[0])]
    print(f"{len(todo)} unindexed PDFs to process")

    new_chunks = []
    fps = fingerprint.load_store()
    for path in todo:
        book = os.path.splitext(os.path.basename(path))[0]
        grade = book.split("-")[0]
        subject = book.split("-", 1)[1] if "-" in book else book
        try:
            data = open(path, "rb").read()
            parsed = parser.parse(data, book + ".pdf")
        except Exception as e:
            print(f"  [skip] {book}: {e}")
            continue
        n = 0
        for ch in parsed["chunks"]:
            ch.update({"book": book, "subject": subject, "grade": grade})
            new_chunks.append(ch); n += 1
        # refresh fingerprint tokens from the freshly parsed text
        try:
            fps[book] = fingerprint.fingerprint(data)
            fps[book]["file"] = os.path.relpath(path, ROOT)
        except Exception:
            pass
        print(f"  {book:22} {n:4} chunks  (ocr_pages={parsed['ocr_pages']}, lang={parsed['language']})")

    if not new_chunks:
        print("nothing added."); return

    from rank_bm25 import BM25Okapi
    with open(PKL, "rb") as f:
        cur = pickle.load(f)["chunks"]
    merged = cur + new_chunks
    print(f"\nrebuilding BM25: {len(cur)} + {len(new_chunks)} = {len(merged)} chunks")
    bm25 = BM25Okapi([tokenize(c["text"]) for c in merged])
    shutil.copy2(PKL, PKL + ".preremaining.bak")
    with open(PKL, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": merged}, f)
    fingerprint.save_store(fps)
    print(f"saved bm25_v2.pkl + fingerprints  (backup: bm25_v2.pkl.preremaining.bak)")


if __name__ == "__main__":
    main()
