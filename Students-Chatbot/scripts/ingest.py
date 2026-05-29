"""
One-time ingestion script.
Reads PDFs from data/, extracts text, chunks, and builds the Qdrant + BM25 index.

Usage:
    python scripts/ingest.py              # all grades
    python scripts/ingest.py --grade 12   # Grade 12 only
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from chatbot.config import DATA_DIR
from chatbot.ocr.pipeline import extract_pages
from chatbot.indexing.chunker import chunk_pages
from chatbot.indexing.indexer import build_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grade", type=str, default=None,
                        help="Process only this grade folder (e.g. --grade 12)")
    args = parser.parse_args()

    if args.grade:
        grade_dir = DATA_DIR / f"Grade_{args.grade}"
        if not grade_dir.exists():
            print(f"Grade folder not found: {grade_dir}")
            sys.exit(1)
        pdfs = sorted(grade_dir.rglob("*.pdf"))
        print(f"Processing Grade {args.grade} only: {len(pdfs)} PDFs\n")
    else:
        pdfs = sorted(DATA_DIR.rglob("*.pdf"))
        print(f"Found {len(pdfs)} PDFs across {len(set(p.parent for p in pdfs))} grades.\n")

    if not pdfs:
        print(f"No PDFs found under {DATA_DIR}")
        sys.exit(1)

    all_chunks = []
    for pdf in tqdm(pdfs, desc="Processing PDFs"):
        pages = list(extract_pages(pdf))
        ocr_count = sum(1 for p in pages if p["used_ocr"])
        if ocr_count:
            print(f"  {pdf.name}: {ocr_count}/{len(pages)} pages via OCR fallback")
        chunks = list(chunk_pages(pages, pdf))
        all_chunks.extend(chunks)
        print(f"  {pdf.name}: {len(pages)} pages -> {len(chunks)} chunks")

    print(f"\nTotal chunks: {len(all_chunks)}")
    build_index(all_chunks)


if __name__ == "__main__":
    main()
