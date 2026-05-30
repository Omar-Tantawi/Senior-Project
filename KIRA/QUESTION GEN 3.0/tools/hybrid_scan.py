"""
Selective Marker scan (Idea 2): re-do ONLY the equation-garbled pages of a book
(within an optional page range), so OCR stays for prose and Marker fixes equations.

Reads the flagged pages from qdrant_storage/equation_pages.json (built by
flag_equation_pages.py), runs Marker on just those pages, and writes
output/marker/<book>.md. Then run `python tools/ingest_marker.py` to merge
(page-level) into the index.

Run from the Marker venv (GPU):
  .marker_venv\\Scripts\\python tools\\hybrid_scan.py --book 12-physics-Sci --pages 63-92 --max 12
"""
import os
import sys
import json
import time
import argparse

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import glob

EQ_JSON    = os.path.join(ROOT, "qdrant_storage", "equation_pages.json")
MARKER_DIR = os.path.join(ROOT, "output", "marker")
DATA       = os.path.join(ROOT, "Data")


def _compress(nums) -> str:
    nums = sorted(set(nums))
    out, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}-{nums[j]}" if j > i else f"{nums[i]}")
        i = j + 1
    return ",".join(out)


def _find_pdf(book: str) -> str | None:
    hits = glob.glob(os.path.join(DATA, "**", f"{book}.pdf"), recursive=True)
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--pages", default="", help="1-indexed range 'A-B' to limit to (optional)")
    ap.add_argument("--max", type=int, default=12, help="cap number of pages (bounds GPU time)")
    args = ap.parse_args()

    flagged = json.load(open(EQ_JSON, encoding="utf-8"))["books"].get(args.book, [])
    if args.pages:
        a, _, b = args.pages.partition("-")
        a, b = int(a), int(b or a)
        flagged = [p for p in flagged if a <= p <= b]
    sel = sorted(flagged)[:args.max]
    if not sel:
        print(f"[hybrid] no flagged equation pages for {args.book} in range {args.pages or 'all'}")
        return
    page_range = _compress(p - 1 for p in sel)          # marker uses 0-indexed
    print(f"[hybrid] {args.book}: {len(sel)} equation pages → {sel}")
    print(f"[hybrid] marker page_range(0-idx)= {page_range}")

    pdf = _find_pdf(args.book)
    if not pdf:
        print(f"[hybrid] PDF not found for {args.book}"); return

    import torch
    print(f"[hybrid] torch {torch.__version__} cuda={torch.cuda.is_available()}")
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser
    from marker.output import text_from_rendered

    cp = ConfigParser({"output_format": "markdown", "page_range": page_range})
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config=cp.generate_config_dict(),
        processor_list=cp.get_processors(),
        renderer=cp.get_renderer(),
    )
    t0 = time.time()
    text, _ext, _imgs = text_from_rendered(converter(pdf))
    dt = time.time() - t0

    os.makedirs(MARKER_DIR, exist_ok=True)
    out = os.path.join(MARKER_DIR, f"{args.book}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)
    eqs = text.count("$") // 2
    print(f"[hybrid] {args.book}: scanned {len(sel)} pages in {dt:.0f}s "
          f"({dt/len(sel):.0f}s/pg), ~{eqs} equations → {out}")


if __name__ == "__main__":
    main()
