"""
GPU batch scan of the math/physics textbooks with Marker, then rebuild the index.

Run this ONCE from the dedicated Marker venv (which has torch>=2.7 + CUDA):

    .marker_venv\\Scripts\\python tools\\marker_batch.py

What it does:
  1. For each math/physics book in Data/, run Marker (GPU) → output/marker/<book>.md
     (skips books whose .md already exists, so it's resumable).
  2. Calls ingest_marker.rebuild() to replace those books' garbled chunks with the
     clean text+LaTeX chunks and refresh their fingerprints.

After this, uploading any of those books (or requesting them by name) generates
questions from real equations — no re-scanning.
"""
import os
import sys
import time
import glob

# force offline so cached models are used and a slow network can't stall us
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTHONUTF8", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from ingest_marker import EQUATION_BOOKS, rebuild

DATA       = os.path.join(ROOT, "Data")
MARKER_DIR = os.path.join(ROOT, "output", "marker")


def _find_pdf(book: str) -> str | None:
    hits = glob.glob(os.path.join(DATA, "**", f"{book}.pdf"), recursive=True)
    return hits[0] if hits else None


def _convert(converter, pdf_path: str) -> str:
    from marker.output import text_from_rendered
    rendered = converter(pdf_path)
    text, _ext, _imgs = text_from_rendered(rendered)
    return text


def main():
    os.makedirs(MARKER_DIR, exist_ok=True)
    import torch
    print(f"[batch] torch {torch.__version__} cuda={torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("[batch] WARNING: CUDA not available — this will be very slow on CPU.")

    todo = []
    for book in EQUATION_BOOKS:
        out_md = os.path.join(MARKER_DIR, f"{book}.md")
        if os.path.exists(out_md):
            print(f"[batch] skip {book} (already scanned)")
            continue
        pdf = _find_pdf(book)
        if not pdf:
            print(f"[batch] WARN: PDF not found for {book}")
            continue
        todo.append((book, pdf, out_md))

    if todo:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.config.parser import ConfigParser
        print("[batch] loading Marker models (cached)...")
        cp = ConfigParser({"output_format": "markdown"})
        converter = PdfConverter(
            artifact_dict=create_model_dict(),
            config=cp.generate_config_dict(),
            processor_list=cp.get_processors(),
            renderer=cp.get_renderer(),
        )
        for book, pdf, out_md in todo:
            print(f"\n[batch] scanning {book} ...")
            t0 = time.time()
            try:
                md = _convert(converter, pdf)
            except Exception as e:
                print(f"[batch] FAILED {book}: {e}")
                continue
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"[batch] {book}: {len(md)} chars in {time.time()-t0:.0f}s → {out_md}")

    print("\n[batch] rebuilding index from markdown ...")
    rebuild()
    print("[batch] done.")


if __name__ == "__main__":
    main()
