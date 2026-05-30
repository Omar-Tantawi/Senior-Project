"""
Build the content-fingerprint store from the Data/ PDFs.

For every PDF under Data/, compute its fingerprint (sha256 + text-hash + tokens)
and map it to the canonical book name (matched to the curriculum index when
possible, else the filename stem). Saves qdrant_storage/fingerprints.json.

Run once, and again whenever you add new PDFs:
  python tools/build_fingerprints.py
"""
import os
import sys
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fingerprint as fp
from index_store import get_store

DATA = os.path.join(ROOT, "Data")


def main():
    store = get_store()
    out = fp.load_store()
    pdfs = glob.glob(os.path.join(DATA, "**", "*.pdf"), recursive=True)
    print(f"found {len(pdfs)} PDFs under Data/")

    for path in sorted(pdfs):
        stem = os.path.splitext(os.path.basename(path))[0]
        book = store.resolve_book(stem) or stem      # canonical name if in index
        try:
            with open(path, "rb") as f:
                data = f.read()
            f_print = fp.fingerprint(data)
        except Exception as e:
            print(f"  [skip] {stem}: {e}")
            continue
        f_print["file"] = os.path.relpath(path, ROOT)
        out[book] = f_print
        in_index = "in-index" if store.resolve_book(stem) else "NOT-in-index"
        print(f"  {book:24} {in_index:12} tokens={f_print['n_tokens']:5} sha={f_print['sha256'][:10]}")

    fp.save_store(out)
    print(f"\nwrote {len(out)} fingerprints → {fp.FP_PATH}")


if __name__ == "__main__":
    main()
