"""
Idea 2 analysis: for each equation book, find which PAGES are equation-garbled in
the current OCR (and would benefit from Marker), vs clean prose pages (keep OCR).

A page is flagged when its OCR text is symbol/digit heavy and word-poor — the
signature of mangled equations (e.g. '+7(-3) 4 + ^0, + $03', '4.5 >10 238').

Prints per-book counts at several thresholds + total Marker time estimate, and
writes the chosen page lists to qdrant_storage/equation_pages.json.
"""
import os
import re
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from index_store import get_store
from arabic_norm import tokenize
from tools.ingest_marker import EQUATION_BOOKS

SEC_PER_PAGE = 250        # measured ~220-350 s/page on the RTX 2060


def page_metrics(text: str) -> dict:
    nospace = re.sub(r"\s", "", text)
    n = len(nospace) or 1
    letters = len(re.findall(r"[A-Za-z؀-ۿ]", nospace))
    digits  = len(re.findall(r"\d|[٠-٩]", nospace))
    symbols = n - letters - digits
    toks = tokenize(text)
    realwords = [t for t in toks if not t.isdigit() and len(t) >= 3]
    return {
        "sym_ratio": symbols / n,
        "dig_ratio": digits / n,
        "garble":    (symbols + digits) / n,          # symbol+digit density
        "realwords": len(realwords),
        "chars":     len(nospace),
    }


def main():
    store = get_store()
    thresholds = [0.15, 0.20, 0.25]
    grand = {t: 0 for t in thresholds}
    grand_pages = 0
    chosen = {}                                        # book -> [pages] at 0.20
    CUT = 0.20

    print(f"{'book':20} {'pages':>5} " + " ".join(f">{t}" for t in thresholds))
    for book in EQUATION_BOOKS:
        idxs = store.book_indices(book)
        if not idxs:
            print(f"{book:20}  (not indexed)")
            continue
        by_page = {}
        for i in idxs:
            c = store.chunk(i)
            by_page.setdefault(c["page_num"], []).append(c["text"])
        page_scores = {}
        for pg, texts in by_page.items():
            joined = "\n".join(texts)
            if len(re.sub(r"\s", "", joined)) < 40:     # near-empty page → skip (likely image)
                page_scores[pg] = 1.0                    # treat as needs-marker (image/equation)
            else:
                page_scores[pg] = page_metrics(joined)["garble"]
        counts = {t: sum(1 for s in page_scores.values() if s >= t) for t in thresholds}
        for t in thresholds:
            grand[t] += counts[t]
        grand_pages += len(by_page)
        chosen[book] = sorted(pg for pg, s in page_scores.items() if s >= CUT)
        print(f"{book:20} {len(by_page):>5} " + " ".join(f"{counts[t]:>3}" for t in thresholds))

    print("-" * 50)
    print(f"{'TOTAL':20} {grand_pages:>5} " + " ".join(f"{grand[t]:>3}" for t in thresholds))
    print()
    for t in thresholds:
        hrs = grand[t] * SEC_PER_PAGE / 3600
        print(f"  flag>{t}:  {grand[t]:>4} pages  → ~{hrs:.1f} h of Marker  "
              f"({100*grand[t]/grand_pages:.0f}% of pages)")

    with open(os.path.join(ROOT, "qdrant_storage", "equation_pages.json"), "w", encoding="utf-8") as f:
        json.dump({"cut": CUT, "sec_per_page": SEC_PER_PAGE, "books": chosen}, f, ensure_ascii=False, indent=1)
    print(f"\nwrote equation_pages.json (cut={CUT}: "
          f"{sum(len(v) for v in chosen.values())} pages total)")


if __name__ == "__main__":
    main()
