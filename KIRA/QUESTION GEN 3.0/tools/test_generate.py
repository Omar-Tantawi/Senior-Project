"""
End-to-end generation test (offline, no HTTP server).

Builds a grounding source for a known book + page range, runs the full engine
(plan → grounded generate → verify/regenerate), evaluates the result, and writes
the questions to a UTF-8 JSON file so Arabic renders correctly.

Usage:
  python tools/test_generate.py --book 12-physics-Sci --pages 63-82 --mcq 3 --tf 2
  python tools/test_generate.py --book 10-arabic --mcq 3 --sa 2 --lang ar
"""

import os
import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from index_store import get_store
from retriever import CurriculumSource
from question_engine import QuestionEngine
from evaluator import evaluate_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--pages", default="")          # e.g. "63-82"
    ap.add_argument("--mcq", type=int, default=3)
    ap.add_argument("--tf",  type=int, default=2)
    ap.add_argument("--sa",  type=int, default=0)
    ap.add_argument("--fb",  type=int, default=0)
    ap.add_argument("--essay", type=int, default=0)
    ap.add_argument("--lang", default="auto")       # ar | en | auto
    ap.add_argument("--bloom", default="remember,understand,apply")
    ap.add_argument("--topic", default="")
    args = ap.parse_args()

    store = get_store()
    book = store.resolve_book(args.book)
    if not book:
        print("UNKNOWN book. Available:", [b["book"] for b in store.list_books()])
        sys.exit(1)

    ps = pe = None
    if args.pages:
        a, _, b = args.pages.partition("-")
        ps, pe = int(a), int(b or a)

    src = CurriculumSource(store, book, ps, pe)
    print(f"book={book} pages={args.pages or 'all'} candidate_chunks={len(src.indices)} "
          f"dense={'on' if store.dense_enabled else 'off'}")

    # resolve language
    lang = args.lang
    if lang == "auto":
        sample = " ".join(store.text(i) for i in src.indices[:8])[:1500]
        import re
        lang = "ar" if len(re.findall(r"[؀-ۿ]", sample)) / max(len(sample), 1) > 0.2 else "en"
    print("language:", lang)

    qlist = (["mcq"] * args.mcq + ["true_false"] * args.tf + ["short_answer"] * args.sa
             + ["fill_blank"] * args.fb + ["essay"] * args.essay)
    blooms = [b.strip() for b in args.bloom.split(",") if b.strip()]

    eng = QuestionEngine()
    t0 = time.time()
    questions = eng.generate(src, len(qlist), qlist, blooms, "medium", lang, args.topic)
    dt = time.time() - t0

    contexts = [q.pop("_context", "") for q in questions]
    result = evaluate_batch(questions, contexts=contexts, language=lang)
    summary = result["summary"]

    print(f"\n=== generated {len(questions)}/{len(qlist)} in {dt:.0f}s ===")
    print("quality:", json.dumps(summary, ensure_ascii=False))
    for i, q in enumerate(result["questions"], 1):
        qq = q.get("quality", {})
        ov = qq.get("checks", {}).get("relevance", {}).get("overlap")
        print(f"  Q{i} {q.get('type'):12} score={qq.get('score')} grade={qq.get('grade')} overlap={ov}")

    out = os.path.join(ROOT, "output",
                       f"test_{book}_{time.strftime('%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"book": book, "pages": args.pages, "language": lang,
                   "seconds": round(dt, 1), "summary": summary,
                   "questions": result["questions"]}, f, ensure_ascii=False, indent=2)
    print("wrote:", out)


if __name__ == "__main__":
    main()
