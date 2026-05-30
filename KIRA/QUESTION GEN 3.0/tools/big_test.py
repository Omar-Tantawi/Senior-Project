"""
Big (exam-sized) generation test for one subject:
  generate mixed-type questions from a verified chapter range → evaluate →
  export to .docx → VALIDATE the .docx opens and contains the questions.

Usage:
  python tools/big_test.py --book 12-physics-Sci --pages 5-13 --lang ar \
      --title "اختبار الفيزياء" --mcq 6 --tf 4 --sa 2 --fb 2 --essay 1
"""
import os
import sys
import json
import time
import zipfile
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from index_store import get_store
from retriever import CurriculumSource
from question_engine import QuestionEngine
from evaluator import evaluate_batch
from docx_exporter import export_to_docx

OUT = os.path.join(ROOT, "output", "exams")

# Arabic/English titles kept here (UTF-8 in the file) so they aren't mangled by the shell.
TITLES = {
    "12-physics-Sci":     "اختبار الفيزياء — الحركة الاهتزازية التوافقية",
    "12-sci-math-1":      "اختبار الرياضيات — المتتاليات",
    "12-science":         "اختبار علم الأحياء — الجهاز العصبي",
    "12-chemistry-Sci":   "اختبار الكيمياء — النشاط الإشعاعي",
    "10-arabic":          "اختبار اللغة العربية",
    "12-English-Sci-Sb":  "English Exam — Learning for Life",
}


def validate_docx(path: str, questions) -> dict:
    """Open the .docx as a zip, confirm it's structurally valid and has content."""
    info = {"valid": False}
    try:
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            assert "word/document.xml" in names, "no document.xml"
            xml = z.read("word/document.xml").decode("utf-8")
        info["paragraphs"] = xml.count("<w:p ") + xml.count("<w:p>")
        info["rtl_marks"]  = xml.count("<w:bidi")
        # spot-check: at least one question's text is present in the doc
        sample = next((q.get("question", "")[:20] for q in questions if q.get("question")), "")
        info["sample_present"] = bool(sample) and sample in xml
        info["valid"] = info["paragraphs"] > 5 and info["sample_present"]
    except Exception as e:
        info["error"] = str(e)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True)
    ap.add_argument("--pages", default="")
    ap.add_argument("--lang", default="auto")
    ap.add_argument("--title", default="Exam")
    ap.add_argument("--mcq", type=int, default=6)
    ap.add_argument("--tf", type=int, default=4)
    ap.add_argument("--sa", type=int, default=2)
    ap.add_argument("--fb", type=int, default=2)
    ap.add_argument("--essay", type=int, default=1)
    ap.add_argument("--bloom", default="remember,understand,apply,analyze")
    args = ap.parse_args()

    store = get_store()
    book = store.resolve_book(args.book)
    ps = pe = None
    if args.pages:
        a, _, b = args.pages.partition("-"); ps, pe = int(a), int(b or a)
    src = CurriculumSource(store, book, ps, pe)

    lang = args.lang
    if lang == "auto":
        import re
        sample = " ".join(store.text(i) for i in src.indices[:8])[:1500]
        lang = "ar" if len(re.findall(r"[؀-ۿ]", sample)) / max(len(sample), 1) > 0.2 else "en"

    qlist = (["mcq"]*args.mcq + ["true_false"]*args.tf + ["short_answer"]*args.sa
             + ["fill_blank"]*args.fb + ["essay"]*args.essay)
    blooms = [b.strip() for b in args.bloom.split(",") if b.strip()]
    print(f"[big] {book} pages={args.pages or 'all'} lang={lang} "
          f"chunks={len(src.indices)} requesting {len(qlist)} questions")

    eng = QuestionEngine()
    t0 = time.time()
    questions = eng.generate(src, len(qlist), qlist, blooms, "medium", lang, "")
    dt = time.time() - t0

    contexts = [q.pop("_context", "") for q in questions]
    res = evaluate_batch(questions, contexts=contexts, language=lang)
    summary = res["summary"]
    print(f"[big] generated {len(questions)}/{len(qlist)} in {dt:.0f}s  quality={summary}")

    os.makedirs(OUT, exist_ok=True)
    base = f"{book}_{time.strftime('%H%M%S')}"
    title = TITLES.get(book, args.title)
    docx_bytes = export_to_docx(questions=res["questions"], title=title,
                                document_id=book, difficulty="medium", language=lang,
                                include_answers=True)
    docx_path = os.path.join(OUT, base + ".docx")
    open(docx_path, "wb").write(docx_bytes)
    val = validate_docx(docx_path, questions)
    print(f"[big] DOCX {os.path.basename(docx_path)}  valid={val['valid']}  {val}")

    json.dump({"book": book, "pages": args.pages, "lang": lang, "seconds": round(dt),
               "summary": summary, "docx": os.path.basename(docx_path), "docx_valid": val,
               "questions": res["questions"]},
              open(os.path.join(OUT, base + ".json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[big] wrote {base}.docx + .json")


if __name__ == "__main__":
    main()
