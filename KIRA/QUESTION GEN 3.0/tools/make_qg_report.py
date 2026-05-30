"""
Regenerate the Question Generator feature report (Word) reflecting the rebuilt
architecture: grounded BM25 retrieval, OCR+Marker hybrid for equations, chapter
selection, content fingerprinting, verify/regenerate, and .docx export.

Saves D:\\Senior\\reports\\report_question_generator.docx
"""
import os
import sys

sys.path.insert(0, r"D:\Senior")                       # reuse the shared report helpers
from generate_reports import (                          # noqa: E402
    new_doc, cover_page, section_heading, body, bullet,
    kv_table, results_table, spacer, OUT_DIR,
)


def main():
    doc = new_doc()

    cover_page(
        doc,
        "Question Generator",
        "AI-Powered Exam Question Generation from the Curriculum (Arabic & English)",
        "8002",
        "FastAPI · Ollama (Command-R7B-Arabic) · BM25 · Marker/Surya · python-docx",
    )

    # 1. Overview
    section_heading(doc, "1. Overview")
    body(doc, "The Question Generator is an AI microservice that automatically produces "
              "exam questions from the official curriculum textbooks (grades 10–12), in both "
              "Arabic and English. A teacher selects a subject and the chapters they want, "
              "chooses how many questions of each type, and receives a ready-to-print Word "
              "exam with an answer key — generated entirely on a local machine, with no "
              "internet or paid API required.")
    body(doc, "The system generates five question types — Multiple Choice, True/False, Short "
              "Answer, Fill-in-the-Blank, and Essay — across all six levels of Bloom's "
              "taxonomy, at three difficulty levels.")

    # 2. The problem and the approach
    section_heading(doc, "2. Problem & Approach")
    body(doc, "A naïve generator that simply asks a language model to 'write questions about "
              "this subject' produces vague, off-topic, or invented content. The core design "
              "principle here is GROUNDING: every question is tied to a specific passage of the "
              "real textbook, and is verified against that passage before being accepted.")
    bullet(doc, "The full curriculum is pre-indexed once (text extracted, chunked, searchable).",
           "Pre-indexing")
    bullet(doc, "For each question, the system retrieves the most relevant textbook passage and "
                "instructs the model to use only that passage.", "Retrieval-grounded")
    bullet(doc, "Each generated question is scored for relevance; weak ones are regenerated.",
           "Verify & retry")

    # 3. Architecture / pipeline
    section_heading(doc, "3. System Architecture")
    body(doc, "End-to-end pipeline for a single request:")
    for step in [
        "Teacher picks a subject + chapters (or uploads a PDF).",
        "The system identifies the book — by name, or by content fingerprint if the file was "
        "renamed — and reuses the existing index (no re-scanning).",
        "It retrieves grounding passages from the curriculum index, scoped to the chosen "
        "book and chapters (BM25 keyword search).",
        "It plans the exam: the model lists the real topics present, and an even-coverage "
        "selector guarantees questions span the whole selection.",
        "For each slot it builds a grounded prompt (the passage + the topic) and the local "
        "model writes one question as structured JSON.",
        "An evaluator scores each question (structure, answer validity, relevance 0–100); "
        "weak questions are regenerated once.",
        "The questions are exported to a formatted Word document (right-to-left for Arabic, "
        "left-to-right for English) with an optional answer key.",
    ]:
        bullet(doc, step)

    # 4. Curriculum index
    section_heading(doc, "4. Curriculum Index", level=2)
    body(doc, "The textbooks are processed once into a searchable index of ~12,000+ passages, "
              "each tagged with its book, subject, grade, and page number. Retrieval uses BM25 "
              "(a proven lexical ranking algorithm) with an Arabic-aware tokenizer that "
              "normalizes letter variants, strips diacritics, and removes the definite article "
              "so that morphological variants still match.")

    # 5. Equation handling — the OCR + Marker hybrid
    section_heading(doc, "5. Equation Handling (Math / Physics / Chemistry)", level=2)
    body(doc, "Standard OCR cannot read equations — fractions, exponents, integrals, and Greek "
              "symbols come out as noise, which previously made science questions unreliable. "
              "The system therefore uses a HYBRID per page:")
    bullet(doc, "Prose pages keep fast standard OCR text.", "OCR")
    bullet(doc, "Equation-dense pages are re-processed with Marker (Surya OCR + an equation "
                "model) which converts formulas to LaTeX, e.g. the spring-pendulum period "
                "T = 2π√(m/k) and the arithmetic-series sum S = n(a+ℓ)/2.", "Marker")
    body(doc, "Only the equation pages are sent to the slower Marker model, so the cost is "
              "minimized. The model then generates genuine equation-based questions.")

    # 6. Chapter selection
    section_heading(doc, "6. Chapter Selection", level=2)
    body(doc, "Instead of typing page numbers, teachers tick the chapters they want. The system "
              "reads each book's printed table of contents to recover accurate chapter names and "
              "page ranges (computing each chapter's end from the next chapter's start), and "
              "falls back to a heuristic for books without a parseable contents page. This also "
              "prevents a selection from ever starting in the middle of a chapter.")

    # 7. Output
    section_heading(doc, "7. Word Export", level=2)
    body(doc, "Exams are delivered as polished .docx files: a student-info header, questions "
              "grouped by type with per-section instructions, ruled answer lines, and an answer "
              "key on a new page. Arabic documents are fully right-aligned (RTL); English "
              "documents are left-aligned — chosen automatically from the detected language.")

    doc.add_page_break()

    # 8. Question types & Bloom
    section_heading(doc, "8. Question Types & Pedagogy")
    results_table(doc,
        ["Question Type", "Description"],
        [["Multiple Choice", "One stem, four options (A–D), one correct + 3 distractors"],
         ["True / False", "A statement that is definitively true or false"],
         ["Short Answer", "Requires a 1–3 sentence response, with a model answer"],
         ["Fill in the Blank", "A sentence with a key term removed"],
         ["Essay", "Open-ended; includes the key points a good answer should cover"]])
    body(doc, "Every question is tagged with a Bloom's taxonomy level (Remember, Understand, "
              "Apply, Analyze, Evaluate, Create) and a difficulty (easy / medium / hard), both "
              "selectable by the teacher.")

    # 9. Results
    section_heading(doc, "9. Validation Results")
    body(doc, "Exam-sized generations (15 mixed-type questions each, on verified chapter ranges) "
              "were produced for six subjects and scored by the built-in quality evaluator. "
              "Every set exported to a valid Word document.")
    results_table(doc,
        ["Subject (Grade 12 unless noted)", "Source", "Avg. Quality / 100", "Result"],
        [["Arabic (Grade 10)",        "OCR",          "92.6", "Excellent"],
         ["English",                   "OCR",          "97.9", "Excellent"],
         ["Science (Biology)",         "OCR + ToC",    "95.5", "Excellent"],
         ["Mathematics (Sequences)",   "OCR + Marker", "89.7", "Good"],
         ["Physics (Oscillations)",    "OCR + Marker", "89.8", "Good"],
         ["Chemistry (Radioactivity)", "OCR",          "96.6", "Excellent"]])
    body(doc, "Quality score = structure validity + answer correctness + content depth + "
              "relevance of the question to its source passage.")

    # 10. Tech stack
    section_heading(doc, "10. Technology Stack", level=2)
    kv_table(doc, [
        ("API",            "FastAPI (Python), port 8002, X-API-Key auth"),
        ("Language Model", "Command-R7B-Arabic via Ollama (local, offline)"),
        ("Retrieval",      "BM25 (rank-bm25) over a pre-built curriculum index"),
        ("Text / OCR",     "PyMuPDF text extraction + EasyOCR fallback"),
        ("Equations",      "Marker (Surya OCR + equation model) → LaTeX"),
        ("Export",         "python-docx (RTL/LTR Word documents)"),
    ])

    # 11. Limitations & future work
    section_heading(doc, "11. Limitations & Future Work", level=2)
    bullet(doc, "The local 7B model excels at conceptual questions; complex multi-step "
                "computations may need a larger model or a symbolic checker.")
    bullet(doc, "Marker equation extraction is GPU-intensive, so it is applied selectively to "
                "equation pages of key chapters; it scales to more content as hardware allows.")
    bullet(doc, "Chapter detection is exact where a book has a machine-readable table of "
                "contents and best-effort otherwise.")

    path = os.path.join(OUT_DIR, "report_question_generator.docx")
    doc.save(path)
    print("OK  ->", path)


if __name__ == "__main__":
    main()
