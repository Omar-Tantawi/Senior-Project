"""
Question Generator microservice — FastAPI app.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8002 --reload

Environment variables:
    AI_API_KEY       Shared secret for X-API-Key header (between this service and Laravel)
    OLLAMA_URL       Ollama base URL (default: http://localhost:11434)
    OLLAMA_MODEL     Model to use   (default: smartschool)

Retrieval is grounded in the prebuilt curriculum index (qdrant_storage/). When a
teacher uploads a known textbook, its OCR'd chunks are reused (no re-OCR) and
questions are generated strictly from that book. Unknown PDFs are parsed + OCR'd
on the fly and indexed in-memory.

Requires Ollama running locally:
    ollama serve
    ollama create smartschool -f Modelfile
"""

import os
import re
import uuid
from dotenv import load_dotenv
load_dotenv()
from typing import Literal
from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel, Field

from pdf_parser      import PDFParser
from index_store     import get_store
from retriever       import CurriculumSource, MemSource
from question_engine import QuestionEngine
from chapters        import detect_chapters, pages_for_chapters
from toc_chapters    import chapters_for_book
import fingerprint

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Question Generator Service",
    version="2.0.0",
    description="AI-powered, retrieval-grounded exam question generator. Arabic & English.",
)

from starlette.formparsers import MultiPartParser
MultiPartParser.max_file_size = 400 * 1024 * 1024   # 400 MB

AI_API_KEY = os.getenv("AI_API_KEY", "change-me-shared-secret")

pdf_parser = PDFParser()
store      = get_store()
engine: QuestionEngine | None = None


def get_engine() -> QuestionEngine:
    global engine
    if engine is None:
        engine = QuestionEngine()
    return engine


# ── Auth ───────────────────────────────────────────────────────────────────────

async def verify_key(request: Request):
    if request.headers.get("X-API-Key", "") != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")


# ── Document registry (in-memory) ───────────────────────────────────────────────

_documents: dict[str, dict] = {}        # document_id → meta (+ source spec)
_fingerprints = fingerprint.load_store()   # book → content fingerprint (built from Data/)
print(f"[main] loaded {len(_fingerprints)} content fingerprints")


def _lang_of_book(book: str) -> str:
    idxs = store.book_indices(book)[:8]
    sample = " ".join(store.text(i) for i in idxs)[:2000]
    ar = len(re.findall(r'[؀-ۿ]', sample)) / max(len(sample), 1)
    en = len(re.findall(r'[a-zA-Z]', sample)) / max(len(sample), 1)
    if ar > 0.2:
        return "ar"
    if en > 0.2:
        return "en"
    return "en"


def _ingest(contents: bytes, filename: str,
            page_start: int | None, page_end: int | None) -> dict:
    """Resolve an upload to a grounding source, reusing the curriculum index when possible."""
    # 1) filename match (fast)   2) content fingerprint (rename-proof: sha256 / text / tokens)
    book = store.resolve_book(filename)
    match_method = "filename" if book else None
    if not book:
        try:
            fp = fingerprint.fingerprint(contents)
            book, score, match_method = fingerprint.match(fp, _fingerprints)
            if book and not store.resolve_book(book):
                book = None            # matched a book we don't actually have indexed
            if book:
                print(f"[main] content-matched '{filename}' → {book} "
                      f"({match_method}, score={score})")
        except Exception as e:
            print(f"[main] fingerprint failed: {e}")
            book = None

    if book:                                              # KNOWN textbook → reuse, no OCR
        idxs = store.book_indices(book, page_start, page_end)
        meta = {
            "kind": "curriculum", "book": book,
            "page_start": page_start, "page_end": page_end,
            "reused_index": True, "used_ocr": False, "match_method": match_method,
            "chunks": len(idxs), "language": _lang_of_book(book),
            "filename": filename, "subject": store.chunk(idxs[0])["subject"] if idxs else None,
            "grade": store.chunk(idxs[0])["grade"] if idxs else None,
        }
        doc_id = book
    else:                                                 # UNKNOWN PDF → parse + OCR
        parsed = pdf_parser.parse(contents, filename, page_start, page_end)
        if not parsed["chunks"]:
            raise HTTPException(
                status_code=422,
                detail=(f"No text could be extracted from pages "
                        f"{page_start or 1}–{page_end or parsed['pages']}. "
                        "These pages may be scanned images with very little text."),
            )
        meta = {
            "kind": "memory", "book": filename,
            "page_start": page_start, "page_end": page_end,
            "reused_index": False, "used_ocr": parsed["ocr_pages"] > 0,
            "chunks": len(parsed["chunks"]), "language": parsed["language"],
            "filename": filename, "_chunks_data": parsed["chunks"],
            "pages": parsed["pages"],
        }
        doc_id = str(uuid.uuid4())[:12]

    meta["document_id"] = doc_id
    _documents[doc_id] = meta
    return meta


def _build_source(doc_id: str):
    meta = _documents[doc_id]
    if meta["kind"] == "curriculum":
        return CurriculumSource(store, meta["book"], meta["page_start"], meta["page_end"])
    return MemSource(meta["_chunks_data"], book=meta.get("filename", "uploaded"))


def _public_meta(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if not k.startswith("_")}


# ── Schemas ────────────────────────────────────────────────────────────────────

QuestionType = Literal["mcq", "true_false", "short_answer", "fill_blank", "essay"]
BloomLevel   = Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]
Difficulty   = Literal["easy", "medium", "hard"]
Language     = Literal["ar", "en", "auto"]


class GenerateRequest(BaseModel):
    document_id:     str
    question_counts: dict[QuestionType, int] | None = Field(
        default=None,
        description="Exact count per type. e.g. {\"mcq\": 5, \"short_answer\": 3}")
    num_questions:   int = Field(5, ge=1, le=50)
    question_types:  list[QuestionType] = Field(default=["mcq"])
    bloom_levels:    list[BloomLevel]   = Field(default=["remember", "understand"])
    difficulty:      Difficulty = "medium"
    language:        Language   = "auto"
    topic_hint:      str  = Field(default="")
    evaluate:        bool = Field(default=True)
    # optional scoping when generating from a known book by name:
    #   chapters wins over page_start/page_end if both are given
    page_start:      int | None = Field(default=None)
    page_end:        int | None = Field(default=None)
    chapters:        list[int] | None = Field(
        default=None, description="Chapter indices from GET /books/{book}/chapters")

    def resolve_question_list(self) -> list[str]:
        if self.question_counts:
            out = []
            for q_type, count in self.question_counts.items():
                out.extend([q_type] * max(1, count))
            return out
        return [self.question_types[i % len(self.question_types)]
                for i in range(self.num_questions)]


class ExportRequest(BaseModel):
    questions:       list[dict]
    title:           str  = Field("Exam Questions")
    document_id:     str  = Field("")
    difficulty:      str  = Field("")
    language:        Language = Field("en")
    include_answers: bool = Field(True)


# ── Generation core (shared by /generate and /from-pdf) ─────────────────────────

def _resolve_language(requested: str, meta_language: str) -> str:
    lang = requested
    if lang == "auto":
        lang = meta_language or "en"
    if lang == "mixed":
        lang = "ar"
    return lang if lang in {"ar", "en"} else "en"


def _generate(source, question_list, bloom_list, difficulty, language, topic_hint,
              do_eval: bool):
    eng = get_engine()
    questions = eng.generate(
        source=source,
        num_questions=len(question_list),
        question_types=question_list,
        bloom_levels=bloom_list,
        difficulty=difficulty,
        language=language,
        topic_hint=topic_hint,
    )
    if not questions:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate questions. Check that Ollama is running "
                   "(ollama serve) and the model is available.")

    quality = None
    if do_eval:
        from evaluator import evaluate_batch
        contexts = [q.pop("_context", "") for q in questions]
        result   = evaluate_batch(questions, contexts=contexts, language=language)
        questions, quality = result["questions"], result["summary"]
    else:
        for q in questions:
            q.pop("_context", None)
    return questions, quality


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "question-generator",
            "books": len(store.book_to_idxs), "dense": store.dense_enabled}


@app.get("/books", dependencies=[Depends(verify_key)])
def list_books():
    """Textbooks already in the curriculum index (uploading any of these reuses them)."""
    return {"books": store.list_books()}


@app.get("/books/{book}/chapters", dependencies=[Depends(verify_key)])
def book_chapters(book: str):
    """
    Chapters/units detected for a book — for the frontend's chapter checkboxes.
    The user ticks chapters; send their `index` values as `chapters` to
    /questions/generate (no manual page typing, never starts mid-chapter).
    Reliable for Marker-processed books; best-effort (with equal-band fallback)
    on image-scanned ones.
    """
    canonical = store.resolve_book(book)
    if not canonical:
        raise HTTPException(status_code=404, detail=f"Unknown book '{book}'. See GET /books.")
    return chapters_for_book(store, canonical)   # ToC if parseable, else heuristic


@app.post("/documents/upload", dependencies=[Depends(verify_key)])
async def upload_document(file: UploadFile = File(...),
                          page_start: int | None = None,
                          page_end:   int | None = None):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    contents = await file.read()
    if len(contents) > 400 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 400MB.")
    meta = _ingest(contents, file.filename, page_start, page_end)
    return _public_meta(meta)


@app.post("/questions/generate", dependencies=[Depends(verify_key)])
def generate_questions(body: GenerateRequest):
    # Generate directly from a known textbook by NAME — no upload needed. The backend
    # should prefer this for curriculum books (nothing is sent over the network).
    if body.document_id not in _documents:
        book = store.resolve_book(body.document_id)
        if book:
            ps, pe = body.page_start, body.page_end
            if body.chapters:                      # chapter selection → page range
                chs = chapters_for_book(store, book)["chapters"]
                sel = [c for c in chs if c["index"] in body.chapters]
                if sel:
                    ps = min(c["start_page"] for c in sel)
                    pe = max(c["end_page"] for c in sel)
            idxs = store.book_indices(book, ps, pe)
            _documents[body.document_id] = {
                "kind": "curriculum", "book": book, "document_id": body.document_id,
                "page_start": ps, "page_end": pe, "chapters": body.chapters,
                "reused_index": True, "used_ocr": False, "match_method": "book_name",
                "chunks": len(idxs), "language": _lang_of_book(book), "filename": book,
                "subject": store.chunk(idxs[0])["subject"] if idxs else None,
                "grade": store.chunk(idxs[0])["grade"] if idxs else None,
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"'{body.document_id}' not found. Upload it first, or pass a known "
                       f"book name as document_id (see GET /books).")
    meta     = _documents[body.document_id]
    source   = _build_source(body.document_id)
    language = _resolve_language(body.language, meta.get("language", "en"))

    questions, quality = _generate(
        source, body.resolve_question_list(), body.bloom_levels,
        body.difficulty, language, body.topic_hint, body.evaluate)

    response = {
        "document_id":   body.document_id,
        "book":          meta.get("book"),
        "reused_index":  meta.get("reused_index"),
        "language":      language,
        "num_requested": len(body.resolve_question_list()),
        "num_generated": len(questions),
        "difficulty":    body.difficulty,
        "questions":     questions,
    }
    if quality:
        response["quality"] = quality
    return response


@app.post("/questions/export/docx", dependencies=[Depends(verify_key)])
def export_questions_docx(body: ExportRequest):
    try:
        from docx_exporter import export_to_docx
        docx_bytes = export_to_docx(
            questions=body.questions, title=body.title, document_id=body.document_id,
            difficulty=body.difficulty, language=body.language,
            include_answers=body.include_answers)
    except ImportError:
        raise HTTPException(status_code=500, detail="python-docx is not installed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate Word file: {e}")

    safe_id  = body.document_id.replace("/", "_") or "export"
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f"attachment; filename=questions_{safe_id}.docx"})


@app.post("/questions/from-pdf", dependencies=[Depends(verify_key)])
async def generate_from_pdf(
    file:               UploadFile = File(...),
    page_start:         int | None = Form(None),
    page_end:           int | None = Form(None),
    mcq_count:          int  = Form(5),
    true_false_count:   int  = Form(3),
    short_answer_count: int  = Form(2),
    fill_blank_count:   int  = Form(3),
    essay_count:        int  = Form(0),
    bloom_levels:       str  = Form("remember,understand,apply"),
    difficulty:         str  = Form("medium"),
    language:           str  = Form("auto"),
    title:              str  = Form("Exam Questions"),
    include_answers:    bool = Form(True),
    output:             str  = Form("docx"),
):
    """All-in-one: upload → (reuse index or OCR) → generate → docx/json."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    contents = await file.read()
    if len(contents) > 400 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 400MB.")

    counts = {"mcq": max(0, mcq_count), "true_false": max(0, true_false_count),
              "short_answer": max(0, short_answer_count), "fill_blank": max(0, fill_blank_count),
              "essay": max(0, essay_count)}
    total = sum(counts.values())
    if total == 0:
        raise HTTPException(status_code=400, detail="At least one question count must be > 0.")
    if total > 50:
        raise HTTPException(status_code=400, detail="Total questions cannot exceed 50.")

    meta     = _ingest(contents, file.filename, page_start, page_end)
    source   = _build_source(meta["document_id"])
    language = _resolve_language(language, meta.get("language", "en"))

    question_list = []
    for q_type, n in counts.items():
        question_list.extend([q_type] * n)
    bloom_list = [b.strip() for b in bloom_levels.split(",") if b.strip()] or \
                 ["remember", "understand", "apply"]

    questions, _ = _generate(source, question_list, bloom_list, difficulty,
                             language, topic_hint="", do_eval=False)

    if output.lower() == "json":
        return {
            "document_id":   meta["document_id"], "book": meta.get("book"),
            "reused_index":  meta.get("reused_index"), "used_ocr": meta.get("used_ocr"),
            "language":      language, "difficulty": difficulty,
            "num_generated": len(questions), "num_requested": len(question_list),
            "questions":     questions,
        }

    try:
        from docx_exporter import export_to_docx
        docx_bytes = export_to_docx(
            questions=questions, title=title, document_id=meta["document_id"],
            difficulty=difficulty, language=language, include_answers=include_answers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word export failed: {e}")

    ascii_title = re.sub(r'[^\x00-\x7F]+', '', title)
    ascii_title = re.sub(r'[^\w\-]+', '_', ascii_title)[:50].strip('_') or "exam"
    plain_name  = f"{ascii_title}_{meta['document_id']}.docx"
    from urllib.parse import quote
    utf8_name   = quote(f"{title}_{meta['document_id']}.docx", safe='')
    disposition = f"attachment; filename=\"{plain_name}\"; filename*=UTF-8''{utf8_name}"
    return Response(
        content=docx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": disposition})


@app.get("/documents", dependencies=[Depends(verify_key)])
def list_documents():
    return {"documents": [_public_meta(m) for m in _documents.values()]}


@app.get("/documents/{document_id}", dependencies=[Depends(verify_key)])
def get_document(document_id: str):
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return _public_meta(doc)


@app.delete("/documents/{document_id}", dependencies=[Depends(verify_key)])
def delete_document(document_id: str):
    if document_id not in _documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    _documents.pop(document_id)
    return {"status": "deleted", "document_id": document_id}


@app.get("/reference/types")
def question_types_reference():
    return {
        "question_types": {
            "mcq": "Multiple Choice (4 options, 1 correct)",
            "true_false": "True or False statement",
            "short_answer": "1-3 sentence answer",
            "fill_blank": "Fill in the blank",
            "essay": "Open-ended essay question",
        },
        "bloom_levels": ["remember", "understand", "apply", "analyze", "evaluate", "create"],
        "difficulty": ["easy", "medium", "hard"],
        "languages": ["ar", "en", "auto"],
    }
