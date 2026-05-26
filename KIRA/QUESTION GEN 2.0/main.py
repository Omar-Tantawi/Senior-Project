"""
Question Generator microservice — FastAPI app.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8002 --reload

Environment variables:
    AI_API_KEY       Shared secret for X-API-Key header (between this service and Laravel)
    OLLAMA_URL       Ollama base URL (default: http://localhost:11434)
    OLLAMA_MODEL     Model to use   (default: command-r7b-arabic)

Requires Ollama running locally:
    ollama serve
    ollama pull command-r7b-arabic
"""

import os
import re
import uuid
from dotenv import load_dotenv
load_dotenv()   # loads .env file if present
from typing import Literal
from fastapi import FastAPI, HTTPException, Request, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from pdf_parser     import PDFParser
from embedder       import Embedder
from question_engine import QuestionEngine

# ── App setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Question Generator Service",
    version="1.0.0",
    description="AI-powered exam question generator. Supports Arabic and English PDFs.",
)

# Allow uploads up to 400 MB
from starlette.formparsers import MultiPartParser
MultiPartParser.max_file_size = 400 * 1024 * 1024   # 400 MB

AI_API_KEY = os.getenv("AI_API_KEY", "change-me-shared-secret")

pdf_parser = PDFParser()
embedder   = Embedder(persist_dir="./chroma_store")
engine     = None   # lazy-init so startup doesn't fail if key missing

def get_engine() -> QuestionEngine:
    global engine
    if engine is None:
        engine = QuestionEngine(embedder)
    return engine


# ── Auth ───────────────────────────────────────────────────────────────────────

async def verify_key(request: Request):
    key = request.headers.get("X-API-Key", "")
    if key != AI_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key.")


# ── In-memory document registry ───────────────────────────────────────────────
# In production this would be in the DB, but for the project this is fine.

_documents: dict[str, dict] = {}   # document_id → metadata


# ── Schemas ────────────────────────────────────────────────────────────────────

QuestionType  = Literal["mcq", "true_false", "short_answer", "fill_blank", "essay"]
BloomLevel    = Literal["remember", "understand", "apply", "analyze", "evaluate", "create"]
Difficulty    = Literal["easy", "medium", "hard"]
Language      = Literal["ar", "en", "auto"]


class GenerateRequest(BaseModel):
    document_id:     str

    # ── Option A: specific count per type (recommended) ──────────────────────
    # e.g. {"mcq": 5, "short_answer": 3, "true_false": 2}
    question_counts: dict[QuestionType, int] | None = Field(
        default=None,
        description="Exact count per question type. e.g. {\"mcq\": 5, \"short_answer\": 3}"
    )

    # ── Option B: legacy — total count + cycling types ────────────────────────
    num_questions:   int        = Field(5, ge=1, le=50)
    question_types:  list[QuestionType] = Field(
        default=["mcq"],
        description="Used only when question_counts is not provided. Cycled evenly."
    )

    bloom_levels:    list[BloomLevel] = Field(
        default=["remember", "understand"],
        description="Bloom's taxonomy levels. Cycled across all questions."
    )
    difficulty:      Difficulty = "medium"
    language:        Language   = "auto"
    topic_hint:      str        = Field(
        default="",
        description="Optional topic focus (e.g. 'photosynthesis', 'chapter 3')"
    )
    evaluate:        bool       = Field(
        default=True,
        description="Automatically score each question for quality (0-100). Set false to skip."
    )

    def resolve_question_list(self) -> list[str]:
        """Returns the full ordered list of question types to generate."""
        if self.question_counts:
            # Expand counts into a flat list: {"mcq":2,"tf":1} → ["mcq","mcq","tf"]
            result = []
            for q_type, count in self.question_counts.items():
                result.extend([q_type] * max(1, count))
            return result
        # Legacy: cycle question_types up to num_questions
        return [
            self.question_types[i % len(self.question_types)]
            for i in range(self.num_questions)
        ]


class ExportRequest(BaseModel):
    questions:       list[dict]
    title:           str      = Field("Exam Questions", description="Document title shown at the top of the Word file.")
    document_id:     str      = Field("", description="Optional — used in the filename only.")
    difficulty:      str      = Field("", description="Difficulty label shown in the subtitle.")
    language:        Language = Field("en", description="'ar' for Arabic RTL layout, 'en' for English.")
    include_answers: bool     = Field(True, description="Set to false to generate a student copy without answers.")


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "question-generator"}


@app.post("/documents/upload", dependencies=[Depends(verify_key)])
async def upload_document(
    file:       UploadFile = File(...),
    page_start: int | None = None,
    page_end:   int | None = None,
):
    """
    Upload a PDF file. Returns document_id to use in /questions/generate.
    Supports Arabic and English PDFs.

    Optional page range (1-based, inclusive):
      - page_start: first page to include (default: 1)
      - page_end:   last page to include  (default: last page)

    Example: page_start=10&page_end=25 → only pages 10–25 are indexed.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    if len(contents) > 400 * 1024 * 1024:   # 400MB limit
        raise HTTPException(status_code=400, detail="File too large. Max 400MB.")

    try:
        parsed = pdf_parser.parse(contents, file.filename, page_start, page_end)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not parsed["chunks"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No text could be extracted from pages {page_start or 1}–{page_end or parsed['pages']}. "
                "These pages may be scanned images. Try different page numbers or a text-based PDF."
            )
        )

    document_id = str(uuid.uuid4())[:12]

    # Store chunks in vector DB
    embedder.add_document(
        document_id=document_id,
        chunks=parsed["chunks"],
        metadata={
            "filename": file.filename,
            "language": parsed["language"],
            "pages":    parsed["pages"],
        },
    )

    # Save metadata
    _documents[document_id] = {
        "document_id": document_id,
        "filename":    file.filename,
        "pages":       parsed["pages"],
        "pages_used":  parsed["pages_used"],
        "page_start":  page_start or 1,
        "page_end":    page_end   or parsed["pages"],
        "chunks":      len(parsed["chunks"]),
        "language":    parsed["language"],
        "ocr_pages":   parsed["ocr_pages"],
    }

    return _documents[document_id]


@app.post("/questions/generate", dependencies=[Depends(verify_key)])
def generate_questions(body: GenerateRequest):
    """
    Generate exam questions from a previously uploaded document.

    - Choose question types: mcq, true_false, short_answer, fill_blank, essay
    - Choose Bloom's levels: remember, understand, apply, analyze, evaluate, create
    - Choose difficulty: easy, medium, hard
    - Language is auto-detected from the PDF, or set manually to "ar" or "en"
    """
    if not embedder.document_exists(body.document_id):
        raise HTTPException(
            status_code=404,
            detail=f"Document '{body.document_id}' not found. Upload it first."
        )

    # Resolve language
    language = body.language
    if language == "auto":
        meta     = _documents.get(body.document_id, {})
        language = meta.get("language", "en")
        if language == "mixed":
            language = "ar"   # default to Arabic for mixed docs

    try:
        eng = get_engine()
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))

    question_list = body.resolve_question_list()

    questions = eng.generate(
        document_id=body.document_id,
        num_questions=len(question_list),
        question_types=question_list,
        bloom_levels=body.bloom_levels,
        difficulty=body.difficulty,
        language=language,
        topic_hint=body.topic_hint,
    )

    if not questions:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate questions. Make sure Ollama is running (ollama serve) and the model is pulled (ollama pull command-r7b-arabic)."
        )

    # ── Evaluation ────────────────────────────────────────────────────────────
    quality_summary = None
    if body.evaluate:
        from evaluator import evaluate_batch
        contexts = [q.pop("_context", "") for q in questions]
        eval_result     = evaluate_batch(questions, contexts=contexts, language=language)
        questions       = eval_result["questions"]
        quality_summary = eval_result["summary"]
    else:
        # Still strip internal _context field even when evaluation is skipped
        for q in questions:
            q.pop("_context", None)

    response = {
        "document_id":    body.document_id,
        "language":       language,
        "num_requested":  len(question_list),
        "num_generated":  len(questions),
        "difficulty":     body.difficulty,
        "questions":      questions,
    }
    if quality_summary:
        response["quality"] = quality_summary

    return response


@app.post("/questions/export/docx", dependencies=[Depends(verify_key)])
def export_questions_docx(body: ExportRequest):
    """
    Export a list of questions to a formatted Word (.docx) file.

    Workflow:
      1. Call /questions/generate  → get the questions array
      2. Call /questions/export/docx with that array → download the .docx

    Set include_answers=false to generate a student exam sheet (no answers).
    Set include_answers=true  to generate a teacher answer key.
    """
    try:
        from docx_exporter import export_to_docx
        docx_bytes = export_to_docx(
            questions       = body.questions,
            title           = body.title,
            document_id     = body.document_id,
            difficulty      = body.difficulty,
            language        = body.language,
            include_answers = body.include_answers,
        )
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="python-docx is not installed. Run: pip install python-docx"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate Word file: {e}")

    safe_id  = body.document_id.replace("/", "_") or "export"
    filename = f"questions_{safe_id}.docx"

    return Response(
        content     = docx_bytes,
        media_type  = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers     = {"Content-Disposition": f"attachment; filename={filename}"},
    )


# ── All-in-one endpoint (frontend / backend integration) ──────────────────────

@app.post("/questions/from-pdf", dependencies=[Depends(verify_key)])
async def generate_from_pdf(
    file:               UploadFile = File(..., description="The PDF file"),
    page_start:         int  | None = Form(None, description="First page (1-based, inclusive)"),
    page_end:           int  | None = Form(None, description="Last page (1-based, inclusive)"),
    mcq_count:          int  = Form(5,  description="Number of MCQ questions"),
    true_false_count:   int  = Form(3,  description="Number of True/False questions"),
    short_answer_count: int  = Form(2,  description="Number of Short Answer questions"),
    fill_blank_count:   int  = Form(3,  description="Number of Fill-in-the-Blank questions"),
    essay_count:        int  = Form(0,  description="Number of Essay questions"),
    bloom_levels:       str  = Form("remember,understand,apply", description="Comma-separated"),
    difficulty:         str  = Form("medium", description="easy | medium | hard"),
    language:           str  = Form("auto",   description="ar | en | auto"),
    title:              str  = Form("Exam Questions", description="Title shown in the Word file"),
    include_answers:    bool = Form(True, description="Append Answer Key at the end"),
    output:             str  = Form("docx", description="'docx' to download Word file, 'json' for raw questions"),
):
    """
    All-in-one endpoint that does the full flow in a single request:

        1. Upload + parse PDF (with OCR fallback for image pages)
        2. Index chunks into the vector store
        3. Generate the requested questions
        4. Either:
             - output=docx → return the formatted Word file (default)
             - output=json → return the questions JSON

    This is the recommended endpoint for backend/frontend integration.

    Frontend sends one multipart/form-data request:
      file              → the PDF
      page_start, page_end → optional page range
      mcq_count, true_false_count, ... → how many of each type
      difficulty        → easy / medium / hard
      language          → ar / en / auto
      title             → Word file title
      output            → "docx" (file download) or "json" (questions list)
    """
    # ── Validate ──────────────────────────────────────────────────────────────
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    if len(contents) > 400 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 400MB.")

    counts = {
        "mcq":          max(0, mcq_count),
        "true_false":   max(0, true_false_count),
        "short_answer": max(0, short_answer_count),
        "fill_blank":   max(0, fill_blank_count),
        "essay":        max(0, essay_count),
    }
    total = sum(counts.values())
    if total == 0:
        raise HTTPException(status_code=400, detail="At least one question count must be > 0.")
    if total > 50:
        raise HTTPException(status_code=400, detail="Total questions cannot exceed 50.")

    # ── Step 1: Parse PDF ─────────────────────────────────────────────────────
    try:
        parsed = pdf_parser.parse(contents, file.filename, page_start, page_end)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not parsed["chunks"]:
        raise HTTPException(
            status_code=422,
            detail=(
                f"No text could be extracted from pages "
                f"{page_start or 1}–{page_end or parsed['pages']}. "
                "These pages may be scanned images with very little text."
            ),
        )

    # ── Step 2: Index in vector store ─────────────────────────────────────────
    document_id = str(uuid.uuid4())[:12]
    embedder.add_document(
        document_id = document_id,
        chunks      = parsed["chunks"],
        metadata    = {
            "filename": file.filename,
            "language": parsed["language"],
            "pages":    parsed["pages"],
        },
    )
    _documents[document_id] = {
        "document_id": document_id,
        "filename":    file.filename,
        "pages":       parsed["pages"],
        "pages_used":  parsed["pages_used"],
        "page_start":  page_start or 1,
        "page_end":    page_end or parsed["pages"],
        "chunks":      len(parsed["chunks"]),
        "language":    parsed["language"],
        "ocr_pages":   parsed["ocr_pages"],
    }

    # ── Step 3: Resolve language ──────────────────────────────────────────────
    resolved_lang = language
    if resolved_lang == "auto":
        resolved_lang = parsed["language"]
        if resolved_lang == "mixed":
            resolved_lang = "ar"
        if resolved_lang not in {"ar", "en"}:
            resolved_lang = "en"

    # ── Step 4: Build question list and bloom list ────────────────────────────
    question_list: list[str] = []
    for q_type, n in counts.items():
        question_list.extend([q_type] * n)

    bloom_list = [b.strip() for b in bloom_levels.split(",") if b.strip()]
    if not bloom_list:
        bloom_list = ["remember", "understand", "apply"]

    # ── Step 5: Generate ──────────────────────────────────────────────────────
    try:
        eng = get_engine()
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))

    questions = eng.generate(
        document_id    = document_id,
        num_questions  = len(question_list),
        question_types = question_list,
        bloom_levels   = bloom_list,
        difficulty     = difficulty,
        language       = resolved_lang,
        topic_hint     = "",
    )

    if not questions:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate questions. Check that Ollama is running and the model is available."
        )

    # Strip internal context field before returning
    for q in questions:
        q.pop("_context", None)

    # ── Step 6: Return JSON or Word file ──────────────────────────────────────
    if output.lower() == "json":
        return {
            "document_id":   document_id,
            "language":      resolved_lang,
            "difficulty":    difficulty,
            "num_generated": len(questions),
            "num_requested": len(question_list),
            "questions":     questions,
        }

    # Otherwise return the .docx file directly
    try:
        from docx_exporter import export_to_docx
        docx_bytes = export_to_docx(
            questions       = questions,
            title           = title,
            document_id     = document_id,
            difficulty      = difficulty,
            language        = resolved_lang,
            include_answers = include_answers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word export failed: {e}")

    # HTTP headers must be ASCII — strip non-ASCII for the plain filename fallback
    ascii_title = re.sub(r'[^\x00-\x7F]+', '', title)          # drop Arabic / non-ASCII
    ascii_title = re.sub(r'[^\w\-]+', '_', ascii_title)[:50]   # keep word chars & dash
    ascii_title = ascii_title.strip('_') or "exam"
    plain_name  = f"{ascii_title}_{document_id}.docx"

    # RFC 5987 lets us send the real UTF-8 filename for browsers that support it
    from urllib.parse import quote
    utf8_name   = quote(f"{title}_{document_id}.docx", safe='')
    disposition = f"attachment; filename=\"{plain_name}\"; filename*=UTF-8''{utf8_name}"

    return Response(
        content    = docx_bytes,
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers    = {"Content-Disposition": disposition},
    )


@app.get("/documents", dependencies=[Depends(verify_key)])
def list_documents():
    """List all uploaded documents."""
    return {"documents": list(_documents.values())}


@app.get("/documents/{document_id}", dependencies=[Depends(verify_key)])
def get_document(document_id: str):
    """Get metadata for a specific document."""
    doc = _documents.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")
    return doc


@app.delete("/documents/{document_id}", dependencies=[Depends(verify_key)])
def delete_document(document_id: str):
    """Delete a document and its stored embeddings."""
    if document_id not in _documents:
        raise HTTPException(status_code=404, detail="Document not found.")
    embedder.delete_document(document_id)
    _documents.pop(document_id)
    return {"status": "deleted", "document_id": document_id}


# ── Question type reference ───────────────────────────────────────────────────

@app.get("/reference/types")
def question_types_reference():
    """Returns all supported question types, Bloom's levels, and difficulty options."""
    return {
        "question_types": {
            "mcq":          "Multiple Choice (4 options, 1 correct)",
            "true_false":   "True or False statement",
            "short_answer": "1-3 sentence answer",
            "fill_blank":   "Fill in the blank",
            "essay":        "Open-ended essay question",
        },
        "bloom_levels": [
            "remember", "understand", "apply", "analyze", "evaluate", "create"
        ],
        "difficulty": ["easy", "medium", "hard"],
        "languages":  ["ar", "en", "auto"],
    }
