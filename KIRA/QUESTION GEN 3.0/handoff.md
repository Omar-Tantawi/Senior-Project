# Question Generator — Integration Handoff

AI microservice that generates exam questions (Arabic + English) from the Syrian
curriculum and exports them to Word. This document is everything the frontend/backend
developer needs to integrate it.

---

## 1. What it does

- Generates 5 question types — **mcq, true_false, short_answer, fill_blank, essay** —
  grounded **strictly** in real textbook content (no hallucinated topics).
- Works **by book name with no file upload** for the 23+ indexed curriculum books, or by
  **uploading a PDF** for anything else.
- Lets the user pick **chapters** (not raw page numbers).
- Exports a formatted **.docx** exam (RTL for Arabic, LTR for English) with an answer key.
- 100% local: questions are written by a local LLM (Ollama), retrieval by a local index.

---

## 2. Running the service

**Prereqs**
- Python **3.12** at `C:\Users\Hussin\AppData\Local\Programs\Python\Python312\python.exe`
- Ollama running **with the models on D:** — this is required:
  ```bat
  set OLLAMA_MODELS=D:\Models
  ollama serve
  ```
  (Models: `smartschool` — the question generator — and its base `command-r7b-arabic`.)

**Start the API**
```bat
cd D:\Senior\question-generator
set OLLAMA_MODELS=D:\Models
set AI_API_KEY=<shared-secret>            REM same value the backend will send
<python312> -m uvicorn main:app --host 0.0.0.0 --port 8002
```
Health check: `GET http://localhost:8002/health` →
`{"status":"ok","service":"question-generator","books":39,"dense":false}`

**Auth:** every endpoint except `/health` and `/reference/types` requires header
`X-API-Key: <shared-secret>`.

---

## 3. Frontend — exactly what to build

For curriculum books **nothing is uploaded**; the UI just drives the API by name.

**UI components (in order):**
1. **Grade dropdown** — 10 / 11 / 12. (Use the `grade` field from `GET /books` to populate.)
2. **Subject dropdown** — filtered by the chosen grade, from `GET /books` (`book` = the id to
   send; show `subject`).
3. **Chapter selector** — call `GET /books/{book}/chapters` and render the returned chapters as
   **checkboxes** (label = `title`, value = `index`). User ticks any subset (can skip middle
   chapters). *Fallback:* if you prefer, offer a page-range input (`page_start`/`page_end`).
4. **Question counts** — five number inputs: MCQ, True/False, Short answer, Fill-blank, Essay.
5. **Bloom levels** — multi-select: remember, understand, apply, analyze, evaluate, create.
6. **Difficulty** — dropdown: easy / medium / hard.
7. **Language** — auto (default) / Arabic / English.
8. **Include answer key** — toggle (for teacher copy vs student copy).
9. **Generate** button → `POST /questions/generate` (body in §4) → render the returned
   `questions` (and `quality` scores) as a preview.
10. **Download Word** button → `POST /questions/export/docx` with the questions → save the
    returned `.docx`.
11. *(Optional)* **Upload PDF** control — only for material **not** in the curriculum; calls
    `POST /questions/from-pdf` (multipart) and returns the `.docx` directly.

**Backend wiring:** send the user's selections straight through to `/questions/generate`
with `document_id` = the chosen `book`. No file transfer for curriculum books — this is the
fix for the "uploads the file over the internet" behavior you saw; the content is already
indexed server-side.

---

## 4. Endpoints

### `GET /books`
```json
{ "books": [ {"book":"12-physics-Sci","subject":"physics-Sci","grade":"12",
              "chunks":831,"pages":268}, ... ] }
```

### `GET /books/{book}/chapters`
```json
{ "book":"12-science", "source":"toc",
  "chapters":[ {"index":1,"title":"الجهاز العصبي","start_page":9,"end_page":19},
               {"index":2,"title":"النسيج العصبي","start_page":20,"end_page":25}, ... ] }
```
`source` is `"toc"` (parsed from the book's table of contents — accurate names/pages) or
`"heuristic"` (best-effort) for books without a parseable ToC.

### `POST /questions/generate`  ← main endpoint
Body (all fields except `document_id` optional):
```json
{
  "document_id": "12-physics-Sci",          // a book name (no upload) OR an uploaded doc id
  "chapters": [1, 2],                        // optional: indices from /books/{book}/chapters
  "page_start": null, "page_end": null,      // optional alternative to chapters
  "question_counts": {"mcq":6,"true_false":4,"short_answer":2,"fill_blank":2,"essay":1},
  "bloom_levels": ["remember","understand","apply","analyze"],
  "difficulty": "medium",                    // easy | medium | hard
  "language": "auto",                        // ar | en | auto
  "evaluate": true                           // attach a 0-100 quality score per question
}
```
Response:
```json
{ "document_id":"12-physics-Sci", "book":"12-physics-Sci", "reused_index":true,
  "language":"ar", "num_generated":15, "difficulty":"medium",
  "questions":[ {"type":"mcq","question":"...","options":{"A":"...","B":"..."},
                 "correct_answer":"B","explanation":"...","quality":{"score":92,...}}, ... ],
  "quality":{"average_score":89.8,"pass_rate":100.0,"overall_grade":"Good"} }
```

### `POST /questions/export/docx`
Body: `{ "questions":[...], "title":"اختبار الفيزياء", "language":"ar", "include_answers":true }`
→ returns the `.docx` file (Content-Disposition attachment). Set `include_answers:false` for a
student copy. Direction (RTL/LTR) is automatic from `language`.

### `POST /questions/from-pdf`  (all-in-one, multipart) — for NON-curriculum PDFs
Form fields: `file` (PDF), `page_start`, `page_end`, `mcq_count`, `true_false_count`,
`short_answer_count`, `fill_blank_count`, `essay_count`, `bloom_levels` (csv), `difficulty`,
`language`, `title`, `include_answers`, `output` (`docx` | `json`).
Uploads → parses/OCRs → generates → returns the `.docx` (or JSON).

### Others
- `POST /documents/upload` — upload a PDF, returns a `document_id` (then use `/questions/generate`).
- `GET /documents`, `GET /documents/{id}`, `DELETE /documents/{id}`.
- `GET /reference/types` — lists all question types, bloom levels, difficulties, languages.

---

## 5. Upload behavior & rename-proof matching

If the backend *does* upload a PDF, the service identifies it by:
1. **filename** match to a known book, else
2. **content fingerprint** — `sha256` of bytes (same file, any name), then text-hash, then token
   overlap.
So **the same PDF uploaded under a different name is still recognized** and reuses the index
(no re-OCR). The response includes `reused_index` and `match_method`.

> The "file uploaded from the internet instead of disk" issue is on the **backend/frontend**
> side (how the file is fetched). For curriculum books, don't upload at all — generate by name.

---

## 6. Equations (math / physics / chemistry)

Standard OCR mangles equations. We added a **Marker** pass that converts equation pages to
**LaTeX** and merges them into the index page-by-page (OCR for prose, Marker for equation
pages). For the demo this is applied to **physics (SHM chapter)** and **math (Sequences
chapter)** in grade 12 — those generate real equation questions (`$T_0=2\pi\sqrt{m/k}$`,
`$S=\frac{n(a+\ell)}{2}$`, …). It scales to more chapters/books via
`tools/hybrid_scan.py` + `tools/ingest_marker.py` (one-time GPU job; ~250 s/page on the
current GPU, so it's run selectively).

---

## 7. Known limitations

- Local **7B** model → strong on **conceptual** questions; **computational** math answers can
  occasionally be wrong (no symbolic solver). Favor conceptual/definitional for math.
- Marker (equation upgrade) is GPU-bound (~250 s/page) → applied to selected chapters, not all
  books, on this hardware.
- Chapter detection is accurate where a book has a parseable ToC (science, English, …),
  best-effort otherwise.
- Ollama must be started with `OLLAMA_MODELS=D:\Models` or it won't find `smartschool`.

---

## 8. Quick test (verified)

Exam-sized generation (15 questions, mixed types) scored **89–98 / 100** across Arabic,
English, Science, Math, Physics, Chemistry, and each exported to a valid `.docx`
(see `output/exams/`).
