# Arabic Educational Chatbot — Backend Integration Guide

This document covers everything needed to deploy and integrate the chatbot into the school system. Follow the steps in order on a first-time setup.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Configuration](#4-configuration)
5. [Starting the Server](#5-starting-the-server)
6. [API Reference](#6-api-reference)
7. [Behavior the Frontend Must Handle](#7-behavior-the-frontend-must-handle)
8. [File Structure](#8-file-structure)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Architecture Overview

```
Student question
      │
      ▼
POST /chat  (FastAPI)
      │
      ├─► Normalize query (CAMeL Tools — Arabic morphology)
      │
      ├─► Dense retrieval (Qdrant local, cosine similarity ≥ 0.60)
      │         if no hits → return "هذا السؤال خارج المنهج الذي لديّ." immediately
      │
      ├─► BM25 lexical retrieval (rank-bm25, pickle index)
      │
      ├─► RRF merge → top-5 chunks
      │
      └─► Ollama LLM (command-r7b-arabic:7b) → answer
```

**Everything runs locally. No external API calls are made at query time.** The embedding model and LLM are served from the same machine.

---

## 2. Prerequisites

### Hardware
- NVIDIA GPU with at least 8 GB VRAM (the LLM alone needs ~6 GB at 4-bit)
- 16 GB RAM minimum

### Software

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 or 3.11 | 3.12+ not tested |
| Ollama | Latest | [ollama.com](https://ollama.com) |
| CUDA toolkit | 11.8 or 12.x | Must match your PyTorch build |

### Ollama model

After installing Ollama, pull the LLM:

```bash
ollama pull command-r7b-arabic:7b
```

Verify it loaded:

```bash
ollama list
# Should show: command-r7b-arabic:7b
```

### Python packages

```bash
pip install fastapi uvicorn[standard]
pip install qdrant-client>=1.17
pip install sentence-transformers
pip install rank-bm25
pip install camel-tools
pip install ollama
pip install PyMuPDF pillow
pip install surya-ocr          # only needed if re-indexing; not required at query time
pip install tqdm
```

Or install everything at once if a `requirements.txt` is present:

```bash
pip install -r requirements.txt
```

---

## 3. Installation

### Step 1 — Clone / copy the project

Place the project at any path. The guide assumes `Senior/` as the root. All paths in code are relative to this root.

### Step 2 — Verify the index exists

The Qdrant vector index and BM25 index are pre-built and must be present on the server. Check:

```
Senior/
└── qdrant_storage/
    ├── collection/          ← Qdrant files
    │   └── curriculum/
    └── bm25.pkl             ← BM25 lexical index (pickle)
```

If either of these is missing, the server will fail to start. See [Troubleshooting](#9-troubleshooting).

### Step 3 — Download the embedding model

The embedding model (`jinaai/jina-embeddings-v3`) is downloaded automatically from HuggingFace on first run. This requires internet access on first startup only (~550 MB). After that it is cached locally at:

```
~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v3/
```

If the server will run in an air-gapped environment, pre-cache the model on a connected machine and copy the cache directory over.

### Step 4 — Set the CORS origin

Open `chatbot/config.py` and set `ALLOWED_ORIGINS` to the URL of your LMS or frontend:

```python
ALLOWED_ORIGINS: list[str] = ["https://lms.yourschool.edu.sy"]
```

Leave as `["*"]` only for local development.

---

## 4. Configuration

All tunable values live in one file: `chatbot/config.py`.

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `"command-r7b-arabic:7b"` | Ollama model name. Must match `ollama list`. |
| `SIMILARITY_THRESHOLD` | `0.60` | Cosine similarity cutoff. Queries with no chunk above this are blocked as off-curriculum. Lower = more permissive retrieval. |
| `TOP_K_DENSE` | `10` | Candidates fetched from Qdrant before RRF merge. |
| `TOP_K_BM25` | `10` | Candidates fetched from BM25 before RRF merge. |
| `TOP_K_FINAL` | `5` | Chunks passed to the LLM after merging. |
| `ALLOWED_ORIGINS` | `["*"]` | CORS origins. Set to your LMS URL before deploying. |
| `EMBEDDING_MODEL` | `"jinaai/jina-embeddings-v3"` | Do not change unless re-indexing with a different model. |
| `QDRANT_DIR` | `<root>/qdrant_storage` | Path to the Qdrant on-disk index. |
| `BM25_PATH` | `<root>/qdrant_storage/bm25.pkl` | Path to the BM25 pickle. |

---

## 5. Starting the Server

From the project root:

```bash
uvicorn chatbot.api.main:app --host 0.0.0.0 --port 8000
```

For production (multiple workers are **not recommended** — Qdrant local storage only supports one process at a time):

```bash
uvicorn chatbot.api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Confirm it is up:

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

The first request after startup takes 10–20 seconds longer than usual — the embedding model and Ollama are loaded into memory on first call. Subsequent requests are fast.

---

## 6. API Reference

Base URL: `http://<server>:8000`

---

### `GET /health`

Liveness check.

**Response `200`**
```json
{ "status": "ok" }
```

---

### `POST /chat`

Sends a question and returns the full answer when generation is complete.

**Request body**
```json
{
  "question": "ما هو قانون نيوتن الأول؟",
  "grade": "10",
  "subject": "فيزياء"
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | The student's question in Arabic (Syrian dialect or MSA). |
| `grade` | string | No | `"10"`, `"11"`, or `"12"`. Scopes retrieval to that grade's curriculum only. |
| `subject` | string | No | Partial subject name (Arabic). Matched case-insensitively. E.g. `"فيزياء"`, `"كيمياء"`, `"رياضيات"`. |

**Response `200`**
```json
{
  "answer": "🗣️ شرح بسيط: ...\n\n📖 تعريف رسمي: ...",
  "sources": [
    {
      "subject": "الفيزياء",
      "grade": "10",
      "page": 34,
      "source": "10-sci-physics"
    }
  ]
}
```

| Field | Description |
|---|---|
| `answer` | Full LLM response. Always contains two sections (see §7). Or the off-curriculum reply string. |
| `sources` | The curriculum chunks used to generate the answer. Show these as citations if your UI supports it. Empty list if the question was off-curriculum. |

**Off-curriculum response** — when no chunk clears the similarity threshold, `answer` is exactly:
```
هذا السؤال خارج المنهج الذي لديّ.
```
`sources` will be `[]`. The HTTP status is still `200`.

---

### `POST /chat/stream`

Same as `/chat` but returns tokens progressively via Server-Sent Events (SSE). Use this for a real-time typing effect in the UI.

**Request body** — identical to `/chat`.

**Response** — `text/event-stream`

```
data: أهلاً
data:  بك
data: ،
data:  إليك
...
data: [DONE]
```

Each `data:` line is one token. The stream ends with `data: [DONE]`.

**Off-curriculum case** — the stream emits a single `data:` with the off-curriculum string, then `data: [DONE]`.

**Frontend implementation note:** Use the browser's `EventSource` API or `fetch` with a streaming reader. The `[DONE]` sentinel signals the stream is finished.

---

## 7. Behavior the Frontend Must Handle

### Dual-layer answer format

Every on-curriculum answer follows this structure:

```
🗣️ شرح بسيط:
<explanation in Syrian colloquial Arabic>

📖 تعريف رسمي:
<formal definition in Modern Standard Arabic>
```

The frontend should render both sections. Splitting on the `📖` marker gives you the two parts if you want to display them separately.

### Off-curriculum detection

Check `sources.length === 0` or check if `answer === "هذا السؤال خارج المنهج الذي لديّ."` to detect and style off-curriculum replies differently (e.g. a yellow warning card instead of a chat bubble).

### Grade and subject filtering

- If the student's current session has a known grade (e.g. from the LMS user profile), always pass `grade` in the request. This prevents a Grade 10 student from receiving Grade 12 content.
- `subject` is optional and useful if the student is inside a specific subject module in the LMS.
- Both filters are applied inside the vector index — they do not silently drop results after retrieval.

### Response latency

| Mode | Typical latency |
|---|---|
| On-curriculum, `/chat` | 15–90 seconds (LLM generation time) |
| Off-curriculum, `/chat` | < 2 seconds (blocked before LLM) |
| `/chat/stream` | First token in ~5 seconds, then streaming |

Show a loading indicator immediately on submit. For `/chat/stream`, showing tokens as they arrive makes the wait feel shorter.

### CORS

The API uses the `ALLOWED_ORIGINS` list from `config.py`. If the frontend gets a CORS error, confirm the LMS origin is in that list and restart the server.

---

## 8. File Structure

```
Senior/
├── chatbot/
│   ├── config.py              ← All configuration (edit this)
│   ├── api/
│   │   └── main.py            ← FastAPI app (entry point)
│   ├── retrieval/
│   │   └── retriever.py       ← Hybrid dense + BM25 retrieval
│   ├── llm/
│   │   └── client.py          ← Ollama wrapper + system prompt
│   ├── indexing/
│   │   ├── indexer.py         ← Builds Qdrant + BM25 indexes (one-time)
│   │   ├── chunker.py         ← PDF text → overlapping chunks
│   │   └── normalizer.py      ← CAMeL Tools Arabic normalization
│   └── ocr/
│       └── pipeline.py        ← PyMuPDF + Surya OCR fallback (one-time)
├── scripts/
│   └── ingest.py              ← Run once to build the index from PDFs
├── qdrant_storage/            ← Pre-built index (do not delete)
│   ├── collection/
│   └── bm25.pkl
└── Data/                      ← Source PDFs (not needed at query time)
    ├── Grade_10/
    ├── Grade_11/
    └── Grade_12/
```

The `Data/` folder and `scripts/` are only needed if the index ever needs to be rebuilt. At query time only `qdrant_storage/` and the `chatbot/` package are required.

---

## 9. Troubleshooting

### Server fails to start — `bm25.pkl not found`

The BM25 index file is missing from `qdrant_storage/`. Either copy it from the source machine or re-run the ingestion:

```bash
python scripts/ingest.py
```

This takes 30–90 minutes and requires the PDFs in `Data/`.

### Server fails to start — Qdrant lock error

Another Python process is holding the Qdrant storage lock. Only one process may access the storage at a time.

```bash
# Find and kill the stale process (Windows)
Get-Process python | Stop-Process -Force

# Linux / macOS
pkill -f python
```

Then restart the server.

### `answer` is always the off-curriculum string for valid questions

The similarity threshold (`SIMILARITY_THRESHOLD = 0.60`) may be too strict for the deployed embedding model version. Lower it to `0.50` in `config.py` and restart. Do not go below `0.45` or the guardrail will start accepting genuinely off-curriculum queries.

### Embedding model fails to download

The server is air-gapped. Copy the HuggingFace cache directory from a connected machine:

```
~/.cache/huggingface/hub/models--jinaai--jina-embeddings-v3/
```

to the same path on the server.

### Ollama not reachable — `Connection refused`

Ollama must be running before the server starts:

```bash
ollama serve          # starts the Ollama daemon
ollama list           # confirm command-r7b-arabic:7b is present
```

### High memory usage / OOM

The GPU must hold both the embedding model (~1 GB) and the LLM (~6 GB). If you get out-of-memory errors, ensure no other GPU workload is running on the same machine. The attention detection model (Feature 1) must run on a separate GPU or machine.

### Slow first response after server restart

Expected. The embedding model is loaded lazily on the first request. Subsequent requests reuse the cached model. Optionally send a warm-up request to `/health` or a dummy `/chat` request on server startup to pre-load the model before real traffic arrives.
