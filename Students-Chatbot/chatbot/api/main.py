"""
FastAPI backend for the Arabic educational chatbot.
Endpoints:
  POST /chat          — returns full response
  POST /chat/stream   — returns SSE token stream
  POST /session/clear — clears a session's history (New Chat)
  GET  /health        — liveness check
"""
import logging
import uuid
from typing import Optional

from fastapi import FastAPI

log = logging.getLogger("chatbot.api")
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chatbot.config import ALLOWED_ORIGINS
from chatbot.retrieval.retriever import retrieve
from chatbot.llm.client import chat, chat_stream, _ABOVE_GRADE_REPLY, _NO_CONTEXT_REPLY

app = FastAPI(title="Arabic Educational Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ────────────────────────────────────────────────────
# { session_id: [ {"role": "user"|"assistant", "content": str}, ... ] }
# Only bare Q/A pairs are stored — NOT RAG contexts.
_sessions: dict[str, list[dict]] = {}


# ── Schemas ───────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    grade: Optional[str] = None
    subject: Optional[str] = None
    session_id: Optional[str] = None   # omit on first message; echo back on follow-ups


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    session_id: str                    # always returned so client can store it


class ClearRequest(BaseModel):
    session_id: str


def _last_user_message(history: list) -> Optional[str]:
    """Return the most recent user message from history, or None."""
    return next(
        (m["content"] for m in reversed(history) if m["role"] == "user"),
        None,
    )


# ── Grade-restriction helper ───────────────────────────────────────────────────

def _resolve_chunks(question: str, grade: Optional[str], subject: Optional[str]):
    """
    Two-pass retrieval that distinguishes "above grade" from "off curriculum":

    Pass 1: retrieve with grade filter
      → chunks found  → use them (normal flow)
      → empty:
          Pass 2: retrieve without grade filter
            → still empty → off-curriculum  (return [], None)
            → chunks found, any chunk.grade > req.grade → above-grade (return [], "above")
            → chunks found, none above grade → off-curriculum (return [], None)

    Returns (chunks, flag) where flag is None | "above_grade".
    """
    # Pass 1: grade-filtered retrieval
    chunks = retrieve(question, grade=grade, subject=subject)
    if chunks:
        return chunks, None

    # Pass 2: unfiltered retrieval (same subject if provided, no grade filter)
    all_chunks = retrieve(question, grade=None, subject=subject)
    if not all_chunks:
        return [], None   # genuinely off-curriculum

    # Check whether any returned chunk belongs to a higher grade
    if grade:
        try:
            req_grade_num = int(grade)
            for c in all_chunks:
                chunk_grade = c.get("grade", "")
                try:
                    if int(chunk_grade) > req_grade_num:
                        return [], "above_grade"
                except (ValueError, TypeError):
                    pass
        except (ValueError, TypeError):
            pass

    return [], None   # content exists but not for this grade/subject combo


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    # Session management
    session_id = req.session_id or str(uuid.uuid4())
    history    = _sessions.setdefault(session_id, [])

    # Attempt 1: retrieve with the bare question
    chunks, flag = _resolve_chunks(req.question, req.grade, req.subject)

    # Attempt 2: if nothing found AND we have history, the question is likely
    # a follow-up ("give me examples", "explain more", etc.).
    # Prepend the last user message so the retriever knows the topic.
    # The LLM still receives the ORIGINAL question — it already has history.
    if not chunks and flag is None and history:
        last_q = _last_user_message(history)
        if last_q:
            enriched = f"{last_q} {req.question}"
            log.info("session=%s  retrying with enriched query: %r", session_id[:8], enriched[:100])
            chunks, flag = _resolve_chunks(enriched, req.grade, req.subject)

    if flag == "above_grade":
        return ChatResponse(answer=_ABOVE_GRADE_REPLY, sources=[], session_id=session_id)

    log.info("session=%s  history_turns=%d  chunks=%d", session_id[:8], len(history) // 2, len(chunks))

    # LLM gets the original question — history already provides the topic context
    answer = chat(req.question, chunks, history=history, grade=req.grade)

    # Persist bare Q/A to session (no RAG context — keeps tokens low)
    history.append({"role": "user",      "content": req.question})
    history.append({"role": "assistant", "content": answer})
    log.info("session=%s  history now %d messages", session_id[:8], len(history))

    sources = [
        {"subject": c.get("subject"), "grade": c.get("grade"),
         "page": c.get("page_num"), "source": c.get("book") or c.get("source")}
        for c in chunks
    ]
    return ChatResponse(answer=answer, sources=sources, session_id=session_id)


@app.post("/chat/stream")
def chat_stream_endpoint(req: ChatRequest):
    # Session management
    session_id = req.session_id or str(uuid.uuid4())
    history    = _sessions.setdefault(session_id, [])

    # Attempt 1: bare question
    chunks, flag = _resolve_chunks(req.question, req.grade, req.subject)

    # Attempt 2: enrich with last user message if follow-up
    if not chunks and flag is None and history:
        last_q = _last_user_message(history)
        if last_q:
            chunks, flag = _resolve_chunks(f"{last_q} {req.question}", req.grade, req.subject)

    if flag == "above_grade":
        def _above():
            yield f"data: {_ABOVE_GRADE_REPLY}\n\n"
            yield f"data: [DONE]\n\n"
        return StreamingResponse(_above(), media_type="text/event-stream")

    # Collect streamed tokens so we can save the full answer to history
    collected: list[str] = []

    def generate():
        for token in chat_stream(req.question, chunks, history=history, grade=req.grade):
            collected.append(token)
            yield f"data: {token}\n\n"

        # Update history after stream completes
        full_answer = "".join(collected)
        history.append({"role": "user",      "content": req.question})
        history.append({"role": "assistant", "content": full_answer})

        yield f"data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/session/clear")
def clear_session(req: ClearRequest):
    """Clears conversation history for a session. Call this for 'New Chat'."""
    _sessions.pop(req.session_id, None)
    return {"cleared": True, "session_id": req.session_id}


@app.get("/session/{session_id}")
def get_session(session_id: str):
    """Debug: returns the stored history for a session."""
    history = _sessions.get(session_id)
    if history is None:
        return {"session_id": session_id, "found": False, "turns": 0, "history": []}
    return {
        "session_id": session_id,
        "found": True,
        "turns": len(history) // 2,
        "history": [
            {"role": m["role"], "preview": m["content"][:80]}
            for m in history
        ],
    }
