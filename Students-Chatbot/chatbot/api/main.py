"""
FastAPI backend for the Arabic educational chatbot.
Endpoints:
  POST /chat        — returns full response
  POST /chat/stream — returns SSE token stream
  GET  /health      — liveness check
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
from pydantic import BaseModel

from chatbot.config import ALLOWED_ORIGINS
from chatbot.retrieval.retriever import retrieve
from chatbot.llm.client import chat, chat_stream

app = FastAPI(title="Arabic Educational Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    grade: Optional[str] = None
    subject: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    chunks = retrieve(req.question, grade=req.grade, subject=req.subject)
    answer = chat(req.question, chunks)
    sources = [
        {"subject": c.get("subject"), "grade": c.get("grade"),
         "page": c.get("page_num"), "source": c.get("book") or c.get("source")}
        for c in chunks
    ]
    return ChatResponse(answer=answer, sources=sources)


@app.post("/chat/stream")
def chat_stream_endpoint(req: ChatRequest):
    chunks = retrieve(req.question, grade=req.grade, subject=req.subject)

    def generate():
        for token in chat_stream(req.question, chunks):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
