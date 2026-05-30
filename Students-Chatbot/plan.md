# Plan: Chat Memory + Grade-Level Restriction

## Context

The chatbot currently treats every request as a cold start — no conversation history is preserved between messages in the same session. This means follow-up questions ("explain that more", "give me another example") fail because the LLM has no prior context.

Additionally, the grade filter today is purely a retrieval filter: if a 10th grader asks a 12th-grade question, the retriever finds no chunks for grade 10 and the bot replies "هذا السؤال خارج المنهج الذي لديّ." — which is misleading. The content *is* in the curriculum, just not for that grade. We need a distinct, accurate rejection message for this case.

---

## Feature 1: Conversation Memory (Session History)

### Approach

Add a server-side in-memory session store keyed by `session_id` (UUID). The client echoes this ID on every subsequent message in the same chat. History is stored as bare question/answer pairs — **not** the full RAG context — to avoid context window bloat.

**Message array sent to Ollama per turn:**
```
[system_prompt]              ← unchanged
[user: Q1]                   ← history (bare question only, no RAG context)
[assistant: A1]              ← history
[user: Q2]                   ← history
[assistant: A2]              ← history
...
[user: Q_new + RAG context]  ← current turn (RAG context only attached here)
```

### Changes

**`chatbot/config.py`** — add one constant:
```python
MAX_HISTORY_TURNS = 10   # 10 messages = 5 back-and-forth turns kept in context
```

**`chatbot/llm/client.py`**

- `chat(question, chunks, history=[])`: build messages as `[system] + history[-MAX_HISTORY_TURNS:] + [current_user_msg]`
- `chat_stream(question, chunks, history=[])`: same; history parameter passed through
- Import `MAX_HISTORY_TURNS` from config

**`chatbot/api/main.py`**

- Add `_sessions: dict[str, list[dict]] = {}` module-level store
- `ChatRequest`: add `session_id: Optional[str] = None`
- `ChatResponse`: add `session_id: str`
- In `chat_endpoint`:
  1. If `session_id` is None → `session_id = str(uuid.uuid4())`
  2. `history = _sessions.setdefault(session_id, [])`
  3. Call `chat(req.question, chunks, history)`
  4. Append `{"role": "user", "content": req.question}` and `{"role": "assistant", "content": answer}` to history
  5. Return `session_id` in response
- In `chat_stream_endpoint`: same setup, but collect tokens inside `generate()` so history is updated after the stream ends
- Add `POST /session/clear` endpoint: `{"session_id": "..."}` → clears that session's history (for a "New Chat" button in the frontend)
- Add `from typing import Optional` and `import uuid`

---

## Feature 2: Grade-Level Restriction

### Approach

Two-pass retrieval in the API layer. No LLM involvement — the decision is made purely from metadata on the retrieved chunks.

**Logic (only runs when `grade` is set in the request):**

```
Pass 1: retrieve(query, grade=req.grade, subject)  →  grade_chunks
  ├─ grade_chunks not empty  →  normal flow (existing behaviour, unchanged)
  └─ grade_chunks empty:
       Pass 2: retrieve(query, subject=subject)    →  all_chunks (no grade filter)
         ├─ all_chunks empty          →  _NO_CONTEXT_REPLY  (not in curriculum at all)
         └─ all_chunks not empty:
              Check grades of returned chunks
              ├─ any chunk.grade > req.grade  →  _ABOVE_GRADE_REPLY
              └─ otherwise                   →  _NO_CONTEXT_REPLY
```

**New constant** (add to `chatbot/llm/client.py`):
```python
_ABOVE_GRADE_REPLY = (
    "هذا الموضوع يُدرَّس في صف أعلى من صفك الحالي. "
    "يُنصح بالتركيز على منهج صفك أولاً."
)
```

**Dynamic system prompt** — inject the student's grade so the LLM is also aware of it as a secondary guardrail:

```python
_SYSTEM_PROMPT_BASE = """..."""  # existing prompt text, renamed

def _build_system_prompt(grade: Optional[str]) -> str:
    grade_line = (
        f"الطالب في الصف {grade}. أجب فقط على الأسئلة المتعلقة بمنهج هذا الصف.\n"
        if grade else ""
    )
    return grade_line + _SYSTEM_PROMPT_BASE
```

The two-pass retrieval is the primary enforcement. The system prompt injection is a reinforcing secondary guard.

### Changes

**`chatbot/llm/client.py`**
- Rename `_SYSTEM_PROMPT` → `_SYSTEM_PROMPT_BASE`
- Add `_build_system_prompt(grade)` function
- Add `_ABOVE_GRADE_REPLY` constant
- `chat()` and `chat_stream()` accept `grade: Optional[str] = None` and pass it to `_build_system_prompt`

**`chatbot/api/main.py`**
- Replace single `retrieve()` call with the two-pass logic above
- Import `_ABOVE_GRADE_REPLY` from `chatbot.llm.client`
- Short-circuit to `ChatResponse(answer=_ABOVE_GRADE_REPLY, sources=[], session_id=session_id)` when above-grade is detected

---

## Files Modified

| File | Change |
|------|--------|
| `chatbot/config.py` | Add `MAX_HISTORY_TURNS = 10` |
| `chatbot/llm/client.py` | History param, dynamic grade-aware system prompt, `_ABOVE_GRADE_REPLY` |
| `chatbot/api/main.py` | Session store, two-pass retrieval, new `/session/clear` endpoint |

---

## Verification

1. **Memory — follow-up works**: Send two requests with the same `session_id` where the second is "اشرحه أكثر" (explain more). The LLM should reference the first topic without it being re-stated.
2. **Cold start preserved**: Send a request with no `session_id` → a new UUID is returned and the response is correct.
3. **Session clear**: Call `/session/clear`, then send a follow-up → the LLM treats it as a brand new conversation.
4. **Above-grade detected**: Send `{"question": "<grade 12 topic>", "grade": "10"}` → response is `_ABOVE_GRADE_REPLY`, not `_NO_CONTEXT_REPLY`.
5. **Truly off-curriculum**: Send a non-curriculum question with `grade: "10"` → response is `_NO_CONTEXT_REPLY`.
6. **Existing behaviour unchanged**: Send a valid grade-10 question with `grade: "10"` → normal dual-layer answer returned.
