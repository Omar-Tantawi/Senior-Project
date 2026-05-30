"""
Ollama LLM client with the Syrian Arabic educational system prompt.
Enforces CoT reasoning and dual-layer responses (Syrian colloquial + MSA).
Supports conversation history for follow-up questions.
"""
from typing import List, Optional

import ollama

from chatbot.config import LLM_MODEL, SIMILARITY_THRESHOLD, MAX_HISTORY_TURNS

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_BASE = """\
أنت مساعد تعليمي ذكي لطلاب المرحلة الثانوية في سوريا.
مهمتك الإجابة على أسئلة الطلاب بناءً على المناهج الدراسية السورية المقدمة فقط.

قواعد صارمة:
١. أجب دائماً بطريقتين متتاليتين:
   🗣️ شرح بسيط: اشرح الفكرة بالعامية السورية بأسلوب واضح وقريب من الطالب.
   📖 تعريف رسمي: قدم التعريف أو الشرح الأكاديمي بالعربية الفصحى.
٢. للمسائل الرياضية والعلمية: اتبع أسلوب التفكير خطوة بخطوة قبل الإجابة النهائية.
٣. لا تخرج عن المحتوى المقدم أدناه. إذا لم يكن الجواب في المحتوى، أجب بالجملة التالية فقط:
   "هذا السؤال خارج المنهج الذي لديّ."
٤. لا تخترع أرقاماً أو معادلات أو معلومات غير موجودة في المحتوى.
"""

_NO_CONTEXT_REPLY = "هذا السؤال خارج المنهج الذي لديّ."

_ABOVE_GRADE_REPLY = (
    "هذا الموضوع يُدرَّس في صف أعلى من صفك الحالي. "
    "يُنصح بالتركيز على منهج صفك أولاً."
)


def _build_system_prompt(grade: Optional[str]) -> str:
    """Inject the student's grade into the system prompt as a secondary guardrail."""
    if grade:
        grade_line = f"الطالب في الصف {grade}. أجب فقط على الأسئلة المتعلقة بمنهج هذا الصف.\n"
        return grade_line + _SYSTEM_PROMPT_BASE
    return _SYSTEM_PROMPT_BASE


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        header = f"[{c.get('book') or c.get('source', '')} — الصف {c.get('grade', '')} — صفحة {c.get('page_num', '')}]"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# ── Public API ─────────────────────────────────────────────────────────────────

def chat(
    question: str,
    chunks: list[dict],
    history: Optional[List[dict]] = None,
    grade: Optional[str] = None,
) -> str:
    """
    Sends question + retrieved context to Ollama.
    Returns the model's reply. If no chunks passed, returns the out-of-scope reply.

    history: list of {"role": "user"|"assistant", "content": str} from prior turns.
             Only the bare question/answer pairs are passed — NOT previous RAG contexts.
             The RAG context is only attached to the current user message.
    """
    if not chunks:
        return _NO_CONTEXT_REPLY

    context      = _build_context(chunks)
    user_message = f"المحتوى المرجعي:\n{context}\n\nسؤال الطالب: {question}"
    history      = history or []

    messages = (
        [{"role": "system", "content": _build_system_prompt(grade)}]
        + history[-MAX_HISTORY_TURNS:]          # trim to last N messages
        + [{"role": "user", "content": user_message}]
    )

    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"]


def chat_stream(
    question: str,
    chunks: list[dict],
    history: Optional[List[dict]] = None,
    grade: Optional[str] = None,
):
    """Generator variant — yields text tokens for streaming API responses."""
    if not chunks:
        yield _NO_CONTEXT_REPLY
        return

    context      = _build_context(chunks)
    user_message = f"المحتوى المرجعي:\n{context}\n\nسؤال الطالب: {question}"
    history      = history or []

    messages = (
        [{"role": "system", "content": _build_system_prompt(grade)}]
        + history[-MAX_HISTORY_TURNS:]
        + [{"role": "user", "content": user_message}]
    )

    stream = ollama.chat(model=LLM_MODEL, messages=messages, stream=True)
    for chunk in stream:
        yield chunk["message"]["content"]
