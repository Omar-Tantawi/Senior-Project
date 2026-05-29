"""
Ollama LLM client with the Syrian Arabic educational system prompt.
Enforces CoT reasoning and dual-layer responses (Syrian colloquial + MSA).
"""
import ollama

from chatbot.config import LLM_MODEL, SIMILARITY_THRESHOLD

_SYSTEM_PROMPT = """\
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


def _build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        header = f"[{c.get('book') or c.get('source', '')} — الصف {c.get('grade', '')} — صفحة {c.get('page_num', '')}]"
        parts.append(f"{header}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def chat(question: str, chunks: list[dict]) -> str:
    """
    Sends question + retrieved context to Ollama.
    Returns the model's reply. If no chunks passed, returns the out-of-scope reply.
    """
    if not chunks:
        return _NO_CONTEXT_REPLY

    context = _build_context(chunks)
    user_message = f"المحتوى المرجعي:\n{context}\n\nسؤال الطالب: {question}"

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"]


def chat_stream(question: str, chunks: list[dict]):
    """Generator variant — yields text tokens for streaming API responses."""
    if not chunks:
        yield _NO_CONTEXT_REPLY
        return

    context = _build_context(chunks)
    user_message = f"المحتوى المرجعي:\n{context}\n\nسؤال الطالب: {question}"

    stream = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]
