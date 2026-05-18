"""Parse Arabic teacher queries into structured intent using Google Gemini."""
import asyncio
import json
import logging

import google.generativeai as genai

from config import settings
from models.schemas import ParsedIntent

logger = logging.getLogger(__name__)


_PARSE_PROMPT = """You are an educational content assistant.
The teacher writes in Arabic describing a topic they want to teach.
Return STRICT JSON (no markdown fences, no extra text) with this exact schema:

{{
  "topic_ar": "<the main topic in Arabic>",
  "topic_en": "<same topic translated to English>",
  "keywords_ar": ["<3-5 Arabic search keywords>"],
  "keywords_en": ["<3-5 English search keywords>"],
  "grade_level": "<elementary|middle_school|high_school|university|general>",
  "subject": "<physics|chemistry|biology|math|history|geography|language|art|computer_science|general>",
  "preferred_media": ["<video|animation|pdf|article|image>"],
  "search_query_ar": "<single best Arabic search phrase ~6 words>",
  "search_query_en": "<single best English search phrase ~6 words>"
}}

Teacher input:
{query}
"""


def _configure() -> None:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in.")
    genai.configure(api_key=settings.gemini_api_key)


async def parse_arabic_query(text: str) -> ParsedIntent:
    """Call Gemini and return a validated ParsedIntent."""
    _configure()
    model = genai.GenerativeModel(
        settings.gemini_model,
        generation_config={
            "response_mime_type": "application/json",
            "temperature": 0.2,
        },
    )
    prompt = _PARSE_PROMPT.format(query=text)
    logger.debug("Sending parse prompt to Gemini")

    # The SDK is sync; offload to thread so we don't block the event loop.
    response = await asyncio.to_thread(model.generate_content, prompt)
    raw = response.text.strip()
    logger.debug("Gemini raw: %s", raw[:300])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        # Sometimes models still wrap in fences despite mime hint
        cleaned = raw.strip("`").lstrip("json").strip()
        data = json.loads(cleaned)

    return ParsedIntent(**data)
