"""Gemini-as-judge: re-rank candidate results by relevance to the Arabic query."""
import asyncio
import json
import logging
from typing import Any

import google.generativeai as genai

from config import settings

logger = logging.getLogger(__name__)


_RANK_PROMPT = """You are ranking educational content for an Arabic-speaking teacher.

Original teacher request (Arabic):
{query_ar}

Candidate items (numbered, each with title and snippet):
{candidates}

Return STRICT JSON only:
{{
  "ranked_indices": [<list of candidate numbers from best to worst>],
  "top_reason_ar": "<one short Arabic sentence explaining why the top pick is best>"
}}

Rules:
- Prefer Arabic-language content, but English is acceptable if clearly more relevant.
- Prefer clear educational/explanatory content over news or ads.
- Penalize off-topic or low-quality items.
"""


def _configure() -> None:
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=settings.gemini_api_key)


def _format_candidates(items: list[dict[str, Any]]) -> str:
    lines = []
    for i, it in enumerate(items):
        title = it.get("title", "")
        snippet = it.get("snippet") or it.get("description") or it.get("channel") or ""
        lines.append(f"[{i}] {title}\n     {snippet[:200]}")
    return "\n".join(lines)


async def rank_results(
    query_ar: str,
    candidates: list[Any],
    *,
    fallback_on_error: bool = True,
) -> list[int]:
    """Return candidate indices sorted by relevance.

    `candidates` is a list of Pydantic models or dicts that have .title and
    .snippet/.description.
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return [0]

    dicts = [c.model_dump() if hasattr(c, "model_dump") else dict(c) for c in candidates]

    try:
        _configure()
        model = genai.GenerativeModel(
            settings.gemini_model,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        )
        prompt = _RANK_PROMPT.format(
            query_ar=query_ar,
            candidates=_format_candidates(dicts),
        )
        resp = await asyncio.to_thread(model.generate_content, prompt)
        data = json.loads(resp.text.strip())
        ranked = data.get("ranked_indices", [])
        # Filter to valid indices, append any missing at the end
        seen = set()
        clean: list[int] = []
        for idx in ranked:
            if isinstance(idx, int) and 0 <= idx < len(candidates) and idx not in seen:
                clean.append(idx)
                seen.add(idx)
        for i in range(len(candidates)):
            if i not in seen:
                clean.append(i)
        return clean
    except Exception as e:
        logger.warning("Ranker failed (%s); falling back to original order", e)
        if fallback_on_error:
            return list(range(len(candidates)))
        raise
