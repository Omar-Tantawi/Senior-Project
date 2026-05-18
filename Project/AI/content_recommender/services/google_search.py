"""Google Programmable Search Engine (Custom Search JSON API) wrapper."""
import logging
from typing import Iterable, Optional

import httpx

from config import settings
from models.schemas import ImageResult, WebResult

logger = logging.getLogger(__name__)

_CSE_URL = "https://www.googleapis.com/customsearch/v1"


async def search_web(
    query: str,
    *,
    filetype: Optional[str] = None,        # e.g. "pdf"
    exclude_youtube: bool = True,
    start: int = 1,                          # 1-based; CSE returns 10/page, max start=91
    exclude: Optional[Iterable[str]] = None,
    num: int = 10,
) -> list[WebResult]:
    """Generic web search for articles/PDFs."""
    if not (settings.google_api_key and settings.google_cse_id):
        raise RuntimeError("GOOGLE_API_KEY and GOOGLE_CSE_ID are required for Custom Search.")

    exclude_set = set(exclude or [])

    q = query
    if filetype:
        q += f" filetype:{filetype}"
    if exclude_youtube:
        q += " -site:youtube.com"

    params = {
        "key": settings.google_api_key,
        "cx": settings.google_cse_id,
        "q": q,
        "num": str(min(num, 10)),
        "start": str(start),
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(_CSE_URL, params=params)
        if resp.status_code == 429 or resp.status_code == 403:
            logger.warning("Custom Search quota/auth issue: %s", resp.text[:200])
            return []
        resp.raise_for_status()
        payload = resp.json()

    results: list[WebResult] = []
    for it in payload.get("items", []):
        url = it.get("link", "")
        if url in exclude_set:
            continue
        results.append(WebResult(
            title=it.get("title", ""),
            url=url,
            snippet=it.get("snippet", ""),
            source=it.get("displayLink", ""),
        ))
    return results


async def search_images(
    query: str,
    *,
    start: int = 1,
    exclude: Optional[Iterable[str]] = None,
    num: int = 10,
) -> list[ImageResult]:
    if not (settings.google_api_key and settings.google_cse_id):
        raise RuntimeError("GOOGLE_API_KEY and GOOGLE_CSE_ID are required.")

    exclude_set = set(exclude or [])

    params = {
        "key": settings.google_api_key,
        "cx": settings.google_cse_id,
        "q": query,
        "searchType": "image",
        "safe": "active",
        "num": str(min(num, 10)),
        "start": str(start),
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(_CSE_URL, params=params)
        if resp.status_code in (403, 429):
            logger.warning("Custom Search (images) quota/auth issue")
            return []
        resp.raise_for_status()
        payload = resp.json()

    results: list[ImageResult] = []
    for it in payload.get("items", []):
        url = it.get("link", "")
        if url in exclude_set:
            continue
        img = it.get("image", {})
        results.append(ImageResult(
            title=it.get("title", ""),
            url=url,
            thumbnail=img.get("thumbnailLink"),
            source=img.get("contextLink") or it.get("displayLink"),
        ))
    return results
