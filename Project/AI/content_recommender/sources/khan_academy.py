"""
Khan Academy search — uses their internal search API (no API key required)
plus a DuckDuckGo site:khanacademy.org fallback.

Returns videos and articles from Khan Academy in both English and Arabic.
Khan Academy Arabic: https://ar.khanacademy.org
"""

import json
import urllib.parse
import urllib.request
import urllib.error
from typing import Optional

from config import SEARCH_TIMEOUT
from models import ContentItem
from sources._util import detect_language, make_query

KA_API      = "https://www.khanacademy.org/api/v1/search"
KA_AR_API   = "https://ar.khanacademy.org/api/v1/search"
KA_BASE_URL = "https://www.khanacademy.org"
KA_AR_URL   = "https://ar.khanacademy.org"
USER_AGENT  = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


def _ka_search(query: str, lang: str = "en", limit: int = 10) -> list[dict]:
    """Hit Khan Academy internal search API."""
    if not query.strip():
        return []

    base = KA_AR_API if lang == "ar" else KA_API
    params = urllib.parse.urlencode({
        "q":       query,
        "lang":    lang,
        "limit":   limit,
    })
    url = f"{base}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            data = json.loads(resp.read())
            return data.get("results") or data.get("hits") or []
    except Exception as e:
        print(f"[khan_academy] API failed (lang={lang}): {e}")
        return []


def _ddg_ka_search(query: str, lang: str = "en", limit: int = 10) -> list[ContentItem]:
    """Fallback: DuckDuckGo site:khanacademy.org search."""
    try:
        from ddgs import DDGS
    except ImportError:
        return []

    site = "ar.khanacademy.org" if lang == "ar" else "khanacademy.org"
    full_query = f"site:{site} {query}"

    try:
        with DDGS() as ddgs:
            raw = list(ddgs.text(full_query, max_results=limit, region="wt-wt"))
    except Exception as e:
        print(f"[khan_academy] DDG fallback failed: {e}")
        return []

    items = []
    for r in raw:
        url   = r.get("href") or r.get("url") or ""
        title = (r.get("title") or "").strip()
        desc  = (r.get("body")  or "").strip()
        if not url or "khanacademy.org" not in url:
            continue

        # Determine content type from URL
        ctype = "video" if "/v/" in url else "article"

        items.append(ContentItem(
            content_type=ctype,
            title=title,
            url=url,
            description=desc,
            source="khan_academy",
            language=detect_language(f"{title} {desc}"),
        ))
    return items


def _parse_ka_result(r: dict, lang: str) -> Optional[ContentItem]:
    """Parse a single Khan Academy API result into a ContentItem."""
    kind  = r.get("kind") or r.get("node_type") or ""
    title = (r.get("title") or r.get("translated_title") or "").strip()
    desc  = (r.get("description") or r.get("translated_description") or "").strip()
    slug  = r.get("relative_url") or r.get("ka_url") or ""

    if not title or not slug:
        return None

    base_url = KA_AR_URL if lang == "ar" else KA_BASE_URL
    url = slug if slug.startswith("http") else f"{base_url}{slug}"

    # Map KA kinds to our content types
    if any(k in kind.lower() for k in ["video", "exercise"]):
        ctype = "video"
    else:
        ctype = "article"

    # Thumbnail from YouTube if it's a video
    yt_id = r.get("youtube_id")
    thumb = f"https://img.youtube.com/vi/{yt_id}/mqdefault.jpg" if yt_id else None

    return ContentItem(
        content_type=ctype,
        title=title,
        url=url,
        description=desc,
        thumbnail_url=thumb,
        source="khan_academy",
        language="ar" if lang == "ar" else "en",
    )


def search(keywords_en: list[str], keywords_ar: list[str], max_results: int) -> list[ContentItem]:
    items: list[ContentItem] = []
    seen:  set[str] = set()

    # 1. Try KA API for English
    if keywords_en:
        query_en = make_query(keywords_en)
        raw_en   = _ka_search(query_en, lang="en", limit=max_results)

        if raw_en:
            for r in raw_en:
                item = _parse_ka_result(r, lang="en")
                if item and item.url not in seen:
                    seen.add(item.url)
                    items.append(item)
        else:
            # Fallback to DDG
            for item in _ddg_ka_search(query_en, lang="en", limit=max_results):
                if item.url not in seen:
                    seen.add(item.url)
                    items.append(item)

    # 2. Try KA API for Arabic
    if keywords_ar:
        query_ar = make_query(keywords_ar)
        raw_ar   = _ka_search(query_ar, lang="ar", limit=max_results)

        if raw_ar:
            for r in raw_ar:
                item = _parse_ka_result(r, lang="ar")
                if item and item.url not in seen:
                    seen.add(item.url)
                    items.append(item)
        else:
            # Fallback to DDG Arabic KA
            for item in _ddg_ka_search(query_ar, lang="ar", limit=max_results):
                if item.url not in seen:
                    seen.add(item.url)
                    items.append(item)

    return items
