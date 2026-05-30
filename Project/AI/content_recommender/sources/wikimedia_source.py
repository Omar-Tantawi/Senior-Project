"""Wikimedia Commons image search (free, no API key)."""

import json
import urllib.parse
import urllib.request
import urllib.error

from config import SEARCH_TIMEOUT
from models import ContentItem
from sources._util import detect_language, make_query

API_URL    = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "ContentRecommenderBot/1.0 (educational-recommender)"


def _api_call(params: dict) -> dict:
    qs  = urllib.parse.urlencode(params)
    url = f"{API_URL}?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"[wikimedia_source] API call failed: {e}")
        return {}


def _search_titles(query: str, limit: int) -> list[dict]:
    if not query.strip():
        return []
    data = _api_call({
        "action":     "query",
        "list":       "search",
        "srsearch":   query,
        "srnamespace": 6,          # File namespace = images/media
        "srlimit":    limit,
        "format":     "json",
    })
    return (data.get("query") or {}).get("search") or []


def _resolve_images(titles: list[str]) -> dict[str, dict]:
    """Batch-resolve File: titles to image URLs."""
    if not titles:
        return {}
    data = _api_call({
        "action": "query",
        "titles": "|".join(titles),
        "prop":   "imageinfo",
        "iiprop": "url|thumbnail|extmetadata",
        "iiurlwidth": 800,
        "format": "json",
    })
    pages = (data.get("query") or {}).get("pages") or {}
    out: dict[str, dict] = {}
    for page in pages.values():
        title = page.get("title")
        info  = (page.get("imageinfo") or [{}])[0]
        if title and info:
            out[title] = info
    return out


def search(keywords_en: list[str], keywords_ar: list[str], max_results: int) -> list[ContentItem]:
    hits: list[dict] = []
    if keywords_en:
        hits += _search_titles(make_query(keywords_en), max_results)
    if keywords_ar:
        hits += _search_titles(make_query(keywords_ar), max_results)

    if not hits:
        return []

    # Deduplicate hit titles, preserve order
    titles: list[str] = []
    seen: set[str] = set()
    for h in hits:
        t = h.get("title")
        if t and t not in seen:
            seen.add(t)
            titles.append(t)

    image_info = _resolve_images(titles[: max_results * 2])

    items: list[ContentItem] = []
    for title in titles:
        info = image_info.get(title)
        if not info:
            continue
        img_url   = info.get("thumburl") or info.get("url")
        desc_url  = info.get("descriptionurl") or img_url
        if not img_url:
            continue

        clean_title = title.replace("File:", "").rsplit(".", 1)[0].replace("_", " ")
        items.append(ContentItem(
            content_type="image",
            title=clean_title,
            url=desc_url,
            description="",
            thumbnail_url=img_url,
            source="wikimedia",
            language=detect_language(clean_title),
        ))

    return items
