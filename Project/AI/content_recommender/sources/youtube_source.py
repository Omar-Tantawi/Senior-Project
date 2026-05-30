"""YouTube search via YouTube Data API v3."""
import json
import urllib.request
import urllib.parse
import urllib.error
from typing import Optional

from config import YOUTUBE_API_KEY
from models import ContentItem
from sources._util import detect_language, make_query

_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


def _api_get(url: str, params: dict) -> Optional[dict]:
    full_url = f"{url}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(full_url, timeout=10) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError) as e:
        print(f"[youtube_source] API request failed: {e}")
        return None


def _fetch_durations(video_ids: list[str]) -> dict[str, str]:
    if not video_ids:
        return {}
    data = _api_get(_VIDEOS_URL, {
        "key":  YOUTUBE_API_KEY,
        "id":   ",".join(video_ids),
        "part": "contentDetails",
    })
    if not data:
        return {}
    result: dict[str, str] = {}
    for item in data.get("items", []):
        vid_id = item.get("id", "")
        iso = item.get("contentDetails", {}).get("duration", "")
        result[vid_id] = _parse_iso_duration(iso)
    return result


def _parse_iso_duration(iso: str) -> str:
    import re
    if not iso:
        return ""
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso)
    if not m:
        return ""
    h, mn, s = int(m.group(1) or 0), int(m.group(2) or 0), int(m.group(3) or 0)
    if h:
        return f"{h}:{mn:02d}:{s:02d}"
    return f"{mn}:{s:02d}"


def _search_one(query: str, limit: int) -> list[dict]:
    if not query.strip():
        return []
    data = _api_get(_SEARCH_URL, {
        "key":        YOUTUBE_API_KEY,
        "q":          query,
        "part":       "snippet",
        "type":       "video",
        "maxResults": min(limit, 50),
    })
    return data.get("items", []) if data else []


def search(keywords_en: list[str], keywords_ar: list[str], max_results: int) -> list[ContentItem]:
    raw: list[dict] = []
    if keywords_en:
        raw += _search_one(make_query(keywords_en, "educational"), max_results)
    if keywords_ar:
        raw += _search_one(make_query(keywords_ar), max_results)

    seen: set[str] = set()
    video_ids: list[str] = []
    pending: list[tuple[str, dict]] = []

    for r in raw:
        if not r:
            continue
        vid_id = (r.get("id") or {}).get("videoId") or ""
        if not vid_id or vid_id in seen:
            continue
        seen.add(vid_id)
        video_ids.append(vid_id)
        pending.append((vid_id, r))

    durations = _fetch_durations(video_ids)

    items: list[ContentItem] = []
    for vid_id, r in pending:
        snippet = r.get("snippet") or {}
        title   = (snippet.get("title") or "").strip()
        desc    = (snippet.get("description") or "").strip()
        thumbs  = snippet.get("thumbnails") or {}
        thumb   = (
            (thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {})
            .get("url")
        )
        url = f"https://www.youtube.com/watch?v={vid_id}"
        items.append(ContentItem(
            content_type="video",
            title=title or "(untitled)",
            url=url,
            description=desc,
            thumbnail_url=thumb,
            source="youtube",
            language=detect_language(f"{title} {desc}"),
            duration=durations.get(vid_id, ""),
        ))

    return items
