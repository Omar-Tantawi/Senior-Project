"""YouTube search via yt-dlp (no API key, no quota, actively maintained)."""
from __future__ import annotations
from models import ContentItem
from sources._util import detect_language, make_query


def _search_one(query: str, limit: int) -> list[dict]:
    if not query.strip():
        return []
    try:
        import yt_dlp
    except ImportError:
        print("[youtube_source] yt-dlp not installed.")
        return []

    opts = {
        "quiet":         True,
        "no_warnings":   True,
        "extract_flat":  True,
        "skip_download": True,
        "default_search": f"ytsearch{limit}",
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(query, download=False)
            return info.get("entries") or []
    except Exception as e:
        print(f"[youtube_source] search failed for '{query}': {e}")
        return []


def _fmt_duration(seconds) -> str | None:
    if seconds is None:
        return None
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return None
    if s < 0:
        return None
    h, rem = divmod(s, 3600)
    m, ss  = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{ss:02d}"
    return f"{m}:{ss:02d}"


def search(keywords_en: list[str], keywords_ar: list[str], max_results: int) -> list[ContentItem]:
    raw: list[dict] = []
    if keywords_en:
        raw += _search_one(make_query(keywords_en, "educational"), max_results)
    if keywords_ar:
        raw += _search_one(make_query(keywords_ar), max_results)

    seen: set[str] = set()
    items: list[ContentItem] = []

    for r in raw:
        if not r:
            continue
        vid   = r.get("id") or ""
        url   = r.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else "")
        if not url or vid in seen:
            continue
        seen.add(vid or url)

        title = (r.get("title") or "").strip()
        desc  = (r.get("description") or "").strip()

        thumbs = r.get("thumbnails") or []
        thumb = None
        if thumbs and isinstance(thumbs, list):
            last = thumbs[-1]
            if isinstance(last, dict):
                thumb = last.get("url")

        items.append(ContentItem(
            content_type="video",
            title=title or "(untitled)",
            url=url,
            description=desc,
            thumbnail_url=thumb,
            source="youtube",
            language=detect_language(f"{title} {desc}"),
            duration=_fmt_duration(r.get("duration")),
        ))

    return items
