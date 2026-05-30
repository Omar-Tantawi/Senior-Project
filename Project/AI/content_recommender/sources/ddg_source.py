"""DuckDuckGo text search for articles and PDFs (no API key, no quota)."""

from models import ContentItem
from sources._util import detect_language, make_query


def _ddg_text(query: str, limit: int) -> list[dict]:
    if not query.strip():
        return []
    try:
        from ddgs import DDGS
    except ImportError:
        print("[ddg_source] ddgs not installed.")
        return []
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=limit, region="wt-wt"))
    except Exception as e:
        print(f"[ddg_source] search failed for '{query}': {e}")
        return []


def _to_items(raw: list[dict], content_type: str) -> list[ContentItem]:
    seen: set[str] = set()
    items: list[ContentItem] = []
    for r in raw:
        url   = r.get("href") or r.get("url") or ""
        if not url or url in seen:
            continue
        seen.add(url)

        title = (r.get("title") or "").strip()
        body  = (r.get("body")  or "").strip()

        items.append(ContentItem(
            content_type=content_type,
            title=title or url,
            url=url,
            description=body,
            source="duckduckgo",
            language=detect_language(f"{title} {body}"),
        ))
    return items


def search_articles(keywords_en: list[str], keywords_ar: list[str], max_results: int) -> list[ContentItem]:
    raw: list[dict] = []
    if keywords_en:
        raw += _ddg_text(make_query(keywords_en), max_results)
    if keywords_ar:
        raw += _ddg_text(make_query(keywords_ar), max_results)
    return _to_items(raw, "article")


def search_pdfs(keywords_en: list[str], keywords_ar: list[str], max_results: int) -> list[ContentItem]:
    raw: list[dict] = []
    if keywords_en:
        raw += _ddg_text(make_query(keywords_en, "filetype:pdf"), max_results)
    if keywords_ar:
        raw += _ddg_text(make_query(keywords_ar, "filetype:pdf"), max_results)

    items = _to_items(raw, "pdf")
    # Keep only entries that look like PDFs
    return [it for it in items if it.url.lower().endswith(".pdf") or "/pdf/" in it.url.lower()]
