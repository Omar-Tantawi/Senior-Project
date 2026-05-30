"""
Arabic educational sources — Sabaq.net and Edraak.org.

Sabaq.net:  Saudi K-12 curriculum videos (free, Arabic only)
Edraak.org: Arabic MOOC platform by Queen Rania Foundation

Both are searched via DuckDuckGo (no API key needed, no broken internal APIs).
"""

from models import ContentItem
from sources._util import detect_language, make_query

_SOURCES = [
    {
        "site":   "sabaq.net",
        "name":   "sabaq",
        "ctype":  "video",
    },
    {
        "site":   "edraak.org",
        "name":   "edraak",
        "ctype":  "article",
    },
]


def _ddg_search(query: str, site: str, limit: int) -> list[dict]:
    if not query.strip():
        return []
    try:
        from ddgs import DDGS
    except ImportError:
        print("[arabic_edu_source] ddgs not installed.")
        return []
    full_query = f"site:{site} {query}"
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(full_query, max_results=limit, region="wt-wt"))
    except Exception as e:
        print(f"[arabic_edu_source] DDG failed for site:{site} — {e}")
        return []


def search(keywords_en: list[str], keywords_ar: list[str], max_results: int) -> list[ContentItem]:
    items: list[ContentItem] = []
    seen:  set[str] = set()

    for source in _SOURCES:
        # Arabic keywords first — these are Arabic-language sites
        queries = []
        if keywords_ar:
            queries.append(make_query(keywords_ar))
        if keywords_en:
            queries.append(make_query(keywords_en))

        for query in queries:
            raw = _ddg_search(query, source["site"], max_results)
            for r in raw:
                url   = r.get("href") or r.get("url") or ""
                title = (r.get("title") or "").strip()
                desc  = (r.get("body")  or "").strip()

                if not url or url in seen:
                    continue
                if source["site"] not in url:
                    continue

                seen.add(url)
                ctype = "video" if (
                    source["ctype"] == "video" or
                    "/v/" in url or
                    "video" in url.lower()
                ) else source["ctype"]

                items.append(ContentItem(
                    content_type=ctype,
                    title=title or url,
                    url=url,
                    description=desc,
                    source=source["name"],
                    language=detect_language(f"{title} {desc}"),
                ))

    return items
