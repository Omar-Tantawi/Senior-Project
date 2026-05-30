"""OpenAlex academic search (free, no API key, ~250M papers, no aggressive rate limit)."""

import json
import urllib.parse
import urllib.request
import urllib.error
from typing import Optional

from config import SEARCH_TIMEOUT
from models import ContentItem
from sources._util import make_query

API_URL    = "https://api.openalex.org/works"
USER_AGENT = "ContentRecommenderBot/1.0 (educational-recommender)"


def search(keywords_en: list[str], max_results: int) -> list[ContentItem]:
    query = make_query(keywords_en)
    if not query:
        return []

    params = urllib.parse.urlencode({
        "search":   query,
        "per-page": max_results,
        "select":   "title,abstract_inverted_index,doi,publication_year,open_access,primary_location,language",
    })
    # Putting an email in the request gives access to the "polite pool" — better rate limits
    url = f"{API_URL}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    try:
        with urllib.request.urlopen(req, timeout=SEARCH_TIMEOUT) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        print(f"[openalex] search failed: {e}")
        return []

    items: list[ContentItem] = []
    for r in data.get("results") or []:
        title = (r.get("title") or "").strip()
        if not title:
            continue

        # Pick the best URL: OA PDF → DOI → landing page
        pdf_url     = (r.get("open_access") or {}).get("oa_url")
        landing_url = ((r.get("primary_location") or {}).get("landing_page_url"))
        doi         = r.get("doi") or ""
        url_        = pdf_url or landing_url or doi
        if not url_:
            continue

        is_pdf = bool(pdf_url) and pdf_url.lower().endswith(".pdf")

        # OpenAlex returns abstracts as an "inverted index" — reconstruct it
        abstract = _reconstruct_abstract(r.get("abstract_inverted_index"))

        items.append(ContentItem(
            content_type="pdf" if is_pdf else "article",
            title=title,
            url=url_,
            description=abstract,
            source="openalex",
            language=r.get("language") or "en",
            year=r.get("publication_year") if isinstance(r.get("publication_year"), int) else None,
        ))

    return items


def _reconstruct_abstract(inverted: Optional[dict]) -> str:
    """OpenAlex stores abstracts as {word: [positions]}. Reconstruct the original text."""
    if not isinstance(inverted, dict) or not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        if not isinstance(idxs, list):
            continue
        for i in idxs:
            if isinstance(i, int):
                positions.append((i, word))
    positions.sort(key=lambda p: p[0])
    return " ".join(w for _, w in positions)
