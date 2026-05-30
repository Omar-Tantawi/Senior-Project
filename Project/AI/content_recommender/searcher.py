"""
Parallel orchestrator that fans out to all source modules and merges results.
Each source runs in a thread (sources are synchronous I/O bound) so a slow
source can't block the others.
"""

import asyncio

from config import MAX_RESULTS_PER_SOURCE
from models import ContentItem
from sources import youtube_source, ddg_source, openalex, wikimedia_source, khan_academy


class Searcher:
    async def search(
        self,
        intent:        dict,
        content_types: list[str],
        language_pref: str,
        per_source:    int,
        exclude_urls:  set[str],
    ) -> list[ContentItem]:

        kw_en = intent.get("keywords_en") or []
        kw_ar = intent.get("keywords_ar") or []

        if language_pref == "ar":
            kw_en = []
        elif language_pref == "en":
            kw_ar = []

        per_source = max(per_source, MAX_RESULTS_PER_SOURCE)
        loop = asyncio.get_running_loop()
        tasks: list[asyncio.Future] = []

        def run(fn, *args):
            return loop.run_in_executor(None, fn, *args)

        if "video" in content_types:
            tasks.append(run(youtube_source.search,   kw_en, kw_ar, per_source))
            tasks.append(run(khan_academy.search,     kw_en, kw_ar, per_source))

        if "article" in content_types:
            tasks.append(run(ddg_source.search_articles, kw_en, kw_ar, per_source))
            tasks.append(run(khan_academy.search,        kw_en, kw_ar, per_source))

        if "pdf" in content_types:
            tasks.append(run(ddg_source.search_pdfs, kw_en, kw_ar, per_source))
            tasks.append(run(openalex.search, kw_en, per_source))

        if "image" in content_types:
            tasks.append(run(wikimedia_source.search, kw_en, kw_ar, per_source))

        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[ContentItem] = []
        seen: set[str] = set(exclude_urls)
        for batch in gathered:
            if isinstance(batch, Exception):
                print(f"[Searcher] source error: {batch}")
                continue
            for item in batch:
                if not item.url or item.url in seen:
                    continue
                if language_pref in ("ar", "en") and item.language not in (language_pref, "unknown"):
                    continue
                seen.add(item.url)
                merged.append(item)

        return merged
