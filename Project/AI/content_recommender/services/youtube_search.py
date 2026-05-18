"""YouTube Data API v3 wrapper. Async, exclusion-aware for 'show more'."""
import logging
from typing import Iterable, Optional

import httpx

from config import settings
from models.schemas import VideoResult

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


async def search_videos(
    query: str,
    *,
    max_results: int = 10,
    page_token: Optional[str] = None,
    exclude: Optional[Iterable[str]] = None,
    relevance_language: str = "ar",
) -> tuple[list[VideoResult], Optional[str]]:
    """Search YouTube for videos.

    Returns (results, next_page_token). `exclude` is a set of video URLs to skip
    (used by /recommend/more to avoid showing the same video again).
    """
    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY missing (used for YouTube Data API).")

    exclude_set = set(exclude or [])

    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoDuration": "medium",
        "videoEmbeddable": "true",
        "safeSearch": "strict",
        "maxResults": str(max_results),
        "relevanceLanguage": relevance_language,
        "key": settings.google_api_key,
    }
    if page_token:
        params["pageToken"] = page_token

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(_SEARCH_URL, params=params)
        if resp.status_code == 403:
            logger.warning("YouTube API quota exceeded or forbidden: %s", resp.text)
            return [], None
        resp.raise_for_status()
        payload = resp.json()

        items = payload.get("items", [])
        next_token = payload.get("nextPageToken")

        # Second call to videos.list to fetch duration (contentDetails)
        ids = [it["id"]["videoId"] for it in items if "videoId" in it.get("id", {})]
        durations: dict[str, str] = {}
        if ids:
            d_resp = await client.get(
                _VIDEOS_URL,
                params={
                    "part": "contentDetails",
                    "id": ",".join(ids),
                    "key": settings.google_api_key,
                },
            )
            if d_resp.status_code == 200:
                for v in d_resp.json().get("items", []):
                    durations[v["id"]] = v.get("contentDetails", {}).get("duration", "")

    results: list[VideoResult] = []
    for it in items:
        vid = it.get("id", {}).get("videoId")
        if not vid:
            continue
        url = f"https://www.youtube.com/watch?v={vid}"
        if url in exclude_set:
            continue
        sn = it.get("snippet", {})
        results.append(VideoResult(
            title=sn.get("title", ""),
            url=url,
            channel=sn.get("channelTitle"),
            duration=_iso_duration_to_human(durations.get(vid, "")),
            thumbnail=sn.get("thumbnails", {}).get("high", {}).get("url"),
            description=sn.get("description", ""),
        ))

    return results, next_token


def _iso_duration_to_human(iso: str) -> str:
    """PT8M42S -> 8:42, PT1H5M -> 1:05:00."""
    if not iso or not iso.startswith("PT"):
        return ""
    body = iso[2:]
    hours = minutes = seconds = 0
    num = ""
    for ch in body:
        if ch.isdigit():
            num += ch
        elif ch == "H":
            hours = int(num or 0)
            num = ""
        elif ch == "M":
            minutes = int(num or 0)
            num = ""
        elif ch == "S":
            seconds = int(num or 0)
            num = ""
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"
