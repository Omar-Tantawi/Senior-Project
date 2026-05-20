"""Pydantic schemas for requests, responses, and internal data structures."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

MediaType = Literal["video", "pdf", "article", "image"]


# ---------- Requests ----------
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Arabic text from the teacher")


class MoreRequest(BaseModel):
    session_id: str
    media_type: MediaType


# ---------- Internal: parsed intent ----------
class ParsedIntent(BaseModel):
    topic_ar: str
    topic_en: str
    keywords_ar: list[str] = []
    keywords_en: list[str] = []
    grade_level: str = "general"
    subject: str = "general"
    preferred_media: list[str] = []
    search_query_ar: str
    search_query_en: str


# ---------- Result items ----------
class VideoResult(BaseModel):
    title: str
    url: str
    channel: Optional[str] = None
    duration: Optional[str] = None
    thumbnail: Optional[str] = None
    description: Optional[str] = None


class WebResult(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    source: Optional[str] = None


class ImageResult(BaseModel):
    title: str
    url: str
    thumbnail: Optional[str] = None
    source: Optional[str] = None


# ---------- Bundled results ----------
class RecommendResponse(BaseModel):
    session_id: str
    parsed_intent: ParsedIntent
    results: dict  # {"video": VideoResult|None, "pdf": WebResult|None, ...}


class MoreResponse(BaseModel):
    result: Optional[dict] = None
    has_more: bool
    media_type: MediaType


class HealthResponse(BaseModel):
    status: str
    apis: dict[str, bool]
