from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field

ContentType  = Literal["video", "article", "pdf", "image"]
LanguagePref = Literal["ar", "en", "both"]


class RecommendRequest(BaseModel):
    query: str
    content_types: list[ContentType] = ["video", "article", "pdf", "image"]
    language_preference: LanguagePref = "both"
    max_results: int = Field(15, ge=1, le=50)
    exclude_urls: list[str] = []
    refinement: str = ""
    session_id: str = ""


class ExtractRequest(BaseModel):
    query: str
    refinement: str = ""


class Intent(BaseModel):
    topic_ar: str
    topic_en: str
    keywords_ar: list[str]
    keywords_en: list[str]
    content_type_hints: list[str] = []


class ContentItem(BaseModel):
    content_type: ContentType
    title: str
    url: str
    description: str = ""
    thumbnail_url: Optional[str] = None
    source: str
    language: str = "unknown"
    relevance_score: float = 0.0
    duration: Optional[str] = None
    year: Optional[int] = None


class RecommendResponse(BaseModel):
    query_original: str
    topic_ar: str
    topic_en: str
    keywords_ar: list[str]
    keywords_en: list[str]
    results: list[ContentItem]
    total_found: int
    search_time_ms: int
    session_id: str
