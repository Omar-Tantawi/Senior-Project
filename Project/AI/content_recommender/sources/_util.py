"""Shared helpers for source modules."""

import re

_AR_RE = re.compile(r"[\u0600-\u06FF]")
_EN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Quick language tag based on character ratio. Returns 'ar', 'en', or 'unknown'."""
    if not text:
        return "unknown"
    ar = len(_AR_RE.findall(text))
    en = len(_EN_RE.findall(text))
    total = ar + en
    if total == 0:
        return "unknown"
    if ar / total > 0.3:
        return "ar"
    if en / total > 0.3:
        return "en"
    return "unknown"


def make_query(keywords: list[str], suffix: str = "") -> str:
    """Build a search query string from keywords."""
    base = " ".join(k.strip() for k in keywords if k and k.strip())
    if suffix:
        base = f"{base} {suffix}".strip()
    return base
