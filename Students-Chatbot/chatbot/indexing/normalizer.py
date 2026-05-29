"""
Arabic text normalization using CAMeL Tools.
Normalizes alef forms, strips diacritics, and normalizes ta marbuta / ya.
Applied during indexing only — query text is also normalized before retrieval.
"""
from camel_tools.utils.normalize import (
    normalize_alef_maksura_ar,
    normalize_alef_ar,
    normalize_teh_marbuta_ar,
    normalize_unicode,
)
from camel_tools.utils.dediac import dediac_ar


def normalize(text: str) -> str:
    text = normalize_unicode(text)
    text = dediac_ar(text)
    text = normalize_alef_ar(text)
    text = normalize_alef_maksura_ar(text)
    text = normalize_teh_marbuta_ar(text)
    return text
