"""
Arabic-aware normalization + tokenization.

This module is the SINGLE source of truth for how text becomes BM25 tokens.
It is shared by:
  - tools/build_index.py  → tokenizes the corpus when rebuilding the BM25 index
  - retriever.py          → tokenizes queries the SAME way

If these two ever disagree, BM25 scores stop lining up. So both import from here.

Why we re-tokenize at all
-------------------------
The original index used a plain `str.split()` — no normalization. That misses:
  • Arabic diacritics / tatweel  (مُعَلِّم vs معلم)
  • alef / ya / taa-marbuta variants  (أحمد/احمد, مدرسى/مدرسي, مدرسة/مدرسه)
  • the definite article  (الطاقة vs طاقة)
All of these cause a query term to silently miss the matching chunk.
We normalize both sides identically so they match.
"""

import re

# ── Character classes ────────────────────────────────────────────────────────

# Arabic diacritics (harakat, tanwin, shadda, sukun, superscript alef …) + tatweel ـ
_DIACRITICS = re.compile(
    r"[ؐ-ًؚ-ٰٟۖ-ۜ۟-۪ۨ-ۭـ]"
)

# alef variants  آ أ إ ٱ  → ا
_ALEF = re.compile(r"[آأإٱ]")

# Token = run of Arabic letters, Latin letters, or digits.
# Punctuation and whitespace act as separators (and are dropped).
_TOKEN = re.compile(r"[A-Za-zء-ي٠-٩ٱ-ۓ0-9]+")

_ARABIC_CHAR = re.compile(r"[؀-ۿ]")

# Definite-article prefixes ONLY (longest first). These are safe to strip because
# they are almost always the article ال and its clitic-combined forms.
# We deliberately do NOT strip bare single clitics (و ف ب ك ل) — that would wreck
# real words like وزارة ("ministry") → زارة.
_ARTICLE_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "ال")

_MIN_TOKEN_LEN = 2     # drop single characters
_MIN_STEM_LEN  = 3     # only strip a prefix if a meaningful stem remains


def normalize_text(text: str) -> str:
    """Unicode-level normalization (no tokenization)."""
    text = _DIACRITICS.sub("", text)
    text = _ALEF.sub("ا", text)              # → ا
    text = (text
            .replace("ى", "ي")          # ى → ي
            .replace("ة", "ه")          # ة → ه  (matches the OCR's own spelling)
            .replace("ؤ", "و")          # ؤ → و
            .replace("ئ", "ي"))         # ئ → ي
    return text


def _strip_article(tok: str) -> str:
    """Strip a leading definite article from an Arabic token, if safe."""
    if not _ARABIC_CHAR.match(tok):
        return tok
    for pre in _ARTICLE_PREFIXES:
        if tok.startswith(pre) and len(tok) - len(pre) >= _MIN_STEM_LEN:
            return tok[len(pre):]
    return tok


def tokenize(text: str) -> list[str]:
    """
    Normalize → split → light-stem (article only) → filter.
    Lowercases Latin so 'Energy' and 'energy' match. Returns a token list
    ready for rank_bm25 (both at index-build time and query time).
    """
    text = normalize_text(text.lower())
    out: list[str] = []
    for tok in _TOKEN.findall(text):
        tok = _strip_article(tok)
        if len(tok) >= _MIN_TOKEN_LEN:
            out.append(tok)
    return out
