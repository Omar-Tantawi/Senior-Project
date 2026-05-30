"""
Extracts educational intent (topic + bilingual keywords) from a teacher's
free-form Arabic input using Ollama (command-r7b-arabic).

Falls back to raw passthrough if Ollama is unreachable so search still works.
"""

import json
import re
import urllib.request
import urllib.error
from typing import Optional

from config import OLLAMA_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT


_SYSTEM_PROMPT = (
    "You are a search query optimizer for an Arabic-language school management system. "
    "Given a teacher's input, extract the exact educational topic and produce bilingual keywords.\n\n"
    "CRITICAL RULES:\n"
    "1. NEVER paraphrase, generalize, or replace technical terms. "
    "If the teacher writes 'النواس المرن' keep BOTH words — do NOT replace with 'المرونة' alone.\n"
    "2. Each Arabic keyword must be a direct substring or close form of the original query's words.\n"
    "3. For science/math topics (physics, chemistry, biology, math), preserve the exact Arabic term "
    "as the first keyword — do not substitute synonyms.\n"
    "4. Always respond with a single JSON object and nothing else."
)

_USER_TEMPLATE = """Teacher input: "{query}"

Return a JSON object with these exact keys:
- "topic_ar":  the exact Arabic topic phrase from the input (3-8 words, remove only filler verbs like أريد/ابحث عن)
- "topic_en":  precise English translation of that exact topic (3-8 words)
- "keywords_ar": list of 3-5 Arabic search keywords — first keyword MUST be the exact topic phrase
- "keywords_en": list of 3-5 English search keywords — first keyword MUST be the direct translation
- "content_type_hints": list of any explicitly requested types from {{"video","article","pdf","image"}}; empty list if none

Respond with the JSON only, no markdown, no commentary."""


class IntentExtractor:
    def __init__(self):
        self._check_ollama()

    # ── Public ────────────────────────────────────────────────────────────────

    def extract(self, query: str, refinement: str = "") -> dict:
        combined = query.strip()
        if refinement.strip():
            combined = f"{combined}  ||  {refinement.strip()}"

        raw = self._call_ollama(combined)
        parsed = self._parse_json(raw) if raw else None
        if parsed:
            return self._normalize(parsed, combined)
        return self._fallback(combined)

    def warmup(self):
        try:
            self._call_ollama("warmup", num_predict=8)
            print("[IntentExtractor] Ollama warmed up.")
        except Exception as e:
            print(f"[IntentExtractor] Warmup failed: {e}")

    # ── Internals ─────────────────────────────────────────────────────────────

    def _check_ollama(self):
        try:
            with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                base = OLLAMA_MODEL.split(":")[0]
                if not any(base in m for m in models):
                    print(f"[IntentExtractor] WARNING: '{OLLAMA_MODEL}' not pulled. Run: ollama pull {OLLAMA_MODEL}")
                else:
                    print(f"[IntentExtractor] Ollama ready — model: {OLLAMA_MODEL}")
        except urllib.error.URLError:
            print("[IntentExtractor] WARNING: Ollama not running. Will use raw passthrough.")

    def _call_ollama(self, query: str, num_predict: int = 512) -> Optional[str]:
        payload = json.dumps({
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _USER_TEMPLATE.format(query=query)},
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": num_predict},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                data = json.loads(resp.read())
                return data["message"]["content"].strip()
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as e:
            print(f"[IntentExtractor] Ollama call failed: {e}")
            return None

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _normalize(data: dict, original: str) -> dict:
        def _as_list(v):
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
            if isinstance(v, str) and v.strip():
                return [v.strip()]
            return []

        return {
            "topic_ar":           str(data.get("topic_ar")  or original).strip(),
            "topic_en":           str(data.get("topic_en")  or original).strip(),
            "keywords_ar":        _as_list(data.get("keywords_ar")) or [original],
            "keywords_en":        _as_list(data.get("keywords_en")) or [original],
            "content_type_hints": _as_list(data.get("content_type_hints")),
        }

    @staticmethod
    def _fallback(query: str) -> dict:
        return {
            "topic_ar":           query,
            "topic_en":           query,
            "keywords_ar":        [query],
            "keywords_en":        [query],
            "content_type_hints": [],
        }
