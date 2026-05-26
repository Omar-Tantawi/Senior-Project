"""
Export an existing questions JSON file to Word without regenerating.

Usage:
    python export_from_json.py output\questions_xxx.json
"""

import json
import os
import sys
import urllib.request

API_BASE = "http://localhost:8002"
API_KEY  = os.getenv("AI_API_KEY", "change-me-shared-secret")


def main():
    if len(sys.argv) < 2:
        print("Usage: python export_from_json.py <path_to_json>")
        sys.exit(1)

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        sys.exit(1)

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    questions   = data.get("questions", [])
    language    = data.get("language", "en")
    difficulty  = data.get("difficulty", "medium")
    document_id = data.get("document_id", "export")

    # Strip quality field from questions for export (exporter doesn't need it)
    clean_questions = [{k: v for k, v in q.items() if k != "quality"} for q in questions]

    payload = json.dumps({
        "questions":       clean_questions,
        "title":           "Exam Questions",
        "document_id":     document_id,
        "difficulty":      difficulty,
        "language":        language if language != "auto" else "en",
        "include_answers": True,
    }, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        f"{API_BASE}/questions/export/docx",
        data    = payload,
        headers = {"Content-Type": "application/json", "X-API-Key": API_KEY},
        method  = "POST",
    )

    docx_path = json_path.replace(".json", ".docx")
    with urllib.request.urlopen(req, timeout=60) as resp:
        with open(docx_path, "wb") as f:
            f.write(resp.read())

    print(f"Word file saved: {docx_path}")


if __name__ == "__main__":
    main()
