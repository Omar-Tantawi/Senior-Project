"""
Upload a PDF + generate questions + export to Word — all in one call.

Edit the SETTINGS section below, then run:
    python generate_and_export.py
"""

import os
import sys
import urllib.request
import urllib.error
import mimetypes
import uuid
from datetime import datetime


# ── SETTINGS — edit these ──────────────────────────────────────────────────────

PDF_PATH    = r"D:\Senior\question-generator\Data\Class 12\12-physics-Sci.pdf"
PAGE_START  = 63       # first page (1-based, inclusive). Use None for first page.
PAGE_END    = 82       # last  page (1-based, inclusive). Use None for last page.

# How many of each type
QUESTION_COUNTS = {
    "mcq":          5,
    "true_false":   4,
    "short_answer": 2,
    "fill_blank":   4,
    "essay":        0,
}

BLOOM_LEVELS    = "remember,understand,apply"   # comma-separated
DIFFICULTY      = "medium"                       # easy | medium | hard
LANGUAGE        = "auto"                         # ar | en | auto
TITLE           = "اختبار فيزياء - الصف الثاني عشر"
INCLUDE_ANSWERS = True                           # adds Answer Key at the end


# ── Don't edit below ──────────────────────────────────────────────────────────

API_BASE   = "http://localhost:8002"
API_KEY    = os.getenv("AI_API_KEY", "change-me-shared-secret")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _build_multipart(fields: dict, file_path: str, file_field: str = "file") -> tuple[bytes, str]:
    """Manually build a multipart/form-data body so we don't need requests."""
    boundary = f"----QGFormBoundary{uuid.uuid4().hex}"
    parts: list[bytes] = []

    # Regular fields
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        parts.append(f"{value}\r\n".encode("utf-8"))

    # File field
    filename = os.path.basename(file_path)
    mime, _  = mimetypes.guess_type(file_path)
    mime     = mime or "application/octet-stream"

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    parts.append(f"--{boundary}\r\n".encode())
    parts.append(
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'.encode()
    )
    parts.append(f"Content-Type: {mime}\r\n\r\n".encode())
    parts.append(file_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(parts)
    return body, boundary


def main():
    if not os.path.exists(PDF_PATH):
        print(f"PDF not found: {PDF_PATH}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(PDF_PATH))[0]
    docx_path = os.path.join(OUTPUT_DIR, f"{base_name}_{timestamp}.docx")

    # Build form fields
    fields = {
        "mcq_count":          QUESTION_COUNTS["mcq"],
        "true_false_count":   QUESTION_COUNTS["true_false"],
        "short_answer_count": QUESTION_COUNTS["short_answer"],
        "fill_blank_count":   QUESTION_COUNTS["fill_blank"],
        "essay_count":        QUESTION_COUNTS["essay"],
        "bloom_levels":       BLOOM_LEVELS,
        "difficulty":         DIFFICULTY,
        "language":           LANGUAGE,
        "title":              TITLE,
        "include_answers":    str(INCLUDE_ANSWERS).lower(),
        "output":             "docx",
    }
    if PAGE_START is not None:
        fields["page_start"] = PAGE_START
    if PAGE_END is not None:
        fields["page_end"] = PAGE_END

    print(f"\n{'='*60}")
    print(f"  Uploading PDF + generating + exporting in one call...")
    print(f"  File:        {PDF_PATH}")
    print(f"  Pages:       {PAGE_START} to {PAGE_END}")
    print(f"  Questions:   " + ", ".join(f"{k}={v}" for k, v in QUESTION_COUNTS.items() if v))
    print(f"  Language:    {LANGUAGE}")
    print(f"{'='*60}\n")

    body, boundary = _build_multipart(fields, PDF_PATH)
    req = urllib.request.Request(
        f"{API_BASE}/questions/from-pdf",
        data    = body,
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "X-API-Key":    API_KEY,
        },
        method  = "POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=None) as resp:
            content_type = resp.headers.get("Content-Type", "")
            payload      = resp.read()

            if "json" in content_type:
                # Server returned JSON (probably an error)
                print("Server returned JSON:")
                print(payload.decode("utf-8"))
                sys.exit(1)

            # Otherwise it's a docx file
            with open(docx_path, "wb") as f:
                f.write(payload)

    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"\nERROR {e.code}: {body}")
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"\nERROR: Could not reach server. Is uvicorn running?\n  {e}")
        sys.exit(1)

    size_kb = os.path.getsize(docx_path) / 1024
    print(f"  Word file saved: {docx_path}  ({size_kb:.1f} KB)\n")


if __name__ == "__main__":
    main()
