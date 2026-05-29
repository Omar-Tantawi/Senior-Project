"""
Extracts text from curriculum PDFs.
Strategy: PyMuPDF for digital PDFs (fast). Falls back to Surya OCR for
scanned pages where extracted text is below OCR_FALLBACK_THRESHOLD chars.
"""
from pathlib import Path
from typing import Generator
import fitz  # PyMuPDF
from PIL import Image

from chatbot.config import OCR_FALLBACK_THRESHOLD

# Lazy module-level predictors — loaded once on first OCR call.
# surya 0.17.x API: FoundationPredictor is the shared backbone;
# DetectionPredictor and RecognitionPredictor both wrap it.
_foundation = None
_det_predictor = None
_rec_predictor = None


def _get_predictors():
    global _foundation, _det_predictor, _rec_predictor
    if _foundation is None:
        from surya.foundation import FoundationPredictor
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        _foundation = FoundationPredictor()
        _det_predictor = DetectionPredictor()
        _rec_predictor = RecognitionPredictor(foundation_predictor=_foundation)
    return _det_predictor, _rec_predictor


def _page_to_pil(page: fitz.Page) -> Image.Image:
    pix = page.get_pixmap(dpi=200, colorspace=fitz.csRGB)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _surya_ocr_page(page: fitz.Page) -> str:
    """Render page to PIL image and run Surya OCR."""
    det_predictor, rec_predictor = _get_predictors()
    img = _page_to_pil(page)
    results = rec_predictor([img], det_predictor=det_predictor)
    lines = [line.text for line in results[0].text_lines if line.text.strip()]
    return "\n".join(lines)


def extract_pages(pdf_path: Path) -> Generator[dict, None, None]:
    """
    Yields one dict per page: {page_num, text, used_ocr}.
    Skips pages with no recoverable text even after OCR fallback.
    """
    doc = fitz.open(str(pdf_path))
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text").strip()
        used_ocr = False

        if len(text) < OCR_FALLBACK_THRESHOLD:
            try:
                ocr_text = _surya_ocr_page(page)
                if ocr_text.strip():
                    text = ocr_text
                    used_ocr = True
            except Exception as e:
                print(f"  [OCR warn] page {page_num}: {e}")

        if text.strip():
            yield {"page_num": page_num, "text": text, "used_ocr": used_ocr}

    doc.close()
