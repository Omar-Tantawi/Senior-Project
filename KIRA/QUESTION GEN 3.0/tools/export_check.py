"""
Verify the Word export: Arabic title renders (not '?????'), non-English is
right-aligned (RTL), English stays left-aligned (LTR).
"""
import os
import sys
import json
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from docx_exporter import export_to_docx

OUT = os.path.join(ROOT, "output")
data = json.load(open(os.path.join(OUT, "test_10-arabic_182527.json"), encoding="utf-8"))
qs = data["questions"]

# UTF-8 Arabic title (literal lives in this .py file → no shell mangling)
ar_title = "اختبار اللغة العربية - الصف العاشر"


def inspect(path, label):
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    print(f"\n[{label}] {os.path.basename(path)}")
    print("  title text present in doc:", ar_title in xml if label == "AR" else "English Test" in xml)
    print("  right-aligned paragraphs (w:jc val=right):", xml.count('w:val="right"'))
    print("  RTL marks (w:bidi):", xml.count("<w:bidi"))
    print("  left-aligned (w:jc val=left):", xml.count('w:val="left"'))


# Arabic export → expect Arabic title + many right/bidi
ar_path = os.path.join(OUT, "verify_export_arabic.docx")
open(ar_path, "wb").write(
    export_to_docx(qs, title=ar_title, document_id="10-arabic",
                   language="ar", include_answers=True))
inspect(ar_path, "AR")

# English export → expect 0 right-aligned (stays LTR)
en_path = os.path.join(OUT, "verify_export_english.docx")
open(en_path, "wb").write(
    export_to_docx(qs, title="English Test", document_id="x",
                   language="en", include_answers=True))
inspect(en_path, "EN")

print("\nwrote:", ar_path)
print("wrote:", en_path)
