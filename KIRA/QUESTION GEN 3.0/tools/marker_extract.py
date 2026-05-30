"""
Extract a PDF (or page range) to Markdown + LaTeX using Marker (Surya + Texify).
Used to re-process equation-heavy books so equations become LaTeX instead of OCR noise.

Usage:
  python tools/marker_extract.py --pdf "Data/Class 12/12-physics-Sci.pdf" --pages 63-82 --out output/marker_physics.md
"""
import os
import sys
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages", default="")          # 1-indexed "63-82" (inclusive)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pdf = args.pdf if os.path.isabs(args.pdf) else os.path.join(ROOT, args.pdf)

    config = {"output_format": "markdown"}
    if args.pages:
        a, _, b = args.pages.partition("-")
        a, b = int(a), int(b or a)
        config["page_range"] = f"{a-1}-{b-1}"        # marker uses 0-indexed pages
    print(f"[marker] pdf={pdf}")
    print(f"[marker] page_range(0-idx)={config.get('page_range','all')}")

    import torch
    print(f"[marker] torch {torch.__version__} cuda={torch.cuda.is_available()}")

    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser
    from marker.output import text_from_rendered

    t0 = time.time()
    print("[marker] loading models (first run downloads Surya/Texify weights)...")
    cp = ConfigParser(config)
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
        config=cp.generate_config_dict(),
        processor_list=cp.get_processors(),
        renderer=cp.get_renderer(),
    )
    print(f"[marker] models ready in {time.time()-t0:.0f}s; converting...")

    t1 = time.time()
    rendered = converter(pdf)
    text, ext, images = text_from_rendered(rendered)
    print(f"[marker] converted in {time.time()-t1:.0f}s")

    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    # quick diagnostics on equation extraction
    n_inline = text.count("$") // 2
    n_block  = text.count("$$") // 2
    n_latex  = sum(text.count(t) for t in ("\\frac", "\\vec", "\\sqrt", "\\sum", "\\int", "\\mu", "_{", "^{"))
    print(f"[marker] wrote {out}  ({len(text)} chars)")
    print(f"[marker] LaTeX signals: $-spans~{n_inline}, $$-blocks~{n_block}, latex-cmds~{n_latex}")


if __name__ == "__main__":
    main()
