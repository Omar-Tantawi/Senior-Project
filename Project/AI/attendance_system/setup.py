"""
setup.py — Download all pretrained model weights.

Run this ONCE before anything else:
    python setup.py

What it downloads:
  1. InsightFace models (RetinaFace + ArcFace) — auto-downloaded on first use
  2. MiniFASNet anti-spoofing weights           — downloaded here manually
  3. Verifies CUDA is available
"""

import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
MODELS_DIR  = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── MiniFASNet weights (Silent-Face-Anti-Spoofing) ───────────────────────────
# Two model files: 2.7M parameter model (high accuracy) and a lightweight one.
FASNET_URLS = {
    "2.7_80x80_MiniFASNetV2.pth": (
        "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
        "/raw/master/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    ),
    "4_0_0_80x80_MiniFASNetV1SE.pth": (
        "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
        "/raw/master/resources/anti_spoof_models/4_0_0_80x80_MiniFASNetV1SE.pth"
    ),
}

FAS_DIR = MODELS_DIR / "anti_spoof"
FAS_DIR.mkdir(exist_ok=True)


def download_file(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  [SKIP] Already exists: {dest.name}")
        return
    print(f"  [DOWN] {dest.name} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [ OK ] {dest.name}")
    except Exception as e:
        print(f"  [FAIL] {dest.name}: {e}")
        print(f"         Please download manually from:\n         {url}")
        print(f"         and place at: {dest}")


def check_cuda() -> None:
    print("\n── CUDA check ──────────────────────────────────────────")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available — {torch.cuda.get_device_name(0)}")
            print(f"       VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  [WARN] CUDA not available — will run on CPU (slower)")
    except ImportError:
        print("  [WARN] PyTorch not installed yet. Run: pip install -r requirements.txt")


def warm_up_insightface() -> None:
    """
    InsightFace downloads its models (RetinaFace + ArcFace) automatically
    on the first call to FaceAnalysis.prepare(). We trigger that here so
    it happens at setup time, not at runtime.
    """
    print("\n── InsightFace model warm-up ────────────────────────────")
    try:
        import insightface
        from insightface.app import FaceAnalysis

        # 'buffalo_l' is the large pack: RetinaFace-R50 + ArcFace-R100
        # It will be saved to ~/.insightface/models/buffalo_l/
        print("  Downloading buffalo_l (RetinaFace + ArcFace) ...")
        app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("  [OK] InsightFace models ready")
    except ImportError:
        print("  [SKIP] insightface not installed yet.")
        print("         Run: pip install -r requirements.txt  then  python setup.py again")
    except Exception as e:
        print(f"  [WARN] InsightFace warm-up failed: {e}")


def main() -> None:
    print("=" * 55)
    print("  Face Attendance System — Setup")
    print("=" * 55)

    # 1. Check CUDA
    check_cuda()

    # 2. Download MiniFASNet weights
    print("\n── Anti-spoofing model weights ─────────────────────────")
    for filename, url in FASNET_URLS.items():
        download_file(url, FAS_DIR / filename)

    # 3. Download / cache InsightFace models
    warm_up_insightface()

    print("\n── Directory layout ────────────────────────────────────")
    for p in sorted(ROOT.rglob("*")):
        if p.is_file():
            rel = p.relative_to(ROOT)
            print(f"  {rel}")

    print("\n[DONE] Setup complete. You can now run:")
    print("       python enroll.py --student_dir data/students/")
    print("       python inference.py --camera 0")


if __name__ == "__main__":
    main()
