"""Download required ONNX models for face detection and recognition.

Downloads:
  1. det_10g.onnx  - RetinaFace detection model (~16MB)
     Detects face locations, confidence scores, and 5-point landmarks

  2. w600k_r50.onnx - ArcFace recognition model (~167MB)
     Generates 512-dim face embeddings for identity matching

Models are from the InsightFace 'buffalo_l' model pack.
Source: https://github.com/deepinsight/insightface/tree/master/model_zoo
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Installing required packages...")
    os.system(f"{sys.executable} -m pip install requests tqdm")
    import requests
    from tqdm import tqdm


MODELS_DIR = Path(__file__).parent / "models"

# InsightFace buffalo_l model pack (contains all models in a zip)
# This is the official release from the InsightFace GitHub
BUFFALO_L_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_L_SIZE_MB = 330

# The models we need from the pack
REQUIRED_MODELS = {
    "det_10g.onnx": "RetinaFace face detection model",
    "w600k_r50.onnx": "ArcFace face recognition model (trained on 600K identities)",
}


def download_file(url: str, dest: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(dest, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))


def download_models():
    """Download the buffalo_l model pack and extract required models."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("  Downloading Face Recognition Models")
    print("=" * 50)

    # Check if all required models already exist
    all_exist = True
    for filename, desc in REQUIRED_MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"\n  [OK] {filename} already exists ({size_mb:.1f}MB)")
            print(f"       {desc}")
        else:
            all_exist = False

    if all_exist:
        print("\nAll models already downloaded!")
        return True

    # Download the full buffalo_l zip
    zip_path = MODELS_DIR / "buffalo_l.zip"

    print(f"\nDownloading buffalo_l model pack (~{BUFFALO_L_SIZE_MB}MB)...")
    print(f"  Source: {BUFFALO_L_URL}")

    try:
        download_file(BUFFALO_L_URL, zip_path, desc="buffalo_l.zip")
    except Exception as e:
        print(f"\n  [ERROR] Download failed: {e}")
        print(f"\n  *** MANUAL DOWNLOAD INSTRUCTIONS ***")
        print(f"  1. Go to: {BUFFALO_L_URL}")
        print(f"  2. Your browser will download 'buffalo_l.zip'")
        print(f"  3. Extract the zip file")
        print(f"  4. Copy these files into: {MODELS_DIR}")
        for filename in REQUIRED_MODELS:
            print(f"     - {filename}")
        return False

    # Extract required models from zip
    print("\nExtracting models...")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for filename in REQUIRED_MODELS:
                # Models may be at root or in a subfolder in the zip
                extracted = False
                for zip_entry in zf.namelist():
                    if zip_entry.endswith(filename):
                        data = zf.read(zip_entry)
                        dest = MODELS_DIR / filename
                        dest.write_bytes(data)
                        size_mb = len(data) / (1024 * 1024)
                        print(f"  [OK] Extracted {filename} ({size_mb:.1f}MB)")
                        extracted = True
                        break
                if not extracted:
                    print(f"  [WARNING] {filename} not found in zip")
    except zipfile.BadZipFile:
        print("  [ERROR] Downloaded file is not a valid zip.")
        print("  The download may have been incomplete. Delete buffalo_l.zip and retry.")
        return False

    # Clean up zip
    zip_path.unlink()
    print("  Cleaned up zip file")

    # Verify
    for filename in REQUIRED_MODELS:
        if not (MODELS_DIR / filename).exists():
            print(f"\n  [ERROR] {filename} is missing after extraction!")
            return False

    print(f"\n{'='*50}")
    print(f"  All models ready in: {MODELS_DIR}")
    print(f"{'='*50}")
    return True


if __name__ == "__main__":
    success = download_models()
    if not success:
        sys.exit(1)
