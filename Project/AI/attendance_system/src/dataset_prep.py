"""
src/dataset_prep.py — Download & prepare VGGFace2 or LFW dataset

Usage:
    python src/dataset_prep.py --dataset vggface2   # Full training set
    python src/dataset_prep.py --dataset lfw        # Lighter alternative
"""

import os
import sys
import argparse
import zipfile
import tarfile
import shutil
import subprocess
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DIR, PROCESSED_DIR

# ─── VGGFace2 via Kaggle ──────────────────────────────────────────────────────
def download_vggface2():
    """
    VGGFace2 dataset: 3.3M images, 8631 identities.
    Requires a Kaggle account + API key (~/.kaggle/kaggle.json).
    """
    print("[INFO] Downloading VGGFace2 from Kaggle...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Check kaggle credentials
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("[ERROR] Kaggle credentials not found.")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Click 'Create New API Token' → downloads kaggle.json")
        print("  3. Place it at ~/.kaggle/kaggle.json")
        print("  4. Run: chmod 600 ~/.kaggle/kaggle.json")
        sys.exit(1)

    cmd = [
        "kaggle", "datasets", "download",
        "-d", "greatgamedota/vggface2-face-recognition-dataset",
        "-p", str(RAW_DIR), "--unzip"
    ]
    subprocess.run(cmd, check=True)
    print(f"[OK] VGGFace2 downloaded to {RAW_DIR}")


def download_lfw():
    """
    LFW (Labeled Faces in the Wild) — lighter alternative.
    No credentials needed.
    """
    import urllib.request
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    dest = RAW_DIR / "lfw.tgz"

    print("[INFO] Downloading LFW dataset (~173 MB)...")
    urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)

    print("\n[INFO] Extracting LFW...")
    with tarfile.open(dest, "r:gz") as tar:
        tar.extractall(RAW_DIR)
    dest.unlink()
    print(f"[OK] LFW extracted to {RAW_DIR / 'lfw'}")


def _progress_hook(count, block_size, total_size):
    pct = count * block_size * 100 // total_size
    sys.stdout.write(f"\r  Progress: {pct}%")
    sys.stdout.flush()


# ─── Verify Dataset Structure ─────────────────────────────────────────────────
def verify_structure(dataset: str):
    """
    Expected structure:
        data/raw/
          vggface2/
            train/
              n000001/ (identity folder)
                0001_01.jpg
                ...
            test/
              ...
        OR
          lfw/
            Aaron_Eckhart/
              Aaron_Eckhart_0001.jpg
    """
    if dataset == "vggface2":
        train_dir = RAW_DIR / "vggface2" / "train"
        if not train_dir.exists():
            # Try flat layout
            train_dir = RAW_DIR / "train"
        assert train_dir.exists(), f"Expected {train_dir} to exist after download."
        identities = list(train_dir.iterdir())
        print(f"[OK] VGGFace2: {len(identities)} identities found in {train_dir}")
    elif dataset == "lfw":
        lfw_dir = RAW_DIR / "lfw"
        assert lfw_dir.exists(), f"Expected {lfw_dir} to exist."
        identities = list(lfw_dir.iterdir())
        print(f"[OK] LFW: {len(identities)} identities found in {lfw_dir}")
    return True


# ─── Filter to identities with enough images ──────────────────────────────────
def filter_dataset(dataset: str, min_images: int = 10):
    """
    Remove identities with fewer than min_images images.
    This is important for metric learning — need enough examples per class.
    """
    if dataset == "vggface2":
        src = RAW_DIR / "vggface2" / "train"
        if not src.exists():
            src = RAW_DIR / "train"
    else:
        src = RAW_DIR / "lfw"

    kept, removed = 0, 0
    for identity in src.iterdir():
        if not identity.is_dir():
            continue
        imgs = list(identity.glob("*.jpg")) + list(identity.glob("*.png"))
        if len(imgs) < min_images:
            shutil.rmtree(identity)
            removed += 1
        else:
            kept += 1

    print(f"[OK] Filtered dataset: kept {kept} identities, removed {removed} (< {min_images} images)")
    return src


# ─── Create train/val split ───────────────────────────────────────────────────
def create_split(src_dir: Path, val_ratio: float = 0.1):
    """
    Create train/val split inside PROCESSED_DIR.
    Structure:
        data/processed/
            train/ {identity}/ *.jpg
            val/   {identity}/ *.jpg
    """
    import random
    from PIL import Image

    train_out = PROCESSED_DIR / "train"
    val_out   = PROCESSED_DIR / "val"
    train_out.mkdir(parents=True, exist_ok=True)
    val_out.mkdir(parents=True, exist_ok=True)

    identities = [d for d in src_dir.iterdir() if d.is_dir()]
    print(f"[INFO] Creating train/val split for {len(identities)} identities...")

    for identity in identities:
        imgs = sorted(list(identity.glob("*.jpg")) + list(identity.glob("*.png")))
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_ratio))
        val_imgs   = imgs[:n_val]
        train_imgs = imgs[n_val:]

        (train_out / identity.name).mkdir(exist_ok=True)
        (val_out   / identity.name).mkdir(exist_ok=True)

        for img_path in train_imgs:
            shutil.copy2(img_path, train_out / identity.name / img_path.name)
        for img_path in val_imgs:
            shutil.copy2(img_path, val_out / identity.name / img_path.name)

    train_count = sum(1 for _ in train_out.rglob("*.jpg"))
    val_count   = sum(1 for _ in val_out.rglob("*.jpg"))
    print(f"[OK] Split complete — Train: {train_count} images | Val: {val_count} images")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["vggface2", "lfw"], default="vggface2")
    parser.add_argument("--min_images", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip download if already present")
    args = parser.parse_args()

    if not args.skip_download:
        if args.dataset == "vggface2":
            download_vggface2()
        else:
            download_lfw()

    verify_structure(args.dataset)
    src = filter_dataset(args.dataset, min_images=args.min_images)
    create_split(src, val_ratio=args.val_ratio)
    print("\n[DONE] Dataset ready. Next step: run src/preprocess.py")
