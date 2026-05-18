"""
src/preprocess.py — Face Detection, Alignment & Normalization Pipeline

Handles all four challenge conditions:
  1. Angled faces        → 5-point landmark alignment
  2. Low/uneven lighting → CLAHE + histogram equalization
  3. Multiple faces      → RetinaFace multi-face detection
  4. Partial occlusion   → Robust RetinaFace detector

Usage:
    python src/preprocess.py --split train
    python src/preprocess.py --split val
    python src/preprocess.py --split all
"""

import os
import sys
import cv2
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).parent.parent))
from config import (
    PROCESSED_DIR, AUGMENTED_DIR, IMAGE_SIZE,
    PIXEL_MEAN, PIXEL_STD, FACE_MARGIN
)

# ─── Detector Setup ───────────────────────────────────────────────────────────
def load_detector():
    """
    Load RetinaFace detector via InsightFace.
    RetinaFace is chosen over MTCNN because:
      - Better accuracy on angled faces (wall-mounted camera)
      - More robust to partial occlusion
      - Handles low-resolution / compressed frames
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",           # Large model: best accuracy
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("[OK] RetinaFace (InsightFace buffalo_l) loaded on GPU")
        return app, "insightface"
    except Exception as e:
        print(f"[WARN] InsightFace not available: {e}")
        print("[INFO] Falling back to MTCNN (facenet-pytorch)...")

    try:
        from facenet_pytorch import MTCNN
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mtcnn = MTCNN(
            image_size=IMAGE_SIZE,
            margin=int(IMAGE_SIZE * FACE_MARGIN),
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            keep_all=True,
            device=device
        )
        print(f"[OK] MTCNN loaded on {device}")
        return mtcnn, "mtcnn"
    except Exception as e:
        raise RuntimeError(f"No face detector available: {e}")


# ─── Face Alignment ───────────────────────────────────────────────────────────
# Reference 5-point landmark positions (ArcFace standard 112×112)
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose tip
    [41.5493, 92.3655],   # left mouth corner
    [70.7299, 92.2041],   # right mouth corner
], dtype=np.float32)


def align_face(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """
    Align face using 5-point landmarks → similarity transform to 112×112.
    This corrects for:
      - Head tilt (wall camera angle)
      - In-plane rotation
      - Scale variation
    """
    from skimage import transform as sk_transform

    tform = sk_transform.SimilarityTransform()
    tform.estimate(landmarks, REFERENCE_LANDMARKS)
    M = tform.params[0:2, :]

    aligned = cv2.warpAffine(
        image, M, (IMAGE_SIZE, IMAGE_SIZE),
        borderMode=cv2.BORDER_REFLECT
    )
    return aligned


# ─── Lighting Normalization ───────────────────────────────────────────────────
def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space.
    Much better than global histogram eq — handles uneven classroom lighting.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge([l_eq, a, b])
    result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    return result


# ─── Pixel Normalization ──────────────────────────────────────────────────────
def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """
    Normalize to [-1, 1] range (ArcFace / InsightFace standard).
    """
    img = image.astype(np.float32) / 255.0
    mean = np.array(PIXEL_MEAN, dtype=np.float32)
    std  = np.array(PIXEL_STD,  dtype=np.float32)
    img = (img - mean) / std
    return img


# ─── Single Image Processing ──────────────────────────────────────────────────
def process_image_insightface(detector, image_path: Path, out_path: Path) -> bool:
    """Process using InsightFace RetinaFace detector."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return False

    # Apply CLAHE before detection (helps in low light)
    img_bgr = normalize_lighting(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    faces = detector.get(img_rgb)
    if not faces:
        return False

    # Pick the largest face (assumes one face per training image)
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))

    # 5-point landmarks
    kps = face.kps.astype(np.float32)
    aligned = align_face(img_rgb, kps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), aligned_bgr)
    return True


def process_image_mtcnn(detector, image_path: Path, out_path: Path) -> bool:
    """Process using MTCNN detector (fallback)."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")

    try:
        # MTCNN returns aligned face directly
        face_tensor = detector(img)
        if face_tensor is None:
            return False

        # Take first face if multiple returned
        if face_tensor.dim() == 4:
            face_tensor = face_tensor[0]

        # Convert tensor → numpy → save
        import torch
        face_np = face_tensor.permute(1, 2, 0).numpy()
        face_np = ((face_np * 0.5 + 0.5) * 255).astype(np.uint8)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        face_bgr = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
        face_bgr = normalize_lighting(face_bgr)
        cv2.imwrite(str(out_path), face_bgr)
        return True
    except Exception:
        return False


# ─── Process Full Split ───────────────────────────────────────────────────────
def process_split(split: str, detector, detector_type: str):
    src_dir = PROCESSED_DIR / split
    dst_dir = AUGMENTED_DIR / split      # Preprocessed images saved here

    if not src_dir.exists():
        print(f"[WARN] {src_dir} does not exist — skipping.")
        return

    identity_dirs = sorted([d for d in src_dir.iterdir() if d.is_dir()])
    print(f"\n[INFO] Processing {split} split: {len(identity_dirs)} identities")

    total_ok, total_fail = 0, 0

    for identity_dir in tqdm(identity_dirs, desc=f"Processing {split}"):
        img_paths = list(identity_dir.glob("*.jpg")) + \
                    list(identity_dir.glob("*.jpeg")) + \
                    list(identity_dir.glob("*.png"))

        for img_path in img_paths:
            out_path = dst_dir / identity_dir.name / img_path.name

            if out_path.exists():
                total_ok += 1
                continue

            if detector_type == "insightface":
                ok = process_image_insightface(detector, img_path, out_path)
            else:
                ok = process_image_mtcnn(detector, img_path, out_path)

            if ok:
                total_ok += 1
            else:
                total_fail += 1

    print(f"[OK] {split}: {total_ok} processed | {total_fail} failed (no face detected)")


# ─── Quality Check ────────────────────────────────────────────────────────────
def quality_check(split: str):
    """Remove identities with too few successfully preprocessed images."""
    out_dir = AUGMENTED_DIR / split
    if not out_dir.exists():
        return

    removed_ids = 0
    for identity_dir in out_dir.iterdir():
        imgs = list(identity_dir.glob("*.jpg"))
        if len(imgs) < 5:
            import shutil
            shutil.rmtree(identity_dir)
            removed_ids += 1

    remaining = len(list(out_dir.iterdir()))
    print(f"[OK] Quality check: removed {removed_ids} identities with < 5 faces | {remaining} remaining")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    args = parser.parse_args()

    detector, det_type = load_detector()

    splits = ["train", "val"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, detector, det_type)
        quality_check(split)

    print("\n[DONE] Preprocessing complete. Next step: run src/train.py")
