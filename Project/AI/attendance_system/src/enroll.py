"""
src/enroll.py — Student Enrollment & Embedding Database Builder

This script:
  1. Takes student photos (from folder or webcam)
  2. Detects & aligns faces via RetinaFace
  3. Generates 512-d embeddings via the trained backbone
  4. Stores mean embedding per student in a fast-lookup database

Usage:
    # Enroll from existing photos
    python src/enroll.py --name "John Smith" --photos path/to/photos/

    # Enroll from webcam (captures 10 frames)
    python src/enroll.py --name "John Smith" --webcam

    # Rebuild full embedding database from student_db folder
    python src/enroll.py --rebuild_all
"""

import os
import sys
import cv2
import pickle
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.model import iresnet50
from src.augmentation import get_val_transforms


# ─── Load Model ───────────────────────────────────────────────────────────────
def load_backbone(device):
    backbone_file = MODEL_DIR / "finetuned_backbone.pth"
    if not backbone_file.exists():
        backbone_file = BACKBONE_PATH    # Fall back to pretrained only

    if not backbone_file.exists():
        print("[ERROR] No trained model found. Run train.py first.")
        sys.exit(1)

    backbone = iresnet50(embedding_dim=EMBEDDING_DIM)
    backbone.load_state_dict(torch.load(backbone_file, map_location=device))
    backbone = backbone.to(device).eval()
    print(f"[OK] Backbone loaded from {backbone_file}")
    return backbone


# ─── Load Detector ────────────────────────────────────────────────────────────
def load_detector():
    try:
        import insightface
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app, "insightface"
    except Exception:
        from facenet_pytorch import MTCNN
        mtcnn = MTCNN(image_size=IMAGE_SIZE, keep_all=False,
                      device="cuda" if torch.cuda.is_available() else "cpu")
        return mtcnn, "mtcnn"


# ─── Detect & Align Single Image ─────────────────────────────────────────────
def detect_and_align(detector, det_type, image_bgr: np.ndarray):
    """Returns aligned face as numpy RGB array, or None if no face found."""
    from src.preprocess import align_face, normalize_lighting, REFERENCE_LANDMARKS

    image_bgr = normalize_lighting(image_bgr)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if det_type == "insightface":
        faces = detector.get(image_rgb)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        kps  = face.kps.astype(np.float32)
        aligned = align_face(image_rgb, kps)
        return aligned

    else:  # MTCNN
        from PIL import Image
        pil_img = Image.fromarray(image_rgb)
        face_tensor = detector(pil_img)
        if face_tensor is None:
            return None
        face_np = face_tensor.permute(1, 2, 0).numpy()
        face_np = ((face_np * 0.5 + 0.5) * 255).astype(np.uint8)
        return face_np


# ─── Get Embedding ────────────────────────────────────────────────────────────
def get_embedding(backbone, face_rgb: np.ndarray, transform, device) -> np.ndarray:
    """Convert aligned face image to normalized 512-d embedding."""
    augmented = transform(image=face_rgb)
    tensor = augmented["image"].unsqueeze(0).to(device)  # [1, 3, 112, 112]

    with torch.no_grad():
        emb = backbone(tensor)
        emb = F.normalize(emb, p=2, dim=1)
    return emb.cpu().numpy()[0]   # shape: (512,)


# ─── Process Photos Folder ────────────────────────────────────────────────────
def process_photos(detector, det_type, backbone, transform, device,
                   photos_dir: Path, student_name: str):
    """
    Process all images in a folder for one student.
    Returns list of embeddings (one per valid face detected).
    """
    img_paths = list(photos_dir.glob("*.jpg")) + \
                list(photos_dir.glob("*.jpeg")) + \
                list(photos_dir.glob("*.png"))

    if not img_paths:
        print(f"[WARN] No images found in {photos_dir}")
        return []

    embeddings = []
    print(f"[INFO] Processing {len(img_paths)} images for '{student_name}'...")

    for img_path in img_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        face = detect_and_align(detector, det_type, img_bgr)
        if face is None:
            print(f"  [SKIP] No face in {img_path.name}")
            continue

        emb = get_embedding(backbone, face, transform, device)
        embeddings.append(emb)

    print(f"  → {len(embeddings)} valid embeddings from {len(img_paths)} images")
    return embeddings


# ─── Capture from Webcam ──────────────────────────────────────────────────────
def capture_from_webcam(detector, det_type, backbone, transform, device,
                         student_name: str, n_frames: int = 15):
    """Live webcam capture for enrollment."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return []

    print(f"\n[INFO] Webcam enrollment for '{student_name}'")
    print(f"  Press SPACE to capture a frame ({n_frames} needed) | Q to quit\n")

    embeddings = []
    captured = 0

    while captured < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"Captured: {captured}/{n_frames}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display, "SPACE=capture  Q=quit", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
        cv2.imshow("Enrollment", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            face = detect_and_align(detector, det_type, frame)
            if face is not None:
                emb = get_embedding(backbone, face, transform, device)
                embeddings.append(emb)
                captured += 1
                print(f"  ✓ Captured frame {captured}/{n_frames}")
            else:
                print("  [WARN] No face detected in frame. Try again.")

    cap.release()
    cv2.destroyAllWindows()
    return embeddings


# ─── Save Embeddings to DB ────────────────────────────────────────────────────
def save_to_database(student_name: str, embeddings: list):
    """
    Load existing DB, add/update student, save back.
    DB format: dict[student_name] = mean_embedding (512-d numpy array)
    """
    if not embeddings:
        print(f"[ERROR] No valid embeddings for '{student_name}'. Enrollment failed.")
        return

    db = {}
    if EMBEDDINGS_PATH.exists():
        with open(EMBEDDINGS_PATH, "rb") as f:
            db = pickle.load(f)

    # Store mean embedding (more robust than storing all)
    db[student_name] = np.mean(embeddings, axis=0)
    # Also store all embeddings for threshold calibration
    all_emb_path = DATA_DIR / "all_embeddings.pkl"
    all_db = {}
    if all_emb_path.exists():
        with open(all_emb_path, "rb") as f:
            all_db = pickle.load(f)
    all_db[student_name] = embeddings

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(db, f)
    with open(all_emb_path, "wb") as f:
        pickle.dump(all_db, f)

    print(f"[OK] '{student_name}' enrolled. DB now has {len(db)} students.")
    print(f"     Embeddings saved to {EMBEDDINGS_PATH}")


# ─── Rebuild Full Database ────────────────────────────────────────────────────
def rebuild_all(detector, det_type, backbone, transform, device):
    """
    Re-process all students in STUDENT_DB_DIR and rebuild embeddings DB.
    Run this after fine-tuning to refresh embeddings with the new model.
    """
    if not STUDENT_DB_DIR.exists():
        print(f"[ERROR] {STUDENT_DB_DIR} not found.")
        return

    student_dirs = [d for d in STUDENT_DB_DIR.iterdir() if d.is_dir()]
    print(f"[INFO] Rebuilding DB for {len(student_dirs)} students...")

    db = {}
    for student_dir in student_dirs:
        embs = process_photos(detector, det_type, backbone, transform, device,
                              student_dir, student_dir.name)
        if embs:
            db[student_dir.name] = np.mean(embs, axis=0)

    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(db, f)
    print(f"[OK] Database rebuilt with {len(db)} students → {EMBEDDINGS_PATH}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name",        type=str, default=None, help="Student full name")
    parser.add_argument("--photos",      type=str, default=None, help="Path to photo folder")
    parser.add_argument("--webcam",      action="store_true",    help="Use webcam capture")
    parser.add_argument("--n_frames",    type=int, default=15,   help="Webcam frames to capture")
    parser.add_argument("--rebuild_all", action="store_true",    help="Rebuild full database")
    args = parser.parse_args()

    device    = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    backbone  = load_backbone(device)
    detector, det_type = load_detector()
    transform = get_val_transforms(IMAGE_SIZE)

    if args.rebuild_all:
        rebuild_all(detector, det_type, backbone, transform, device)

    elif args.name:
        if args.webcam:
            embs = capture_from_webcam(detector, det_type, backbone, transform, device,
                                       args.name, args.n_frames)
        elif args.photos:
            photos_dir = Path(args.photos)
            assert photos_dir.exists(), f"Folder not found: {photos_dir}"
            embs = process_photos(detector, det_type, backbone, transform, device,
                                  photos_dir, args.name)
        else:
            print("[ERROR] Provide --photos or --webcam")
            sys.exit(1)

        save_to_database(args.name, embs)

    else:
        print("[ERROR] Provide --name or --rebuild_all")
        parser.print_help()
