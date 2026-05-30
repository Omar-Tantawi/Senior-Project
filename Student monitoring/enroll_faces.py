"""
enroll_faces.py — Enroll a student's face for the attention monitor.

Uses the attendance system's OWN face models (RetinaFace det_10g + AdaFace
iresnet50) and storage code, so a student enrolled here is recognised
identically by both the attention monitor and the attendance system. The
embeddings land in the shared folder:

    {attendance_system}/data/embeddings/{student_id}/embeddings.npy
    {attendance_system}/data/embeddings/{student_id}/name.txt

`student_id` MUST be the student's database ID (students.id) — that is what the
backend expects in alerts.

Usage:
    # All images inside a folder (replaces any existing enrollment for this id)
    python enroll_faces.py --student-id 12 --name "Ali Mohammed" --photos "C:/path/to/ali_photos"

    # Specific image files
    python enroll_faces.py --student-id 14 --name "Omar T" --photos a.jpg b.jpg c.jpg

    # Add more photos to an already-enrolled student (keeps existing embeddings)
    python enroll_faces.py --student-id 14 --name "Omar T" --photos new.jpg --append

Tips for good recognition:
    - Use 3–5 clear, front-facing photos per student, different lighting/angles.
    - One face per photo is best. If several faces are present, the largest is used.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("enroll")

THIS_DIR       = Path(__file__).parent
ATTENDANCE_DIR = THIS_DIR.parent / "Project" / "AI" / "attendance_system"
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(photo_args: list[str]) -> list[Path]:
    """Expand folder arguments into a flat list of image files."""
    images: list[Path] = []
    for arg in photo_args:
        p = Path(arg)
        if p.is_dir():
            images.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS))
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            images.append(p)
        else:
            log.warning("Skipping (not an image or folder): %s", arg)
    return images


def main(args: argparse.Namespace) -> None:
    images = collect_images(args.photos)
    if not images:
        log.error("No images found. Pass image files or a folder of images via --photos.")
        sys.exit(1)
    log.info("Found %d photo(s) to process.", len(images))

    # ── Load the attendance system's detector + recognizer ───────────────────────
    sys.path.insert(0, str(ATTENDANCE_DIR))
    try:
        from core.detector import FaceDetector
        from core.recognizer import FaceRecognizer
    except ImportError as exc:
        log.error("Could not import attendance face modules: %s", exc)
        sys.exit(1)

    att_cfg = yaml.safe_load(open(ATTENDANCE_DIR / "config.yaml"))
    det_cfg = att_cfg["detection"]
    embeddings_dir = Path(args.embeddings_dir) if args.embeddings_dir else (ATTENDANCE_DIR / att_cfg["storage"]["embeddings_dir"])

    log.info("Loading face models (RetinaFace + AdaFace)…")
    detector = FaceDetector(
        det_model_path=str(ATTENDANCE_DIR / det_cfg["det_model"]),
        rec_model_path=str(ATTENDANCE_DIR / det_cfg["rec_model"]),
        det_size=tuple(det_cfg["det_size"]),
        det_thresh=det_cfg["det_thresh"],
    )
    recognizer = FaceRecognizer(
        embeddings_dir=str(embeddings_dir),
        similarity_threshold=att_cfg["recognition"]["similarity_threshold"],
        unknown_threshold=att_cfg["recognition"]["unknown_threshold"],
    )

    # ── Extract one embedding per photo (largest face) ───────────────────────────
    embeddings: list[np.ndarray] = []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            log.warning("Could not read %s — skipping.", img_path.name)
            continue

        faces = [f for f in detector.detect_faces(img) if f.embedding is not None]
        if not faces:
            log.warning("No face detected in %s — skipping.", img_path.name)
            continue

        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        embeddings.append(np.asarray(largest.embedding, dtype=np.float32))
        log.info("✓ %s (det_score=%.2f)", img_path.name, largest.det_score)

    if not embeddings:
        log.error("No usable faces found in any photo. Nothing enrolled.")
        sys.exit(1)

    # ── Append to existing embeddings if requested ───────────────────────────────
    if args.append and recognizer.is_enrolled(str(args.student_id)):
        existing = recognizer.database[str(args.student_id)]["embeddings"]
        embeddings = [*list(existing), *embeddings]
        log.info("Appending to %d existing embedding(s).", len(existing))

    # ── Save via the attendance recognizer (writes shared {id}/ layout) ──────────
    recognizer.enroll_student(str(args.student_id), args.name, embeddings)

    log.info("=" * 50)
    log.info("Enrolled: %s (student_id=%s) with %d embedding(s)", args.name, args.student_id, len(embeddings))
    log.info("Folder: %s", embeddings_dir / str(args.student_id))
    log.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enroll a student's face for the attention monitor (uses attendance models).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python enroll_faces.py --student-id 12 --name "Ali Mohammed" --photos "C:/photos/ali"
  python enroll_faces.py --student-id 14 --name "Omar T" --photos a.jpg b.jpg c.jpg
        """,
    )
    parser.add_argument("--student-id",     required=True, help="Database student ID (students.id)")
    parser.add_argument("--name",           required=True, help="Student display name")
    parser.add_argument("--photos",         required=True, nargs="+", help="Image file(s) or a folder of images")
    parser.add_argument("--embeddings-dir", default=None,  help="Override embeddings directory (defaults to attendance system's)")
    parser.add_argument("--append",         action="store_true", help="Add to existing embeddings instead of replacing")
    main(parser.parse_args())
