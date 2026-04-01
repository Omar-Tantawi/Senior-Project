"""
enroll.py — Build the student embedding database from enrollment photos.

HOW TO USE:
    1. Organize your student photos like this:

        data/students/
            S001_Alice_Johnson/
                photo1.jpg
                photo2.jpg
                photo3.jpg   ← ideally 5-10 photos per student
            S002_Bob_Smith/
                front.jpg
                side.jpg
            ...

       Folder name format: <StudentID>_<FirstName>_<LastName>
       (underscores separate ID from name parts)

    2. Run:
        python enroll.py --student_dir data/students/

       Or to add a single new student without re-enrolling everyone:
        python enroll.py --student_dir data/students/ --student_id S003

    3. The script saves the FAISS index to data/embeddings/.

DATA AUGMENTATION:
    If a student has fewer than 5 photos, we augment each photo with:
      - Horizontal flip
      - Slight brightness variations (±20%)
    This improves recognition robustness in varying classroom lighting.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("enroll")

# ── Local imports ─────────────────────────────────────────────────────────────
from face_analyzer  import FaceAnalyzer
from embedding_db   import EmbeddingDatabase


# ── Augmentation helpers ───────────────────────────────────────────────────────

def augment_image(img: np.ndarray) -> List[np.ndarray]:
    """
    Return a list of augmented versions of `img`.
    Used when a student has very few enrollment photos.
    """
    augmented = [img]  # always include original

    # Horizontal flip — captures slightly different face angle
    augmented.append(cv2.flip(img, 1))

    # Brightness variations — simulate different classroom lighting
    for factor in [0.8, 1.2]:
        bright = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        augmented.append(bright)

    # Mild Gaussian blur — simulates slight defocus
    augmented.append(cv2.GaussianBlur(img, (3, 3), 0))

    return augmented


# ── Student folder parser ──────────────────────────────────────────────────────

def parse_student_folder(folder: Path) -> Tuple[str, str]:
    """
    Extract student_id and name from a folder name.

    Expected format: "<ID>_<Name>" or "<ID>_<FirstName>_<LastName>"
    Examples:
        S001_Alice           → ("S001", "Alice")
        S001_Alice_Johnson   → ("S001", "Alice Johnson")
        2024001_John_Doe     → ("2024001", "John Doe")

    Falls back to using the whole folder name if no underscore found.
    """
    name = folder.name
    parts = name.split("_", 1)  # split on first underscore only
    if len(parts) == 2:
        student_id = parts[0]
        display_name = parts[1].replace("_", " ")
    else:
        student_id   = name
        display_name = name

    return student_id, display_name


def get_image_files(folder: Path) -> List[Path]:
    """Return all image files in a folder."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in extensions]


# ── Core enrollment logic ──────────────────────────────────────────────────────

def enroll_student(
    student_dir:  Path,
    analyzer:     FaceAnalyzer,
    database:     EmbeddingDatabase,
    augment:      bool,
    min_photos_warn: int,
) -> dict:
    """
    Process all photos for one student and add their embeddings to the DB.

    Returns a summary dict with enrollment statistics.
    """
    student_id, name = parse_student_folder(student_dir)
    image_files = get_image_files(student_dir)

    if not image_files:
        logger.warning("[%s] No images found in %s", student_id, student_dir)
        return {"student_id": student_id, "name": name, "success": 0, "failed": 0}

    if len(image_files) < min_photos_warn:
        logger.warning(
            "[%s] Only %d photo(s) — recommend at least %d for better accuracy.",
            student_id, len(image_files), min_photos_warn
        )

    embeddings: List[np.ndarray] = []
    photo_paths: List[str]       = []
    failed: int                  = 0

    for photo_path in image_files:
        img = cv2.imread(str(photo_path))
        if img is None:
            logger.warning("[%s] Cannot read image: %s", student_id, photo_path)
            failed += 1
            continue

        # Decide which images to embed
        images_to_process = augment_image(img) if augment else [img]

        for aug_img in images_to_process:
            # Run the full detection + embedding pipeline
            faces = analyzer.get_faces(aug_img)

            if not faces:
                logger.warning(
                    "[%s] No face detected in %s (augmented)",
                    student_id, photo_path.name
                )
                failed += 1
                continue

            if len(faces) > 1:
                # Multiple faces — pick the largest (most prominent)
                face = max(faces, key=lambda f: f.height)
                logger.warning(
                    "[%s] Multiple faces in %s — using largest.",
                    student_id, photo_path.name
                )
            else:
                face = faces[0]

            embeddings.append(face.embedding)
            photo_paths.append(str(photo_path))

    if not embeddings:
        logger.error("[%s] Could not extract ANY embeddings. Check photo quality.", student_id)
        return {"student_id": student_id, "name": name, "success": 0, "failed": failed}

    # Add all embeddings to database
    database.add_student_embeddings(student_id, name, embeddings, photo_paths)

    logger.info(
        "[%s] %-25s → %d embeddings enrolled (%d failed)",
        student_id, name, len(embeddings), failed
    )
    return {
        "student_id": student_id,
        "name":       name,
        "success":    len(embeddings),
        "failed":     failed,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize models
    logger.info("Loading face detection + recognition models...")
    analyzer = FaceAnalyzer(config)
    database = EmbeddingDatabase(config)

    student_root = Path(args.student_dir)
    if not student_root.exists():
        logger.error("Student directory not found: %s", student_root)
        sys.exit(1)

    # Determine which students to enroll
    if args.student_id:
        # Enroll a single student by ID prefix
        student_dirs = [
            d for d in student_root.iterdir()
            if d.is_dir() and d.name.startswith(args.student_id)
        ]
        if not student_dirs:
            logger.error("No folder found for student ID: %s", args.student_id)
            sys.exit(1)
    else:
        # Enroll all students
        student_dirs = [d for d in sorted(student_root.iterdir()) if d.is_dir()]

    logger.info("Found %d student folder(s) to process.", len(student_dirs))

    augment = config["enrollment"]["augment"]
    min_warn = config["enrollment"]["min_photos_warning"]

    summaries = []
    for student_dir in tqdm(student_dirs, desc="Enrolling students"):
        summary = enroll_student(student_dir, analyzer, database, augment, min_warn)
        summaries.append(summary)

    # Save database
    database.save()

    # Print summary
    total_ok   = sum(s["success"] for s in summaries)
    total_fail = sum(s["failed"]  for s in summaries)
    enrolled   = [s for s in summaries if s["success"] > 0]

    print("\n" + "="*55)
    print(f"  Enrollment Complete")
    print("="*55)
    print(f"  Students enrolled : {len(enrolled)}")
    print(f"  Total embeddings  : {total_ok}")
    print(f"  Failed images     : {total_fail}")
    print(f"  Database saved to : {config['paths']['faiss_index']}")
    print("="*55)

    if args.list:
        print("\nEnrolled students:")
        for s in database.student_list:
            print(f"  {s['student_id']:10s}  {s['name']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enroll students into the face recognition database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll all students in data/students/
  python enroll.py

  # Enroll all with custom student directory
  python enroll.py --student_dir /path/to/photos/

  # Re-enroll a single student
  python enroll.py --student_id S001

  # List all enrolled students after enrollment
  python enroll.py --list
        """
    )
    parser.add_argument("--config",      default="config.yaml",    help="Path to config.yaml")
    parser.add_argument("--student_dir", default="data/students/", help="Root folder of student photos")
    parser.add_argument("--student_id",  default=None,             help="Enroll only this student ID")
    parser.add_argument("--list",        action="store_true",       help="List all enrolled students after done")
    parser.add_argument("--debug",       action="store_true",       help="Enable debug logging")
    args = parser.parse_args()
    main(args)
