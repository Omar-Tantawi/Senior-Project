"""Dataset Preparation Script.

Prepares a face dataset for fine-tuning ArcFace.

Expected input structure (standard face dataset format):
    dataset_root/
        person_001/
            img1.jpg
            img2.jpg
        person_002/
            img1.jpg
            img2.jpg
        ...

What this script does:
    1. Detects faces in each image using RetinaFace
    2. Aligns faces to 112x112 (ArcFace standard)
    3. Saves aligned faces to output directory
    4. Creates a label mapping file (identity_name -> numeric_id)
    5. Optionally limits to N identities (for using a subset)

Usage:
    python prepare_dataset.py --input path/to/raw_dataset --output path/to/aligned_dataset
    python prepare_dataset.py --input path/to/raw_dataset --output path/to/aligned_dataset --max-identities 2000
    python prepare_dataset.py --input path/to/raw_dataset --output path/to/aligned_dataset --max-images-per-id 10
"""

import argparse
import sys
import os
import json
import random
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.detector import FaceDetector, _align_face


def prepare_dataset(input_dir: str, output_dir: str, det_model: str, rec_model: str,
                    max_identities: int = None, max_images_per_id: int = None,
                    min_images_per_id: int = 2):
    """Prepare aligned face dataset for training.

    Args:
        input_dir: Raw dataset with person_name/image.jpg structure
        output_dir: Where to save aligned 112x112 faces
        det_model: Path to RetinaFace ONNX model
        rec_model: Path to ArcFace ONNX model (not used here, but needed by FaceDetector)
        max_identities: Limit number of identities (None = use all)
        max_images_per_id: Limit images per identity (None = use all)
        min_images_per_id: Skip identities with fewer than this many images
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize detector (we only need detection + landmarks, not recognition)
    print("[Prep] Loading face detector...")
    detector = FaceDetector(
        det_model_path=det_model,
        rec_model_path=None,  # Don't need recognition model for alignment
        det_size=(640, 640),
        det_thresh=0.5,
    )

    # Get all identity directories
    identity_dirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
    print(f"[Prep] Found {len(identity_dirs)} identities in {input_dir}")

    # Limit identities if requested
    if max_identities and len(identity_dirs) > max_identities:
        random.seed(42)  # Reproducible
        identity_dirs = random.sample(identity_dirs, max_identities)
        print(f"[Prep] Limited to {max_identities} identities")

    label_map = {}  # identity_name -> numeric_id
    stats = {"total_images": 0, "successful": 0, "failed": 0, "skipped_identities": 0}
    current_label = 0

    for identity_dir in tqdm(identity_dirs, desc="Processing identities"):
        identity_name = identity_dir.name

        # Get image files
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(identity_dir.glob(ext))

        if len(image_files) < min_images_per_id:
            stats["skipped_identities"] += 1
            continue

        # Limit images per identity
        if max_images_per_id and len(image_files) > max_images_per_id:
            random.seed(42)
            image_files = random.sample(image_files, max_images_per_id)

        # Create output directory for this identity
        out_id_dir = output_path / f"{current_label:06d}"
        out_id_dir.mkdir(exist_ok=True)

        saved_count = 0
        for img_file in image_files:
            stats["total_images"] += 1

            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    stats["failed"] += 1
                    continue

                # Detect faces
                faces = detector.detect_faces(img)

                if not faces:
                    stats["failed"] += 1
                    continue

                # Take the largest face (most likely the subject)
                largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

                # Align face to 112x112
                aligned = _align_face(img, largest.landmarks, image_size=112)

                # Save
                out_file = out_id_dir / f"{saved_count:04d}.jpg"
                cv2.imwrite(str(out_file), aligned)
                saved_count += 1
                stats["successful"] += 1

            except Exception as e:
                stats["failed"] += 1
                continue

        if saved_count >= min_images_per_id:
            label_map[identity_name] = current_label
            current_label += 1
        else:
            # Remove directory if not enough successful alignments
            import shutil
            shutil.rmtree(out_id_dir, ignore_errors=True)
            stats["skipped_identities"] += 1

    # Save label mapping
    label_map_file = output_path / "label_map.json"
    with open(label_map_file, "w") as f:
        json.dump(label_map, f, indent=2)

    # Save stats
    stats["total_identities"] = current_label
    stats_file = output_path / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  Dataset Preparation Complete")
    print(f"{'='*50}")
    print(f"  Identities: {current_label}")
    print(f"  Total images processed: {stats['total_images']}")
    print(f"  Successful alignments: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Skipped identities: {stats['skipped_identities']}")
    print(f"  Output: {output_path}")
    print(f"  Label map: {label_map_file}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Prepare face dataset for ArcFace fine-tuning")
    parser.add_argument("--input", required=True, help="Path to raw dataset (person_name/image.jpg)")
    parser.add_argument("--output", required=True, help="Path to save aligned dataset")
    parser.add_argument("--max-identities", type=int, default=None,
                        help="Limit number of identities (for using a subset)")
    parser.add_argument("--max-images-per-id", type=int, default=None,
                        help="Limit images per identity")
    parser.add_argument("--min-images-per-id", type=int, default=2,
                        help="Skip identities with fewer images (default: 2)")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    det_model = str(base_dir / "models" / "det_10g.onnx")
    rec_model = str(base_dir / "models" / "w600k_r50.onnx")

    prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        det_model=det_model,
        rec_model=rec_model,
        max_identities=args.max_identities,
        max_images_per_id=args.max_images_per_id,
        min_images_per_id=args.min_images_per_id,
    )


if __name__ == "__main__":
    main()
