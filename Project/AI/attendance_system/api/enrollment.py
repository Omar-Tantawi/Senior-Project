"""Photo Upload Enrollment API.

Allows enrollment from uploaded photo files (not just live webcam).
This is what the backend will call when students enroll through the app.

Usage from backend:
    from api.enrollment import enroll_from_photos

    result = enroll_from_photos(
        student_id="12345",
        student_name="John Doe",
        photo_paths=["uploads/photo1.jpg", "uploads/photo2.jpg"],
    )

    if result["success"]:
        print(f"Enrolled with {result['photos_used']} valid photos")

Usage from command line:
    python api/enrollment.py --student-id 12345 --student-name "John Doe" \
        --photos photo1.jpg photo2.jpg photo3.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.detector import FaceDetector
from core.recognizer import FaceRecognizer


def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_largest_face(faces):
    """Return the face with the largest bounding box (closest/biggest face in photo)."""
    if not faces:
        return None
    largest = None
    max_area = 0
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            largest = face
    return largest


def enroll_from_photos(
    student_id: str,
    student_name: str,
    photo_paths: list,
    min_photos: int = 1,
    config: dict = None,
    overwrite: bool = False,
) -> dict:
    """Enroll a student from uploaded photo files.

    Args:
        student_id: Unique student ID
        student_name: Full name of the student
        photo_paths: List of paths to photo files (jpg, png, etc.)
        min_photos: Minimum valid photos required to enroll (default 1)
        config: Optional pre-loaded config (loads from config.yaml if None)
        overwrite: If True, replace existing enrollment for this student

    Returns:
        dict with keys:
            success (bool): Whether enrollment succeeded
            student_id (str): The student ID
            student_name (str): The student name
            photos_used (int): Number of photos successfully processed
            photos_failed (int): Number of photos that failed (no face, etc.)
            errors (list): Detailed errors per photo
            message (str): Human-readable status
    """
    if config is None:
        config = load_config()

    result = {
        "success": False,
        "student_id": student_id,
        "student_name": student_name,
        "photos_used": 0,
        "photos_failed": 0,
        "errors": [],
        "message": "",
    }

    # Initialize face detector and recognizer
    det_cfg = config["detection"]
    detector = FaceDetector(
        det_model_path=det_cfg["det_model"],
        rec_model_path=det_cfg["rec_model"],
        det_size=tuple(det_cfg["det_size"]),
        det_thresh=det_cfg["det_thresh"],
    )
    recognizer = FaceRecognizer(
        embeddings_dir=config["storage"]["embeddings_dir"],
        similarity_threshold=config["recognition"]["similarity_threshold"],
        unknown_threshold=config["recognition"]["unknown_threshold"],
    )

    # Check if already enrolled
    if not overwrite and student_id in recognizer.database:
        result["message"] = f"Student {student_id} already enrolled. Use overwrite=True to replace."
        return result

    # Process each photo
    embeddings = []
    for photo_path in photo_paths:
        photo_path = Path(photo_path)

        # Check file exists
        if not photo_path.exists():
            result["photos_failed"] += 1
            result["errors"].append({
                "photo": str(photo_path),
                "error": "File not found",
            })
            continue

        # Load image
        img = cv2.imread(str(photo_path))
        if img is None:
            result["photos_failed"] += 1
            result["errors"].append({
                "photo": str(photo_path),
                "error": "Could not load image (invalid format)",
            })
            continue

        # Detect faces
        faces = detector.detect_and_recognize(img)
        if not faces:
            result["photos_failed"] += 1
            result["errors"].append({
                "photo": str(photo_path),
                "error": "No face detected in photo",
            })
            continue

        # Get the largest face (assume it's the student)
        face = find_largest_face(faces)
        if face is None or face.embedding is None:
            result["photos_failed"] += 1
            result["errors"].append({
                "photo": str(photo_path),
                "error": "Could not extract face embedding",
            })
            continue

        # Add embedding
        embeddings.append(face.embedding)
        result["photos_used"] += 1
        print(f"[Enroll]   {photo_path.name}: face extracted (bbox area: "
              f"{int((face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))})")

    # Check minimum photos
    if len(embeddings) < min_photos:
        result["message"] = (
            f"Not enough valid photos: got {len(embeddings)}, need at least {min_photos}"
        )
        return result

    # Save enrollment
    try:
        recognizer.enroll_student(student_id, student_name, embeddings)
        result["success"] = True
        result["message"] = (
            f"Successfully enrolled {student_name} with {len(embeddings)} photos "
            f"({result['photos_failed']} failed)"
        )
    except Exception as e:
        result["message"] = f"Enrollment save failed: {e}"

    return result


def main():
    """Command-line interface for photo enrollment."""
    parser = argparse.ArgumentParser(
        description="Enroll student from uploaded photos"
    )
    parser.add_argument("--student-id", required=True, help="Unique student ID")
    parser.add_argument("--student-name", required=True, help="Full name")
    parser.add_argument("--photos", required=True, nargs="+",
                        help="Paths to photo files")
    parser.add_argument("--min-photos", type=int, default=1,
                        help="Minimum valid photos required (default 1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing enrollment if present")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Photo Enrollment")
    print(f"{'='*60}")
    print(f"Student ID:   {args.student_id}")
    print(f"Student Name: {args.student_name}")
    print(f"Photos:       {len(args.photos)} files")
    print(f"{'='*60}\n")

    result = enroll_from_photos(
        student_id=args.student_id,
        student_name=args.student_name,
        photo_paths=args.photos,
        min_photos=args.min_photos,
        overwrite=args.overwrite,
    )

    print(f"\n{'='*60}")
    if result["success"]:
        print(f"[SUCCESS] {result['message']}")
    else:
        print(f"[FAILED] {result['message']}")

    if result["errors"]:
        print(f"\nErrors:")
        for err in result["errors"]:
            print(f"  - {err['photo']}: {err['error']}")
    print(f"{'='*60}\n")

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
