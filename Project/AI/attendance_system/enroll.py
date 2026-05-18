"""Student Enrollment Script.

Usage:
    python enroll.py --student-id 12345 --student-name "John Doe"
    python enroll.py --student-id 12345 --student-name "John Doe" --photos 5

Captures face photos from webcam, extracts embeddings, and saves them to disk.
"""

import argparse
import time
import cv2
import yaml
import numpy as np
from pathlib import Path

from core.detector import FaceDetector
from core.recognizer import FaceRecognizer


def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_largest_face(faces):
    """Return the face with the largest bounding box area (closest to camera)."""
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def enroll_student(student_id: str, student_name: str, config: dict):
    cam_cfg = config["camera"]
    enroll_cfg = config["enrollment"]
    num_photos = enroll_cfg["photos_per_student"]
    delay = enroll_cfg["capture_delay_sec"]

    print(f"\n{'='*50}")
    print(f"  Enrolling: {student_name} (ID: {student_id})")
    print(f"  Photos to capture: {num_photos}")
    print(f"{'='*50}")
    print(f"\nInstructions:")
    print(f"  1. Face the camera directly")
    print(f"  2. When prompted, turn slightly LEFT")
    print(f"  3. Then turn slightly RIGHT")
    print(f"  4. Stay natural - blink normally")
    print(f"\nPress SPACE to capture each photo. Press Q to cancel.\n")

    # Initialize
    base_dir = Path(__file__).parent
    det_model = str(base_dir / config["detection"]["det_model"])
    rec_model = str(base_dir / config["detection"]["rec_model"])

    detector = FaceDetector(
        det_model_path=det_model,
        rec_model_path=rec_model,
        det_size=tuple(config["detection"]["det_size"]),
        det_thresh=config["detection"]["det_thresh"],
    )
    recognizer = FaceRecognizer(
        embeddings_dir=config["storage"]["embeddings_dir"],
        similarity_threshold=config["recognition"]["similarity_threshold"],
    )

    if recognizer.is_enrolled(student_id):
        print(f"[WARNING] Student {student_id} is already enrolled.")
        resp = input("Overwrite? (y/n): ").strip().lower()
        if resp != "y":
            print("Enrollment cancelled.")
            return

    cap = cv2.VideoCapture(cam_cfg["source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    embeddings = []
    poses = ["FRONT", "SLIGHT LEFT", "SLIGHT RIGHT", "FRONT (natural)", "FRONT (different expression)"]

    captured = 0
    faces = []
    print("  [INFO] Camera is opening... (detection runs on CPU, may be slow)")

    while captured < num_photos:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break

        display = frame.copy()
        pose_label = poses[captured] if captured < len(poses) else "FRONT"
        info = f"Photo {captured + 1}/{num_photos} - Look: {pose_label} - SPACE to capture, Q to quit"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw previous detection results if any
        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        if faces:
            largest = find_largest_face(faces)
            bbox = largest.bbox.astype(int)
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 3)
            cv2.putText(display, "TARGET", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Enrollment", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("\nEnrollment cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            return

        if key == ord(" "):  # Space bar
            # Run detection only when SPACE is pressed
            print("  [*] Detecting face... (please wait)")
            faces = detector.detect_faces(frame)

            if not faces:
                print("  [!] No face detected. Try again.")
                continue

            largest = find_largest_face(faces)
            if largest.embedding is None:
                print("  [!] Could not extract embedding. Try again.")
                continue

            embeddings.append(largest.embedding)
            captured += 1
            print(f"  [+] Captured photo {captured}/{num_photos} ({pose_label})")

            if captured < num_photos:
                # Brief pause between captures
                time.sleep(delay)

    cap.release()
    cv2.destroyAllWindows()

    if len(embeddings) < 3:
        print(f"\n[ERROR] Need at least 3 photos, got {len(embeddings)}. Enrollment failed.")
        return

    # Save embeddings
    recognizer.enroll_student(student_id, student_name, embeddings)
    print(f"\n[SUCCESS] {student_name} enrolled with {len(embeddings)} photos!")


def main():
    parser = argparse.ArgumentParser(description="Enroll a student for face recognition")
    parser.add_argument("--student-id", required=True, help="Student ID number")
    parser.add_argument("--student-name", required=True, help="Student full name")
    parser.add_argument("--photos", type=int, help="Number of photos to capture (overrides config)")
    args = parser.parse_args()

    config = load_config()
    if args.photos:
        config["enrollment"]["photos_per_student"] = args.photos

    enroll_student(args.student_id, args.student_name, config)


if __name__ == "__main__":
    main()
