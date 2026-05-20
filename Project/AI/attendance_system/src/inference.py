"""
src/inference.py — Real-Time Multi-Face Attendance System

Handles:
  - Multiple faces per frame (classroom)
  - Wall-mounted angled camera
  - Low/uneven lighting (CLAHE preprocessing)
  - Partial occlusion (robust RetinaFace)
  - Attendance cooldown (no duplicate marking)
  - Consecutive-frame confirmation (reduces false positives)

Usage:
    # Run on webcam
    python src/inference.py

    # Run on video file
    python src/inference.py --source path/to/video.mp4

    # Run on RTSP stream (IP camera)
    python src/inference.py --source rtsp://192.168.1.100/stream
"""

import os
import sys
import cv2
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.model import iresnet50
from src.augmentation import get_val_transforms
from src.preprocess import normalize_lighting, align_face


# ─── Cosine Similarity ────────────────────────────────────────────────────────
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b))


# ─── Attendance Tracker ───────────────────────────────────────────────────────
class AttendanceTracker:
    """
    Manages attendance state:
      - Cooldown: prevents re-marking same student within N seconds
      - Consecutive frames: requires N consecutive detections before marking
      - CSV logging
    """

    def __init__(self, session_id: str = None):
        self.session_id    = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.marked        = {}             # student_name → timestamp of last mark
        self.frame_buffer  = defaultdict(deque)  # student → deque of recent detections
        self.confirmed     = set()          # Students confirmed present
        self.session_log   = []             # For CSV export

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.csv_path = OUTPUT_DIR / f"attendance_{self.session_id}.csv"
        print(f"[INFO] Attendance CSV: {self.csv_path}")

    def update(self, student_name: str, similarity: float) -> bool:
        """
        Returns True if this detection triggers a new attendance mark.
        """
        now = time.time()

        # Add to rolling buffer
        self.frame_buffer[student_name].append(now)

        # Keep only recent frames (within 2 seconds)
        cutoff = now - 2.0
        while self.frame_buffer[student_name] and self.frame_buffer[student_name][0] < cutoff:
            self.frame_buffer[student_name].popleft()

        # Need MIN_CONSECUTIVE_FRAMES detections in the buffer
        if len(self.frame_buffer[student_name]) < MIN_CONSECUTIVE_FRAMES:
            return False

        # Cooldown check
        if student_name in self.marked:
            elapsed = now - self.marked[student_name]
            if elapsed < ATTENDANCE_COOLDOWN_SEC:
                return False   # Already marked recently

        # ── Mark attendance ──────────────────────────────────────────────────
        self.marked[student_name] = now
        self.confirmed.add(student_name)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_log.append({
            "session_id":  self.session_id,
            "student":     student_name,
            "timestamp":   timestamp,
            "similarity":  round(similarity, 4),
            "status":      "Present"
        })
        self._save_csv()
        return True

    def _save_csv(self):
        df = pd.DataFrame(self.session_log)
        df.to_csv(self.csv_path, index=False)

    def get_summary(self) -> dict:
        return {
            "total_marked":  len(self.confirmed),
            "students":      sorted(self.confirmed),
            "session_id":    self.session_id,
            "csv_path":      str(self.csv_path)
        }


# ─── Face Recognizer ─────────────────────────────────────────────────────────
class FaceRecognizer:
    """
    Wraps backbone + embedding DB for real-time recognition.
    """

    def __init__(self, device):
        self.device    = device
        self.backbone  = self._load_backbone()
        self.transform = get_val_transforms(IMAGE_SIZE)
        self.db        = self._load_db()
        self.detector, self.det_type = self._load_detector()

    def _load_backbone(self):
        path = MODEL_DIR / "finetuned_backbone.pth"
        if not path.exists():
            path = BACKBONE_PATH
        if not path.exists():
            raise FileNotFoundError("No model found. Run train.py and enroll.py first.")

        backbone = iresnet50(embedding_dim=EMBEDDING_DIM)
        backbone.load_state_dict(torch.load(path, map_location=self.device))
        backbone = backbone.to(self.device).eval()
        print(f"[OK] Model loaded from {path}")
        return backbone

    def _load_db(self) -> dict:
        if not EMBEDDINGS_PATH.exists():
            raise FileNotFoundError("Embedding DB not found. Run enroll.py first.")
        with open(EMBEDDINGS_PATH, "rb") as f:
            db = pickle.load(f)
        print(f"[OK] Embedding DB loaded: {len(db)} students")
        return db

    def _load_detector(self):
        try:
            import insightface
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            print("[OK] RetinaFace detector loaded (GPU)")
            return app, "insightface"
        except Exception:
            from facenet_pytorch import MTCNN
            detector = MTCNN(keep_all=True, device=str(self.device))
            print("[OK] MTCNN detector loaded")
            return detector, "mtcnn"

    def detect_faces(self, frame_bgr: np.ndarray):
        """
        Returns list of dicts: [{bbox, landmarks, conf}]
        Handles multiple faces for crowded classroom.
        """
        frame_bgr = normalize_lighting(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        results = []

        if self.det_type == "insightface":
            faces = self.detector.get(frame_rgb)
            for face in faces[:MAX_FACES]:
                if face.det_score < CONFIDENCE_THRESHOLD:
                    continue
                results.append({
                    "bbox":       face.bbox.astype(int),
                    "landmarks":  face.kps.astype(np.float32),
                    "conf":       float(face.det_score),
                    "aligned":    align_face(frame_rgb, face.kps.astype(np.float32))
                })

        else:  # MTCNN
            from PIL import Image
            pil = Image.fromarray(frame_rgb)
            boxes, probs, landmarks = self.detector.detect(pil, landmarks=True)
            if boxes is None:
                return []
            for box, prob, lm in zip(boxes, probs, landmarks):
                if prob < CONFIDENCE_THRESHOLD:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                face_crop = frame_rgb[max(0,y1):y2, max(0,x1):x2]
                if face_crop.size == 0:
                    continue
                face_resized = cv2.resize(face_crop, (IMAGE_SIZE, IMAGE_SIZE))
                results.append({
                    "bbox":      np.array([x1, y1, x2, y2]),
                    "landmarks": None,
                    "conf":      float(prob),
                    "aligned":   face_resized
                })

        return results

    def get_embedding(self, face_rgb: np.ndarray) -> np.ndarray:
        aug    = self.transform(image=face_rgb)
        tensor = aug["image"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.backbone(tensor)
            emb = F.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy()[0]

    def recognize(self, embedding: np.ndarray) -> tuple[str, float]:
        """
        Compare embedding against all enrolled students.
        Returns (name, similarity_score).
        """
        best_name  = UNKNOWN_LABEL
        best_score = 0.0

        for name, db_emb in self.db.items():
            score = cosine_similarity(embedding, db_emb)
            if score > best_score:
                best_score = score
                best_name  = name

        if best_score < RECOGNITION_THRESHOLD:
            best_name = UNKNOWN_LABEL

        return best_name, best_score


# ─── Drawing ──────────────────────────────────────────────────────────────────
def draw_face(frame, bbox, name, score, just_marked: bool):
    x1, y1, x2, y2 = bbox

    if name == UNKNOWN_LABEL:
        color = (0, 0, 255)       # Red = Unknown
    elif just_marked:
        color = (0, 255, 0)       # Green = Just marked
    else:
        color = (255, 200, 0)     # Blue = Already marked

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{name} ({score:.2f})"
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + len(label) * 10, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    if just_marked:
        cv2.putText(frame, "✓ MARKED", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def draw_hud(frame, tracker: AttendanceTracker, fps: float):
    h, w = frame.shape[:2]
    summary = tracker.get_summary()

    # Semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 120), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"FPS: {fps:.1f}",                     (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    cv2.putText(frame, f"Present: {summary['total_marked']}",  (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),    1)
    cv2.putText(frame, f"Session: {summary['session_id']}",    (10, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.45,(200,200,200),1)
    cv2.putText(frame, "Q: quit | S: summary",                 (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150),1)


# ─── Main Inference Loop ──────────────────────────────────────────────────────
def run_inference(source):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    recognizer = FaceRecognizer(device)
    tracker    = AttendanceTracker()

    cap = cv2.VideoCapture(source if isinstance(source, str) else int(source))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    # Optionally set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n[INFO] Starting attendance system. Press Q to quit, S for summary.\n")

    frame_count = 0
    fps_timer   = time.time()
    fps         = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        frame_count += 1

        # FPS calculation
        if frame_count % 30 == 0:
            elapsed = time.time() - fps_timer
            fps = 30.0 / elapsed
            fps_timer = time.time()

        # Skip frames for performance
        if frame_count % FRAME_SKIP != 0:
            draw_hud(frame, tracker, fps)
            cv2.imshow("Attendance System", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        # ── Detect all faces in frame ──────────────────────────────────────
        face_results = recognizer.detect_faces(frame)

        for face_data in face_results:
            aligned  = face_data["aligned"]
            bbox     = face_data["bbox"]
            conf     = face_data["conf"]

            # ── Get embedding & recognize ──────────────────────────────────
            embedding        = recognizer.get_embedding(aligned)
            name, similarity = recognizer.recognize(embedding)

            # ── Update attendance tracker ──────────────────────────────────
            just_marked = False
            if name != UNKNOWN_LABEL:
                just_marked = tracker.update(name, similarity)
                if just_marked:
                    print(f"  ✓ MARKED: {name} (similarity={similarity:.3f})")

            draw_face(frame, bbox, name, similarity, just_marked)

        draw_hud(frame, tracker, fps)
        cv2.imshow("Attendance System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            summary = tracker.get_summary()
            print("\n─── ATTENDANCE SUMMARY ─────────────────────────")
            for s in summary["students"]:
                print(f"  ✓ {s}")
            print(f"Total present: {summary['total_marked']}")
            print(f"CSV saved:     {summary['csv_path']}")
            print("────────────────────────────────────────────────\n")

    cap.release()
    cv2.destroyAllWindows()

    # Final summary
    summary = tracker.get_summary()
    print("\n═══ SESSION COMPLETE ════════════════════════════")
    print(f"  Session ID: {summary['session_id']}")
    print(f"  Students marked present: {summary['total_marked']}")
    for s in summary["students"]:
        print(f"    ✓ {s}")
    print(f"  Saved to: {summary['csv_path']}")
    print("═════════════════════════════════════════════════\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=0,
                        help="Camera index (0), video file path, or RTSP URL")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run_inference(source)
