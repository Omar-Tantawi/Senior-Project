"""
monitor_with_faces.py — Per-student attention monitor.

Combines two models in one pipeline:
  1. Attendance face detector (RetinaFace det_10g + AdaFace iresnet50) —
     identifies WHICH student each face is. Reuses the EXACT same models and
     code as the attendance system so embeddings are directly comparable.
  2. YOLO sms_v1 — classifies each detected person as High / Low Attention

When a known student shows sustained Low Attention, it:
  - Saves a short video clip of the last ~30 seconds
  - POSTs an alert to the backend with the student ID and footage path

Enrollment database is shared with the attendance system — no separate
enrollment needed. Just make sure students are already enrolled there.

Usage:
    python monitor_with_faces.py --camera 0 --camera-code classroom-cam-01 --section-id 3
    python monitor_with_faces.py --video clip.mp4 --camera-code cam-02 --section-id 5
    python monitor_with_faces.py --camera 0 --headless

Environment variables (or CLI args):
    BACKEND_URL        e.g. http://localhost:8000
    AI_API_KEY         must match ATTENDANCE_AI_KEY in backend .env
    EMBEDDINGS_DIR     path to attendance_system/data/embeddings (auto-detected if omitted)
    FOOTAGE_DIR        where to save alert clips
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("monitor_faces")

# ── Paths ──────────────────────────────────────────────────────────────────────

THIS_DIR        = Path(__file__).parent
YOLO_MODEL_PATH = THIS_DIR / "weights" / "sms_v1" / "weights" / "best.pt"
ATTENDANCE_DIR  = THIS_DIR.parent / "Project" / "AI" / "attendance_system"
DEFAULT_EMBEDDINGS_DIR = ATTENDANCE_DIR / "data" / "embeddings"
DEFAULT_FOOTAGE_DIR    = THIS_DIR / "footage"

# ── Thresholds ─────────────────────────────────────────────────────────────────

YOLO_CONF          = 0.35    # YOLO detection confidence
YOLO_IMGSZ         = 480     # inference size for YOLO (smaller = faster on CPU)
FACE_DET_THRESH    = 0.50    # InsightFace detection confidence
FACE_DET_SIZE      = 640     # face detector input size (must match training size to avoid shape warnings)
MIN_FACE_HEIGHT    = 40      # pixels — skip distant/tiny faces
SIM_THRESHOLD      = 0.25    # cosine similarity to accept a face match (matches attendance config)
FRAME_SKIP         = 10      # run models every Nth frame (raise on CPU for smoother video)
RECOG_CACHE_TTL    = 20      # frames before re-running face recognition
IOU_MATCH_THRESH   = 0.20    # IoU to associate a face box with a YOLO box

# Per-student alert logic
WINDOW_SECONDS       = 12    # TEST: short window
MIN_FRAMES_TO_ALERT  = 2     # TEST: only need 2 frames
LOW_ATTN_RATIO       = 0.30  # TEST: 30% low attention triggers alert
ALERT_COOLDOWN       = 20    # TEST: 20s cooldown so you can trigger it multiple times

# Clip recording
CLIP_BUFFER_SECONDS = 35     # keep this many seconds of frames in rolling buffer
CLIP_FPS            = 15     # playback fps for saved clips

LOW_ATTENTION_CLASS = 1
CLASS_NAMES = {0: "High Attention", 1: "Low Attention"}


# ── Enrollment database ────────────────────────────────────────────────────────

class EnrollmentDB:
    """
    Loads face embeddings from the attendance system's per-student folders.

    Layout expected:
        embeddings_dir/
            {student_id}/
                embeddings.npy   — shape (N, 512) float32
                name.txt         — student display name
    """

    def __init__(self, embeddings_dir: Path) -> None:
        self.records: List[Tuple[str, str, np.ndarray]] = []  # (student_id, name, embedding)
        self._load(embeddings_dir)

    def _load(self, base: Path) -> None:
        if not base.exists():
            log.warning("Embeddings dir not found: %s — running without face recognition.", base)
            return

        for student_dir in base.iterdir():
            if not student_dir.is_dir():
                continue
            emb_file  = student_dir / "embeddings.npy"
            name_file = student_dir / "name.txt"
            if not emb_file.exists():
                continue

            student_id = student_dir.name
            name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else student_id

            embeddings = np.load(str(emb_file)).astype(np.float32)
            if embeddings.ndim == 1:
                embeddings = embeddings[np.newaxis, :]   # single embedding

            for emb in embeddings:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb = emb / norm
                self.records.append((student_id, name, emb))

        log.info("Loaded %d embedding(s) for %d student(s).",
                 len(self.records),
                 len({r[0] for r in self.records}))

    def search(self, query: np.ndarray) -> Tuple[Optional[str], Optional[str], float]:
        """Return (student_id, name, similarity) of best match, or (None, None, 0)."""
        if not self.records:
            return None, None, 0.0

        q = query.astype(np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        best_sim = -1.0
        best_id  = None
        best_name = None

        for sid, name, emb in self.records:
            sim = float(np.dot(q, emb))
            if sim > best_sim:
                best_sim  = sim
                best_id   = sid
                best_name = name

        if best_sim < SIM_THRESHOLD:
            return None, None, best_sim

        return best_id, best_name, best_sim

    @property
    def is_empty(self) -> bool:
        return len(self.records) == 0


# ── IoU helper ─────────────────────────────────────────────────────────────────

def iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    xi1 = max(a[0], b[0]); yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2]); yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def face_contained_in_yolo(face_box: np.ndarray, yolo_box: np.ndarray) -> bool:
    """True if the face bbox center lies inside the YOLO person bbox."""
    cx = (face_box[0] + face_box[2]) / 2
    cy = (face_box[1] + face_box[3]) / 2
    return yolo_box[0] <= cx <= yolo_box[2] and yolo_box[1] <= cy <= yolo_box[3]


# ── Alert sender ───────────────────────────────────────────────────────────────

def send_alert(
    backend_url: str,
    api_key: str,
    camera_code: str,
    section_id: Optional[int],
    student_id: str,
    student_name: str,
    low_ratio: float,
    duration_seconds: float,
    footage_path: Optional[str],
) -> bool:
    url = f"{backend_url.rstrip('/')}/api/webhooks/attention-alert"
    payload = {
        "type":                "low_attention",
        "camera_id":           camera_code,
        "student_id":          student_id,
        "student_name":        student_name,
        "low_attention_ratio": round(low_ratio, 4),
        "student_count":       1,
        "duration_seconds":    int(duration_seconds),
        "timestamp":           datetime.now(timezone.utc).isoformat(),
        "section_id":          section_id,
        "footage_path":        footage_path,
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"X-AI-Key": api_key, "Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            log.info("Alert sent — student=%s ratio=%.0f%%", student_name, low_ratio * 100)
            return True
        log.warning("Alert rejected: %d %s", resp.status_code, resp.text[:200])
    except requests.RequestException as exc:
        log.error("Failed to send alert: %s", exc)
    return False


# ── Clip recorder ──────────────────────────────────────────────────────────────

class ClipRecorder:
    """
    Maintains a rolling frame buffer and can flush the last N seconds to a
    video file on demand.
    """

    def __init__(self, footage_dir: Path, buffer_seconds: int = CLIP_BUFFER_SECONDS) -> None:
        footage_dir.mkdir(parents=True, exist_ok=True)
        self.footage_dir = footage_dir
        self.max_frames  = buffer_seconds * CLIP_FPS
        self.buffer: collections.deque[np.ndarray] = collections.deque(maxlen=self.max_frames)

    def push(self, frame: np.ndarray) -> None:
        self.buffer.append(frame.copy())

    def save(self, student_id: str, camera_code: str) -> Optional[str]:
        if not self.buffer:
            return None
        h, w = self.buffer[0].shape[:2]
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"low_attention_{camera_code}_{student_id}_{ts}.mp4"
        path = self.footage_dir / name

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, CLIP_FPS, (w, h))
        for frame in self.buffer:
            writer.write(frame)
        writer.release()
        log.info("Clip saved: %s", path)
        return str(path)


# ── Per-student tracker ────────────────────────────────────────────────────────

class StudentTracker:
    """
    Maintains a rolling attention window and cooldown per student.
    """

    def __init__(self) -> None:
        # student_id → deque of (timestamp, is_low_attention)
        self.windows: Dict[str, collections.deque] = collections.defaultdict(
            lambda: collections.deque()
        )
        self.last_alert: Dict[str, float] = {}

    def update(self, student_id: str, is_low: bool) -> None:
        now = time.time()
        dq  = self.windows[student_id]
        dq.append((now, is_low))
        # evict old entries
        while dq and (now - dq[0][0]) > WINDOW_SECONDS:
            dq.popleft()

    def should_alert(self, student_id: str) -> Tuple[bool, float]:
        """Returns (should_alert, low_ratio)."""
        dq  = self.windows[student_id]
        now = time.time()

        if len(dq) < MIN_FRAMES_TO_ALERT:
            return False, 0.0

        low_ratio = sum(1 for _, low in dq if low) / len(dq)
        if low_ratio < LOW_ATTN_RATIO:
            return False, low_ratio

        last = self.last_alert.get(student_id, 0.0)
        if (now - last) < ALERT_COOLDOWN:
            return False, low_ratio

        return True, low_ratio

    def mark_alerted(self, student_id: str) -> None:
        self.last_alert[student_id] = time.time()


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    backend_url    = args.backend_url    or os.environ.get("BACKEND_URL", "http://localhost:8000")
    api_key        = args.api_key        or os.environ.get("AI_API_KEY",  "")
    embeddings_dir = Path(args.embeddings_dir or os.environ.get("EMBEDDINGS_DIR", str(DEFAULT_EMBEDDINGS_DIR)))
    footage_dir    = Path(args.footage_dir    or os.environ.get("FOOTAGE_DIR",    str(DEFAULT_FOOTAGE_DIR)))

    if not api_key:
        log.warning("No AI_API_KEY set — alerts will be rejected by the backend.")

    # ── Load models ────────────────────────────────────────────────────────────
    log.info("Loading YOLO model…")
    yolo = YOLO(str(YOLO_MODEL_PATH))

    log.info("Loading face detector (attendance system: RetinaFace + AdaFace)…")
    try:
        import yaml
        sys.path.insert(0, str(ATTENDANCE_DIR))
        from core.detector import FaceDetector
        with open(ATTENDANCE_DIR / "config.yaml", "r") as f:
            att_cfg = yaml.safe_load(f)
        det_cfg = att_cfg["detection"]
        detector = FaceDetector(
            det_model_path=str(ATTENDANCE_DIR / det_cfg["det_model"]),
            rec_model_path=str(ATTENDANCE_DIR / det_cfg["rec_model"]),
            det_size=(FACE_DET_SIZE, FACE_DET_SIZE),
            det_thresh=det_cfg["det_thresh"],
        )
        use_face = True
        log.info("Face detector ready — same model as attendance enrollment.")
    except Exception as exc:
        log.warning("Face detector unavailable (%s) — attention-only mode (no student IDs).", exc)
        detector = None
        use_face = False

    db       = EnrollmentDB(embeddings_dir)
    recorder = ClipRecorder(footage_dir)
    tracker  = StudentTracker()

    # Recognition cache: face_box_key → (student_id, name, sim, last_frame)
    rec_cache: Dict[str, Tuple[Optional[str], Optional[str], float, int]] = {}

    # ── Open source ────────────────────────────────────────────────────────────
    source = args.video if args.video else int(args.camera)
    cap    = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open source: %s", source)
        sys.exit(1)

    session_start = time.time()
    frame_idx     = 0

    # Latest detections — kept between inference frames so boxes stay on screen
    # instead of flickering once every FRAME_SKIP frames.
    yolo_boxes: List[Tuple[np.ndarray, int]] = []                                  # (bbox, class_id)
    identified: List[Tuple[np.ndarray, Optional[str], Optional[str], float]] = []  # (face_bbox, sid, name, sim)

    log.info("Monitoring started. Camera: %s | Section: %s", args.camera_code, args.section_id)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        recorder.push(frame)

        if frame_idx % FRAME_SKIP != 0:
            if not args.headless:
                _draw(frame, yolo_boxes, identified)
                cv2.imshow("Student Attention Monitor — Q to quit", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
            continue

        # ── YOLO: attention detection ──────────────────────────────────────────
        yolo_results = yolo.predict(frame, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False, workers=0)
        yolo_boxes = []   # (bbox, class_id)
        for box in yolo_results[0].boxes:
            yolo_boxes.append((box.xyxy[0].cpu().numpy(), int(box.cls[0])))
        if yolo_boxes:
            log.info("YOLO: %s", [CLASS_NAMES[c] for _, c in yolo_boxes])
        else:
            log.debug("YOLO: no detections this frame")

        # ── Face detection + recognition ───────────────────────────────────────
        identified = []
        # identified: (face_bbox, student_id, name, similarity)

        if use_face and detector is not None:
            raw_faces = detector.detect_faces(frame)
            for f in raw_faces:
                if f.det_score < FACE_DET_THRESH:
                    continue
                bbox = np.asarray(f.bbox, dtype=np.float32)
                if (bbox[3] - bbox[1]) < MIN_FACE_HEIGHT:
                    continue
                if f.embedding is None:
                    continue

                # Recognition cache key: rounded face center
                cx = int((bbox[0] + bbox[2]) / 2 / 20) * 20
                cy = int((bbox[1] + bbox[3]) / 2 / 20) * 20
                cache_key = f"{cx}_{cy}"

                cached = rec_cache.get(cache_key)
                if cached and (frame_idx - cached[3]) <= RECOG_CACHE_TTL:
                    sid, name, sim = cached[0], cached[1], cached[2]
                else:
                    emb = np.asarray(f.embedding, dtype=np.float32)
                    sid, name, sim = db.search(emb)
                    rec_cache[cache_key] = (sid, name, sim, frame_idx)

                identified.append((bbox, sid, name, sim))

        # ── Associate faces with YOLO boxes ────────────────────────────────────
        # For each face, find the YOLO attention box that best matches it.
        # If only one YOLO box exists and only one face exists, associate them
        # directly — on a close-up webcam the boxes rarely overlap perfectly.
        for face_bbox, sid, name, sim in identified:
            attn_class = None
            best_iou   = IOU_MATCH_THRESH

            for ybox, ycls in yolo_boxes:
                if face_contained_in_yolo(face_bbox, ybox):
                    attn_class = ycls
                    break
                score = iou(face_bbox, ybox)
                if score > best_iou:
                    best_iou   = score
                    attn_class = ycls

            # Fallback A: one face + one YOLO box → assume same person
            if attn_class is None and len(yolo_boxes) == 1 and len(identified) == 1:
                attn_class = yolo_boxes[0][1]
                log.debug("Association fallback A: single face + single YOLO box → linked")
            # Fallback B: multiple faces but no overlap found → assign nearest YOLO box by center distance
            elif attn_class is None and yolo_boxes:
                fx = (face_bbox[0] + face_bbox[2]) / 2
                fy = (face_bbox[1] + face_bbox[3]) / 2
                nearest_cls = min(
                    yolo_boxes,
                    key=lambda yb: ((yb[0][0]+yb[0][2])/2 - fx)**2 + ((yb[0][1]+yb[0][3])/2 - fy)**2
                )[1]
                attn_class = nearest_cls
                log.debug("Association fallback B: nearest YOLO box → linked")

            if attn_class is None:
                log.debug("No YOLO box matched face (sid=%s) — skipping", sid)
                continue

            # Unknown students still tracked — give them a STABLE id based on
            # coarse face position so they accumulate a window across frames
            # (a per-frame id would reset every frame and never alert).
            if sid is not None:
                effective_sid  = sid
                effective_name = name
            else:
                ux = int((face_bbox[0] + face_bbox[2]) / 2 / 80) * 80
                uy = int((face_bbox[1] + face_bbox[3]) / 2 / 80) * 80
                effective_sid  = f"unknown_{ux}_{uy}"
                effective_name = "Unknown Student"

            is_low = (attn_class == LOW_ATTENTION_CLASS)
            log.info("Tracking %s → %s", effective_name, CLASS_NAMES[attn_class])
            tracker.update(effective_sid, is_low)

            # ── Check alert condition ──────────────────────────────────────────
            should, low_ratio = tracker.should_alert(effective_sid)
            if should:
                clip_path = recorder.save(effective_sid, args.camera_code)
                sent = send_alert(
                    backend_url      = backend_url,
                    api_key          = api_key,
                    camera_code      = args.camera_code,
                    section_id       = args.section_id,
                    student_id       = sid,          # None for unknowns — backend accepts nullable
                    student_name     = effective_name,
                    low_ratio        = low_ratio,
                    duration_seconds = time.time() - session_start,
                    footage_path     = clip_path,
                )
                if sent:
                    tracker.mark_alerted(effective_sid)

        # ── Draw ───────────────────────────────────────────────────────────────
        if not args.headless:
            _draw(frame, yolo_boxes, identified)
            cv2.imshow("Student Attention Monitor — Q to quit", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()
    log.info("Session ended after %.0fs.", time.time() - session_start)


def _draw(
    frame: np.ndarray,
    yolo_boxes: List[Tuple[np.ndarray, int]],
    identified: List[Tuple[np.ndarray, Optional[str], Optional[str], float]],
) -> None:
    def label(text: str, x: int, y: int, color: tuple) -> None:
        """Draw text with a filled background pill so it's always readable."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        yt = max(y, th + 6)
        cv2.rectangle(frame, (x, yt - th - 6), (x + tw + 6, yt + 2), color, -1)
        cv2.putText(frame, text, (x + 3, yt - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)

    # Draw attention boxes (green = High, red = Low)
    for bbox, cls in yolo_boxes:
        color = (50, 200, 50) if cls == 0 else (0, 60, 220)
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label(CLASS_NAMES[cls], x1, y1 - 4, color)

    # Draw face identity labels on top (name for known, "Unknown" otherwise)
    for face_bbox, sid, name, sim in identified:
        x1, y1 = int(face_bbox[0]), int(face_bbox[1])
        if sid is not None:
            label(f"{name} {sim:.2f}", x1, y1 - 26, (0, 220, 220))   # yellow = recognized
        else:
            label("Unknown", x1, y1 - 26, (120, 120, 120))           # gray = no match


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-student attention monitor — face recognition + YOLO sms_v1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python monitor_with_faces.py --camera 0 --camera-code classroom-cam-01 --section-id 3
  python monitor_with_faces.py --video clip.mp4 --camera-code cam-02 --section-id 5 --headless
        """,
    )
    parser.add_argument("--camera",          default=0,    type=int)
    parser.add_argument("--video",           default=None)
    parser.add_argument("--camera-code",     default="classroom-cam-01")
    parser.add_argument("--section-id",      default=None, type=int)
    parser.add_argument("--backend-url",     default=None)
    parser.add_argument("--api-key",         default=None)
    parser.add_argument("--embeddings-dir",  default=None, help="Path to attendance_system/data/embeddings")
    parser.add_argument("--footage-dir",     default=None, help="Where to save alert clips")
    parser.add_argument("--headless",        action="store_true")
    run(parser.parse_args())
