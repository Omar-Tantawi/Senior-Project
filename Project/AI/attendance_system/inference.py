"""
inference.py — Live classroom camera → attendance sheet.

This is the main inference loop. It:
  1. Opens the classroom camera (or a video file for testing).
  2. For each frame:
       a. Detects all faces with RetinaFace.
       b. Checks liveness with MiniFASNet (rejects photos/screens).
       c. Matches each face embedding against the FAISS database.
       d. Assigns SORT track IDs so we don't re-identify on every frame.
       e. Uses temporal voting to confirm attendance.
  3. Writes the final attendance list to a CSV when the session ends.

HOW TO RUN:
    # Live camera (default device 0)
    python inference.py

    # Specific camera index
    python inference.py --camera 1

    # Video file (for testing)
    python inference.py --video path/to/classroom_video.mp4

    # RTSP stream (IP camera)
    python inference.py --video rtsp://192.168.1.10:554/stream

    # Run for 10 minutes then auto-exit
    python inference.py --duration 600

KEYBOARD CONTROLS (while window is open):
    Q or ESC  — quit and save attendance
    S         — save current attendance snapshot to CSV
    R         — reset all tracks and votes (start new session)
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("inference")

# ── Local imports ──────────────────────────────────────────────────────────────
from face_analyzer import FaceAnalyzer, DetectedFace
from embedding_db  import EmbeddingDatabase
from liveness      import LivenessDetector
from tracker       import FaceTracker, AttendanceVoter


# ── Visualization helpers ──────────────────────────────────────────────────────

# Color scheme (BGR)
COLOR_CONFIRMED = (50,  200, 50)   # Green  — confirmed present
COLOR_PENDING   = (50,  200, 255)  # Amber  — recognized, awaiting votes
COLOR_UNKNOWN   = (120, 120, 120)  # Gray   — face detected, no match
COLOR_SPOOF     = (0,   0,   220)  # Red    — liveness rejected

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 2


def draw_face(
    frame:       np.ndarray,
    face:        DetectedFace,
    label:       str,
    color:       tuple,
    similarity:  Optional[float],
    cfg_output:  dict,
) -> None:
    """Draw bounding box, name label, and similarity score on frame."""
    x1, y1, x2, y2 = face.bbox_int

    if cfg_output["draw_boxes"]:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)

    if cfg_output["draw_names"] and label:
        # Background pill behind text for readability
        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, label, (x1 + 3, y1 - 4), FONT, FONT_SCALE, (20, 20, 20), THICKNESS)

    if cfg_output["draw_similarity"] and similarity is not None:
        sim_text = f"{similarity:.2f}"
        cv2.putText(frame, sim_text, (x1 + 3, y2 - 4), FONT, 0.45, color, 1)


def draw_overlay(
    frame:      np.ndarray,
    confirmed:  Dict[str, str],  # {student_id: name}
    fps:        float,
    session_elapsed: float,
) -> None:
    """Draw top-right HUD: FPS, session time, confirmed count."""
    h, w = frame.shape[:2]
    panel_x = w - 250

    info_lines = [
        f"FPS: {fps:.1f}",
        f"Time: {int(session_elapsed // 60):02d}:{int(session_elapsed % 60):02d}",
        f"Present: {len(confirmed)}",
    ]

    for i, line in enumerate(info_lines):
        y = 25 + i * 22
        cv2.putText(frame, line, (panel_x, y), FONT, 0.55, (200, 200, 200), 1)

    # Confirmed student list on left side
    y_start = 30
    cv2.putText(frame, "Confirmed present:", (10, y_start), FONT, 0.5, (50, 200, 50), 1)
    for i, (sid, name) in enumerate(sorted(confirmed.items())):
        y = y_start + 20 + i * 20
        cv2.putText(frame, f"  {sid}: {name}", (10, y), FONT, 0.45, (200, 255, 200), 1)


# ── Attendance CSV ─────────────────────────────────────────────────────────────

def save_attendance_csv(
    confirmed:   Dict[str, str],   # {student_id: name}
    all_students: List[dict],       # full enrolled list [{student_id, name}]
    output_path:  str,
    session_start: datetime,
) -> None:
    """
    Write attendance to CSV with columns:
        student_id, name, status, timestamp
    All enrolled students appear — marked present or absent.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    timestamp_str = session_start.strftime("%Y-%m-%d %H:%M:%S")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["student_id", "name", "status", "session_date", "timestamp"]
        )
        writer.writeheader()

        for student in sorted(all_students, key=lambda s: s["student_id"]):
            sid   = student["student_id"]
            name  = student["name"]
            status = "PRESENT" if sid in confirmed else "ABSENT"
            writer.writerow({
                "student_id":   sid,
                "name":         name,
                "status":       status,
                "session_date": session_start.strftime("%Y-%m-%d"),
                "timestamp":    timestamp_str,
            })

    logger.info("Attendance saved → %s", output_path)

    # Print summary to terminal
    present = sum(1 for s in all_students if s["student_id"] in confirmed)
    absent  = len(all_students) - present
    print(f"\n{'='*50}")
    print(f"  Attendance Summary — {session_start.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*50}")
    print(f"  Present : {present}")
    print(f"  Absent  : {absent}")
    print(f"  Total   : {len(all_students)}")
    print(f"  Saved   : {output_path}")
    print(f"{'='*50}")
    if confirmed:
        print("\n  Present students:")
        for sid in sorted(confirmed):
            print(f"    ✓ {sid}: {confirmed[sid]}")


# ── Recognition cache ─────────────────────────────────────────────────────────

class RecognitionCache:
    """
    Caches the last recognition result for each track ID.
    Avoids running FAISS search on every frame for the same face.
    Cache is invalidated after `ttl_frames` frames or if the track is lost.
    """
    def __init__(self, ttl_frames: int = 10) -> None:
        self.ttl     = ttl_frames
        self._cache: Dict[int, Tuple[Optional[str], Optional[str], float, int]] = {}
        # track_id → (student_id, name, similarity, last_frame)

    def get(self, track_id: int, current_frame: int) -> Optional[Tuple[Optional[str], Optional[str], float]]:
        if track_id not in self._cache:
            return None
        sid, name, sim, last_frame = self._cache[track_id]
        if current_frame - last_frame > self.ttl:
            return None  # stale
        return sid, name, sim

    def set(self, track_id: int, student_id: Optional[str], name: Optional[str],
            similarity: float, current_frame: int) -> None:
        self._cache[track_id] = (student_id, name, similarity, current_frame)

    def invalidate(self, track_ids: set) -> None:
        """Remove entries for track IDs that no longer exist."""
        for tid in list(self._cache):
            if tid not in track_ids:
                del self._cache[tid]


# ── Main inference loop ────────────────────────────────────────────────────────

def run_inference(args: argparse.Namespace) -> None:
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Initialize components ─────────────────────────────────────────────────
    logger.info("Loading models...")
    analyzer  = FaceAnalyzer(config)
    database  = EmbeddingDatabase(config)
    liveness  = LivenessDetector(config)
    tracker   = FaceTracker(config)
    voter     = AttendanceVoter(config)
    rec_cache = RecognitionCache(ttl_frames=15)

    all_students = database.student_list
    if not all_students:
        logger.error("Database is empty! Run: python enroll.py  first.")
        sys.exit(1)
    logger.info("Loaded %d enrolled students.", len(all_students))

    # ── Open video source ─────────────────────────────────────────────────────
    source = args.video if args.video else int(args.camera)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        logger.error("Cannot open video source: %s", source)
        sys.exit(1)

    # Try to set higher resolution for better face detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera resolution: %dx%d", actual_w, actual_h)

    cfg_output = config["output"]
    output_csv = args.output or cfg_output["csv_path"]

    session_start = datetime.now()
    frame_idx     = 0
    fps_timer     = time.time()
    fps           = 0.0
    confirmed_ids: Dict[str, str] = {}   # {student_id: name} — final present list

    logger.info("Session started. Press Q or ESC to quit and save attendance.")
    logger.info("Waiting for faces... (students need to face the camera)")

    max_duration = args.duration if args.duration else float("inf")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("Video stream ended.")
            break

        frame_idx += 1
        elapsed = time.time() - fps_timer

        # ── FPS calculation (every 30 frames) ─────────────────────────────────
        if frame_idx % 30 == 0:
            fps = 30.0 / elapsed if elapsed > 0 else 0.0
            fps_timer = time.time()

        # ── Check session duration ────────────────────────────────────────────
        session_elapsed = (datetime.now() - session_start).total_seconds()
        if session_elapsed > max_duration:
            logger.info("Session duration reached (%ds). Saving attendance.", int(max_duration))
            break

        # ── Step 1: Detect all faces in frame ─────────────────────────────────
        faces: List[DetectedFace] = analyzer.get_faces(frame)

        # ── Step 2: Track faces across frames ─────────────────────────────────
        tracked_faces = tracker.update(faces)

        # Clean up stale cache entries
        active_track_ids = {f.track_id for f in tracked_faces}
        rec_cache.invalidate(active_track_ids)

        # ── Step 3: Liveness + recognition for each tracked face ───────────────
        for face in tracked_faces:
            draw_color  = COLOR_UNKNOWN
            draw_label  = "unknown"
            draw_sim    = None

            # 3a. Liveness check — skip if spoof detected
            is_live, live_score = liveness.check(frame, face.bbox)
            if not is_live:
                face.is_live = False
                draw_color   = COLOR_SPOOF
                draw_label   = f"SPOOF ({live_score:.2f})"
                draw_face(frame, face, draw_label, draw_color, None, cfg_output)
                continue
            face.is_live = True

            # 3b. Check recognition cache (avoids FAISS search every frame)
            cached = rec_cache.get(face.track_id, frame_idx)
            if cached is not None:
                student_id, name, similarity = cached
            else:
                # 3c. FAISS nearest-neighbour search
                student_id, name, similarity = database.search(face.embedding)
                rec_cache.set(face.track_id, student_id, name, similarity, frame_idx)

            face.identity   = student_id
            face.similarity = similarity

            # 3d. Temporal voting — only confirm after N consistent recognitions
            confirmed_id = voter.vote(face.track_id, student_id, similarity)

            if confirmed_id is not None:
                # This student is confirmed present
                confirmed_name = name or confirmed_id
                confirmed_ids[confirmed_id] = confirmed_name
                draw_color = COLOR_CONFIRMED
                draw_label = f"{confirmed_name} ✓"
                draw_sim   = similarity
            elif student_id is not None:
                # Recognized but not yet confirmed (still collecting votes)
                draw_color = COLOR_PENDING
                draw_label = f"{name}?"
                draw_sim   = similarity
            else:
                draw_color = COLOR_UNKNOWN
                draw_label = "unknown"
                draw_sim   = similarity if similarity > 0 else None

            draw_face(frame, face, draw_label, draw_color, draw_sim, cfg_output)

        # ── Step 4: Draw HUD overlay ───────────────────────────────────────────
        draw_overlay(frame, confirmed_ids, fps, session_elapsed)

        # ── Step 5: Show frame ────────────────────────────────────────────────
        if not args.headless:
            cv2.imshow("Attendance System — press Q to quit", frame)

        # ── Optional: save annotated frame ────────────────────────────────────
        if cfg_output.get("save_annotated_frames"):
            out_dir = Path(cfg_output["output_dir"]) / "frames"
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_dir / f"frame_{frame_idx:06d}.jpg"), frame)

        # ── Keyboard controls ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):   # Q or ESC
            logger.info("Quit requested.")
            break
        elif key in (ord("s"), ord("S")):      # S = snapshot
            snap_path = output_csv.replace(".csv", f"_snap_{frame_idx}.csv")
            save_attendance_csv(confirmed_ids, all_students, snap_path, session_start)
        elif key in (ord("r"), ord("R")):      # R = reset session
            logger.info("Session reset.")
            tracker.reset()
            voter.reset()
            confirmed_ids.clear()
            rec_cache.invalidate(set())

    # ── Session end ────────────────────────────────────────────────────────────
    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()

    # Save final attendance
    save_attendance_csv(confirmed_ids, all_students, output_csv, session_start)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run face recognition attendance on live camera or video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Live webcam
  python inference.py

  # USB camera index 1
  python inference.py --camera 1

  # Video file
  python inference.py --video classroom.mp4

  # RTSP IP camera
  python inference.py --video rtsp://192.168.1.10:554/stream

  # Auto-quit after 45 minutes
  python inference.py --duration 2700

  # No display (server mode)
  python inference.py --headless --output attendance_today.csv
        """
    )
    parser.add_argument("--config",   default="config.yaml",      help="Path to config.yaml")
    parser.add_argument("--camera",   default=0,     type=int,    help="Camera device index")
    parser.add_argument("--video",    default=None,               help="Video file or RTSP URL instead of camera")
    parser.add_argument("--output",   default=None,               help="Output CSV path (overrides config)")
    parser.add_argument("--duration", default=None, type=float,   help="Auto-quit after N seconds")
    parser.add_argument("--headless", action="store_true",        help="No display window (for server use)")
    parser.add_argument("--debug",    action="store_true",        help="Enable debug logging")
    args = parser.parse_args()
    run_inference(args)
