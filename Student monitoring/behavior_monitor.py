"""
behavior_monitor.py — Live classroom attention monitor using sms_v1 YOLO model.

Runs the student monitoring model on a camera/video feed and POSTs an alert to
the school backend whenever the classroom shows sustained low attention.

Usage:
    python behavior_monitor.py --camera 0 --camera-code classroom-cam-01 --section-id 3
    python behavior_monitor.py --video path/to/clip.mp4 --camera-code cam-02 --section-id 5
    python behavior_monitor.py --camera 0 --headless  # no display window

Environment variables (or pass as CLI args):
    BACKEND_URL   e.g. http://localhost:8000
    AI_API_KEY    shared secret — must match backend .env AI_API_KEY
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

import cv2
import requests
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("behavior_monitor")

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_PATH   = Path(__file__).parent / "weights" / "sms_v1" / "weights" / "best.pt"
CONF_THRESH  = 0.35          # detection confidence threshold
FRAME_SKIP   = 5             # run model every Nth frame (reduces GPU load)

# Attention tracking window
WINDOW_SECONDS    = 60       # rolling window for computing attention ratio
FPS_ESTIMATE      = 30       # used only to size the window deque before first FPS sample

# Alert thresholds
LOW_ATTENTION_TRIGGER = 0.60  # alert when ≥60% of detections are Low Attention
MIN_DETECTIONS        = 5     # need at least this many detections in window before alerting
ALERT_COOLDOWN        = 300   # seconds between successive alerts (avoid spam)

CLASS_NAMES = {0: "High Attention", 1: "Low Attention"}
LOW_ATTENTION_CLASS = 1

# ── Alert sender ───────────────────────────────────────────────────────────────

def send_alert(
    backend_url: str,
    api_key: str,
    camera_code: str,
    section_id: int | None,
    low_ratio: float,
    total_detections: int,
    duration_seconds: float,
) -> bool:
    url = f"{backend_url.rstrip('/')}/api/webhooks/attention-alert"
    payload = {
        "type":               "low_attention",
        "camera_id":          camera_code,
        "low_attention_ratio": round(low_ratio, 4),
        "student_count":      total_detections,
        "duration_seconds":   int(duration_seconds),
        "timestamp":          datetime.now(timezone.utc).isoformat(),
        "section_id":         section_id,
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            headers={"X-AI-Key": api_key, "Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code in (200, 201):
            log.info("Alert sent — low_attention_ratio=%.0f%%", low_ratio * 100)
            return True
        log.warning("Alert rejected: %d %s", resp.status_code, resp.text[:200])
    except requests.RequestException as exc:
        log.error("Failed to send alert: %s", exc)
    return False


# ── Visualization ──────────────────────────────────────────────────────────────

def draw_frame(frame, results, window_ratio: float | None) -> None:
    for box in results[0].boxes:
        cls   = int(box.cls[0])
        conf  = float(box.conf[0])
        label = f"{CLASS_NAMES.get(cls, str(cls))} {conf:.2f}"
        color = (50, 200, 50) if cls == 0 else (0, 60, 220)   # green / red
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if window_ratio is not None:
        bar_text = f"Low attention: {window_ratio * 100:.0f}%"
        bar_color = (0, 60, 220) if window_ratio >= LOW_ATTENTION_TRIGGER else (50, 200, 50)
        cv2.putText(frame, bar_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, bar_color, 2)


# ── Main loop ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    backend_url = args.backend_url or os.environ.get("BACKEND_URL", "http://localhost:8000")
    api_key     = args.api_key     or os.environ.get("AI_API_KEY",  "")

    if not api_key:
        log.warning("No AI_API_KEY set — alerts will be rejected by the backend.")

    log.info("Loading model: %s", MODEL_PATH)
    model = YOLO(str(MODEL_PATH))

    source = args.video if args.video else int(args.camera)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Cannot open source: %s", source)
        sys.exit(1)

    # rolling deque sized for WINDOW_SECONDS worth of sampled frames
    window_frames = int(WINDOW_SECONDS * FPS_ESTIMATE / FRAME_SKIP)
    # each entry: (timestamp, n_low, n_total) per processed frame
    window: collections.deque[tuple[float, int, int]] = collections.deque()

    frame_idx        = 0
    last_alert_time  = 0.0
    current_ratio    = None
    session_start    = time.time()

    log.info("Monitoring started. Camera: %s | Section: %s", args.camera_code, args.section_id)
    if not args.headless:
        log.info("Press Q / ESC to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            log.info("Stream ended.")
            break

        frame_idx += 1
        now = time.time()

        # ── Run model every FRAME_SKIP frames ─────────────────────────────────
        if frame_idx % FRAME_SKIP == 0:
            results = model.predict(frame, conf=CONF_THRESH, verbose=False, workers=0)
            classes = [int(b.cls[0]) for b in results[0].boxes]
            n_total = len(classes)
            n_low   = classes.count(LOW_ATTENTION_CLASS)
            window.append((now, n_low, n_total))

            # evict entries older than WINDOW_SECONDS
            while window and (now - window[0][0]) > WINDOW_SECONDS:
                window.popleft()

            # compute rolling ratio
            total_in_window = sum(t for _, _, t in window)
            low_in_window   = sum(l for _, l, _ in window)
            current_ratio   = (low_in_window / total_in_window) if total_in_window >= MIN_DETECTIONS else None

            # ── Alert logic ───────────────────────────────────────────────────
            if (
                current_ratio is not None
                and current_ratio >= LOW_ATTENTION_TRIGGER
                and (now - last_alert_time) >= ALERT_COOLDOWN
            ):
                sent = send_alert(
                    backend_url   = backend_url,
                    api_key       = api_key,
                    camera_code   = args.camera_code,
                    section_id    = args.section_id,
                    low_ratio     = current_ratio,
                    total_detections = total_in_window,
                    duration_seconds = now - session_start,
                )
                if sent:
                    last_alert_time = now

            if not args.headless:
                draw_frame(frame, results, current_ratio)

        # ── Display ────────────────────────────────────────────────────────────
        if not args.headless:
            cv2.imshow("Student Attention Monitor — Q to quit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    if not args.headless:
        cv2.destroyAllWindows()

    elapsed = time.time() - session_start
    log.info("Session ended after %.0fs.", elapsed)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Live classroom attention monitor — posts alerts to school backend.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python behavior_monitor.py --camera 0 --camera-code classroom-cam-01 --section-id 3
  python behavior_monitor.py --video clip.mp4 --camera-code cam-02 --section-id 5 --headless
  python behavior_monitor.py --camera 0 --backend-url http://192.168.1.10:8000 --camera-code cam-01
        """,
    )
    parser.add_argument("--camera",       default=0,    type=int,   help="Camera device index (default: 0)")
    parser.add_argument("--video",        default=None,             help="Video file or RTSP URL (overrides --camera)")
    parser.add_argument("--camera-code",  default="classroom-cam-01", help="Camera code matching camera.code in backend DB")
    parser.add_argument("--section-id",   default=None, type=int,   help="Section ID for notification targeting")
    parser.add_argument("--backend-url",  default=None,             help="Backend base URL (or set BACKEND_URL env var)")
    parser.add_argument("--api-key",      default=None,             help="AI API key (or set AI_API_KEY env var)")
    parser.add_argument("--headless",     action="store_true",      help="No display window (server mode)")
    run(parser.parse_args())
