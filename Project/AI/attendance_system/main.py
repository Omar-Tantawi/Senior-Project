"""AI Attendance System.

Two ways to run:

  1. MANUAL (from terminal):
     python main.py --duration 2 --interval 5 --class-id "CS101"

  2. SERVER MODE (backend triggers it via HTTP):
     python main.py --server --port 5000
     Then backend calls: POST http://localhost:5000/scan
     with body: {"class_id": "CS101", "duration_sec": 120, "interval_sec": 5}
"""

import argparse
import time
import cv2
import yaml
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

from core.detector import FaceDetector
from core.recognizer import FaceRecognizer
from core.attendance_manager import AttendanceManager
from api.client import BackendAPIClient
from storage.csv_handler import CSVHandler
from utils.visualization import draw_detections, draw_status_bar


def load_config() -> dict:
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_unknown_face(frame, face, save_dir: Path) -> str:
    """Crop and save an unknown face for later review."""
    save_dir.mkdir(parents=True, exist_ok=True)
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    pad = 20
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = frame[y1:y2, x1:x2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = save_dir / f"unknown_{timestamp}.jpg"
    cv2.imwrite(str(path), crop)
    return str(path)


# ============================================================
#  Shared components (loaded once, reused across scans)
# ============================================================

_detector = None
_recognizer = None
_api_client = None
_csv_handler = None
_config = None


def init_system(config: dict):
    """Load models and initialize all components once."""
    global _detector, _recognizer, _api_client, _csv_handler, _config
    _config = config

    det_cfg = config["detection"]
    rec_cfg = config["recognition"]
    api_cfg = config["api"]
    store_cfg = config["storage"]

    base_dir = Path(__file__).parent
    det_model = str(base_dir / det_cfg["det_model"])
    rec_model = str(base_dir / det_cfg["rec_model"])

    _detector = FaceDetector(
        det_model_path=det_model,
        rec_model_path=rec_model,
        det_size=tuple(det_cfg["det_size"]),
        det_thresh=det_cfg["det_thresh"],
        max_faces=det_cfg["max_faces"],
    )
    _recognizer = FaceRecognizer(
        embeddings_dir=store_cfg["embeddings_dir"],
        similarity_threshold=rec_cfg["similarity_threshold"],
        unknown_threshold=rec_cfg["unknown_threshold"],
    )
    _api_client = BackendAPIClient(
        base_url=api_cfg["base_url"],
        timeout=api_cfg["timeout_sec"],
        endpoints=api_cfg["endpoints"],
    )
    _csv_handler = CSVHandler(output_dir=store_cfg["csv_output_dir"])


def run_scan(duration_sec: int = 120, interval_sec: int = 5,
             class_id: str = None, show_display: bool = True) -> dict:
    """Run a single attendance scan session.

    Args:
        duration_sec: How many SECONDS to scan (default 120 = 2 minutes)
        interval_sec: Seconds between face captures (default 5)
        class_id: Optional class identifier
        show_display: Show webcam window

    Returns:
        Attendance result dict (clean format for API)
    """
    config = _config
    cam_cfg = config["camera"]
    att_cfg = config["attendance"]
    store_cfg = config["storage"]

    print(f"\n{'='*50}")
    print(f"  Attendance Scan Starting")
    print(f"  Duration: {duration_sec} seconds")
    print(f"  Capture interval: {interval_sec} seconds")
    print(f"  Class ID: {class_id or 'N/A'}")
    print(f"{'='*50}\n")

    attendance = AttendanceManager(min_confidence=att_cfg["min_confidence"])

    # Fetch expected students from API
    expected_students = {}
    api_available = _api_client.is_available()
    if api_available:
        print("[API] Backend connected")
        students_data = _api_client.get_students()
        if students_data:
            expected_students = {s["student_id"]: s["student_name"] for s in students_data}
            print(f"[API] Loaded {len(expected_students)} expected students")
    else:
        print("[API] Backend not available - offline mode (CSV only)")

    if _recognizer.get_enrolled_count() == 0:
        print("\n[WARNING] No students enrolled! Run enroll.py first.\n")

    # Open camera
    cap = cv2.VideoCapture(cam_cfg["source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return {"error": "Cannot open camera"}

    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=duration_sec)
    last_capture = datetime.min
    frame_count = 0
    fps = 0.0
    fps_start = time.time()
    scan_count = 0

    unknown_dir = Path(store_cfg["unknown_faces_dir"])
    last_faces = []
    last_results = []

    print(f"Running until {end_time.strftime('%H:%M:%S')}... Press Q to stop early.\n")

    try:
        while datetime.now() < end_time:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame")
                break

            frame_count += 1
            now = datetime.now()

            elapsed = time.time() - fps_start
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()

            # Run detection at the configured interval
            if (now - last_capture).total_seconds() >= interval_sec:
                last_capture = now
                scan_count += 1
                remaining_sec = int((end_time - now).total_seconds())
                print(f"  [Scan {scan_count}] Detecting faces... ({remaining_sec}s remaining)")

                last_faces = _detector.detect_faces(frame)
                last_results = []

                for face in last_faces:
                    if face.embedding is None:
                        last_results.append({"student_id": None, "student_name": None, "confidence": 0.0})
                        continue

                    student_id, student_name, confidence = _recognizer.recognize(face.embedding)
                    last_results.append({
                        "student_id": student_id,
                        "student_name": student_name,
                        "confidence": confidence,
                    })

                    if student_id:
                        attendance.mark_detected(student_id, student_name, confidence)
                    else:
                        face_path = save_unknown_face(frame, face, unknown_dir)

                print(f"  [Scan {scan_count}] Found {len(last_faces)} faces | "
                      f"Present so far: {len(attendance.present_students)}")

            # Display
            if show_display:
                display = frame.copy()
                if last_faces and last_results:
                    display = draw_detections(display, last_faces, last_results)
                display = draw_status_bar(
                    display,
                    present_count=len(attendance.present_students),
                    total_enrolled=_recognizer.get_enrolled_count(),
                    fps=fps,
                )

                remaining = end_time - now
                secs_left = max(0, int(remaining.total_seconds()))
                time_text = f"Time left: {secs_left}s"
                cv2.putText(display, time_text, (display.shape[1] - 180, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                cv2.imshow("Attendance System", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("\n[INFO] Stopped by user")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # === Generate results ===

    # Clean API format (sent to backend)
    api_report = attendance.get_attendance_for_api(expected_students or None)
    if class_id:
        api_report["class_id"] = class_id

    # Debug format (saved to CSV for developer)
    debug_report = attendance.get_debug_report(expected_students or None)

    # Save CSV (debug)
    csv_path = _csv_handler.save_report(debug_report)

    # Submit to backend (clean format)
    if api_available:
        _api_client.submit_attendance(api_report)

    # Print summary
    present_count = len(attendance.present_students)
    absent_count = len(expected_students) - present_count if expected_students else 0

    print(f"\n{'='*50}")
    print(f"  Attendance Complete - {api_report['date']}")
    print(f"{'='*50}")
    print(f"  Scans performed: {scan_count}")
    print(f"  Present: {present_count}")
    print(f"  Absent:  {absent_count}")
    print(f"  CSV:     {csv_path}")
    print(f"{'='*50}")

    for entry in api_report.get("attendance", []):
        status_icon = "+" if entry["status"] == "present" else "-"
        print(f"  [{status_icon}] {entry['student_name']} (ID: {entry['student_id']}) - {entry['status']}")

    print()
    return api_report


# ============================================================
#  SERVER MODE - Backend triggers scans via HTTP
# ============================================================

class ScanHandler(BaseHTTPRequestHandler):
    """HTTP handler for backend-triggered attendance scans."""

    def do_POST(self):
        if self.path == "/scan":
            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}

            duration_sec = body.get("duration_sec", 120)
            interval_sec = body.get("interval_sec", 5)
            class_id = body.get("class_id", None)

            print(f"\n[SERVER] Scan requested: {duration_sec}s, every {interval_sec}s, class={class_id}")

            # Run scan (blocks until complete)
            result = run_scan(
                duration_sec=duration_sec,
                interval_sec=interval_sec,
                class_id=class_id,
                show_display=True,
            )

            # Return result to backend
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        elif self.path == "/enroll":
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}

            student_id = body.get("student_id")
            student_name = body.get("student_name")

            if not student_id or not student_name:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "student_id and student_name required"}')
                return

            print(f"\n[SERVER] Enrollment requested for {student_name} ({student_id})")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ready", "message": "Run enrollment on the AI machine"}).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == "/status":
            status = {
                "status": "running",
                "enrolled_students": _recognizer.get_enrolled_count() if _recognizer else 0,
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(status).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP logs."""
        pass


def run_server(port: int = 5000):
    """Start HTTP server that listens for scan requests from the backend."""
    server = HTTPServer(("0.0.0.0", port), ScanHandler)
    print(f"\n{'='*50}")
    print(f"  AI Attendance Server Running")
    print(f"  Listening on port {port}")
    print(f"  Enrolled students: {_recognizer.get_enrolled_count()}")
    print(f"{'='*50}")
    print(f"\n  Backend can now call:")
    print(f"    POST http://localhost:{port}/scan")
    print(f'    Body: {{"class_id": "CS101", "duration_sec": 120, "interval_sec": 5}}')
    print(f"\n    GET  http://localhost:{port}/status")
    print(f"\n  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down")
        server.shutdown()


# ============================================================
#  CLI Entry Point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="AI Attendance System")
    parser.add_argument("--server", action="store_true",
                        help="Run in server mode (backend triggers scans via HTTP)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Server port (default: 5000)")
    parser.add_argument("--duration", type=int, default=120,
                        help="Scan duration in SECONDS (default: 120)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Capture interval in seconds (default: 5)")
    parser.add_argument("--class-id", type=str, help="Class identifier")
    parser.add_argument("--no-display", action="store_true",
                        help="Run without GUI window")
    args = parser.parse_args()

    config = load_config()
    init_system(config)

    if args.server:
        run_server(port=args.port)
    else:
        run_scan(
            duration_sec=args.duration,
            interval_sec=args.interval,
            class_id=args.class_id,
            show_display=not args.no_display,
        )


if __name__ == "__main__":
    main()
