from datetime import datetime, date
from typing import Optional, Dict


class AttendanceManager:
    """Tracks which students have been detected during the current session."""

    def __init__(self, min_confidence: float = 0.4):
        self.min_confidence = min_confidence
        self.session_date: date = date.today()

        # student_id -> student_name (simple: detected = present, no duplicates)
        self.present_students: dict = {}

        # Debug info kept separately (not sent to backend)
        self._debug_info: dict = {}

    def reset_session(self):
        """Start a new attendance session."""
        self.session_date = date.today()
        self.present_students = {}
        self._debug_info = {}
        print(f"[Attendance] New session started for {self.session_date}")

    def mark_detected(self, student_id: str, student_name: str, confidence: float):
        """Record that a student was detected. Only prints once per student."""
        if confidence < self.min_confidence:
            return

        if student_id not in self.present_students:
            self.present_students[student_id] = student_name
            print(f"[Attendance] {student_name} (ID: {student_id}) marked PRESENT")

            # Save debug info (for CSV/developer only, not sent to backend)
            self._debug_info[student_id] = {
                "first_seen": datetime.now(),
                "best_confidence": confidence,
                "detection_count": 1,
            }
        else:
            # Update debug info silently
            debug = self._debug_info.get(student_id, {})
            debug["detection_count"] = debug.get("detection_count", 0) + 1
            if confidence > debug.get("best_confidence", 0):
                debug["best_confidence"] = confidence
            self._debug_info[student_id] = debug

    def get_attendance_for_api(self, expected_students: Optional[Dict[str, str]] = None) -> dict:
        """Generate clean attendance data for the backend API.

        Simple format: each student appears once as present or absent.
        """
        attendance = []

        # Mark present students
        for student_id, student_name in self.present_students.items():
            attendance.append({
                "student_id": student_id,
                "student_name": student_name,
                "status": "present",
            })

        # Mark absent students (those expected but not detected)
        if expected_students:
            for student_id, student_name in expected_students.items():
                if student_id not in self.present_students:
                    attendance.append({
                        "student_id": student_id,
                        "student_name": student_name,
                        "status": "absent",
                    })

        return {
            "date": self.session_date.isoformat(),
            "attendance": attendance,
        }

    def get_debug_report(self, expected_students: Optional[Dict[str, str]] = None) -> dict:
        """Generate detailed report for CSV/developer debugging."""
        present = []
        for student_id, student_name in self.present_students.items():
            debug = self._debug_info.get(student_id, {})
            present.append({
                "student_id": student_id,
                "student_name": student_name,
                "status": "present",
                "first_seen": debug.get("first_seen", "").isoformat() if debug.get("first_seen") else "",
                "best_confidence": debug.get("best_confidence", 0),
                "detection_count": debug.get("detection_count", 0),
            })

        absent = []
        if expected_students:
            for student_id, student_name in expected_students.items():
                if student_id not in self.present_students:
                    absent.append({
                        "student_id": student_id,
                        "student_name": student_name,
                        "status": "absent",
                    })

        return {
            "date": self.session_date.isoformat(),
            "total_present": len(present),
            "total_absent": len(absent),
            "present": present,
            "absent": absent,
        }

    def is_present(self, student_id: str) -> bool:
        return student_id in self.present_students
