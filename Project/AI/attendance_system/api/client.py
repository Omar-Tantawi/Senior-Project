import requests
from typing import Optional


class BackendAPIClient:
    """REST API client for communicating with the school management backend."""

    def __init__(self, base_url: str, timeout: int = 10, endpoints: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.endpoints = endpoints or {
            "students": "/students",
            "schedule": "/schedule",
            "attendance": "/attendance",
            "enroll": "/enroll",
        }
        self._available = None

    def _url(self, endpoint_key: str) -> str:
        return self.base_url + self.endpoints[endpoint_key]

    def is_available(self) -> bool:
        """Check if the backend API is reachable."""
        try:
            resp = requests.get(self.base_url, timeout=3)
            self._available = resp.status_code < 500
        except requests.ConnectionError:
            self._available = False
        return self._available

    def get_students(self) -> Optional[list[dict]]:
        """Fetch student roster. Returns list of {"student_id": str, "student_name": str}."""
        try:
            resp = requests.get(self._url("students"), timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[API] Failed to fetch students: {e}")
            return None

    def get_schedule(self, date: str = None) -> Optional[list[dict]]:
        """Fetch class schedule for a given date."""
        params = {"date": date} if date else {}
        try:
            resp = requests.get(self._url("schedule"), params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[API] Failed to fetch schedule: {e}")
            return None

    def submit_attendance(self, attendance_data: dict) -> bool:
        """Submit attendance report to the backend."""
        try:
            resp = requests.post(self._url("attendance"), json=attendance_data, timeout=self.timeout)
            resp.raise_for_status()
            print(f"[API] Attendance submitted successfully")
            return True
        except Exception as e:
            print(f"[API] Failed to submit attendance: {e}")
            return False

    def submit_enrollment(self, student_id: str, student_name: str) -> bool:
        """Notify backend that a student has been enrolled in face recognition."""
        try:
            resp = requests.post(
                self._url("enroll"),
                json={"student_id": student_id, "student_name": student_name},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return True
        except Exception as e:
            print(f"[API] Failed to submit enrollment: {e}")
            return False
