import requests
from typing import Optional, Dict, List


class BackendAPIClient:
    """REST API client for communicating with the school management backend."""

    def __init__(self, base_url: str, timeout: int = 10,
                 endpoints: Optional[Dict] = None, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self.endpoints = endpoints or {
            "students":   "/ai/attendance/students",
            "attendance": "/ai/attendance",
        }
        self._available = None

    def _url(self, endpoint_key: str) -> str:
        return self.base_url + self.endpoints[endpoint_key]

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["X-AI-Key"] = self.api_key
        return h

    def is_available(self) -> bool:
        try:
            resp = requests.get(self.base_url, timeout=3)
            self._available = resp.status_code < 500
        except requests.ConnectionError:
            self._available = False
        return self._available

    def get_students(self) -> Optional[List[dict]]:
        try:
            resp = requests.get(self._url("students"), headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"[API] Failed to fetch students: {e}")
            return None

    def submit_attendance(self, attendance_data: dict) -> bool:
        try:
            resp = requests.post(self._url("attendance"), json=attendance_data,
                                 headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
            print(f"[API] Attendance submitted successfully")
            return True
        except Exception as e:
            print(f"[API] Failed to submit attendance: {e}")
            return False
