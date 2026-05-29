import cv2
import numpy as np
from typing import List


# Colors (BGR)
COLOR_PRESENT = (0, 200, 0)      # Green - recognized student
COLOR_UNKNOWN = (0, 0, 255)      # Red - unknown face
COLOR_LOW_CONF = (0, 165, 255)   # Orange - low confidence


def draw_detections(frame: np.ndarray, faces: list, results: List[dict]) -> np.ndarray:
    """Draw bounding boxes and labels on the frame.

    Args:
        frame: BGR image
        faces: list of InsightFace Face objects
        results: list of dicts with keys: student_id, student_name, confidence
    """
    annotated = frame.copy()

    for face, result in zip(faces, results):
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        student_id = result.get("student_id")
        student_name = result.get("student_name")
        confidence = result.get("confidence", 0.0)

        if student_id:
            color = COLOR_PRESENT
            label = f"{student_name} ({confidence:.0%})"
        else:
            color = COLOR_UNKNOWN
            label = f"Unknown ({confidence:.0%})"

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 8), (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return annotated


def draw_status_bar(frame: np.ndarray, present_count: int, total_enrolled: int,
                    fps: float = 0.0) -> np.ndarray:
    """Draw a status bar at the top of the frame."""
    h, w = frame.shape[:2]
    bar_height = 35

    # Semi-transparent bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    status_text = f"Present: {present_count}/{total_enrolled} | FPS: {fps:.1f}"
    cv2.putText(frame, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return frame
