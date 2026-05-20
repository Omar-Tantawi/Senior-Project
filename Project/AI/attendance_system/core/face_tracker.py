"""Multi-Frame Face Tracker for CCTV.

Tracks the same person across multiple frames and averages their embeddings
to improve recognition accuracy on low-quality CCTV footage.

Strategy:
    1. Detect faces in each frame
    2. Match detections to existing tracks using IoU (bbox overlap)
    3. Accumulate embeddings for each track
    4. Once enough frames collected, generate averaged embedding for recognition

Why this works:
    Single CCTV frame → noisy embedding → uncertain recognition
    Multiple frames averaged → stable embedding → confident recognition

Example: A blurry frame might match the wrong student 30% of the time,
         but averaging 5 frames drops that to <1%.
"""

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


def iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute Intersection over Union for two bounding boxes [x1,y1,x2,y2]."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    if inter_area == 0:
        return 0.0

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return inter_area / float(box_a_area + box_b_area - inter_area)


@dataclass
class Track:
    """A tracked face across multiple frames."""
    track_id: int
    last_bbox: np.ndarray
    last_seen: float
    embeddings: list = field(default_factory=list)
    quality_scores: list = field(default_factory=list)
    frame_count: int = 0
    recognized_id: str | None = None
    recognized_name: str | None = None
    recognized_confidence: float = 0.0
    marked_attendance: bool = False

    def add_observation(self, bbox: np.ndarray, embedding: np.ndarray | None,
                        quality: float = 1.0):
        """Add a new detection to this track."""
        self.last_bbox = bbox
        self.last_seen = time.time()
        self.frame_count += 1
        if embedding is not None:
            self.embeddings.append(embedding)
            self.quality_scores.append(quality)

    def get_averaged_embedding(self, max_frames: int = 10,
                               weight_by_quality: bool = True) -> np.ndarray | None:
        """Compute weighted average of recent embeddings.

        Args:
            max_frames: Use only the last N frames
            weight_by_quality: Give higher quality frames more weight

        Returns:
            Averaged 512-dim embedding, or None if no embeddings.
        """
        if not self.embeddings:
            return None

        # Use only recent frames
        recent_embs = self.embeddings[-max_frames:]
        recent_qual = self.quality_scores[-max_frames:]

        embs_array = np.stack(recent_embs)  # (N, 512)

        if weight_by_quality:
            weights = np.array(recent_qual)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
            averaged = np.average(embs_array, axis=0, weights=weights)
        else:
            averaged = np.mean(embs_array, axis=0)

        # L2 normalize
        norm = np.linalg.norm(averaged)
        if norm > 0:
            averaged = averaged / norm

        return averaged


class FaceTracker:
    """Simple IoU-based face tracker.

    Args:
        iou_threshold: Minimum IoU to match a detection to a track (0.0-1.0)
        max_age_sec: Drop tracks not seen for this long
        min_frames_for_recognition: How many frames before averaging embeddings
    """

    def __init__(self,
                 iou_threshold: float = 0.3,
                 max_age_sec: float = 2.0,
                 min_frames_for_recognition: int = 3):
        self.iou_threshold = iou_threshold
        self.max_age_sec = max_age_sec
        self.min_frames_for_recognition = min_frames_for_recognition
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 0

    def update(self, detections: list) -> list[Track]:
        """Update tracker with new detections from current frame.

        Args:
            detections: List of objects with .bbox, .embedding attributes
                        (e.g., Face dataclass from detector.py)

        Returns:
            List of currently active tracks (updated with this frame's data)
        """
        now = time.time()

        # 1. Match detections to existing tracks using IoU
        matched_track_ids = set()
        unmatched_detections = []

        for det in detections:
            best_track_id = None
            best_iou = self.iou_threshold

            for track_id, track in self.tracks.items():
                if track_id in matched_track_ids:
                    continue
                score = iou(det.bbox, track.last_bbox)
                if score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is not None:
                # Update existing track
                quality = getattr(det, "quality_score", 1.0)
                self.tracks[best_track_id].add_observation(
                    det.bbox, det.embedding, quality
                )
                matched_track_ids.add(best_track_id)
            else:
                unmatched_detections.append(det)

        # 2. Create new tracks for unmatched detections
        for det in unmatched_detections:
            track = Track(
                track_id=self.next_track_id,
                last_bbox=det.bbox.copy(),
                last_seen=now,
            )
            quality = getattr(det, "quality_score", 1.0)
            track.add_observation(det.bbox, det.embedding, quality)
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

        # 3. Drop stale tracks
        stale_ids = [
            tid for tid, t in self.tracks.items()
            if now - t.last_seen > self.max_age_sec
        ]
        for tid in stale_ids:
            del self.tracks[tid]

        return list(self.tracks.values())

    def get_recognition_ready_tracks(self) -> list[Track]:
        """Return tracks with enough frames to do reliable recognition."""
        return [
            t for t in self.tracks.values()
            if len(t.embeddings) >= self.min_frames_for_recognition
        ]

    def get_track(self, track_id: int) -> Track | None:
        return self.tracks.get(track_id)

    def mark_attendance(self, track_id: int, student_id: str, student_name: str,
                        confidence: float):
        """Mark this track as having been recognized and counted."""
        track = self.tracks.get(track_id)
        if track:
            track.recognized_id = student_id
            track.recognized_name = student_name
            track.recognized_confidence = confidence
            track.marked_attendance = True

    def reset(self):
        """Clear all tracks (e.g., for new attendance session)."""
        self.tracks = {}
        self.next_track_id = 0


if __name__ == "__main__":
    # Quick test
    @dataclass
    class FakeFace:
        bbox: np.ndarray
        embedding: np.ndarray
        quality_score: float = 1.0

    tracker = FaceTracker(iou_threshold=0.3, max_age_sec=2.0)

    # Simulate 5 frames of one person walking
    print("Simulating 5 frames of a tracked person...")
    for i in range(5):
        fake = FakeFace(
            bbox=np.array([100 + i*5, 100, 200 + i*5, 200], dtype=float),
            embedding=np.random.randn(512).astype(np.float32),
            quality_score=0.8 + i * 0.04,
        )
        tracks = tracker.update([fake])
        print(f"  Frame {i+1}: {len(tracks)} active tracks, "
              f"track 0 has {len(tracks[0].embeddings)} embeddings")

    # Get averaged embedding
    avg = tracker.tracks[0].get_averaged_embedding(max_frames=5)
    print(f"\nAveraged embedding shape: {avg.shape}")
    print(f"Averaged embedding norm: {np.linalg.norm(avg):.4f}")
    print(f"Ready for recognition: {len(tracker.get_recognition_ready_tracks())} tracks")
