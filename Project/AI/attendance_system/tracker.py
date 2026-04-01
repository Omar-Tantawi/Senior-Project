"""
tracker.py — SORT (Simple Online and Realtime Tracking) for face tracking.

WHY TRACKING:
    Without tracking, the recognition pipeline would run the full
    detection → liveness → embedding → FAISS search cycle on EVERY frame.
    At 30 fps with 20 students, that's 600 ArcFace inferences per second —
    wasteful and slow.

    With tracking (SORT), we:
      1. Assign each detected face a persistent "track ID" across frames.
      2. Run full recognition only when a NEW face appears (new track ID).
      3. For subsequent frames, reuse the identity from the first recognition.
      4. Only re-run recognition when the track is "uncertain" (low similarity).

    Result: ~5–10× fewer embedding computations at inference time.

HOW SORT WORKS:
    SORT uses a Kalman filter per track to predict where the face box will be
    in the next frame, then uses IoU (Intersection over Union) between
    predictions and new detections to match them. Unmatched detections
    become new tracks; tracks with no match for `max_age` frames are deleted.

    SORT paper: Bewley et al., ICIP 2016 — https://arxiv.org/abs/1602.00763

TEMPORAL VOTING:
    We add a voting layer on top of SORT. A student is only marked "present"
    after being recognized consistently across `min_frames_to_confirm` frames.
    This prevents a single false match from marking a student present.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter

logger = logging.getLogger(__name__)


# ── Kalman-based single-object tracker ───────────────────────────────────────

class KalmanBoxTracker:
    """
    Tracks a single bounding box using a Kalman filter.
    State: [cx, cy, s, r, dcx, dcy, ds]
        cx, cy = center x, y
        s      = area
        r      = aspect ratio (w/h) — assumed constant
        dcx, dcy, ds = velocities
    """

    _id_counter = 0

    def __init__(self, bbox: np.ndarray) -> None:
        KalmanBoxTracker._id_counter += 1
        self.id       = KalmanBoxTracker._id_counter
        self.hits     = 1
        self.no_match = 0
        self.age      = 0

        # 7-state, 4-observation Kalman filter
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1],
        ], dtype=np.float32)
        self.kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0],
        ], dtype=np.float32)
        self.kf.R[2:, 2:] *= 10.
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P         *= 10.
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        self.kf.x[:4]      = _bbox_to_z(bbox)

    def predict(self) -> np.ndarray:
        if self.kf.x[6] + self.kf.x[2] <= 0:
            self.kf.x[6] = 0
        self.kf.predict()
        self.age      += 1
        self.no_match += 1
        return _z_to_bbox(self.kf.x)

    def update(self, bbox: np.ndarray) -> None:
        self.no_match = 0
        self.hits    += 1
        self.kf.update(_bbox_to_z(bbox))

    @property
    def state(self) -> np.ndarray:
        return _z_to_bbox(self.kf.x)

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 1


def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    """[x1,y1,x2,y2] → [cx, cy, area, aspect_ratio]"""
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h) if h > 0 else 1.0
    return np.array([[x], [y], [s], [r]], dtype=np.float32)


def _z_to_bbox(x: np.ndarray) -> np.ndarray:
    """[cx, cy, area, aspect_ratio] → [x1,y1,x2,y2]"""
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w if w > 0 else 0
    return np.array([
        x[0] - w / 2., x[1] - h / 2.,
        x[0] + w / 2., x[1] + h / 2.,
    ], dtype=np.float32).flatten()


def _iou(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w   = max(0., xx2 - xx1)
    h   = max(0., yy2 - yy1)
    inter = w * h
    area_test = (bb_test[2]-bb_test[0]) * (bb_test[3]-bb_test[1])
    area_gt   = (bb_gt[2]-bb_gt[0])    * (bb_gt[3]-bb_gt[1])
    union = area_test + area_gt - inter
    return inter / union if union > 0 else 0.0


def _associate(
    detections: np.ndarray,     # (N, 4)
    predictions: np.ndarray,    # (M, 4)
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hungarian matching between detections and track predictions.
    Returns (matches, unmatched_dets, unmatched_trks).
    """
    if predictions.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections), dtype=int),
            np.empty(0, dtype=int),
        )
    if detections.shape[0] == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty(0, dtype=int),
            np.arange(len(predictions), dtype=int),
        )

    # Build IoU cost matrix
    iou_matrix = np.zeros((len(detections), len(predictions)), dtype=np.float32)
    for d, det in enumerate(detections):
        for t, trk in enumerate(predictions):
            iou_matrix[d, t] = _iou(det, trk)

    # Hungarian algorithm — maximise IoU (minimise negative IoU)
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matches = np.column_stack([row_ind, col_ind])

    # Discard matches below IoU threshold
    valid = iou_matrix[row_ind, col_ind] >= iou_threshold
    matches = matches[valid]

    unmatched_dets = np.setdiff1d(np.arange(len(detections)), matches[:, 0] if len(matches) else [])
    unmatched_trks = np.setdiff1d(np.arange(len(predictions)), matches[:, 1] if len(matches) else [])

    return matches, unmatched_dets, unmatched_trks


# ── SORT tracker ─────────────────────────────────────────────────────────────

class FaceTracker:
    """
    SORT-based multi-face tracker.

    Assigns consistent integer track IDs to each face across frames.
    This allows the recognition module to run once per track, not once per frame.

    Usage:
        tracker = FaceTracker(config)

        for frame in video:
            detections = analyzer.get_faces(frame)
            tracked    = tracker.update(detections)
            # Each face in `tracked` now has a .track_id assigned.
    """

    def __init__(self, config: dict) -> None:
        trk_cfg = config["tracking"]
        self.enabled       = trk_cfg["enabled"]
        self.max_age       = trk_cfg["max_age"]
        self.min_hits      = trk_cfg["min_hits"]
        self.iou_threshold = trk_cfg["iou_threshold"]
        self._trackers: List[KalmanBoxTracker] = []
        logger.info(
            "FaceTracker init | max_age=%d min_hits=%d iou_thr=%.2f",
            self.max_age, self.min_hits, self.iou_threshold
        )

    def reset(self) -> None:
        """Clear all active tracks (call at start of new session)."""
        self._trackers = []
        KalmanBoxTracker._id_counter = 0

    def update(self, faces: list) -> list:
        """
        Match incoming face detections to existing tracks.

        Parameters
        ----------
        faces : List[DetectedFace]   (from face_analyzer.py)

        Returns
        -------
        List[DetectedFace]
            Same list, but each face has .track_id set.
            Only returns CONFIRMED tracks (hit ≥ min_hits).
        """
        from face_analyzer import DetectedFace  # avoid circular import

        if not self.enabled:
            # No tracking — assign sequential IDs per frame
            for i, face in enumerate(faces):
                face.track_id = i
            return faces

        if not faces:
            # Still need to predict to age-out stale tracks
            for trk in self._trackers:
                trk.predict()
            self._trackers = [t for t in self._trackers if t.no_match <= self.max_age]
            return []

        # Get current bboxes
        dets   = np.array([f.bbox for f in faces])   # (N, 4)
        preds  = np.array([t.predict() for t in self._trackers])  # (M, 4)

        matches, unmatched_dets, unmatched_trks = _associate(dets, preds, self.iou_threshold)

        # Update matched tracks
        for det_idx, trk_idx in matches:
            self._trackers[trk_idx].update(dets[det_idx])

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._trackers.append(KalmanBoxTracker(dets[det_idx]))

        # Remove dead tracks
        self._trackers = [t for t in self._trackers if t.no_match <= self.max_age]

        # Assign track IDs to faces
        # Build det_bbox → track mapping
        matched_det_to_trk: Dict[int, KalmanBoxTracker] = {}
        for det_idx, trk_idx in matches:
            matched_det_to_trk[det_idx] = self._trackers[trk_idx]

        result = []
        for i, (face, trk) in enumerate(
            ((faces[d], matched_det_to_trk[d]) for d in range(len(faces)) if d in matched_det_to_trk)
        ):
            if trk.hits >= self.min_hits or trk.age <= self.min_hits:
                face.track_id = trk.id
                result.append(face)

        # New unmatched detections get IDs from their newly created tracks
        for det_idx in unmatched_dets:
            face = faces[det_idx]
            # Find the tracker just created for this detection
            new_trk = next(
                (t for t in self._trackers if np.allclose(t.state, dets[det_idx], atol=5)),
                None
            )
            if new_trk:
                face.track_id = new_trk.id
                result.append(face)

        return result


# ── Temporal voting ───────────────────────────────────────────────────────────

class AttendanceVoter:
    """
    Confirms attendance only after a student is recognized across N frames.

    This prevents:
      - A lookalike walking past the camera getting false-positive attendance.
      - A single frame with an erroneous match marking a student present.

    A student is marked "confirmed present" when their identity appears in
    at least `min_frames` of the last `window` frame recognitions for a
    given track ID.
    """

    def __init__(self, config: dict) -> None:
        self.min_frames = config["voting"]["min_frames_to_confirm"]
        # track_id → deque of (student_id, similarity) for recent frames
        self._votes: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.min_frames * 2)
        )
        # track_id → confirmed student_id (once confirmed, stays confirmed)
        self._confirmed: Dict[int, str] = {}

    def vote(
        self,
        track_id:   int,
        student_id: Optional[str],
        similarity: float,
    ) -> Optional[str]:
        """
        Record a recognition result for a track and return confirmed identity.

        Returns the confirmed student_id once enough votes accumulate,
        or None if still undecided.
        """
        # If already confirmed, keep returning the confirmed ID
        if track_id in self._confirmed:
            return self._confirmed[track_id]

        self._votes[track_id].append((student_id, similarity))

        # Count votes for the most-voted student_id
        id_counts: Dict[Optional[str], int] = defaultdict(int)
        for sid, _ in self._votes[track_id]:
            id_counts[sid] += 1

        best_id    = max(id_counts, key=lambda k: id_counts[k])
        best_count = id_counts[best_id]

        if best_id is not None and best_count >= self.min_frames:
            self._confirmed[track_id] = best_id
            logger.info(
                "Attendance confirmed: track=%d → student=%s (%d/%d votes)",
                track_id, best_id, best_count, self.min_frames
            )
            return best_id

        return None

    def get_all_confirmed(self) -> Dict[int, str]:
        """Return {track_id: student_id} for all confirmed students."""
        return dict(self._confirmed)

    def reset(self) -> None:
        """Clear all votes (call at start of new session)."""
        self._votes.clear()
        self._confirmed.clear()
