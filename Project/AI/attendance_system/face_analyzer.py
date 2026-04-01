"""
face_analyzer.py — Wraps InsightFace (RetinaFace + ArcFace) in a clean API.

This is the heart of the recognition pipeline. It handles:
  - Face detection  : finds all faces in a frame and returns bounding boxes
                      + 5 facial landmarks (eyes, nose, mouth corners)
  - Alignment       : affine-transforms each face to a canonical 112×112 crop
                      (this is done INSIDE insightface — no extra code needed)
  - Embedding       : runs ArcFace on each aligned face → 512-d L2-norm vector

The key insight: insightface's FaceAnalysis does detection + alignment +
embedding in ONE call (app.get(img)). We wrap it to add logging, filtering,
and to expose just what the rest of the pipeline needs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """
    Everything we know about one face found in a frame.

    bbox        : [x1, y1, x2, y2] in pixel coordinates
    landmarks   : 5 × 2 array — [left_eye, right_eye, nose, left_mouth, right_mouth]
    embedding   : 512-d L2-normalized ArcFace vector (None before recognition)
    det_score   : RetinaFace confidence (0–1)
    track_id    : assigned by SORT tracker (-1 = not yet tracked)
    identity    : matched student ID (None = unknown)
    similarity  : cosine similarity to best match (None = not yet matched)
    is_live     : liveness result (None = not yet checked)
    """
    bbox:       np.ndarray                  # shape (4,)  float32
    landmarks:  np.ndarray                  # shape (5,2) float32
    embedding:  Optional[np.ndarray] = None # shape (512,) float32
    det_score:  float = 0.0
    track_id:   int = -1
    identity:   Optional[str] = None
    similarity: Optional[float] = None
    is_live:    Optional[bool] = None

    # Convenience ─────────────────────────────────────────────────────────────
    @property
    def bbox_int(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox.astype(int)
        return x1, y1, x2, y2

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox_int
        return (x1 + x2) // 2, (y1 + y2) // 2

    @property
    def height(self) -> int:
        _, y1, _, y2 = self.bbox_int
        return y2 - y1


class FaceAnalyzer:
    """
    Wraps InsightFace FaceAnalysis to detect + embed all faces in a frame.

    Usage:
        analyzer = FaceAnalyzer(config)
        faces = analyzer.get_faces(bgr_frame)
        # faces is a List[DetectedFace], one per person visible in the frame
    """

    def __init__(self, config: dict) -> None:
        self.cfg = config
        det_cfg  = config["detection"]
        dev_cfg  = config["device"]

        # Choose inference provider based on GPU availability
        if dev_cfg["use_gpu"]:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id    = dev_cfg["gpu_id"]
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id    = -1  # insightface uses -1 to mean CPU

        # InsightFace loads BOTH the detector and the recognition model
        # when you call prepare(). The 'buffalo_l' model pack contains:
        #   - det_10g.onnx       : RetinaFace-R50 detector
        #   - w600k_r50.onnx     : ArcFace-R100 embedder (trained on WebFace600K)
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(
                name=det_cfg["model_pack"],
                providers=providers
            )
            det_size = tuple(det_cfg["det_size"])          # (640, 640)
            self._app.prepare(ctx_id=ctx_id, det_size=det_size)
            logger.info(
                "FaceAnalyzer ready | pack=%s | det_size=%s | providers=%s",
                det_cfg["model_pack"], det_size, providers
            )
        except Exception as e:
            logger.error("Failed to initialize InsightFace: %s", e)
            raise

        self._det_thresh  = det_cfg["det_thresh"]
        self._min_face_sz = det_cfg["min_face_size"]

    # ── Public API ────────────────────────────────────────────────────────────

    def get_faces(self, bgr_frame: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in `bgr_frame` and return their embeddings.

        Parameters
        ----------
        bgr_frame : np.ndarray
            BGR image as returned by cv2.VideoCapture.read()

        Returns
        -------
        List[DetectedFace]
            One DetectedFace per person found. Empty list if no faces detected.
            Each face already has .embedding set (512-d ArcFace vector).
        """
        if bgr_frame is None or bgr_frame.size == 0:
            return []

        # insightface expects BGR (same as OpenCV) — no conversion needed.
        # app.get() runs the full pipeline: detect → align → embed
        raw_faces = self._app.get(bgr_frame)

        results: List[DetectedFace] = []
        for f in raw_faces:
            # Filter by detection confidence
            if f.det_score < self._det_thresh:
                logger.debug("Face skipped: det_score=%.2f < %.2f", f.det_score, self._det_thresh)
                continue

            # Filter by face size (skip tiny/distant faces)
            bbox  = f.bbox  # [x1, y1, x2, y2]
            h     = bbox[3] - bbox[1]
            if h < self._min_face_sz:
                logger.debug("Face skipped: height=%d < min=%d", int(h), self._min_face_sz)
                continue

            # f.normed_embedding is the L2-normalized 512-d ArcFace vector
            # This is what we compare with cosine similarity
            embedding = f.normed_embedding.astype(np.float32)

            face = DetectedFace(
                bbox      = bbox.astype(np.float32),
                landmarks = f.kps.astype(np.float32),   # 5 × 2
                embedding = embedding,
                det_score = float(f.det_score),
            )
            results.append(face)

        logger.debug("Detected %d valid face(s) in frame", len(results))
        return results

    def get_embedding_only(self, aligned_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Run ArcFace on an already-aligned 112×112 BGR crop.
        Used during enrollment when you have pre-cropped images.

        Returns 512-d L2-normalized embedding, or None on failure.
        """
        try:
            faces = self._app.get(aligned_crop)
            if not faces:
                return None
            # Return embedding of the highest-confidence face
            best = max(faces, key=lambda f: f.det_score)
            return best.normed_embedding.astype(np.float32)
        except Exception as e:
            logger.warning("get_embedding_only failed: %s", e)
            return None


# ── Utility: cosine similarity ────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalized vectors.

    Because ArcFace outputs L2-normalized embeddings, this is simply the
    dot product: sim = a · b   (range: -1 to 1, higher = more similar).

    Typical thresholds:
        sim > 0.45 → same person   (tune on your dataset)
        sim < 0.30 → different person
        0.30–0.45 → uncertain
    """
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance = 1 - cosine_similarity. Lower = more similar."""
    return 1.0 - cosine_similarity(a, b)
