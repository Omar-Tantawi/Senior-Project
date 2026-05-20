"""Face detection and embedding extraction using ONNX models directly.

Uses:
- RetinaFace (det_10g.onnx) for face detection
- ArcFace (w600k_r50.onnx) for face embedding extraction

No insightface package required — only onnxruntime + numpy + opencv.
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Face:
    """Detected face with bounding box, score, landmarks, and embedding."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    det_score: float          # detection confidence
    landmarks: np.ndarray     # 5-point landmarks (5, 2): left_eye, right_eye, nose, left_mouth, right_mouth
    embedding: np.ndarray | None = None  # 512-dim ArcFace embedding


def _distance2bbox(points, distance):
    """Convert distances to bounding boxes."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points, distance):
    """Convert distances to keypoints."""
    num_points = distance.shape[1] // 2
    kps = np.zeros((distance.shape[0], num_points, 2), dtype=np.float32)
    for i in range(num_points):
        kps[:, i, 0] = points[:, 0] + distance[:, 2 * i]
        kps[:, i, 1] = points[:, 1] + distance[:, 2 * i + 1]
    return kps


def _nms(bboxes, scores, threshold=0.4):
    """Non-maximum suppression."""
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= threshold)[0]
        order = order[inds + 1]

    return keep


def _align_face(img, landmarks, image_size=112):
    """Align face using 5-point landmarks (ArcFace standard alignment).

    Uses similarity transform to align the face to a standard template.
    """
    # ArcFace standard reference points for 112x112
    dst = np.array([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ], dtype=np.float32)

    src = landmarks.astype(np.float32)

    # Estimate similarity transform using 5-point landmarks
    tform = _estimate_similarity_transform(src, dst)
    aligned = cv2.warpAffine(img, tform, (image_size, image_size), borderValue=0.0)
    return aligned


def _estimate_similarity_transform(src, dst):
    """Estimate 2D similarity transform (rotation, scale, translation)."""
    num = src.shape[0]
    dim = src.shape[1]

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = np.zeros((num * dim, 2 * dim))
    b = np.zeros(num * dim)

    for i in range(num):
        A[2 * i, 0:2] = src_demean[i]
        A[2 * i + 1, 2:4] = src_demean[i]
        A[2 * i, 2] = -src_demean[i, 1]
        A[2 * i, 3] = src_demean[i, 0]  # This is wrong, let me use a simpler method

    # Use cv2.estimateAffinePartial2D for simplicity and correctness
    tform, _ = cv2.estimateAffinePartial2D(src, dst)
    if tform is None:
        # Fallback: just use first 3 points with getAffineTransform
        tform = cv2.getAffineTransform(src[:3], dst[:3])
    return tform


class FaceDetector:
    """Face detection using RetinaFace ONNX model."""

    # RetinaFace uses 3 feature map strides
    _feat_stride_fpn = [8, 16, 32]
    _num_anchors = 2

    def __init__(self, det_model_path: str, rec_model_path: str = None,
                 det_size: tuple = (640, 640), det_thresh: float = 0.5,
                 max_faces: int = 60, nms_thresh: float = 0.4):
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.max_faces = max_faces
        self.nms_thresh = nms_thresh

        # Load detection model
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        print(f"[Detector] Loading detection model: {det_model_path}")
        self.det_session = ort.InferenceSession(det_model_path, providers=providers)
        self.det_input_name = self.det_session.get_inputs()[0].name

        # Load recognition model (optional)
        self.rec_session = None
        if rec_model_path and Path(rec_model_path).exists():
            print(f"[Detector] Loading recognition model: {rec_model_path}")
            self.rec_session = ort.InferenceSession(rec_model_path, providers=providers)
            self.rec_input_name = self.rec_session.get_inputs()[0].name

        active_providers = self.det_session.get_providers()
        if "CUDAExecutionProvider" in active_providers:
            print("[Detector] Running on GPU (CUDA)")
        else:
            print("[Detector] Running on CPU (CUDA not available)")

    def _preprocess_det(self, img: np.ndarray) -> tuple[np.ndarray, float, tuple]:
        """Preprocess image for detection model."""
        input_h, input_w = self.det_size
        img_h, img_w = img.shape[:2]

        # Scale to fit detection input size
        scale = min(input_w / img_w, input_h / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

        resized = cv2.resize(img, (new_w, new_h))

        # Pad to detection input size
        det_img = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        det_img[:new_h, :new_w, :] = resized

        # Normalize: (img - 127.5) / 128.0, then transpose to NCHW
        det_img = (det_img.astype(np.float32) - 127.5) / 128.0
        det_img = det_img.transpose(2, 0, 1)  # HWC -> CHW
        det_img = np.expand_dims(det_img, axis=0)  # Add batch dim

        return det_img, scale, (new_h, new_w)

    def _postprocess_det(self, outputs, scale: float) -> list[Face]:
        """Parse detection model outputs into Face objects."""
        scores_list = []
        bboxes_list = []
        kps_list = []

        num_strides = len(self._feat_stride_fpn)
        # Output layout: [scores_s8, scores_s16, scores_s32,
        #                  bboxes_s8, bboxes_s16, bboxes_s32,
        #                  kps_s8, kps_s16, kps_s32]
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]                    # (H*W*anchors, 1)
            bbox_deltas = outputs[idx + num_strides]     # (H*W*anchors, 4)
            kps_deltas = outputs[idx + num_strides * 2]  # (H*W*anchors, 10)

            height = self.det_size[0] // stride
            width = self.det_size[1] // stride

            # Generate anchor centers
            anchor_centers = np.stack(
                np.mgrid[:height, :width][::-1], axis=-1
            ).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape(-1, 2)

            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1)
                anchor_centers = anchor_centers.reshape(-1, 2)

            # Filter by score
            if scores.ndim == 2:
                scores = scores[:, 0]
            pos_inds = np.where(scores >= self.det_thresh)[0]

            if len(pos_inds) == 0:
                continue

            bboxes = _distance2bbox(anchor_centers[pos_inds], bbox_deltas[pos_inds])
            kps = _distance2kps(anchor_centers[pos_inds], kps_deltas[pos_inds])

            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes / scale)
            kps_list.append(kps / scale)

        if not scores_list:
            return []

        all_scores = np.concatenate(scores_list)
        all_bboxes = np.concatenate(bboxes_list)
        all_kps = np.concatenate(kps_list)

        # NMS
        keep = _nms(all_bboxes, all_scores, self.nms_thresh)
        if len(keep) > self.max_faces:
            keep = keep[:self.max_faces]

        faces = []
        for i in keep:
            face = Face(
                bbox=all_bboxes[i],
                det_score=float(all_scores[i]),
                landmarks=all_kps[i],
            )
            faces.append(face)

        return faces

    def _extract_embedding(self, img: np.ndarray, face: Face) -> np.ndarray | None:
        """Extract ArcFace embedding for a detected face."""
        if self.rec_session is None:
            return None

        # Align face using landmarks
        aligned = _align_face(img, face.landmarks, image_size=112)

        # Preprocess for ArcFace: normalize and transpose
        aligned = aligned.astype(np.float32)
        aligned = (aligned - 127.5) / 127.5  # Normalize to [-1, 1]
        aligned = aligned.transpose(2, 0, 1)  # HWC -> CHW
        aligned = np.expand_dims(aligned, axis=0)  # Add batch dim

        # Run recognition model
        embedding = self.rec_session.run(None, {self.rec_input_name: aligned})[0]
        embedding = embedding.flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def detect_faces(self, frame: np.ndarray) -> list[Face]:
        """Detect faces and extract embeddings from a BGR frame.

        Returns list of Face objects with bbox, score, landmarks, and embedding.
        """
        # Detection
        det_input, scale, _ = self._preprocess_det(frame)
        det_outputs = self.det_session.run(None, {self.det_input_name: det_input})
        faces = self._postprocess_det(det_outputs, scale)

        # Extract embeddings
        for face in faces:
            face.embedding = self._extract_embedding(frame, face)

        return faces
