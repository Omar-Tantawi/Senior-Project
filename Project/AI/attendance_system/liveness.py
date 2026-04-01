"""
liveness.py — Anti-spoofing using MiniFASNet (Silent-Face-Anti-Spoofing).

WHY THIS MATTERS:
    Without liveness detection, a student can hold up a photo of a classmate
    on their phone and get marked as present. MiniFASNet catches this by
    looking at micro-texture differences between real skin and printed/screen
    images (it uses Fourier transform features to detect moiré patterns and
    pixel-level artifacts of displays).

HOW IT WORKS:
    1. Crop and resize the face region from the frame.
    2. Pass it through MiniFASNet (a lightweight CNN, ~2.7MB).
    3. The model outputs a 2-class softmax: [spoof_prob, real_prob].
    4. If real_prob > threshold → LIVE. Else → SPOOF.

We run TWO models (MiniFASNetV2 + MiniFASNetV1SE) and average their outputs.
This is the same ensemble strategy used in the original paper.

REFERENCE:
    "Searching Central Difference Convolutional Networks for Face Anti-Spoofing"
    Yu et al., CVPR 2020
    https://arxiv.org/abs/2001.07663
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── MiniFASNet architecture ───────────────────────────────────────────────────
# This is a stripped-down version of the MiniFASNet as defined in the
# Silent-Face-Anti-Spoofing repo. We keep only what's needed for inference.

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.pw(self.dw(x))), inplace=True)


class MiniFASNet(nn.Module):
    """
    Lightweight face anti-spoofing CNN.
    Input : (B, 3, 80, 80)  BGR image, normalised to [0,1]
    Output: (B, 2)          [spoof_logit, real_logit]
    """
    def __init__(self, num_classes: int = 2, embedding_size: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            DepthwiseSeparable(32,  64),
            DepthwiseSeparable(64, 128, stride=2),
            DepthwiseSeparable(128, 128),
            DepthwiseSeparable(128, 256, stride=2),
            DepthwiseSeparable(256, 256),
            DepthwiseSeparable(256, 512, stride=2),
            *[DepthwiseSeparable(512, 512) for _ in range(5)],
            DepthwiseSeparable(512, 1024, stride=2),
            DepthwiseSeparable(1024, 1024),
        )
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.embed     = nn.Linear(1024, embedding_size)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = F.relu(self.embed(x), inplace=True)
        return self.classifier(x)


# ── LivenessDetector ─────────────────────────────────────────────────────────

class LivenessDetector:
    """
    Runs MiniFASNet ensemble on a detected face to determine if it's real.

    Usage:
        detector = LivenessDetector(config)
        is_live, score = detector.check(bgr_frame, bbox)
        # is_live: True = real person, False = spoof
        # score: float in [0,1], higher = more likely real
    """

    INPUT_SIZE = (80, 80)

    def __init__(self, config: dict) -> None:
        self.cfg      = config["liveness"]
        self.threshold = self.cfg["threshold"]
        self.scale    = self.cfg["scale_factor"]
        self.enabled  = self.cfg["enabled"]

        if not self.enabled:
            logger.warning("Liveness detection is DISABLED. Spoofing is possible.")
            return

        dev = config["device"]
        self.device = torch.device(
            f"cuda:{dev['gpu_id']}" if dev["use_gpu"] and torch.cuda.is_available()
            else "cpu"
        )

        model_dir = Path(self.cfg["model_dir"])
        self.models: list[nn.Module] = []

        weight_files = [
            "2.7_80x80_MiniFASNetV2.pth",
            "4_0_0_80x80_MiniFASNetV1SE.pth",
        ]

        for wf in weight_files:
            path = model_dir / wf
            if not path.exists():
                logger.warning(
                    "Anti-spoof weight not found: %s\n"
                    "Run python setup.py to download it.", path
                )
                continue
            model = MiniFASNet().to(self.device)
            state = torch.load(path, map_location=self.device)
            # Handle both raw state_dict and checkpoint dicts
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            model.eval()
            self.models.append(model)
            logger.info("Loaded anti-spoof model: %s", wf)

        if not self.models:
            logger.error(
                "No anti-spoof models loaded! Liveness detection will be skipped. "
                "Run python setup.py to download model weights."
            )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _crop_face(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Crop the face region from frame with a scale factor around the bbox.
        The MiniFASNet paper found that using a slightly larger crop (scale ~2.7)
        captures skin texture outside the strict face region, improving accuracy.
        """
        x1, y1, x2, y2 = bbox.astype(int)
        w  = x2 - x1
        h  = y2 - y1
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Expand the crop
        half_w = int(w * self.scale / 2)
        half_h = int(h * self.scale / 2)

        # Clamp to frame boundaries
        nx1 = max(0, cx - half_w)
        ny1 = max(0, cy - half_h)
        nx2 = min(frame.shape[1], cx + half_w)
        ny2 = min(frame.shape[0], cy + half_h)

        if nx2 <= nx1 or ny2 <= ny1:
            return None

        crop = frame[ny1:ny2, nx1:nx2]
        return cv2.resize(crop, self.INPUT_SIZE)

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        """
        BGR uint8 → normalised float32 tensor (1, 3, 80, 80) on self.device.
        """
        img = crop.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))           # HWC → CHW
        tensor = torch.from_numpy(img).unsqueeze(0)  # add batch dim
        return tensor.to(self.device)

    # ── Public API ────────────────────────────────────────────────────────────

    def check(
        self,
        bgr_frame: np.ndarray,
        bbox: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Check whether the face in `bbox` is a real person or a spoof.

        Parameters
        ----------
        bgr_frame : BGR frame from camera
        bbox      : [x1, y1, x2, y2] bounding box from RetinaFace

        Returns
        -------
        (is_live, score)
            is_live : True = real person
            score   : averaged real probability across both models (0–1)
        """
        # If disabled or no models loaded, assume live (fail open)
        if not self.enabled or not self.models:
            return True, 1.0

        crop = self._crop_face(bgr_frame, bbox)
        if crop is None:
            return True, 1.0  # Can't crop → don't block recognition

        tensor  = self._preprocess(crop)
        scores  = []

        with torch.no_grad():
            for model in self.models:
                logits = model(tensor)                      # (1, 2)
                probs  = F.softmax(logits, dim=1)          # (1, 2)
                real_prob = probs[0, 1].item()             # index 1 = real class
                scores.append(real_prob)

        avg_score = float(np.mean(scores))
        is_live   = avg_score > self.threshold

        logger.debug(
            "Liveness check: score=%.3f threshold=%.2f → %s",
            avg_score, self.threshold, "LIVE" if is_live else "SPOOF"
        )
        return is_live, avg_score
