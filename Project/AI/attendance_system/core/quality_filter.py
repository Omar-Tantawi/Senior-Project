"""Face Quality Filter for CCTV.

Assesses face image quality and recommends adaptive recognition thresholds.
Instead of dropping low-quality faces, we use stricter thresholds for them.

Quality factors evaluated:
- Sharpness (Laplacian variance) — blurry detection
- Brightness (mean intensity) — too dark/bright detection
- Contrast (standard deviation) — flat/washed-out detection
- Face size (resolution) — too small detection

Returns a quality score 0.0-1.0 and an adaptive threshold recommendation.
"""

import cv2
import numpy as np


# Quality thresholds (tuned for CCTV)
SHARPNESS_GOOD = 100.0      # Laplacian variance > this = sharp
SHARPNESS_POOR = 30.0       # < this = very blurry

BRIGHTNESS_MIN = 40         # Mean intensity (0-255)
BRIGHTNESS_MAX = 215        # Too bright = washed out
BRIGHTNESS_OPT = 127        # Optimal mid-range

CONTRAST_GOOD = 50.0        # Standard deviation
CONTRAST_POOR = 20.0

FACE_SIZE_GOOD = 80         # Minimum side length (px) for good quality
FACE_SIZE_MIN = 30          # Below this = very small face


def assess_sharpness(image: np.ndarray) -> float:
    """Measure sharpness using Laplacian variance.
    Higher = sharper. Blurry images have low variance.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def assess_brightness(image: np.ndarray) -> float:
    """Measure mean brightness (0-255)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(np.mean(gray))


def assess_contrast(image: np.ndarray) -> float:
    """Measure contrast (standard deviation of intensities)."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return float(np.std(gray))


def assess_face_size(face_image: np.ndarray) -> int:
    """Return the smaller dimension of the face crop."""
    return min(face_image.shape[:2])


def assess_quality(face_image: np.ndarray) -> dict:
    """Comprehensive quality assessment of a face crop.

    Args:
        face_image: BGR face crop (any size)

    Returns:
        dict with quality metrics and an overall score 0.0-1.0
    """
    if face_image is None or face_image.size == 0:
        return {
            "score": 0.0,
            "sharpness": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "size": 0,
            "issues": ["empty_image"],
        }

    sharpness = assess_sharpness(face_image)
    brightness = assess_brightness(face_image)
    contrast = assess_contrast(face_image)
    size = assess_face_size(face_image)

    # Score each factor 0.0-1.0
    issues = []

    # Sharpness score
    if sharpness >= SHARPNESS_GOOD:
        sharpness_score = 1.0
    elif sharpness <= SHARPNESS_POOR:
        sharpness_score = 0.0
        issues.append("blurry")
    else:
        sharpness_score = (sharpness - SHARPNESS_POOR) / (SHARPNESS_GOOD - SHARPNESS_POOR)

    # Brightness score (best at BRIGHTNESS_OPT, drops at extremes)
    if BRIGHTNESS_MIN <= brightness <= BRIGHTNESS_MAX:
        # Distance from optimal
        brightness_score = 1.0 - abs(brightness - BRIGHTNESS_OPT) / BRIGHTNESS_OPT
    else:
        brightness_score = 0.0
        if brightness < BRIGHTNESS_MIN:
            issues.append("too_dark")
        else:
            issues.append("too_bright")

    # Contrast score
    if contrast >= CONTRAST_GOOD:
        contrast_score = 1.0
    elif contrast <= CONTRAST_POOR:
        contrast_score = 0.0
        issues.append("low_contrast")
    else:
        contrast_score = (contrast - CONTRAST_POOR) / (CONTRAST_GOOD - CONTRAST_POOR)

    # Size score
    if size >= FACE_SIZE_GOOD:
        size_score = 1.0
    elif size <= FACE_SIZE_MIN:
        size_score = 0.0
        issues.append("face_too_small")
    else:
        size_score = (size - FACE_SIZE_MIN) / (FACE_SIZE_GOOD - FACE_SIZE_MIN)

    # Weighted overall score (sharpness matters most for CCTV)
    overall = (
        sharpness_score * 0.40
        + size_score * 0.30
        + contrast_score * 0.20
        + brightness_score * 0.10
    )

    return {
        "score": float(overall),
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "size": size,
        "sharpness_score": sharpness_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "size_score": size_score,
        "issues": issues,
    }


def adaptive_threshold(quality_score: float, base_threshold: float = 0.4) -> float:
    """Adjust the similarity threshold based on face quality.

    Logic: Higher quality face → trust normal threshold.
           Lower quality face → require higher confidence (stricter).

    Args:
        quality_score: 0.0-1.0 from assess_quality()
        base_threshold: Default similarity threshold from config

    Returns:
        Adjusted threshold to use for this specific face.
    """
    if quality_score >= 0.8:
        # Clear face: use normal threshold
        return base_threshold
    elif quality_score >= 0.5:
        # Medium quality: 25% stricter
        return base_threshold + (1.0 - base_threshold) * 0.25
    elif quality_score >= 0.3:
        # Low quality: 50% stricter
        return base_threshold + (1.0 - base_threshold) * 0.50
    else:
        # Very low quality: 75% stricter (almost won't match anything)
        return base_threshold + (1.0 - base_threshold) * 0.75


def should_skip(quality_score: float, min_quality: float = 0.15) -> bool:
    """Return True if face is too low quality to be useful at all."""
    return quality_score < min_quality


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        if img is None:
            print(f"Cannot load: {sys.argv[1]}")
            sys.exit(1)

        quality = assess_quality(img)
        print(f"Quality Assessment for {sys.argv[1]}:")
        print(f"  Overall Score: {quality['score']:.3f}")
        print(f"  Sharpness: {quality['sharpness']:.1f} ({quality['sharpness_score']:.2f})")
        print(f"  Brightness: {quality['brightness']:.1f} ({quality['brightness_score']:.2f})")
        print(f"  Contrast: {quality['contrast']:.1f} ({quality['contrast_score']:.2f})")
        print(f"  Size: {quality['size']} ({quality['size_score']:.2f})")
        print(f"  Issues: {quality['issues']}")
        print(f"  Adaptive threshold (base 0.4): {adaptive_threshold(quality['score'], 0.4):.3f}")
        print(f"  Should skip: {should_skip(quality['score'])}")
    else:
        print("Usage: python quality_filter.py <face_image.jpg>")
