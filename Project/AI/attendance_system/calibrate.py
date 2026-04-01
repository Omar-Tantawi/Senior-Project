"""
calibrate.py — Tune the similarity threshold (τ) for your specific classroom.

WHY YOU NEED THIS:
    The default threshold (0.45) works well for general populations, but your
    classroom may have:
      - Students who look similar (siblings, twins)
      - Specific lighting conditions (fluorescent, dim, backlit)
      - Students with similar uniforms covering hair/forehead
    Running this script helps you find the OPTIMAL threshold for your setup.

HOW IT WORKS:
    1. You provide "genuine pairs" (same student, different photos) and
       "impostor pairs" (different students).
    2. We compute cosine similarity for all pairs.
    3. We plot the False Accept Rate (FAR) vs False Reject Rate (FRR) curve.
    4. The optimal τ minimizes both errors (Equal Error Rate point).

HOW TO USE:
    # You need some labeled test photos — different from enrollment photos!
    # Organize them like your enrollment data:
    #   data/test_photos/
    #       S001_Alice_Johnson/
    #           test1.jpg
    #           test2.jpg

    python calibrate.py --test_dir data/test_photos/

    The script prints the recommended threshold and plots the EER curve.
"""

from __future__ import annotations

import argparse
import logging
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml
from tqdm import tqdm

logger = logging.getLogger("calibrate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

from face_analyzer import FaceAnalyzer
from enroll import parse_student_folder, get_image_files


def extract_all_embeddings(
    test_dir: Path,
    analyzer: FaceAnalyzer,
) -> dict:
    """
    Returns {student_id: [embedding, embedding, ...]}
    """
    result = {}
    student_dirs = [d for d in sorted(test_dir.iterdir()) if d.is_dir()]

    for student_dir in tqdm(student_dirs, desc="Extracting test embeddings"):
        student_id, name = parse_student_folder(student_dir)
        images = get_image_files(student_dir)
        embeddings = []

        for img_path in images:
            img   = cv2.imread(str(img_path))
            if img is None:
                continue
            faces = analyzer.get_faces(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: f.height)
            embeddings.append(face.embedding)

        if embeddings:
            result[student_id] = {"name": name, "embeddings": embeddings}

    return result


def compute_pairs(
    data: dict,
) -> Tuple[List[float], List[float]]:
    """
    Build genuine and impostor similarity distributions.

    Genuine : pairs of embeddings from the SAME student
    Impostor: pairs of embeddings from DIFFERENT students
    """
    student_ids = list(data.keys())
    genuine_scores  = []
    impostor_scores = []

    # Genuine pairs (same student)
    for sid in student_ids:
        embs = data[sid]["embeddings"]
        for e1, e2 in combinations(embs, 2):
            score = float(np.dot(e1, e2))  # cosine sim for unit vectors
            genuine_scores.append(score)

    # Impostor pairs (different students)
    for s1, s2 in combinations(student_ids, 2):
        embs1 = data[s1]["embeddings"]
        embs2 = data[s2]["embeddings"]
        for e1 in embs1[:3]:        # limit to 3 per student to keep it fast
            for e2 in embs2[:3]:
                score = float(np.dot(e1, e2))
                impostor_scores.append(score)

    return genuine_scores, impostor_scores


def find_eer_threshold(
    genuine_scores:  List[float],
    impostor_scores: List[float],
) -> Tuple[float, float]:
    """
    Find the threshold at the Equal Error Rate (EER) point.

    At EER: FAR (False Accept Rate) ≈ FRR (False Reject Rate).
    This is the single best operating point when you have no preference
    between security and convenience.

    Returns (threshold, eer_value)
    """
    all_scores = sorted(set(genuine_scores + impostor_scores))
    best_threshold = 0.5
    best_eer       = 1.0

    for τ in np.linspace(min(all_scores), max(all_scores), 500):
        # FAR: fraction of impostor pairs ABOVE threshold (false accepts)
        far = sum(1 for s in impostor_scores if s >= τ) / len(impostor_scores)
        # FRR: fraction of genuine pairs BELOW threshold (false rejects)
        frr = sum(1 for s in genuine_scores  if s <  τ) / len(genuine_scores)

        eer_approx = abs(far - frr)
        if eer_approx < best_eer:
            best_eer       = eer_approx
            best_threshold = τ

    eer_value = (
        sum(1 for s in impostor_scores if s >= best_threshold) / len(impostor_scores)
        + sum(1 for s in genuine_scores  if s <  best_threshold) / len(genuine_scores)
    ) / 2.0

    return best_threshold, eer_value


def main(args: argparse.Namespace) -> None:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    analyzer = FaceAnalyzer(config)
    test_dir = Path(args.test_dir)

    if not test_dir.exists():
        logger.error("Test directory not found: %s", test_dir)
        return

    # Extract embeddings from test photos
    data = extract_all_embeddings(test_dir, analyzer)

    if len(data) < 2:
        logger.error("Need at least 2 students with test photos for calibration.")
        return

    total_embeddings = sum(len(d["embeddings"]) for d in data.values())
    logger.info("Extracted embeddings: %d students, %d total", len(data), total_embeddings)

    # Compute similarity distributions
    genuine_scores, impostor_scores = compute_pairs(data)
    logger.info(
        "Pairs: %d genuine, %d impostor",
        len(genuine_scores), len(impostor_scores)
    )

    # Find optimal threshold
    threshold, eer = find_eer_threshold(genuine_scores, impostor_scores)

    # Print statistics
    genuine_arr  = np.array(genuine_scores)
    impostor_arr = np.array(impostor_scores)

    print("\n" + "="*55)
    print("  Calibration Results")
    print("="*55)
    print(f"\n  Genuine  similarity: mean={genuine_arr.mean():.3f}  std={genuine_arr.std():.3f}")
    print(f"  Impostor similarity: mean={impostor_arr.mean():.3f}  std={impostor_arr.std():.3f}")
    print(f"\n  ► Recommended threshold (EER): {threshold:.3f}")
    print(f"    Equal Error Rate           : {eer*100:.1f}%")
    print()
    print("  Interpretation:")
    print(f"    At τ={threshold:.3f}: ~{eer*100:.1f}% of impostors accepted,")
    print(f"                        ~{eer*100:.1f}% of genuine faces rejected.")
    print()
    print("  If you want stricter security  → use a HIGHER threshold (more 'absent')")
    print("  If you want more convenience   → use a LOWER threshold (more 'present')")
    print()
    print(f"  ► Update config.yaml:")
    print(f"    recognition:")
    print(f"      similarity_threshold: {threshold:.3f}")
    print("="*55)

    # Try to plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot 1: score distributions
        axes[0].hist(genuine_scores,  bins=50, alpha=0.7, color="green",  label="Genuine (same person)")
        axes[0].hist(impostor_scores, bins=50, alpha=0.7, color="red",    label="Impostor (diff. person)")
        axes[0].axvline(threshold, color="blue", linestyle="--", label=f"EER threshold = {threshold:.3f}")
        axes[0].set_xlabel("Cosine Similarity")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Score Distribution")
        axes[0].legend()

        # Plot 2: FAR/FRR curve
        taus = np.linspace(min(genuine_scores + impostor_scores), 1.0, 300)
        fars = [sum(1 for s in impostor_scores if s >= t) / len(impostor_scores) for t in taus]
        frrs = [sum(1 for s in genuine_scores  if s <  t) / len(genuine_scores)  for t in taus]

        axes[1].plot(taus, [f*100 for f in fars], label="FAR (False Accept Rate)", color="red")
        axes[1].plot(taus, [f*100 for f in frrs], label="FRR (False Reject Rate)", color="green")
        axes[1].axvline(threshold, color="blue", linestyle="--", label=f"EER τ = {threshold:.3f}")
        axes[1].set_xlabel("Threshold τ")
        axes[1].set_ylabel("Error Rate (%)")
        axes[1].set_title("FAR / FRR Curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = "output/calibration_plot.png"
        Path("output").mkdir(exist_ok=True)
        plt.savefig(plot_path, dpi=150)
        print(f"\n  Plot saved → {plot_path}")
        plt.show()

    except ImportError:
        print("\n  (Install matplotlib to see the score distribution plot)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate similarity threshold using test photos."
    )
    parser.add_argument("--config",   default="config.yaml",       help="Path to config.yaml")
    parser.add_argument("--test_dir", default="data/test_photos/", help="Directory of test photos")
    args = parser.parse_args()
    main(args)
