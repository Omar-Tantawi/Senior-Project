"""Evaluation Script: Compare Original vs Fine-Tuned ArcFace.

Measures face recognition accuracy before and after fine-tuning.

Tests:
    1. Verification accuracy (same-person vs different-person pairs)
    2. Identification accuracy (1:N matching against gallery)
    3. Embedding quality (intra-class vs inter-class distances)

Usage:
    python evaluate.py --original models/w600k_r50.onnx --finetuned fine_tuning/checkpoints/arcface_finetuned.onnx --data path/to/aligned_dataset
    python evaluate.py --original models/w600k_r50.onnx --finetuned fine_tuning/checkpoints/arcface_finetuned.onnx --data path/to/aligned_dataset --pairs 5000
"""

import argparse
import json
import sys
import random
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] onnxruntime is required. Install with: pip install onnxruntime-gpu")
    sys.exit(1)


# ============================================================
#  Embedding Extraction
# ============================================================

class ONNXEmbedder:
    """Extract face embeddings using an ONNX ArcFace model."""

    def __init__(self, model_path: str, name: str = "model"):
        self.name = name
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        active = self.session.get_providers()
        device = "GPU" if "CUDAExecutionProvider" in active else "CPU"
        print(f"[{name}] Loaded {model_path} ({device})")

    def get_embedding(self, aligned_face: np.ndarray) -> np.ndarray:
        """Get 512-dim embedding from a 112x112 aligned BGR face image.

        Args:
            aligned_face: (H, W, 3) BGR uint8 image (will be resized to 112x112)

        Returns:
            (512,) L2-normalized embedding
        """
        # Auto-resize to 112x112 if needed (handles both aligned images and raw images)
        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112), interpolation=cv2.INTER_AREA)

        img = aligned_face.astype(np.float32)
        img = (img - 127.5) / 127.5  # Normalize to [-1, 1]
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # Add batch dim

        embedding = self.session.run(None, {self.input_name: img})[0]
        embedding = embedding.flatten()

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding


# ============================================================
#  Dataset Loading
# ============================================================

def load_evaluation_data(data_dir: str, max_per_id: int = 10):
    """Load aligned face images grouped by identity.

    Supports both:
    - Numeric format: 000000/, 000001/ (from prepare_dataset.py)
    - String format: pins_Adriana Lima/, pins_Alex Lawther/ (from raw PINS)

    Returns:
        dict: {label: [image_path, ...]}
    """
    data_path = Path(data_dir)
    identity_data = {}

    # Find all identity directories (numeric or string names)
    identity_dirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir()
    ])

    for label, identity_dir in enumerate(identity_dirs):
        images = sorted(identity_dir.glob("*.jpg"))
        if len(images) >= 2:  # Need at least 2 for verification pairs
            identity_data[label] = [str(p) for p in images[:max_per_id]]

    print(f"[Eval] Loaded {len(identity_data)} identities "
          f"({sum(len(v) for v in identity_data.values())} images)")
    return identity_data


def generate_pairs(identity_data: dict, num_pairs: int = 3000):
    """Generate positive (same person) and negative (different person) pairs.

    Returns:
        positive_pairs: list of (img_path_1, img_path_2)
        negative_pairs: list of (img_path_1, img_path_2)
    """
    labels = list(identity_data.keys())
    random.seed(42)

    positive_pairs = []
    negative_pairs = []

    half = num_pairs // 2

    # Positive pairs (same identity)
    for _ in range(half):
        label = random.choice(labels)
        images = identity_data[label]
        if len(images) >= 2:
            img1, img2 = random.sample(images, 2)
            positive_pairs.append((img1, img2))

    # Negative pairs (different identities)
    for _ in range(half):
        label1, label2 = random.sample(labels, 2)
        img1 = random.choice(identity_data[label1])
        img2 = random.choice(identity_data[label2])
        negative_pairs.append((img1, img2))

    print(f"[Eval] Generated {len(positive_pairs)} positive + "
          f"{len(negative_pairs)} negative pairs")
    return positive_pairs, negative_pairs


# ============================================================
#  Evaluation Metrics
# ============================================================

def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(emb1, emb2))


def evaluate_verification(embedder: ONNXEmbedder,
                          positive_pairs, negative_pairs,
                          thresholds=None):
    """Evaluate face verification (1:1 matching).

    For each pair, compute similarity and check if it exceeds threshold.

    Returns:
        dict with accuracy, best_threshold, TPR, FPR at various thresholds
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.05)

    # Compute similarities for all pairs
    print(f"  [{embedder.name}] Computing positive pair similarities...")
    pos_sims = []
    for img1_path, img2_path in positive_pairs:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            continue
        emb1 = embedder.get_embedding(img1)
        emb2 = embedder.get_embedding(img2)
        pos_sims.append(cosine_similarity(emb1, emb2))

    print(f"  [{embedder.name}] Computing negative pair similarities...")
    neg_sims = []
    for img1_path, img2_path in negative_pairs:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            continue
        emb1 = embedder.get_embedding(img1)
        emb2 = embedder.get_embedding(img2)
        neg_sims.append(cosine_similarity(emb1, emb2))

    pos_sims = np.array(pos_sims)
    neg_sims = np.array(neg_sims)

    # Find best threshold
    best_acc = 0
    best_thresh = 0
    results_per_thresh = []

    for thresh in thresholds:
        tp = np.sum(pos_sims >= thresh)
        fn = np.sum(pos_sims < thresh)
        tn = np.sum(neg_sims < thresh)
        fp = np.sum(neg_sims >= thresh)

        acc = (tp + tn) / (tp + fn + tn + fp) if (tp + fn + tn + fp) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results_per_thresh.append({
            "threshold": float(thresh),
            "accuracy": float(acc),
            "tpr": float(tpr),
            "fpr": float(fpr),
        })

        if acc > best_acc:
            best_acc = acc
            best_thresh = float(thresh)

    return {
        "best_accuracy": float(best_acc),
        "best_threshold": best_thresh,
        "mean_positive_similarity": float(np.mean(pos_sims)),
        "mean_negative_similarity": float(np.mean(neg_sims)),
        "std_positive_similarity": float(np.std(pos_sims)),
        "std_negative_similarity": float(np.std(neg_sims)),
        "num_positive_pairs": len(pos_sims),
        "num_negative_pairs": len(neg_sims),
        "thresholds": results_per_thresh,
    }


def evaluate_identification(embedder: ONNXEmbedder, identity_data: dict,
                            gallery_size: int = 1):
    """Evaluate face identification (1:N matching).

    Uses first image(s) of each identity as gallery, rest as probes.

    Returns:
        dict with rank-1 and rank-5 accuracy
    """
    # Build gallery: first `gallery_size` images per identity
    gallery_embeddings = {}  # label -> list of embeddings
    probe_items = []  # (label, image_path)

    for label, images in identity_data.items():
        gallery_imgs = images[:gallery_size]
        probe_imgs = images[gallery_size:]

        if not probe_imgs:
            continue

        # Gallery embeddings
        gallery_embs = []
        for img_path in gallery_imgs:
            img = cv2.imread(img_path)
            if img is not None:
                gallery_embs.append(embedder.get_embedding(img))
        if gallery_embs:
            gallery_embeddings[label] = gallery_embs

        for img_path in probe_imgs:
            probe_items.append((label, img_path))

    if not probe_items or not gallery_embeddings:
        return {"rank1_accuracy": 0, "rank5_accuracy": 0, "num_probes": 0}

    print(f"  [{embedder.name}] Running identification on {len(probe_items)} probes "
          f"against {len(gallery_embeddings)} gallery identities...")

    # Flatten gallery for efficient computation
    gallery_labels = []
    gallery_matrix = []
    for label, embs in gallery_embeddings.items():
        for emb in embs:
            gallery_labels.append(label)
            gallery_matrix.append(emb)

    gallery_labels = np.array(gallery_labels)
    gallery_matrix = np.array(gallery_matrix)  # (G, 512)

    rank1_correct = 0
    rank5_correct = 0

    for true_label, img_path in probe_items:
        img = cv2.imread(img_path)
        if img is None:
            continue

        probe_emb = embedder.get_embedding(img)

        # Compute similarities to all gallery entries
        sims = gallery_matrix @ probe_emb  # cosine similarity (all L2-normalized)

        # Get top-5 matches
        top5_indices = np.argsort(sims)[::-1][:5]
        top5_labels = gallery_labels[top5_indices]

        if top5_labels[0] == true_label:
            rank1_correct += 1
        if true_label in top5_labels:
            rank5_correct += 1

    num_probes = len(probe_items)
    return {
        "rank1_accuracy": float(rank1_correct / num_probes) if num_probes > 0 else 0,
        "rank5_accuracy": float(rank5_correct / num_probes) if num_probes > 0 else 0,
        "num_probes": num_probes,
        "num_gallery_identities": len(gallery_embeddings),
    }


def evaluate_embedding_quality(embedder: ONNXEmbedder, identity_data: dict,
                                max_identities: int = 50):
    """Measure intra-class and inter-class embedding distances.

    Good embeddings have:
        - Low intra-class distance (same person = similar)
        - High inter-class distance (different people = far apart)
    """
    labels = list(identity_data.keys())
    if len(labels) > max_identities:
        random.seed(42)
        labels = random.sample(labels, max_identities)

    print(f"  [{embedder.name}] Computing embedding quality on {len(labels)} identities...")

    # Get mean embedding per identity
    identity_embeddings = {}
    all_intra_dists = []

    for label in labels:
        images = identity_data[label]
        embs = []
        for img_path in images:
            img = cv2.imread(img_path)
            if img is not None:
                embs.append(embedder.get_embedding(img))

        if len(embs) >= 2:
            identity_embeddings[label] = embs

            # Intra-class: pairwise cosine distance within same identity
            for i in range(len(embs)):
                for j in range(i + 1, len(embs)):
                    sim = cosine_similarity(embs[i], embs[j])
                    all_intra_dists.append(sim)

    # Inter-class: cosine distance between different identities (using mean embeddings)
    all_inter_dists = []
    label_list = list(identity_embeddings.keys())
    mean_embs = {label: np.mean(embs, axis=0) for label, embs in identity_embeddings.items()}
    # Re-normalize mean embeddings
    for label in mean_embs:
        norm = np.linalg.norm(mean_embs[label])
        if norm > 0:
            mean_embs[label] = mean_embs[label] / norm

    for i in range(len(label_list)):
        for j in range(i + 1, len(label_list)):
            sim = cosine_similarity(mean_embs[label_list[i]], mean_embs[label_list[j]])
            all_inter_dists.append(sim)

    intra = np.array(all_intra_dists) if all_intra_dists else np.array([0])
    inter = np.array(all_inter_dists) if all_inter_dists else np.array([0])

    return {
        "mean_intra_class_similarity": float(np.mean(intra)),
        "std_intra_class_similarity": float(np.std(intra)),
        "mean_inter_class_similarity": float(np.mean(inter)),
        "std_inter_class_similarity": float(np.std(inter)),
        "separability": float(np.mean(intra) - np.mean(inter)),
        "num_identities_evaluated": len(identity_embeddings),
    }


# ============================================================
#  Main
# ============================================================

def run_evaluation(original_path: str, finetuned_path: str, data_dir: str,
                   num_pairs: int = 3000, output_file: str = None):
    """Run full comparison evaluation between original and fine-tuned models."""

    print(f"\n{'='*60}")
    print(f"  ArcFace Evaluation: Original vs Fine-Tuned")
    print(f"{'='*60}\n")

    # Load models
    original = ONNXEmbedder(original_path, name="Original")
    finetuned = ONNXEmbedder(finetuned_path, name="FineTuned")

    # Load data
    identity_data = load_evaluation_data(data_dir)
    if len(identity_data) < 2:
        print("[ERROR] Need at least 2 identities with 2+ images each for evaluation.")
        return

    # Generate pairs
    pos_pairs, neg_pairs = generate_pairs(identity_data, num_pairs)

    results = {"original": {}, "finetuned": {}}

    # === 1. Verification ===
    print(f"\n{'='*40}")
    print(f"  Test 1: Face Verification (1:1)")
    print(f"{'='*40}")

    print(f"\n  Original model:")
    results["original"]["verification"] = evaluate_verification(original, pos_pairs, neg_pairs)

    print(f"\n  Fine-tuned model:")
    results["finetuned"]["verification"] = evaluate_verification(finetuned, pos_pairs, neg_pairs)

    # === 2. Identification ===
    print(f"\n{'='*40}")
    print(f"  Test 2: Face Identification (1:N)")
    print(f"{'='*40}")

    print(f"\n  Original model:")
    results["original"]["identification"] = evaluate_identification(original, identity_data)

    print(f"\n  Fine-tuned model:")
    results["finetuned"]["identification"] = evaluate_identification(finetuned, identity_data)

    # === 3. Embedding Quality ===
    print(f"\n{'='*40}")
    print(f"  Test 3: Embedding Quality")
    print(f"{'='*40}")

    print(f"\n  Original model:")
    results["original"]["embedding_quality"] = evaluate_embedding_quality(original, identity_data)

    print(f"\n  Fine-tuned model:")
    results["finetuned"]["embedding_quality"] = evaluate_embedding_quality(finetuned, identity_data)

    # === Print Summary ===
    print(f"\n\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")

    orig_ver = results["original"]["verification"]
    ft_ver = results["finetuned"]["verification"]
    print(f"\n  Verification Accuracy:")
    print(f"    Original:   {orig_ver['best_accuracy']*100:.1f}% (threshold: {orig_ver['best_threshold']:.2f})")
    print(f"    Fine-tuned: {ft_ver['best_accuracy']*100:.1f}% (threshold: {ft_ver['best_threshold']:.2f})")
    diff = (ft_ver['best_accuracy'] - orig_ver['best_accuracy']) * 100
    print(f"    Change:     {diff:+.1f}%")

    orig_id = results["original"]["identification"]
    ft_id = results["finetuned"]["identification"]
    print(f"\n  Identification (Rank-1) Accuracy:")
    print(f"    Original:   {orig_id['rank1_accuracy']*100:.1f}%")
    print(f"    Fine-tuned: {ft_id['rank1_accuracy']*100:.1f}%")
    diff = (ft_id['rank1_accuracy'] - orig_id['rank1_accuracy']) * 100
    print(f"    Change:     {diff:+.1f}%")

    orig_eq = results["original"]["embedding_quality"]
    ft_eq = results["finetuned"]["embedding_quality"]
    print(f"\n  Embedding Separability (higher = better):")
    print(f"    Original:   {orig_eq['separability']:.4f}")
    print(f"    Fine-tuned: {ft_eq['separability']:.4f}")

    print(f"\n  Mean Intra-class Similarity (higher = better):")
    print(f"    Original:   {orig_eq['mean_intra_class_similarity']:.4f}")
    print(f"    Fine-tuned: {ft_eq['mean_intra_class_similarity']:.4f}")

    print(f"\n  Mean Inter-class Similarity (lower = better):")
    print(f"    Original:   {orig_eq['mean_inter_class_similarity']:.4f}")
    print(f"    Fine-tuned: {ft_eq['mean_inter_class_similarity']:.4f}")

    print(f"\n{'='*60}")

    # Save results to JSON
    if output_file is None:
        output_file = "fine_tuning/evaluation_results.json"
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Full results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Original vs Fine-Tuned ArcFace")
    parser.add_argument("--original", required=True,
                        help="Path to original ArcFace ONNX model")
    parser.add_argument("--finetuned", required=True,
                        help="Path to fine-tuned ArcFace ONNX model")
    parser.add_argument("--data", required=True,
                        help="Path to aligned dataset (output of prepare_dataset.py)")
    parser.add_argument("--pairs", type=int, default=3000,
                        help="Number of verification pairs to generate (default: 3000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results JSON")
    args = parser.parse_args()

    run_evaluation(
        original_path=args.original,
        finetuned_path=args.finetuned,
        data_dir=args.data,
        num_pairs=args.pairs,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
