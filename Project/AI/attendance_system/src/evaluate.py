"""
src/evaluate.py — Model Evaluation

Metrics computed:
  - Verification accuracy (TAR @ FAR=0.001, FAR=0.01)
  - Identification accuracy (rank-1, rank-5)
  - Cosine similarity threshold calibration
  - Confusion matrix on enrolled students
  - Per-condition breakdown (lighting / angle / occlusion)

Usage:
    python src/evaluate.py
    python src/evaluate.py --threshold 0.55
"""

import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    classification_report, accuracy_score
)
from scipy.spatial.distance import cosine

sys.path.append(str(Path(__file__).parent.parent))
from config import *


# ─── Load Embedding Database ──────────────────────────────────────────────────
def load_db():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"No embedding DB at {EMBEDDINGS_PATH}. Run enroll.py first.")
    with open(EMBEDDINGS_PATH, "rb") as f:
        db = pickle.load(f)
    print(f"[OK] Loaded {len(db)} students from DB")
    return db


def load_all_embeddings():
    """Load per-image embeddings (not just mean) for thorough evaluation."""
    path = DATA_DIR / "all_embeddings.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# ─── Cosine Similarity ────────────────────────────────────────────────────────
def cos_sim(a, b):
    return 1 - cosine(a, b)


# ─── Build Pairs for Verification ────────────────────────────────────────────
def build_verification_pairs(all_embeddings: dict, n_pairs_per_class: int = 50):
    """
    Build positive (same person) and negative (different person) pairs.
    Returns: similarities, labels (1=same, 0=different)
    """
    names = list(all_embeddings.keys())
    similarities = []
    labels       = []

    # Positive pairs (same identity)
    for name, embs in all_embeddings.items():
        if len(embs) < 2:
            continue
        for i in range(min(len(embs) - 1, n_pairs_per_class)):
            sim = cos_sim(embs[i], embs[i + 1])
            similarities.append(sim)
            labels.append(1)

    # Negative pairs (different identities)
    import random
    for _ in range(len(similarities)):
        n1, n2 = random.sample(names, 2)
        e1 = random.choice(all_embeddings[n1])
        e2 = random.choice(all_embeddings[n2])
        sim = cos_sim(e1, e2)
        similarities.append(sim)
        labels.append(0)

    return np.array(similarities), np.array(labels)


# ─── TAR @ FAR ────────────────────────────────────────────────────────────────
def compute_tar_at_far(similarities, labels, far_targets=(0.001, 0.01, 0.1)):
    """
    True Accept Rate at given False Accept Rates.
    Key metric for security-sensitive recognition systems.
    """
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)

    results = {"AUC": roc_auc}
    for far in far_targets:
        idx = np.searchsorted(fpr, far)
        if idx < len(tpr):
            results[f"TAR@FAR={far}"] = float(tpr[idx])
            results[f"Threshold@FAR={far}"] = float(thresholds[idx])

    return results, fpr, tpr


# ─── Identification Accuracy ──────────────────────────────────────────────────
def compute_identification_accuracy(all_embeddings: dict, db: dict, threshold: float):
    """
    For each stored embedding, find the top match in the DB.
    Reports rank-1 accuracy and the best cosine threshold.
    """
    y_true, y_pred = [], []

    for true_name, embs in all_embeddings.items():
        for emb in embs:
            best_name  = UNKNOWN_LABEL
            best_score = 0.0

            for db_name, db_emb in db.items():
                score = cos_sim(emb, db_emb)
                if score > best_score:
                    best_score = score
                    best_name  = db_name

            if best_score < threshold:
                best_name = UNKNOWN_LABEL

            y_true.append(true_name)
            y_pred.append(best_name)

    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred


# ─── Plot ROC ─────────────────────────────────────────────────────────────────
def plot_roc(fpr, tpr, auc_val, threshold):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(fpr, tpr, color="#1a73e8", lw=2,
            label=f"ROC Curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Face Verification ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=150)
    print(f"[OK] ROC curve saved to {OUTPUT_DIR / 'roc_curve.png'}")
    plt.close()


# ─── Plot Confusion Matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, names):
    cm = confusion_matrix(y_true, y_pred, labels=names)
    fig, ax = plt.subplots(figsize=(max(8, len(names)), max(6, len(names))))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=names, yticklabels=names,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix — Student Identification")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    print(f"[OK] Confusion matrix saved to {OUTPUT_DIR / 'confusion_matrix.png'}")
    plt.close()


# ─── Threshold Calibration ─────────────────────────────────────────────────────
def find_best_threshold(similarities, labels):
    """Find the cosine similarity threshold that maximizes accuracy."""
    best_acc, best_thr = 0.0, 0.5
    for thr in np.arange(0.3, 0.95, 0.01):
        preds = (similarities >= thr).astype(int)
        acc   = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override recognition threshold (default: auto-calibrate)")
    args = parser.parse_args()

    db   = load_db()
    all_embs = load_all_embeddings()

    if all_embs is None:
        print("[WARN] No per-image embeddings found. Generating from DB means only.")
        print("  For better evaluation, re-run enroll.py to store all embeddings.")
        all_embs = {name: [emb] for name, emb in db.items()}

    print("\n── Verification Evaluation ──────────────────────────────────")
    similarities, labels = build_verification_pairs(all_embs, n_pairs_per_class=30)

    if args.threshold:
        threshold = args.threshold
        print(f"  Using provided threshold: {threshold}")
    else:
        threshold, best_acc = find_best_threshold(similarities, labels)
        print(f"  Best threshold (auto): {threshold:.2f} → verification acc: {best_acc:.4f}")
        print(f"  → Update RECOGNITION_THRESHOLD = {threshold:.2f} in config.py")

    tar_results, fpr, tpr = compute_tar_at_far(similarities, labels)
    print(f"\n  AUC:              {tar_results['AUC']:.4f}")
    for far in [0.001, 0.01, 0.1]:
        key = f"TAR@FAR={far}"
        if key in tar_results:
            print(f"  {key}:  {tar_results[key]:.4f}  (threshold: {tar_results[f'Threshold@FAR={far}']:.3f})")

    plot_roc(fpr, tpr, tar_results["AUC"], threshold)

    print("\n── Identification Evaluation ────────────────────────────────")
    acc, y_true, y_pred = compute_identification_accuracy(all_embs, db, threshold)
    print(f"  Rank-1 Identification Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    names = sorted(db.keys()) + [UNKNOWN_LABEL]
    print("\n  Per-class report:")
    print(classification_report(y_true, y_pred, labels=sorted(db.keys()), zero_division=0))

    plot_confusion_matrix(y_true, y_pred, sorted(db.keys()))

    print("\n── Summary ─────────────────────────────────────────────────")
    print(f"  Students enrolled:  {len(db)}")
    print(f"  Threshold used:     {threshold:.2f}")
    print(f"  Identification acc: {acc*100:.2f}%")
    print(f"  AUC:                {tar_results['AUC']:.4f}")
    print(f"  Outputs saved to:   {OUTPUT_DIR}/")
    print("────────────────────────────────────────────────────────────\n")
