"""
Train EduRanker-Ar — fine-tune a multilingual sentence transformer on Arabic
query-passage pairs for better educational content ranking.

Base model : sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Datasets   : tydiqa (Arabic) + arbml/arcd + xquad (Arabic)
             All standard Parquet — no legacy dataset scripts required.
Loss       : MultipleNegativesRankingLoss  (contrastive learning)
Output     : ./checkpoints/edu_ranker_ar/

Hardware   : RTX 5060, ~15-30 min
"""

import random
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path

import torch
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import (
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL   = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR   = Path(__file__).parent / "checkpoints" / "edu_ranker_ar"
EPOCHS       = 3
BATCH_SIZE   = 32
WARMUP_RATIO = 0.1
LR           = 2e-5
EVAL_SPLIT   = 0.1      # hold out 10% for evaluation
SEED         = 42


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_tydiqa_arabic() -> list[tuple[str, str]]:
    """TyDi QA secondary task — Arabic (question, passage) pairs."""
    print("  [dataset] Loading tydiqa Arabic...")
    try:
        ds = load_dataset("tydiqa", "secondary_task", split="train")
        pairs = []
        for row in ds:
            lang = row.get("id", "")
            if not lang.startswith("arabic"):
                continue
            q   = (row.get("question_text") or "").strip()
            ctx = (row.get("context") or row.get("document_plaintext") or "").strip()[:1000]
            if q and ctx:
                pairs.append((q, ctx))
        print(f"  [dataset] tydiqa Arabic: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"  [dataset] tydiqa failed: {e}")
        return []


def load_arcd() -> list[tuple[str, str]]:
    """ARCD — Arabic Reading Comprehension Dataset (question, context) pairs."""
    print("  [dataset] Loading arbml/arcd...")
    try:
        ds = load_dataset("arbml/arcd", split="train")
        pairs = []
        for row in ds:
            q   = (row.get("question") or "").strip()
            ctx = (row.get("context")  or "").strip()
            if q and ctx:
                pairs.append((q, ctx))
        print(f"  [dataset] arcd: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"  [dataset] arcd failed: {e}")
        return []


def load_xquad_arabic() -> list[tuple[str, str]]:
    """XQuAD Arabic split — (question, context) pairs."""
    print("  [dataset] Loading xquad Arabic...")
    try:
        ds = load_dataset("xquad", "xquad.ar", split="validation")
        pairs = []
        for row in ds:
            q   = (row.get("question") or "").strip()
            ctx = (row.get("context")  or "").strip()
            if q and ctx:
                pairs.append((q, ctx))
        print(f"  [dataset] xquad Arabic: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"  [dataset] xquad failed: {e}")
        return []


def load_arabic_wikipedia_pairs(max_pairs: int = 5000) -> list[tuple[str, str]]:
    """Arabic Wikipedia — use article title as query, intro paragraph as positive."""
    print(f"  [dataset] Loading Arabic Wikipedia (up to {max_pairs} articles)...")
    try:
        ds = load_dataset("wikipedia", "20231101.ar", split="train", streaming=True)
        pairs = []
        for row in ds:
            title = (row.get("title") or "").strip()
            text  = (row.get("text")  or "").strip()
            if not title or not text:
                continue
            # Use first 500 chars of the article as the passage
            passage = text[:500].strip()
            if len(passage) > 50:
                pairs.append((title, passage))
            if len(pairs) >= max_pairs:
                break
        print(f"  [dataset] Wikipedia Arabic: {len(pairs)} pairs")
        return pairs
    except Exception as e:
        print(f"  [dataset] Wikipedia failed: {e}")
        return []


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"[train_ranker] CUDA  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[train_ranker] Device: {torch.cuda.get_device_name(0)}")

    # 1. Collect all pairs
    print("[train_ranker] Loading Arabic QA datasets...")
    all_pairs: list[tuple[str, str]] = []
    all_pairs += load_tydiqa_arabic()
    all_pairs += load_arcd()
    all_pairs += load_xquad_arabic()
    all_pairs += load_arabic_wikipedia_pairs(max_pairs=5000)

    if not all_pairs:
        print("[train_ranker] ERROR: No pairs loaded. Check your internet connection.")
        return

    random.shuffle(all_pairs)
    print(f"[train_ranker] Total pairs: {len(all_pairs)}")

    # 2. Split train/eval
    cut = int(len(all_pairs) * (1 - EVAL_SPLIT))
    train_pairs = all_pairs[:cut]
    eval_pairs  = all_pairs[cut:]
    print(f"[train_ranker] Train: {len(train_pairs)} | Eval: {len(eval_pairs)}")

    train_hf = HFDataset.from_dict({
        "anchor":   [p[0] for p in train_pairs],
        "positive": [p[1] for p in train_pairs],
    })
    eval_hf = HFDataset.from_dict({
        "anchor":   [p[0] for p in eval_pairs],
        "positive": [p[1] for p in eval_pairs],
    })

    # 3. Load base model
    print(f"\n[train_ranker] Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    # 4. Loss
    loss = MultipleNegativesRankingLoss(model)

    # 5. Training arguments
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_ratio=WARMUP_RATIO,
        learning_rate=LR,
        fp16=torch.cuda.is_available(),
        bf16=False,
        eval_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        seed=SEED,
    )

    # 6. Train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_hf,
        eval_dataset=eval_hf,
        loss=loss,
    )

    print(f"[train_ranker] Training {EPOCHS} epochs | batch={BATCH_SIZE} | lr={LR}")
    trainer.train()

    # 7. Save
    model.save_pretrained(str(OUTPUT_DIR))
    print(f"\n[train_ranker] ✓ Done. Model saved → {OUTPUT_DIR}")
    print(f"[train_ranker] Update config.py: EMBED_MODEL = r\"{OUTPUT_DIR}\"")


if __name__ == "__main__":
    main()
