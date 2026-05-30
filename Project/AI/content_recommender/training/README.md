# Training Pipeline

Two custom models for the Content Recommender:

## Model 1 — EduRanker-Ar (fine-tuned)
**File:** `train_ranker.py`
**Base:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
**Dataset:** `unicamp-dl/mmarco` (Arabic) — 500K query-passage pairs
**Loss:** `MultipleNegativesRankingLoss` (contrastive)
**Output:** `./checkpoints/edu_ranker_ar/`
**Hardware:** RTX 5060, ~2 hours
**Evaluation:** `evaluate_ranker.py` on MIRACL-Arabic (MRR@10, Recall@100)

## Model 2 — SubjectClassifier-Ar (from scratch)
**File:** `train_classifier.py`
**Architecture:** 4-layer Transformer encoder, ~10M params, random init
**Dataset:** `arbml/SANAD` (200K Arabic news, 7 classes)
**Output:** `./checkpoints/subject_classifier_ar/`
**Hardware:** RTX 5060, ~1 hour

## How to Run

```bash
pip install -r requirements_train.txt
python training/train_ranker.py        # Model 1
python training/evaluate_ranker.py     # Eval Model 1
python training/train_classifier.py    # Model 2
```

After training, update `config.py`:
```python
EMBED_MODEL = "./training/checkpoints/edu_ranker_ar"
```
