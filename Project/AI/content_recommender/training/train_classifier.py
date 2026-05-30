"""
Train SubjectClassifier-Ar — a small Transformer encoder built and trained
FROM SCRATCH (random init, no pretrained weights) to classify Arabic
text into 7 SANAD categories: Culture, Finance, Medical, Politics,
Religion, Sports, Tech.

Architecture:
  - SentencePiece-like WordPiece tokenizer trained on SANAD corpus
  - Embedding layer + sinusoidal positional encoding
  - 4 Transformer encoder layers, hidden=256, heads=4, ff=512
  - Mean-pool + linear classifier head
  - ~10M parameters total

Dataset: arbml/SANAD (~200K Arabic news articles)
Hardware: RTX 5060 8GB, ~1 hour for 3 epochs
"""

import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path(__file__).parent / "checkpoints" / "subject_classifier_ar"
DATASET     = "arbml/SANAD"
VOCAB_SIZE  = 16_000
MAX_LEN     = 128
D_MODEL     = 256
N_HEADS     = 4
N_LAYERS    = 4
FF_DIM      = 512
DROPOUT     = 0.1
BATCH_SIZE  = 32
EPOCHS      = 3
LR          = 3e-4
SEED        = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def build_tokenizer(texts: list[str], save_path: Path) -> Tokenizer:
    print(f"[classifier] Training WordPiece tokenizer (vocab={VOCAB_SIZE})...")
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordPieceTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"],
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    print(f"[classifier] Tokenizer saved: {save_path}")
    return tokenizer


def encode(tokenizer: Tokenizer, text: str, max_len: int = MAX_LEN) -> list[int]:
    ids = tokenizer.encode(text).ids[: max_len - 2]
    pad_id = tokenizer.token_to_id("[PAD]")
    cls_id = tokenizer.token_to_id("[CLS]") or 2
    sep_id = tokenizer.token_to_id("[SEP]") or 3
    ids = [cls_id] + ids + [sep_id]
    ids += [pad_id] * (max_len - len(ids))
    return ids


# ── Model ─────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class SubjectClassifier(nn.Module):
    """Tiny Transformer encoder. Random init — NO pretrained weights."""

    def __init__(self, vocab_size: int, n_classes: int, pad_id: int = 0):
        super().__init__()
        self.pad_id    = pad_id
        self.embed     = nn.Embedding(vocab_size, D_MODEL, padding_idx=pad_id)
        self.pos       = PositionalEncoding(D_MODEL, max_len=MAX_LEN)
        layer          = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FF_DIM,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.encoder   = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.dropout   = nn.Dropout(DROPOUT)
        self.classifier = nn.Linear(D_MODEL, n_classes)
        # Random init applied automatically by nn modules; no pretrained load.

    def forward(self, ids):
        mask = ids == self.pad_id
        x = self.embed(ids) * math.sqrt(D_MODEL)
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        # Mean-pool over non-pad tokens
        valid = (~mask).unsqueeze(-1).float()
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        return self.classifier(self.dropout(pooled))


# ── Dataset ───────────────────────────────────────────────────────────────────

class SanadDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            torch.tensor(encode(self.tokenizer, self.texts[idx]), dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def load_sanad():
    print("[classifier] Loading SANAD dataset...")
    ds = load_dataset(DATASET, split="train")

    # Find text + label fields (SANAD configs vary)
    text_field  = "text"     if "text"     in ds.column_names else ds.column_names[0]
    label_field = "category" if "category" in ds.column_names else ds.column_names[-1]

    texts  = [r[text_field][:1500] for r in ds]          # cap raw text length
    labels_raw = [r[label_field] for r in ds]

    # Build label index
    unique = sorted(set(labels_raw))
    label_to_id = {lab: i for i, lab in enumerate(unique)}
    labels = [label_to_id[l] for l in labels_raw]

    print(f"[classifier] {len(texts)} examples, {len(unique)} classes: {unique}")

    # 90/10 train/eval
    random.seed(SEED)
    idxs = list(range(len(texts)))
    random.shuffle(idxs)
    cut = int(0.9 * len(idxs))
    train_idx, eval_idx = idxs[:cut], idxs[cut:]

    return (
        [texts[i] for i in train_idx],  [labels[i] for i in train_idx],
        [texts[i] for i in eval_idx],   [labels[i] for i in eval_idx],
        label_to_id,
    )


# ── Train ─────────────────────────────────────────────────────────────────────

def main():
    random.seed(SEED); torch.manual_seed(SEED)
    print(f"[classifier] Device: {DEVICE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_x, train_y, eval_x, eval_y, label_to_id = load_sanad()

    # Tokenizer trained on the training texts (subsample for speed)
    sample_for_tokenizer = train_x[:50_000]
    tokenizer = build_tokenizer(sample_for_tokenizer, OUTPUT_DIR / "tokenizer.json")
    pad_id = tokenizer.token_to_id("[PAD]") or 0

    train_ds = SanadDataset(train_x, train_y, tokenizer)
    eval_ds  = SanadDataset(eval_x,  eval_y,  tokenizer)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    eval_dl  = DataLoader(eval_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SubjectClassifier(VOCAB_SIZE, len(label_to_id), pad_id=pad_id).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[classifier] Model params: {n_params/1e6:.2f}M  (trained from scratch, no pretrained weights)")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=len(train_dl) * EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for ids, labels in pbar:
            ids, labels = ids.to(DEVICE), labels.to(DEVICE)
            logits = model(ids)
            loss = F.cross_entropy(logits, labels)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            loss_sum += loss.item() * labels.size(0)
            correct  += (logits.argmax(dim=-1) == labels).sum().item()
            total    += labels.size(0)
            pbar.set_postfix(loss=f"{loss_sum/total:.4f}", acc=f"{correct/total:.4f}")

        # Eval
        model.eval()
        ec, et = 0, 0
        with torch.no_grad():
            for ids, labels in eval_dl:
                ids, labels = ids.to(DEVICE), labels.to(DEVICE)
                logits = model(ids)
                ec += (logits.argmax(dim=-1) == labels).sum().item()
                et += labels.size(0)
        print(f"[classifier] Epoch {epoch+1} eval acc: {ec/et:.4f}")

    # Save
    torch.save({
        "model_state":  model.state_dict(),
        "label_to_id":  label_to_id,
        "config":       {
            "vocab_size": VOCAB_SIZE, "max_len": MAX_LEN,
            "d_model": D_MODEL, "n_heads": N_HEADS, "n_layers": N_LAYERS,
            "ff_dim": FF_DIM, "pad_id": pad_id,
        },
    }, OUTPUT_DIR / "model.pt")
    print(f"\n[classifier] DONE. Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
