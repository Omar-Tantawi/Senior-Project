"""
src/train.py — Full Training Pipeline

Stages:
  1. Pretrain backbone on VGGFace2 / LFW (or load pretrained weights)
  2. Fine-tune on student enrollment data

Features:
  - Mixed precision (fp16) for faster NVIDIA GPU training
  - Cosine annealing LR with warmup
  - TensorBoard logging
  - Automatic checkpoint saving (best val loss & every N epochs)
  - Early stopping

Usage:
    # Stage 1 — Pretrain on VGGFace2
    python src/train.py --stage pretrain

    # Stage 2 — Fine-tune on students
    python src/train.py --stage finetune

    # Resume from checkpoint
    python src/train.py --stage pretrain --resume models/checkpoint_epoch_10.pth
"""

import os
import sys
import time
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from config import *
from src.model import FaceRecognitionModel, iresnet50
from src.augmentation import FaceDataset, get_train_transforms, get_val_transforms


# ─── Warmup + Cosine Annealing LR Scheduler ──────────────────────────────────
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            factor = (self.last_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / \
                       (self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [base_lr * factor for base_lr in self.base_lrs]


import math


# ─── Training Step ────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scaler, device, epoch, writer):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d} [TRAIN]", leave=False)
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast("cuda", enabled=MIXED_PRECISION):
            embeddings, loss = model(images, labels)

        scaler.scale(loss).backward()
        # Gradient clipping prevents exploding gradients
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item() * images.size(0)
        total_samples += images.size(0)

        global_step = epoch * len(loader) + batch_idx
        writer.add_scalar("Loss/train_batch", loss.item(), global_step)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / total_samples
    return avg_loss


# ─── Validation Step ──────────────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, device, epoch):
    model.eval()
    total_loss, total_samples = 0.0, 0

    for images, labels in tqdm(loader, desc=f"Epoch {epoch:3d} [VAL  ]", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=MIXED_PRECISION):
            _, loss = model(images, labels)

        total_loss    += loss.item() * images.size(0)
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    return avg_loss


# ─── Save Checkpoint ──────────────────────────────────────────────────────────
def save_checkpoint(model, optimizer, epoch, val_loss, path):
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "val_loss":   val_loss,
    }, path)


# ─── Stage 1: Pretrain on VGGFace2 / LFW ─────────────────────────────────────
def run_pretrain(args):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Stage 1: Pretraining on {DATASET_NAME.upper()}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = FaceDataset(
        root_dir=str(AUGMENTED_DIR / "train"),
        transform=get_train_transforms(IMAGE_SIZE),
        mode="train"
    )
    val_dataset = FaceDataset(
        root_dir=str(AUGMENTED_DIR / "val"),
        transform=get_val_transforms(IMAGE_SIZE),
        mode="val"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    num_classes = train_dataset.num_classes
    print(f"[INFO] Training with {num_classes} identities")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = FaceRecognitionModel(
        backbone_name=BACKBONE,
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM,
        loss_type=LOSS_FUNCTION
    ).to(device)

    # Resume from checkpoint
    start_epoch = 1
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[OK] Resumed from {args.resume} (epoch {ckpt['epoch']})")

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)

    scaler = GradScaler(enabled=MIXED_PRECISION)
    writer = SummaryWriter(log_dir=str(TENSORBOARD_DIR / "pretrain"))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    PATIENCE = 7   # Early stopping patience

    # ── Training Loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, writer)
        val_loss   = val_epoch(model, val_loader, device, epoch)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
              f"LR: {lr:.6f} | {elapsed:.1f}s")

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("LR",         lr,         epoch)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss,
                            MODEL_DIR / "best_pretrain.pth")
            print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1

        # Save periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, val_loss,
                            MODEL_DIR / f"checkpoint_epoch_{epoch}.pth")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

    writer.close()

    # Save backbone weights only (for inference / fine-tuning)
    torch.save(model.backbone.state_dict(), BACKBONE_PATH)
    print(f"\n[DONE] Pretraining complete. Backbone saved to {BACKBONE_PATH}")
    print("  Next: python src/train.py --stage finetune")


# ─── Stage 2: Fine-tune on Student Enrollment Data ───────────────────────────
def run_finetune(args):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  Stage 2: Fine-tuning on Student Enrollment Data")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    from src.augmentation import get_enrollment_transforms

    # Student DB: data/student_db/{student_name}/*.jpg
    if not STUDENT_DB_DIR.exists() or not any(STUDENT_DB_DIR.iterdir()):
        print("[ERROR] No student enrollment data found.")
        print(f"  Create: {STUDENT_DB_DIR}/StudentName/*.jpg")
        print("  Min 5 photos per student. Run: python src/enroll.py")
        sys.exit(1)

    train_dataset = FaceDataset(
        root_dir=str(STUDENT_DB_DIR),
        transform=get_train_transforms(IMAGE_SIZE),
        mode="finetune"
    )
    # Use same data for val when few samples (small class scenario)
    val_dataset = FaceDataset(
        root_dir=str(STUDENT_DB_DIR),
        transform=get_val_transforms(IMAGE_SIZE),
        mode="val"
    )

    train_loader = DataLoader(
        train_dataset, batch_size=FINETUNE_BATCH, shuffle=True,
        num_workers=2, pin_memory=PIN_MEMORY, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=FINETUNE_BATCH, shuffle=False,
        num_workers=2, pin_memory=PIN_MEMORY
    )

    num_students = train_dataset.num_classes
    print(f"[INFO] Fine-tuning for {num_students} students")

    # Load pretrained backbone
    from src.model import iresnet50, ArcFaceLoss
    backbone = iresnet50(embedding_dim=EMBEDDING_DIM)
    if BACKBONE_PATH.exists():
        backbone.load_state_dict(torch.load(BACKBONE_PATH, map_location="cpu"))
        print(f"[OK] Loaded pretrained backbone from {BACKBONE_PATH}")
    else:
        print("[WARN] No pretrained backbone found. Training from scratch is slower.")

    model = FaceRecognitionModel(
        backbone_name=BACKBONE,
        num_classes=num_students,
        embedding_dim=EMBEDDING_DIM,
        loss_type=LOSS_FUNCTION
    ).to(device)
    model.backbone.load_state_dict(backbone.state_dict())

    # Fine-tune: lower LR, train full network
    optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)
    scaler    = GradScaler(enabled=MIXED_PRECISION)
    writer    = SummaryWriter(log_dir=str(TENSORBOARD_DIR / "finetune"))

    best_val_loss = float("inf")

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch, writer)
        val_loss   = val_epoch(model, val_loader, device, epoch)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{FINETUNE_EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.backbone.state_dict(), MODEL_DIR / "finetuned_backbone.pth")
            print(f"  ✓ Best fine-tuned model saved")

    writer.close()
    print(f"\n[DONE] Fine-tuning complete.")
    print("  Next: python src/enroll.py — build the student embedding database")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage",  choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    if args.stage == "pretrain":
        run_pretrain(args)
    else:
        run_finetune(args)
