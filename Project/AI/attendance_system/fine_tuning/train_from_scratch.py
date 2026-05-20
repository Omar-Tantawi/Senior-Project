"""Train IResNet-100 + AdaFace from scratch on MS1M-ArcFace dataset.

This script trains a face recognition model FROM SCRATCH (no pretrained weights)
using:
- IResNet-100 backbone (deeper than ResNet-50)
- AdaFace loss (quality-aware, optimized for CCTV/low-quality images)
- MS1M-ArcFace dataset (5.8M images, 85K identities)

Reference:
    Kim et al., "AdaFace: Quality Adaptive Margin for Face Recognition"
    CVPR 2022 (Oral)

Usage:
    python fine_tuning/train_from_scratch.py \
        --data "C:/Users/abdul/OneDrive/Desktop/MS-Celeb-1M/ms1m-arcface" \
        --epochs 25 \
        --batch-size 128

Resume from checkpoint:
    python fine_tuning/train_from_scratch.py \
        --data "C:/Users/abdul/OneDrive/Desktop/MS-Celeb-1M/ms1m-arcface" \
        --resume fine_tuning/checkpoints/scratch/latest.pth
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

# Add fine_tuning to path so we can import models
sys.path.insert(0, str(Path(__file__).parent))
from models import iresnet50, iresnet100, AdaFace


# ============================================================
# Dataset Loader for MS1M-ArcFace format
# ============================================================
class MS1MArcFaceDataset(Dataset):
    """Dataset loader for MS1M-ArcFace folder structure.

    Expected structure:
        root/
            0/      <- identity 0
                1.jpg
                2.jpg
                ...
            1/      <- identity 1
                1.jpg
                ...
            ...
    """

    def __init__(self, root_dir, image_size=112, train=True, max_images_per_id=None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size
        self.train = train
        self.max_images_per_id = max_images_per_id

        print(f"[Dataset] Scanning {self.root_dir}...")
        start = time.time()

        # Get all identity folders (sorted numerically)
        identity_dirs = sorted(
            [d for d in self.root_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x.name) if x.name.isdigit() else x.name,
        )

        # Build sample list: (image_path, label)
        self.samples = []
        self.num_classes = 0

        for label, identity_dir in enumerate(identity_dirs):
            images = list(identity_dir.glob("*.jpg"))
            if len(images) >= 2:  # Need at least 2 images per identity
                # Cap to max_images_per_id if specified
                if self.max_images_per_id is not None:
                    images = images[:self.max_images_per_id]
                for img_path in images:
                    self.samples.append((str(img_path), label))
                self.num_classes = label + 1

        elapsed = time.time() - start
        print(f"[Dataset] Found {self.num_classes} identities, "
              f"{len(self.samples)} images in {elapsed:.1f}s")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        img = cv2.imread(img_path)
        if img is None:
            # Fallback: black image (shouldn't happen with clean dataset)
            img = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # Resize if needed (MS1M is already 112x112 but just in case)
        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size))

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Data augmentation for training
        if self.train:
            # Random horizontal flip
            if random.random() < 0.5:
                img = cv2.flip(img, 1)

        # Normalize to [-1, 1] (standard for face recognition)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        return torch.from_numpy(img), label


# ============================================================
# Training Functions
# ============================================================
def train_one_epoch(
    model, head, loader, optimizer, scaler, device, epoch, total_epochs,
    log_interval=50, writer=None, max_batches=None,
):
    """Train for one epoch."""
    model.train()
    head.train()

    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()
    batch_start = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            print(f"[SmokeTest] Stopping early at batch {batch_idx} (max_batches={max_batches})")
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Mixed precision forward pass
        with torch.amp.autocast("cuda"):
            embeddings = model(images)
            logits = head(embeddings, labels)
            loss = F.cross_entropy(logits, labels)

        # NaN detection - skip this batch if loss is NaN/Inf
        if not torch.isfinite(loss):
            print(f"[WARN] NaN/Inf loss detected at batch {batch_idx}, skipping...")
            optimizer.zero_grad()
            continue

        # Mixed precision backward pass
        scaler.scale(loss).backward()

        # Gradient clipping to prevent explosion (key fix!)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)

        scaler.step(optimizer)
        scaler.update()

        # Stats
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Logging
        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - batch_start
            samples_per_sec = (log_interval * loader.batch_size) / elapsed
            current_lr = optimizer.param_groups[0]["lr"]
            acc = 100.0 * correct / total

            print(
                f"[Epoch {epoch}/{total_epochs}] "
                f"Batch {batch_idx + 1}/{len(loader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {acc:.2f}% | "
                f"LR: {current_lr:.5f} | "
                f"{samples_per_sec:.1f} img/s"
            )

            if writer is not None:
                global_step = (epoch - 1) * len(loader) + batch_idx
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/accuracy", acc, global_step)
                writer.add_scalar("train/lr", current_lr, global_step)

            batch_start = time.time()

    num_batches = batch_idx + 1 if max_batches is None else min(batch_idx + 1, max_batches)
    avg_loss = total_loss / max(num_batches, 1)
    accuracy = 100.0 * correct / max(total, 1)
    elapsed = time.time() - start_time

    print(
        f"\n[Epoch {epoch}/{total_epochs}] Complete | "
        f"Avg Loss: {avg_loss:.4f} | "
        f"Accuracy: {accuracy:.2f}% | "
        f"Time: {elapsed / 60:.1f} min\n"
    )

    return avg_loss, accuracy


def save_checkpoint(
    model, head, optimizer, scaler, epoch, num_classes, save_path, is_best=False
):
    """Save training checkpoint."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "num_classes": num_classes,
        "model_state_dict": model.state_dict(),
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }

    torch.save(checkpoint, save_path)
    print(f"[Save] Checkpoint saved to {save_path}")

    if is_best:
        best_path = save_path.parent / "best.pth"
        torch.save(checkpoint, best_path)
        print(f"[Save] Best model saved to {best_path}")


def load_checkpoint(model, head, optimizer, scaler, checkpoint_path, device):
    """Load training checkpoint to resume."""
    print(f"[Load] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    head.load_state_dict(checkpoint["head_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"[Load] Resuming from epoch {start_epoch}")
    return start_epoch


def export_to_onnx(model, output_path, device):
    """Export trained model to ONNX format."""
    import io as _io
    import sys as _sys

    model.eval()
    dummy_input = torch.randn(1, 3, 112, 112).to(device)

    # Redirect stdout/stderr to avoid Unicode encoding issues on Windows
    old_stdout = _sys.stdout
    old_stderr = _sys.stderr
    _sys.stdout = _io.StringIO()
    _sys.stderr = _io.StringIO()

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=18,
            input_names=["input"],
            output_names=["embedding"],
            dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        )
    finally:
        _sys.stdout = old_stdout
        _sys.stderr = old_stderr

    print(f"[Export] Model saved to {output_path}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Train IResNet-100 + AdaFace from scratch")
    parser.add_argument("--data", required=True, help="Path to MS1M-ArcFace dataset")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers")
    parser.add_argument("--output-dir", default="fine_tuning/checkpoints/scratch",
                        help="Where to save checkpoints")
    parser.add_argument("--log-dir", default="fine_tuning/checkpoints/scratch/logs",
                        help="TensorBoard log directory")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save-every", type=int, default=1, help="Save every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Stop after N batches per epoch (smoke testing)")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Print log every N batches")
    parser.add_argument("--backbone", choices=["iresnet50", "iresnet100"],
                        default="iresnet100",
                        help="Backbone architecture (iresnet50 = ~1.8x faster)")
    parser.add_argument("--lr-milestones", type=int, nargs="+",
                        default=[10, 18, 22],
                        help="Epochs at which to decay LR by 10x")
    parser.add_argument("--max-images-per-id", type=int, default=None,
                        help="Cap images per identity (e.g., 30 for ~2.5M total from 5.8M)")
    args = parser.parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training IResNet-100 + AdaFace from scratch")
    print(f"{'='*60}")
    print(f"[Device] Using {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[Device] CUDA: {torch.version.cuda}")

    # ============================================================
    # Dataset & DataLoader
    # ============================================================
    train_dataset = MS1MArcFaceDataset(
        args.data, image_size=112, train=True,
        max_images_per_id=args.max_images_per_id
    )
    num_classes = train_dataset.num_classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    print(f"\n[Data] Total identities: {num_classes}")
    print(f"[Data] Total images: {len(train_dataset)}")
    print(f"[Data] Batches per epoch: {len(train_loader)}")

    # ============================================================
    # Model: IResNet + AdaFace head
    # ============================================================
    backbone_fn = iresnet50 if args.backbone == "iresnet50" else iresnet100
    print(f"\n[Model] Creating {args.backbone} (from scratch, no pretrained weights)")
    model = backbone_fn(num_features=512, fp16=True).to(device)

    print(f"[Model] Creating AdaFace head with {num_classes} classes")
    head = AdaFace(
        embedding_size=512,
        classnum=num_classes,
        m=0.4,
        h=0.333,
        s=64.0,
        t_alpha=0.01,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] Total parameters: {total_params:.2f}M")

    # ============================================================
    # Optimizer & Scheduler
    # ============================================================
    optimizer = torch.optim.SGD(
        [{"params": model.parameters()}, {"params": head.parameters()}],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Step LR scheduler: decay at user-specified milestones
    print(f"[Optim] LR decay milestones: {args.lr_milestones}")
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=0.1
    )

    # Mixed precision scaler
    scaler = GradScaler()

    # ============================================================
    # TensorBoard (optional)
    # ============================================================
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(args.log_dir)
        print(f"[Log] TensorBoard logs: {args.log_dir}")
    except ImportError:
        print("[Log] TensorBoard not installed (pip install tensorboard) - skipping")

    # ============================================================
    # Resume from checkpoint?
    # ============================================================
    start_epoch = 1
    if args.resume:
        start_epoch = load_checkpoint(model, head, optimizer, scaler, args.resume, device)
        # Step scheduler to correct position
        for _ in range(start_epoch - 1):
            scheduler.step()

    # ============================================================
    # Training Loop
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Starting training: {args.epochs} epochs")
    print(f"{'='*60}\n")

    best_accuracy = 0.0
    training_start = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()

        # Train
        avg_loss, accuracy = train_one_epoch(
            model, head, train_loader, optimizer, scaler,
            device, epoch, args.epochs, log_interval=args.log_interval,
            writer=writer, max_batches=args.max_batches,
        )

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy

        if epoch % args.save_every == 0 or epoch == args.epochs:
            checkpoint_path = Path(args.output_dir) / "latest.pth"
            save_checkpoint(
                model, head, optimizer, scaler, epoch, num_classes,
                checkpoint_path, is_best=is_best,
            )

        # Log epoch summary
        if writer is not None:
            writer.add_scalar("epoch/loss", avg_loss, epoch)
            writer.add_scalar("epoch/accuracy", accuracy, epoch)

        epoch_time = (time.time() - epoch_start) / 60
        total_elapsed = (time.time() - training_start) / 3600
        print(f"[Time] Epoch took {epoch_time:.1f} min | "
              f"Total training time: {total_elapsed:.2f} hrs\n")

    # ============================================================
    # Export final model to ONNX
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"[Result] Best accuracy: {best_accuracy:.2f}%")
    print(f"[Result] Total time: {(time.time() - training_start) / 3600:.2f} hours")

    onnx_path = Path(args.output_dir) / f"{args.backbone}_adaface.onnx"
    print(f"\n[Export] Exporting to ONNX...")
    export_to_onnx(model, onnx_path, device)

    print(f"\n[Next steps]")
    print(f"  1. Test the model: python fine_tuning/evaluate.py")
    print(f"  2. Deploy: copy {onnx_path} to models/ and update config.yaml")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
