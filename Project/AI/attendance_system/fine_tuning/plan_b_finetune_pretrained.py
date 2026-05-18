"""
PLAN B: Fine-tune a Pretrained AdaFace IResNet-50 Model
========================================================

USE THIS IF:
- Plan A (train_from_scratch.py) fails or accuracy plateaus
- Fine-tuning is allowed by your professor (you confirmed it is)

WHAT THIS DOES:
- Downloads/loads a pretrained AdaFace IResNet-50 (already trained on MS1M by the authors)
- Fine-tunes it for ~2 hours on your dataset with CCTV-style augmentation
- Final accuracy: 95-98% expected

PREREQUISITES:
1. Download pretrained weights from:
   https://github.com/mk-minchul/AdaFace
   File: adaface_ir50_ms1mv2.ckpt (or .pth)
   Save it to: fine_tuning/checkpoints/pretrained/adaface_ir50_ms1mv2.ckpt

USAGE:
    python fine_tuning/plan_b_finetune_pretrained.py \
        --data "C:/Users/abdul/OneDrive/Desktop/MS-Celeb-1M/ms1m-arcface" \
        --pretrained "fine_tuning/checkpoints/pretrained/adaface_ir50_ms1mv2.ckpt" \
        --epochs 3 \
        --batch-size 128 \
        --lr 0.001 \
        --max-images-per-id 20

EXPECTED RUNTIME: ~2 hours on RTX 5060
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms

# Reuse existing modules
from models.iresnet import iresnet50
from models.adaface import AdaFaceHead
from train_from_scratch import FaceDataset  # Reuse the dataset class


# ============================================================
# CCTV-style augmentation (the valuable part from external suggestion)
# ============================================================
class CCTVAugmentation:
    """Simulates CCTV conditions: blur, low res, occlusion."""

    def __init__(self, image_size=112):
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            # Simulate motion blur (students moving)
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
            ], p=0.4),
            # Simulate low-res CCTV
            transforms.RandomApply([
                transforms.Resize(56),
                transforms.Resize(image_size),
            ], p=0.3),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            # Random occlusion (hands, hair, masks)
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
        ])

    def __call__(self, img):
        return self.transform(img)


def load_pretrained(model, pretrained_path):
    """Load pretrained AdaFace weights into IResNet-50."""
    print(f"[Pretrained] Loading from {pretrained_path}")
    ckpt = torch.load(pretrained_path, map_location="cpu")

    # AdaFace checkpoints typically have 'state_dict' key with 'model.' prefix
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Strip prefixes like 'model.' or 'backbone.'
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("model.", "").replace("backbone.", "")
        cleaned[new_key] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"[Pretrained] Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    if len(missing) > 10:
        print(f"[WARN] Too many missing keys — check checkpoint format!")
        print(f"  First 5 missing: {missing[:5]}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to MS1M dataset")
    parser.add_argument("--pretrained", required=True, help="Path to pretrained .ckpt/.pth")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tune is fast — 2-3 epochs is enough")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001, help="LOW LR for fine-tuning (10-100x smaller than from-scratch)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-images-per-id", type=int, default=20)
    parser.add_argument("--output-dir", default="fine_tuning/checkpoints/finetuned")
    parser.add_argument("--log-interval", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # Dataset with CCTV augmentation
    print(f"\n[Dataset] Scanning {args.data}...")
    dataset = FaceDataset(
        root_dir=args.data,
        image_size=112,
        train=True,
        max_images_per_id=args.max_images_per_id,
    )
    # Override transform with CCTV augmentation
    dataset.transform = CCTVAugmentation(image_size=112)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    num_classes = len(dataset.identities)
    print(f"[Data] {num_classes} identities, {len(dataset)} images")

    # Model: Load pretrained backbone
    print(f"\n[Model] Creating IResNet-50 backbone")
    model = iresnet50(num_features=512, fp16=True).to(device)
    model = load_pretrained(model, args.pretrained)

    # Fresh AdaFace head (because num_classes is different from pretrained)
    head = AdaFaceHead(embedding_size=512, num_classes=num_classes).to(device)

    # Optimizer: low LR for backbone, higher for new head
    optimizer = torch.optim.SGD([
        {"params": model.parameters(), "lr": args.lr},
        {"params": head.parameters(), "lr": args.lr * 10},  # Head is fresh, needs more learning
    ], momentum=0.9, weight_decay=5e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(loader)
    )
    scaler = GradScaler("cuda")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\nStarting fine-tuning: {args.epochs} epochs\n{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        total_loss = 0
        correct = 0
        total = 0
        start = time.time()

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast("cuda"):
                features = model(images)
                logits = head(features, labels)
                loss = nn.functional.cross_entropy(logits, labels)

            if not torch.isfinite(loss):
                print(f"[WARN] NaN at batch {batch_idx}, skipping")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % args.log_interval == 0:
                elapsed = time.time() - start
                img_per_sec = (batch_idx + 1) * args.batch_size / elapsed
                lr_now = optimizer.param_groups[0]["lr"]
                print(f"[Epoch {epoch}/{args.epochs}] Batch {batch_idx + 1}/{len(loader)} "
                      f"| Loss: {total_loss / (batch_idx + 1):.4f} "
                      f"| Acc: {100 * correct / total:.2f}% "
                      f"| LR: {lr_now:.5f} "
                      f"| {img_per_sec:.1f} img/s")

        # Save checkpoint each epoch
        ckpt_path = Path(args.output_dir) / f"finetuned_epoch{epoch}.pth"
        torch.save({
            "model": model.state_dict(),
            "head": head.state_dict(),
            "epoch": epoch,
        }, ckpt_path)
        print(f"[Save] Checkpoint -> {ckpt_path}")

    # Export to ONNX
    print(f"\n[ONNX] Exporting...")
    model.eval()
    dummy = torch.randn(1, 3, 112, 112).to(device)
    onnx_path = Path(args.output_dir) / "iresnet50_adaface_finetuned.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=14,
    )
    print(f"[ONNX] Saved -> {onnx_path}")
    print(f"\n✅ Fine-tuning complete!")


if __name__ == "__main__":
    main()
