"""
PLAN C: Train EdgeFace from Scratch
====================================

USE THIS IF:
- Plan A (IResNet-50 from scratch) fails AND
- Plan B (fine-tune pretrained) is not allowed by your professor

WHAT THIS DOES:
- Trains EdgeFace (~1.77M params) from scratch on MS1M
- Uses CCTV-style augmentation
- ~10 hours on RTX 5060
- Expected accuracy: 88-93%

USAGE:
    python fine_tuning/plan_c_train_edgeface.py \
        --data "C:/Users/abdul/OneDrive/Desktop/MS-Celeb-1M/ms1m-arcface" \
        --epochs 20 \
        --batch-size 256 \
        --lr 0.1 \
        --max-images-per-id 30

NOTE: EdgeFace is small enough that batch_size 256 fits on RTX 5060.
LR 0.1 works because the model is small and stable.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms

from models.edgeface import edgeface_xs
from models.adaface import AdaFaceHead
from train_from_scratch import FaceDataset
from plan_b_finetune_pretrained import CCTVAugmentation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=[10, 15, 18])
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--max-images-per-id", type=int, default=30)
    parser.add_argument("--output-dir", default="fine_tuning/checkpoints/edgeface")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--use-cctv-aug", action="store_true", help="Use CCTV augmentation")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device}")
    if device.type == "cuda":
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")

    # Dataset
    print(f"\n[Dataset] Scanning {args.data}...")
    dataset = FaceDataset(
        root_dir=args.data,
        image_size=112,
        train=True,
        max_images_per_id=args.max_images_per_id,
    )
    if args.use_cctv_aug:
        dataset.transform = CCTVAugmentation(image_size=112)
        print("[Aug] Using CCTV-style augmentation")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    num_classes = len(dataset.identities)
    print(f"[Data] {num_classes} identities, {len(dataset)} images")

    # Model
    print(f"\n[Model] Creating EdgeFace-XS (from scratch)")
    model = edgeface_xs(num_features=512, fp16=True).to(device)
    head = AdaFaceHead(embedding_size=512, num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] Backbone params: {total_params:.2f}M")

    # Optimizer
    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(head.parameters()),
        lr=args.lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=0.1
    )
    scaler = GradScaler("cuda")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\nStarting EdgeFace training: {args.epochs} epochs\n{'=' * 60}\n")

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

        scheduler.step()
        ckpt_path = Path(args.output_dir) / f"edgeface_epoch{epoch}.pth"
        torch.save({
            "model": model.state_dict(),
            "head": head.state_dict(),
            "epoch": epoch,
        }, ckpt_path)
        print(f"[Save] {ckpt_path}")

    # Export ONNX
    print(f"\n[ONNX] Exporting...")
    model.eval()
    dummy = torch.randn(1, 3, 112, 112).to(device)
    onnx_path = Path(args.output_dir) / "edgeface_adaface.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=14,
    )
    print(f"[ONNX] Saved -> {onnx_path}")
    print(f"\n✅ EdgeFace training complete!")


if __name__ == "__main__":
    main()
