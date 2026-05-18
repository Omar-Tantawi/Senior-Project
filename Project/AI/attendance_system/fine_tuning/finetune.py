"""ArcFace Fine-Tuning Script.

Fine-tunes a pretrained ArcFace (ResNet-50) model on a custom face dataset
using Angular Margin (ArcFace) loss in PyTorch.

How it works:
    1. Loads the pretrained ArcFace ONNX weights into a PyTorch ResNet-50
    2. Replaces the classification head with a new ArcFace head for N identities
    3. Trains with ArcFace loss (angular margin softmax)
    4. Exports the fine-tuned model back to ONNX format

Prerequisites:
    pip install torch torchvision
    python prepare_dataset.py --input <raw_data> --output <aligned_data>

Usage:
    python finetune.py --data path/to/aligned_dataset --epochs 20
    python finetune.py --data path/to/aligned_dataset --epochs 10 --lr 0.001 --batch-size 64
    python finetune.py --data path/to/aligned_dataset --epochs 20 --resume checkpoints/latest.pth
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms
except ImportError:
    print("[ERROR] PyTorch is required for fine-tuning.")
    print("  Install it with: pip install torch torchvision")
    print("  Visit https://pytorch.org/get-started/locally/ for GPU version")
    sys.exit(1)

import cv2


# ============================================================
#  Dataset
# ============================================================

class FaceDataset(Dataset):
    """Face dataset for fine-tuning.

    Expects directory structure from prepare_dataset.py:
        aligned_dataset/
            000000/
                0000.jpg
                0001.jpg
            000001/
                0000.jpg
                0001.jpg
            label_map.json
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # list of (image_path, label)

        # Find all identity directories (numeric or string names)
        # This supports both:
        # - Numeric format: 000000/, 000001/ (from prepare_dataset.py)
        # - String format: pins_Adriana Lima/, pins_Alex Lawther/ (from raw PINS)
        identity_dirs = sorted([
            d for d in self.root_dir.iterdir()
            if d.is_dir()
        ])

        for label, identity_dir in enumerate(identity_dirs):
            for img_file in sorted(identity_dir.glob("*.jpg")):
                self.samples.append((str(img_file), label))

        # Determine number of classes
        self.num_classes = len(identity_dirs) if identity_dirs else 0

        print(f"[Dataset] Loaded {len(self.samples)} images, {self.num_classes} identities")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Read image
        img = cv2.imread(img_path)
        if img is None:
            # Return a blank image if read fails
            img = np.zeros((112, 112, 3), dtype=np.uint8)

        # Resize to 112x112 if needed (handles both aligned images and raw images)
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            # Default: normalize to [-1, 1], convert to tensor
            img = img.astype(np.float32) / 127.5 - 1.0
            img = torch.from_numpy(img.transpose(2, 0, 1))  # HWC -> CHW

        return img, label


# ============================================================
#  Data Augmentation
# ============================================================

def get_train_transforms():
    """Training data augmentation pipeline.

    Since faces are already aligned to 112x112, we apply subtle augmentations
    that preserve facial structure but improve robustness.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1,
        ),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomAffine(
            degrees=10,        # slight rotation
            translate=(0.05, 0.05),  # slight shift
            scale=(0.95, 1.05),      # slight zoom
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
    ])


def get_val_transforms():
    """Validation transforms (no augmentation, just normalize)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ============================================================
#  ResNet-50 Backbone (ArcFace architecture)
# ============================================================

class BottleneckIR(nn.Module):
    """Improved Residual Bottleneck block (used by ArcFace ResNet-50)."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.shortcut_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels or stride != 1 else nn.Identity()

        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class ArcFaceResNet50(nn.Module):
    """ResNet-50 backbone for ArcFace (IResNet-50 architecture).

    Input: (B, 3, 112, 112) normalized face images
    Output: (B, 512) L2-normalized face embeddings
    """

    def __init__(self):
        super().__init__()

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )

        # Residual blocks: [3, 4, 14, 3] for ResNet-50
        # Channel progression: 64 -> 64 -> 128 -> 256 -> 512
        self.layer1 = self._make_layer(64, 64, 3, stride=2)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 14, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # Output layer
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = [BottleneckIR(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(BottleneckIR(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.output_layer(x)
        return x


# ============================================================
#  ArcFace Loss (Angular Margin Softmax)
# ============================================================

class ArcFaceLoss(nn.Module):
    """ArcFace loss: Additive Angular Margin Loss.

    Math:
        L = -log( exp(s * cos(theta_yi + m)) /
                  (exp(s * cos(theta_yi + m)) + sum_j!=yi exp(s * cos(theta_j))) )

    Where:
        theta = angle between feature embedding and class center weight
        m = angular margin (default 0.5 radians)
        s = feature scale (default 64)

    This forces the model to learn more discriminative embeddings by adding
    an angular margin penalty to the correct class.
    """

    def __init__(self, embedding_dim: int, num_classes: int,
                 s: float = 64.0, m: float = 0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.num_classes = num_classes

        # Class center weights (one 512-dim vector per identity)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        # Precompute cos(m) and sin(m)
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)  # threshold
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: (B, 512) face embeddings (not L2-normalized yet)
            labels: (B,) ground truth class labels

        Returns:
            loss: scalar ArcFace loss
        """
        # L2 normalize embeddings and weights
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: cos(theta)
        cosine = F.linear(embeddings_norm, weight_norm)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt(1.0 - cosine.pow(2))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # When cos(theta) < cos(pi - m), use fallback
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to the target class
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        # Cross entropy loss
        loss = F.cross_entropy(output, labels)
        return loss


# ============================================================
#  Load Pretrained Weights from ONNX
# ============================================================

def load_onnx_weights_to_pytorch(model: ArcFaceResNet50, onnx_path: str) -> bool:
    """Attempt to load pretrained ArcFace ONNX weights into PyTorch model.

    This maps ONNX weight names to our PyTorch model's state dict.
    May not map 100% of weights due to naming differences, but the backbone
    weights (convolutions, batch norms) are what matter most.

    Returns:
        True if weights were loaded (even partially), False if failed
    """
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        print("[WARNING] 'onnx' package not installed. Cannot load pretrained weights.")
        print("  Install with: pip install onnx")
        print("  Training will start from random initialization.")
        return False

    print(f"[Model] Loading pretrained weights from {onnx_path}...")

    try:
        onnx_model = onnx.load(onnx_path)
    except Exception as e:
        print(f"[WARNING] Could not load ONNX model: {e}")
        return False

    # Extract weights from ONNX
    onnx_weights = {}
    for initializer in onnx_model.graph.initializer:
        onnx_weights[initializer.name] = numpy_helper.to_array(initializer)

    # Get PyTorch state dict
    state_dict = model.state_dict()
    matched = 0
    total = len(state_dict)

    # Try to match by shape (when names don't match)
    # Group PyTorch params by shape for shape-based matching
    pytorch_by_shape = {}
    for name, param in state_dict.items():
        shape = tuple(param.shape)
        if shape not in pytorch_by_shape:
            pytorch_by_shape[shape] = []
        pytorch_by_shape[shape].append(name)

    # Group ONNX weights by shape
    onnx_by_shape = {}
    for name, weight in onnx_weights.items():
        shape = tuple(weight.shape)
        if shape not in onnx_by_shape:
            onnx_by_shape[shape] = []
        onnx_by_shape[shape].append((name, weight))

    # Match weights by shape (in order)
    new_state_dict = {}
    for shape in pytorch_by_shape:
        if shape in onnx_by_shape:
            pt_names = pytorch_by_shape[shape]
            ox_entries = onnx_by_shape[shape]
            for i, pt_name in enumerate(pt_names):
                if i < len(ox_entries):
                    new_state_dict[pt_name] = torch.from_numpy(ox_entries[i][1].copy())
                    matched += 1

    if matched > 0:
        model.load_state_dict(new_state_dict, strict=False)
        print(f"[Model] Loaded {matched}/{total} weight tensors from pretrained model")
        return True
    else:
        print("[WARNING] Could not match any weights. Training from scratch.")
        return False


# ============================================================
#  Training Loop
# ============================================================

def train_one_epoch(model, arcface_loss, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    arcface_loss.train()

    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        embeddings = model(images)
        loss = arcface_loss(embeddings, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stats
        total_loss += loss.item()
        batch_count += 1

        # Accuracy (using cosine similarity to class centers)
        with torch.no_grad():
            emb_norm = F.normalize(embeddings, p=2, dim=1)
            w_norm = F.normalize(arcface_loss.weight, p=2, dim=1)
            logits = F.linear(emb_norm, w_norm)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            avg_loss = total_loss / batch_count
            acc = 100.0 * correct / total if total > 0 else 0
            print(f"  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.1f}%")

    avg_loss = total_loss / max(batch_count, 1)
    acc = 100.0 * correct / total if total > 0 else 0
    return avg_loss, acc


def validate(model, arcface_loss, dataloader, device):
    """Validate the model."""
    model.eval()
    arcface_loss.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)

            # Compute loss
            emb_norm = F.normalize(embeddings, p=2, dim=1)
            w_norm = F.normalize(arcface_loss.weight, p=2, dim=1)
            cosine = F.linear(emb_norm, w_norm)
            loss = F.cross_entropy(cosine * arcface_loss.s, labels)

            total_loss += loss.item()
            batch_count += 1

            preds = cosine.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(batch_count, 1)
    acc = 100.0 * correct / total if total > 0 else 0
    return avg_loss, acc


# ============================================================
#  Export to ONNX
# ============================================================

def export_to_onnx(model: ArcFaceResNet50, output_path: str, device):
    """Export fine-tuned model to ONNX format for inference."""
    import sys
    import io

    model.eval()
    dummy_input = torch.randn(1, 3, 112, 112).to(device)

    # Redirect stdout/stderr to avoid Unicode encoding issues on Windows
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=18,  # Use newer opset version
            input_names=["input"],
            output_names=["embedding"],
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    print(f"[Export] Model saved to {output_path}")


# ============================================================
#  Main
# ============================================================

def finetune(data_dir: str, epochs: int = 20, batch_size: int = 64,
             lr: float = 0.01, pretrained_onnx: str = None,
             checkpoint_dir: str = "fine_tuning/checkpoints",
             resume: str = None, val_split: float = 0.1):
    """Run ArcFace fine-tuning.

    Args:
        data_dir: Path to aligned dataset (output of prepare_dataset.py)
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        pretrained_onnx: Path to pretrained ArcFace ONNX model
        checkpoint_dir: Where to save checkpoints
        resume: Path to checkpoint to resume from
        val_split: Fraction of data for validation
    """
    # Device (force CPU if CUDA not fully compatible)
    # Check if CUDA is truly available and working
    use_cuda = False
    if torch.cuda.is_available():
        try:
            # Test CUDA capability
            test_tensor = torch.zeros(1, device="cuda")
            use_cuda = True
        except RuntimeError:
            print("[WARNING] CUDA available but incompatible with this GPU. Using CPU instead.")
            use_cuda = False

    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"\n{'='*50}")
    print(f"  ArcFace Fine-Tuning")
    print(f"{'='*50}")
    print(f"  Device: {device}")
    print(f"  Dataset: {data_dir}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"{'='*50}\n")

    # Create dataset
    full_dataset = FaceDataset(data_dir, transform=None)  # transforms applied below
    num_classes = full_dataset.num_classes

    if num_classes == 0:
        print("[ERROR] No data found. Run prepare_dataset.py first.")
        return

    # Split into train/val
    total = len(full_dataset)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Apply different transforms
    # We wrap the datasets to apply transforms
    class TransformDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            if isinstance(img, torch.Tensor):
                # Already transformed by parent dataset, skip
                return img, label
            if self.transform:
                img_np = img if isinstance(img, np.ndarray) else np.array(img)
                img = self.transform(img_np)
            return img, label

    # Since FaceDataset returns tensors, we need to modify approach
    # Create datasets with transforms directly
    train_ds = FaceDataset(data_dir, transform=get_train_transforms())
    val_ds = FaceDataset(data_dir, transform=get_val_transforms())

    # Split indices
    indices = list(range(total))
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(train_ds, train_indices)
    val_subset = torch.utils.data.Subset(val_ds, val_indices)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    print(f"[Data] Train: {len(train_subset)} images | Val: {len(val_subset)} images")
    print(f"[Data] Classes: {num_classes}")

    # Create model
    model = ArcFaceResNet50().to(device)
    arcface_loss = ArcFaceLoss(
        embedding_dim=512,
        num_classes=num_classes,
        s=64.0,
        m=0.5,
    ).to(device)

    # Load pretrained weights
    if pretrained_onnx and Path(pretrained_onnx).exists():
        load_onnx_weights_to_pytorch(model, pretrained_onnx)
    elif pretrained_onnx:
        print(f"[WARNING] Pretrained model not found: {pretrained_onnx}")
        print("  Training from scratch (this will take longer)")

    # Optimizer: lower LR for pretrained backbone, higher for new head
    backbone_params = list(model.parameters())
    head_params = list(arcface_loss.parameters())

    optimizer = torch.optim.SGD([
        {"params": backbone_params, "lr": lr * 0.1},   # Fine-tune backbone slowly
        {"params": head_params, "lr": lr},               # Train new head faster
    ], momentum=0.9, weight_decay=5e-4)

    # Learning rate scheduler: cosine annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0.0

    if resume and Path(resume).exists():
        print(f"[Resume] Loading checkpoint: {resume}")
        checkpoint = torch.load(resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        arcface_loss.load_state_dict(checkpoint["arcface_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_acc = checkpoint.get("best_acc", 0.0)
        print(f"[Resume] Resuming from epoch {start_epoch}, best acc: {best_acc:.1f}%")

    # Checkpoint directory
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\n[Training] Starting from epoch {start_epoch}...\n")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, arcface_loss, train_loader, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, arcface_loss, val_loader, device)

        # Update LR
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        print(f"\n  Epoch {epoch}/{epochs} ({epoch_time:.1f}s)")
        print(f"    Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"    Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        print(f"    LR: {current_lr:.6f}\n")

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "arcface_state_dict": arcface_loss.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "best_acc": max(best_acc, val_acc),
            "num_classes": num_classes,
        }

        # Save latest
        torch.save(checkpoint, ckpt_dir / "latest.pth")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(checkpoint, ckpt_dir / "best.pth")
            print(f"    >> New best model! Val Acc: {val_acc:.1f}%")

        # Save every 5 epochs
        if epoch % 5 == 0:
            torch.save(checkpoint, ckpt_dir / f"epoch_{epoch:03d}.pth")

    # Export best model to ONNX
    print(f"\n{'='*50}")
    print(f"  Training Complete!")
    print(f"  Best Validation Accuracy: {best_acc:.1f}%")
    print(f"{'='*50}\n")

    # Load best checkpoint and export
    best_ckpt = torch.load(ckpt_dir / "best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    output_onnx = str(ckpt_dir / "arcface_finetuned.onnx")
    export_to_onnx(model, output_onnx, device)

    print(f"\n  To use the fine-tuned model:")
    print(f"    1. Copy {output_onnx} to models/w600k_r50.onnx")
    print(f"    2. Or update config.yaml: rec_model: '{output_onnx}'")
    print(f"\n  To evaluate:")
    print(f"    python fine_tuning/evaluate.py --original models/w600k_r50.onnx --finetuned {output_onnx}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ArcFace on custom face dataset")
    parser.add_argument("--data", required=True,
                        help="Path to aligned dataset (output of prepare_dataset.py)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (default: 20)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size (default: 64, reduce if GPU OOM)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained ONNX model (default: auto-detect)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    args = parser.parse_args()

    # Auto-detect pretrained model
    pretrained = args.pretrained
    if pretrained is None:
        base_dir = Path(__file__).parent.parent
        default_model = base_dir / "models" / "w600k_r50.onnx"
        if default_model.exists():
            pretrained = str(default_model)
            print(f"[Auto] Found pretrained model: {pretrained}")

    finetune(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        pretrained_onnx=pretrained,
        resume=args.resume,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()
