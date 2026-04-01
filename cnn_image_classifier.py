"""
CNN Image Classifier using PyTorch
===================================
Supports:
  - Training on a custom dataset (folder structure: dataset/class_name/*.jpg)
  - Transfer learning with ResNet-18 (fine-tune on your data)
  - Predicting / inferring on a single photo
  - GPU acceleration if available

Usage:
  1. Prepare your dataset:
       dataset/
         cats/   image1.jpg  image2.jpg ...
         dogs/   image1.jpg  image2.jpg ...
         ...

  2. Train:
       python cnn_image_classifier.py --mode train --data_dir dataset --epochs 10

  3. Predict on a photo:
       python cnn_image_classifier.py --mode predict --image path/to/photo.jpg --model_path best_model.pth

  4. Train from scratch (no transfer learning):
       python cnn_image_classifier.py --mode train --data_dir dataset --no_pretrained
"""

import os
import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe on all platforms)
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1.  CONFIGURATION
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE    = 224          # ResNet / standard CNN input size
BATCH_SIZE  = 32
VAL_SPLIT   = 0.2          # 20 % of data used for validation
RANDOM_SEED = 42

print(f"[INFO] Using device: {DEVICE}")


# ─────────────────────────────────────────────
# 2.  DATA TRANSFORMS
# ─────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225]),   # ImageNet std
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

infer_transforms = val_transforms   # same as validation


# ─────────────────────────────────────────────
# 3.  CUSTOM CNN (from scratch)
# ─────────────────────────────────────────────
class CustomCNN(nn.Module):
    """
    A small but effective CNN built from scratch.
    Architecture: Conv → BN → ReLU → Pool  (×3)  → FC → Dropout → FC
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 224×224 → 112×112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112×112 → 56×56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56×56 → 28×28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 28×28 → 14×14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),   # → 256×4×4 = 4096
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ─────────────────────────────────────────────
# 4.  TRANSFER-LEARNING MODEL (ResNet-18)
# ─────────────────────────────────────────────
def build_resnet(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Fine-tune a pretrained ResNet-18 for num_classes."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model   = models.resnet18(weights=weights)

    if pretrained:
        # Freeze all layers except the final FC layer
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─────────────────────────────────────────────
# 5.  TRAINING FUNCTION
# ─────────────────────────────────────────────
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs):
    """
    Standard PyTorch training loop.
    Returns the model with the best validation accuracy.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc       = 0.0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}  " + "-" * 30)
        t0 = time.time()

        for phase in ("train", "val"):
            model.train() if phase == "train" else model.eval()

            running_loss = 0.0
            running_corrects = 0
            total = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    preds   = outputs.argmax(dim=1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss     += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                total            += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc  = running_corrects / total

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            print(f"  {phase.upper():5s} — loss: {epoch_loss:.4f}  acc: {epoch_acc:.4f}")

            if phase == "val":
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc       = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

        print(f"  elapsed: {time.time()-t0:.1f}s")

    print(f"\n[DONE] Best val accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model, history


# ─────────────────────────────────────────────
# 6.  PLOT TRAINING CURVES
# ─────────────────────────────────────────────
def plot_history(history: dict, save_path: str = "training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"],   label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] Training curves saved → {save_path}")


# ─────────────────────────────────────────────
# 7.  INFERENCE ON A SINGLE PHOTO
# ─────────────────────────────────────────────
def predict_image(image_path: str, model_path: str, class_names: list = None):
    """
    Load a saved model and predict the class of a single image.

    Args:
        image_path  : path to the .jpg/.png photo
        model_path  : path to the saved .pth file
        class_names : optional list of class names; if None tries to load from
                      class_names.txt in the same folder as model_path
    """
    # ── Load class names ──────────────────────────────
    if class_names is None:
        names_file = os.path.join(os.path.dirname(model_path), "class_names.txt")
        if os.path.isfile(names_file):
            with open(names_file) as f:
                class_names = [l.strip() for l in f if l.strip()]
        else:
            raise ValueError(
                "class_names not provided and class_names.txt not found. "
                "Pass --class_names or retrain the model."
            )

    num_classes = len(class_names)

    # ── Rebuild model ─────────────────────────────────────
    # Try ResNet first (default); fall back to CustomCNN if mismatch.
    try:
        model = build_resnet(num_classes, pretrained=False)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)
    except RuntimeError:
        model = CustomCNN(num_classes)
        state = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state)

    model.to(DEVICE)
    model.eval()

    # ── Pre-process image ─────────────────────────────────
    img = Image.open(image_path).convert("RGB")
    tensor = infer_transforms(img).unsqueeze(0).to(DEVICE)

    # ── Inference ─────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()

    top5_probs, top5_idx = probs.topk(min(5, num_classes))
    print(f"\n[RESULT] Image: {image_path}")
    print(f"{'Rank':<6} {'Class':<25} {'Confidence':>10}")
    print("-" * 45)
    for rank, (p, idx) in enumerate(zip(top5_probs, top5_idx), 1):
        print(f"{rank:<6} {class_names[idx.item()]:<25} {p.item()*100:>9.2f}%")

    predicted_class = class_names[top5_idx[0].item()]
    confidence      = top5_probs[0].item() * 100
    print(f"\n→ Predicted: {predicted_class}  ({confidence:.1f}% confidence)")
    return predicted_class, confidence


# ─────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="CNN image classifier (train / predict)")
    parser.add_argument("--mode", choices=["train", "predict"], required=True,
                        help="train: train model | predict: classify a single image")

    # Training args
    parser.add_argument("--data_dir",    default="dataset",
                        help="Root folder: dataset/class_name/image.jpg")
    parser.add_argument("--epochs",      type=int, default=20)
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--model_out",   default="best_model.pth",
                        help="Where to save the trained model")
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Train CustomCNN from scratch instead of ResNet")

    # Prediction args
    parser.add_argument("--image",       help="Path to image for prediction")
    parser.add_argument("--model_path",  default="best_model.pth",
                        help="Path to a saved .pth model file")
    parser.add_argument("--class_names", nargs="*",
                        help="List of class names in training order")

    args = parser.parse_args()

    # ── TRAIN ──────────────────────────────────────────────────────────────
    if args.mode == "train":
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(
                f"Dataset folder not found: {args.data_dir}\n"
                "Create a folder with sub-folders per class, e.g.:\n"
                "  dataset/cats/  dataset/dogs/  ..."
            )

        # Load full dataset with training transforms first (will override for val)
        full_dataset = datasets.ImageFolder(args.data_dir, transform=train_transforms)
        class_names  = full_dataset.classes
        num_classes  = len(class_names)
        print(f"[INFO] Classes ({num_classes}): {class_names}")

        # Train / val split
        n_val   = int(len(full_dataset) * VAL_SPLIT)
        n_train = len(full_dataset) - n_val
        torch.manual_seed(RANDOM_SEED)
        train_ds, val_ds = random_split(full_dataset, [n_train, n_val])

        # Apply correct transforms to validation subset
        val_ds.dataset = copy.deepcopy(full_dataset)
        val_ds.dataset.transform = val_transforms

        dataloaders = {
            "train": DataLoader(train_ds, batch_size=args.batch_size,
                                shuffle=True,  num_workers=0, pin_memory=True),
            "val":   DataLoader(val_ds,   batch_size=args.batch_size,
                                shuffle=False, num_workers=0, pin_memory=True),
        }
        print(f"[INFO] Train: {n_train} samples | Val: {n_val} samples")

        # Build model
        if args.no_pretrained:
            print("[INFO] Building CustomCNN from scratch ...")
            model = CustomCNN(num_classes)
        else:
            print("[INFO] Building ResNet-18 with ImageNet weights (transfer learning) ...")
            model = build_resnet(num_classes, pretrained=True)
        model.to(DEVICE)

        # Loss, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()
        # Only optimize parameters that require gradients
        params    = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

        # Train
        model, history = train_model(
            model, dataloaders, criterion, optimizer, scheduler, args.epochs
        )

        # Save model + class names
        torch.save(model.state_dict(), args.model_out)
        names_file = os.path.join(os.path.dirname(args.model_out) or ".", "class_names.txt")
        with open(names_file, "w") as f:
            f.write("\n".join(class_names))
        print(f"[INFO] Model saved  → {args.model_out}")
        print(f"[INFO] Classes saved → {names_file}")

        # Plot curves
        plot_history(history)

    # ── PREDICT ────────────────────────────────────────────────────────────
    elif args.mode == "predict":
        if not args.image:
            parser.error("--image is required for predict mode")
        predict_image(args.image, args.model_path, args.class_names)


if __name__ == "__main__":
    main()
