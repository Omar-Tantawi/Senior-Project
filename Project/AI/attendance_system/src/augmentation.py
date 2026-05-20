"""
src/augmentation.py — Augmentation Pipeline

Designed specifically for:
  1. Wall-mounted angled camera       → rotation, perspective warp
  2. Low / uneven classroom lighting  → brightness, contrast, CLAHE, shadows
  3. Multiple faces in frame          → random crop, scale variation
  4. Partial occlusion                → random erasing (masks/glasses/hair)

Used during training via PyTorch Dataset.
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── Training Augmentation Pipeline ──────────────────────────────────────────
def get_train_transforms(image_size: int = 112) -> A.Compose:
    """
    Heavy augmentation for training robustness.
    Each transform is annotated with which challenge it addresses.
    """
    return A.Compose([

        # ── Spatial: Wall-mounted camera angle ──────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.Rotate(
            limit=25,                       # ±25° for head tilt correction
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.6
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.15,
            rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4
        ),
        # Perspective warp simulates camera angle (wall-mounted)
        A.Perspective(
            scale=(0.02, 0.08),
            p=0.3
        ),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.75, 1.0),             # Scale variation from distance
            ratio=(0.9, 1.1),
            p=0.5
        ),

        # ── Photometric: Lighting conditions ────────────────────────────────
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.4,       # Handles dark corners of classroom
                contrast_limit=0.4,
                p=1.0
            ),
            A.RandomGamma(gamma_limit=(60, 140), p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),

        # Shadow simulation (overhead classroom lighting)
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 2),
            shadow_dimension=4,
            p=0.3
        ),

        # Color/tone shifts
        A.HueSaturationValue(
            hue_shift_limit=15,
            sat_shift_limit=30,
            val_shift_limit=30,
            p=0.4
        ),
        A.RGBShift(
            r_shift_limit=15,
            g_shift_limit=15,
            b_shift_limit=15,
            p=0.3
        ),

        # ── Blur & Noise: Camera quality / motion ───────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),   # Student moving
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),

        # ── Occlusion: Masks, glasses, hair ─────────────────────────────────
        A.CoarseDropout(
            max_holes=6,
            max_height=int(image_size * 0.2),    # Up to 20% of face height
            max_width=int(image_size * 0.2),
            min_holes=1,
            min_height=int(image_size * 0.05),
            min_width=int(image_size * 0.05),
            fill_value=0,
            p=0.35
        ),

        # ── Normalize & convert ──────────────────────────────────────────────
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


# ─── Validation / Inference Transform (no augmentation) ──────────────────────
def get_val_transforms(image_size: int = 112) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


# ─── Enrollment Transform (for adding new students) ───────────────────────────
def get_enrollment_transforms(image_size: int = 112) -> A.Compose:
    """
    Light augmentation used when enrolling a new student.
    Generates 10× the images from a few enrollment photos.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.CLAHE(clip_limit=3.0, p=0.4),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        A.CoarseDropout(max_holes=3, max_height=16, max_width=16, p=0.2),
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ])


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────
class FaceDataset(torch.utils.data.Dataset):
    """
    Dataset that loads preprocessed face images.
    Directory structure:
        root/
          identity_name/
            img1.jpg
            img2.jpg
    """

    def __init__(self, root_dir: str, transform=None, mode: str = "train"):
        from pathlib import Path
        self.root_dir  = Path(root_dir)
        self.transform = transform
        self.mode      = mode
        self.samples   = []
        self.labels    = []
        self.class_to_idx = {}

        identity_dirs = sorted([
            d for d in self.root_dir.iterdir() if d.is_dir()
        ])

        for idx, identity_dir in enumerate(identity_dirs):
            self.class_to_idx[identity_dir.name] = idx
            for img_path in identity_dir.glob("*.jpg"):
                self.samples.append(img_path)
                self.labels.append(idx)
            for img_path in identity_dir.glob("*.png"):
                self.samples.append(img_path)
                self.labels.append(idx)

        self.num_classes = len(self.class_to_idx)
        print(f"[Dataset] {mode}: {len(self.samples)} images | {self.num_classes} identities")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label    = self.labels[idx]

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            # Return black image if corrupted
            img_rgb = np.zeros((112, 112, 3), dtype=np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img_rgb)
            img_tensor = augmented["image"]
        else:
            img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0

        return img_tensor, label


# ─── Visualize augmentations (for debugging) ──────────────────────────────────
def visualize_augmentations(image_path: str, n: int = 8):
    """Show augmented versions of one image — useful for sanity checking."""
    import matplotlib.pyplot as plt

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = get_train_transforms()
    fig, axes = plt.subplots(2, n // 2, figsize=(16, 6))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        aug = transform(image=img)["image"]
        # Denormalize for display
        aug_np = aug.permute(1, 2, 0).numpy()
        aug_np = aug_np * 0.5 + 0.5
        aug_np = np.clip(aug_np, 0, 1)
        ax.imshow(aug_np)
        ax.axis("off")
        ax.set_title(f"Aug {i+1}", fontsize=8)

    plt.suptitle("Training Augmentations Preview", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/augmentation_preview.png", dpi=120)
    plt.show()
    print("[OK] Saved to outputs/augmentation_preview.png")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        visualize_augmentations(sys.argv[1])
    else:
        print("Usage: python src/augmentation.py path/to/face.jpg")
