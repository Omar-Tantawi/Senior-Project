"""
Fine-tunes yolo11n.pt on the merged SCB 6-class student behavior dataset.
Optimised for an RTX 5060 (8 GB VRAM) with fp16 mixed precision.

Run on the GPU machine:
    python -m attention.training.train_behavior

After training, copy the output best.pt to:
    attention/weights/behavior_best.pt
"""

from pathlib import Path

from config import (
    MERGED_DATA_DIR,
    WEIGHTS_DIR,
    BEHAVIOR_BASE_WEIGHTS,
)


def train(
    data_yaml: Path | None = None,
    base_weights: str = BEHAVIOR_BASE_WEIGHTS,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 32,
    device: str = "cuda",
    workers: int = 8,
    name: str = "behavior_v1",
    patience: int = 20,
    cls: float = 1.5,
) -> Path:
    """
    Fine-tunes the behavior detector. Returns path to best.pt.

    Key training decisions for academic defense:
    - cls=1.5: higher classification loss to counter the discuss-class imbalance (9:1 ratio)
    - degrees=10: rotation augment mirrors CCTV mounting angle variance
    - half=True: fp16 on RTX 5060 halves memory and speeds up training ~1.8x
    - workers=0 on Windows to avoid DataLoader multiprocessing spawn errors
    """
    from ultralytics import YOLO

    if data_yaml is None:
        data_yaml = MERGED_DATA_DIR / "dataset.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Merged dataset YAML not found at {data_yaml}. "
            "Run `python -m attention.data.merge_dataset` first."
        )

    model = YOLO(base_weights)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers if device != "cpu" else 0,
        half=True,
        cache=True,        # cache images in RAM — skips disk I/O from epoch 2 onward
        patience=patience,
        project=str(WEIGHTS_DIR),
        name=name,
        save=True,
        val=True,
        plots=True,
        verbose=True,
        # Class imbalance: raise classification loss gain
        cls=cls,
        # Augmentation tuned for surveillance-angle classroom footage
        degrees=10,
        translate=0.1,
        scale=0.3,
        flipud=0.0,       # students are never upside-down
        fliplr=0.5,
        mosaic=1.0,
        auto_augment="randaugment",
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best_pt}")
    print(f"mAP@50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    return best_pt


def validate(weights_path: Path | None = None, data_yaml: Path | None = None) -> dict:
    """
    Validates the trained model on the merged val split.
    Returns Ultralytics results_dict with mAP50, mAP50-95 per class.
    """
    from ultralytics import YOLO

    if weights_path is None:
        weights_path = WEIGHTS_DIR / "behavior_best.pt"
    if data_yaml is None:
        data_yaml = MERGED_DATA_DIR / "dataset.yaml"

    model = YOLO(str(weights_path))
    results = model.val(data=str(data_yaml), split="val", verbose=True)
    return results.results_dict


if __name__ == "__main__":
    best = train(
        data_yaml=Path("C:/ML/sms/data.yaml"),
        base_weights="yolo26s.pt",
        name="sms_v1",               # binary: High Attention / Low Attention
        imgsz=640,                   # 993 images total — 640px is sufficient
        batch=16,                    # small dataset, fits easily in VRAM at 640px
        workers=0,
        cls=1.5,
        patience=20,
    )
    print("\nRunning validation on best weights...")
    metrics = validate(best)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
